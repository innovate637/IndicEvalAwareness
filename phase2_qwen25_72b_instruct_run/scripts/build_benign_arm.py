#!/usr/bin/env python3
"""C2 (plan §8.14) — build the benign arm. Implements §3.4 exactly.

The benign arm is HALF the main grid (36,000 of 72,000 generations) and the
entire defence against "the model just became more refusing about everything".
Rev 2 promised benign_200_<lang>.json and shipped no code for it, which meant
the arm got built by hand -- unacceptable for an input the manifest hashes.

§3.4 algorithm, verbatim:
  1. Load Dolly-T, all 6 languages, parallel by doc_id.
  2. Keep only doc_ids present and non-empty in ALL six.
  3. Compute English word-count quartile boundaries of the 200 harmful items.
  4. Stratified-sample 50 benign items from each harmful quartile band, using a
     fixed permutation with seed 2026 (same discipline as D14).
  5. Freeze to data/final_set/benign_200_<lang>.json with itemnum 1..200.

Length stratification matters because refusal and response length are
correlated; an unmatched benign set makes the false-refusal comparison partly
a length comparison.

Required properties (C2):
  * DETERMINISTIC — same seed, same inputs, byte-identical output. Asserted in
    tests/test_build_benign_arm.py. This is a reproducibility claim, not a
    convenience.
  * EMITS A STRATIFICATION REPORT — realised counts per band and the
    harmful-vs-benign English word-count distributions side by side. If it
    silently fails to match, nothing downstream would notice.
  * FAILS LOUDLY ON THIN STRATA — fewer than 50 eligible parallel doc_ids in
    any band stops the build. Never sample with replacement, never borrow from
    an adjacent band.
  * VERIFIES KANNADA COVERAGE at build time rather than assuming it. IndicAlign
    spans 14 Indic languages including Kannada and Dolly-T is one of its splits,
    but verify.

NOTE: requires pyarrow to read data/prompts/Dolly.parquet. Neither environment
on this machine has it (see record.md Q19); install is not authorised.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FINAL = REPO / "data" / "final_set"
DOLLY = REPO / "data" / "prompts" / "Dolly.parquet"

SEED = 2026
LANGS = ["en", "hi", "bn", "ta", "te", "kn"]
N_BANDS = 4
PER_BAND = 50
SOURCE_TAG = "ai4bharat/indic-align/Dolly_T"

# Dolly-T ships language-tagged rows; map study code -> IndicAlign tag.
LANG_TAG = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng",
            "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda"}


class BuildError(RuntimeError):
    pass


def word_count(s: str) -> int:
    return len(s.split())


def quartile_bands(counts: list[int]) -> list[tuple[float, float]]:
    """Closed-open bands from the English word-count quartiles of the harmful
    set. Last band is closed on the right so the maximum lands inside it."""
    s = sorted(counts)
    n = len(s)

    def q(p: float) -> float:
        # nearest-rank, dependency-free and reproducible
        import math
        return float(s[min(n - 1, max(0, math.ceil(p * n) - 1))])

    q1, q2, q3 = q(0.25), q(0.50), q(0.75)
    return [(float("-inf"), q1), (q1, q2), (q2, q3), (q3, float("inf"))]


def band_of(wc: int, bands: list[tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(bands):
        if i == len(bands) - 1:
            if wc > lo:
                return i
        elif lo < wc <= hi if i > 0 else wc <= hi:
            return i
    return len(bands) - 1


def load_dolly() -> dict:
    if not DOLLY.exists():
        raise BuildError(f"{DOLLY} not found")
    try:
        import pandas as pd
    except ImportError as e:
        raise BuildError(f"pandas required: {e}")
    try:
        df = pd.read_parquet(DOLLY)
    except ImportError as e:
        raise BuildError(
            f"reading {DOLLY.name} needs pyarrow, which is not installed in "
            f"either environment on this machine. Install is not authorised "
            f"(CLAUDE.md rule 2). Original error: {e}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harmful", default=str(FINAL / "final_harmful_200_en.json"),
                    help="English harmful set; supplies the quartile boundaries")
    ap.add_argument("--out-dir", default=str(FINAL))
    ap.add_argument("--report", default=str(FINAL / "benign_stratification.json"))
    ap.add_argument("--check", action="store_true",
                    help="build and report, write nothing")
    args = ap.parse_args()

    # ---- 3. English word-count quartile boundaries of the harmful set ------
    harmful = json.loads(Path(args.harmful).read_text())
    h_counts = [word_count(r.get("prompt") or r.get("text") or "")
                for r in harmful]
    bands = quartile_bands(h_counts)
    print(f"harmful n={len(h_counts)}  word-count bands: "
          + ", ".join(f"({lo:.0f},{hi:.0f}]" for lo, hi in bands))

    # ---- 1-2. Dolly-T, parallel by doc_id, non-empty in ALL six -----------
    df = load_dolly()
    cols = set(df.columns)
    lang_col = next((c for c in ("lang", "language", "lang_tag") if c in cols), None)
    text_col = next((c for c in ("text", "prompt", "instruction") if c in cols), None)
    if not lang_col or "doc_id" not in cols or not text_col:
        raise BuildError(f"unexpected Dolly schema: {sorted(cols)}")

    by_lang: dict[str, dict[str, str]] = {}
    for code in LANGS:
        tag = LANG_TAG[code]
        sub = df[df[lang_col] == tag]
        if sub.empty:
            raise BuildError(
                f"Dolly-T has no rows tagged {tag} ({code}). Kannada coverage "
                f"in particular must be VERIFIED, not assumed (C2).")
        by_lang[code] = {r.doc_id: (getattr(r, text_col) or "").strip()
                         for r in sub.itertuples()}
        print(f"  {code} ({tag}): {len(by_lang[code])} rows")

    eligible = [d for d in by_lang["en"]
                if all(by_lang[c].get(d) for c in LANGS)]
    eligible.sort()                                    # determinism before shuffle
    print(f"parallel and non-empty in all six: {len(eligible)}")

    # ---- 4. stratified sample, fixed seed-2026 permutation ---------------
    buckets: dict[int, list[str]] = {i: [] for i in range(N_BANDS)}
    for d in eligible:
        buckets[band_of(word_count(by_lang["en"][d]), bands)].append(d)

    thin = {i: len(v) for i, v in buckets.items() if len(v) < PER_BAND}
    if thin:
        raise BuildError(
            f"thin strata: bands {thin} have fewer than {PER_BAND} eligible "
            f"parallel doc_ids. Stopping. Do NOT sample with replacement and "
            f"do NOT borrow from an adjacent band (C2).")

    rng = random.Random(SEED)
    chosen: list[str] = []
    report_bands = {}
    for i in range(N_BANDS):
        pool = sorted(buckets[i])
        perm = pool[:]
        rng.shuffle(perm)
        take = perm[:PER_BAND]
        chosen.extend(take)
        report_bands[f"band{i}"] = {
            "range": [None if bands[i][0] == float("-inf") else bands[i][0],
                      None if bands[i][1] == float("inf") else bands[i][1]],
            "eligible": len(pool), "taken": len(take),
            "mean_wc": round(sum(word_count(by_lang["en"][d]) for d in take)
                             / len(take), 2)}

    # ---- 5. freeze, itemnum 1..200, §3.1 row schema ----------------------
    out_dir = Path(args.out_dir)
    written = []
    for code in LANGS:
        rows = [{"itemnum": i + 1, "doc_id": d,
                 "lang": code, "text": by_lang[code][d],
                 "prompt": by_lang[code][d],
                 "source": SOURCE_TAG, "harm_category": "benign",
                 "translation_source": "source_en" if code == "en" else "indictrans2",
                 "translation_run_id": f"dolly_t_{SEED}"}
                for i, d in enumerate(chosen)]
        dst = out_dir / f"benign_200_{code}.json"
        if not args.check:
            dst.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        written.append(str(dst.relative_to(REPO)))

    b_counts = [word_count(by_lang["en"][d]) for d in chosen]
    report = {
        "seed": SEED, "n": len(chosen),
        "bands": report_bands,
        "length_match": {
            "harmful_en": {"n": len(h_counts),
                           "mean": round(sum(h_counts) / len(h_counts), 2),
                           "min": min(h_counts), "max": max(h_counts)},
            "benign_en": {"n": len(b_counts),
                          "mean": round(sum(b_counts) / len(b_counts), 2),
                          "min": min(b_counts), "max": max(b_counts)},
        },
        "files": written,
    }
    print(json.dumps(report, indent=2))
    if not args.check:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {len(written)} files + {args.report}")
    else:
        print("\n[check] nothing written")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
