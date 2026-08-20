#!/usr/bin/env python3
"""C1 (plan §8.14) — normalise delivered translations into the canonical layout.

Rev 2 called this "trivial, not reproduced here". It is not trivial: it is the
only thing standing between the delivered files and G0, and every mismatch it
reconciles is SILENT rather than loud.

Reconciles the §3.1 mismatch table:

  itemnum base   delivered 0..199  ->  canonical 1..200   (renumber HERE, once;
                                       the plan is 1-based in six places)
  field names    {translation, status, cometkiwi, prompt_en, lang}
                                   ->  {itemnum, doc_id, prompt,
                                        translation_source, translation_run_id}
  provenance     absent            ->  stamped from --source
  doc_id align   unenforced        ->  enforced HERE, not only at the gate
                                       (a G0 failure here is expensive to
                                       diagnose)

Plus the two C1 requirements not in that table:
  * IDEMPOTENT     — running twice is a no-op, not a double-renumber. Input is
                     asserted 0-based on entry.
  * HASH-PROTECTED — refuses to overwrite a file whose SHA-256 is already
                     recorded in preflight/manifest.json, so a late
                     re-translation cannot silently invalidate a manifest.

Usage
  # dry run, all languages, from the delivered data/harmful_<lang>.json
  python scripts/normalise_translations.py --check

  # write canonical files
  python scripts/normalise_translations.py --source opus --run-id ta_20260812

  # §3.6 Option A: drop refused doc_ids from ALL languages
  python scripts/normalise_translations.py --source opus --run-id r1 \
      --drop-doc-ids e6efb528-...,b6d11d84-...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FINAL = REPO / "data" / "final_set"
INCOMING = FINAL / "_incoming"

# §3.1 allows opus | indictrans2 | google_translate_manual. `source_en` is a
# fourth value for English only: English is the SOURCE text, not a translation,
# and stamping it "opus" would be a false provenance claim in the manifest.
ALLOWED_SOURCES = {"opus", "indictrans2", "google_translate_manual", "source_en"}

CANONICAL_FIELDS = ["itemnum", "doc_id", "prompt",
                    "translation_source", "translation_run_id"]


class NormaliseError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_hashes() -> dict[str, str]:
    """Every input hash recorded in any manifest we can find."""
    hashes: dict[str, str] = {}
    for man in list(REPO.rglob("preflight/manifest.json")):
        try:
            hashes.update(json.loads(man.read_text()).get("input_files", {}))
        except Exception:
            continue
    return hashes


# --------------------------------------------------------------------------
# source readers
# --------------------------------------------------------------------------
def source_path(lang: str, arm: str) -> Path:
    """Where the delivered file actually lives.

    _incoming/ wins if present (that is the documented drop point); otherwise
    fall back to the repository's current layout.
    """
    if arm == "harmful":
        cands = [INCOMING / f"harmful_{lang}.json",
                 INCOMING / f"final_harmful_200_{lang}.json",
                 REPO / "data" / f"harmful_{lang}.json"]
    else:
        cands = [INCOMING / f"benign_{lang}.json",
                 INCOMING / f"benign_200_{lang}.json",
                 FINAL / f"benign_200_{lang}.json"]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"no delivered file for {arm}/{lang}; tried: "
        + ", ".join(str(c.relative_to(REPO)) for c in cands))


def target_path(lang: str, arm: str) -> Path:
    return FINAL / (f"final_harmful_200_{lang}.json" if arm == "harmful"
                    else f"benign_200_{lang}.json")


def extract_prompt(row: dict) -> tuple[str, str | None]:
    """Return (prompt_text, status). Handles every delivered shape."""
    for key in ("prompt", "translation", "text"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip(), row.get("status")
    return "", row.get("status")


def normalise_rows(rows: list[dict], lang: str, arm: str,
                   source: str, run_id: str,
                   drop: set[str]) -> tuple[list[dict], dict]:
    report: dict = {"n_in": len(rows), "dropped": [], "problems": []}

    inums = [r.get("itemnum") for r in rows]
    if any(not isinstance(i, int) for i in inums):
        raise NormaliseError(f"{arm}/{lang}: non-integer itemnum present")

    lo, hi = min(inums), max(inums)
    already_1based = (lo, hi) == (1, len(rows))
    if not already_1based and (lo, hi) != (0, len(rows) - 1):
        raise NormaliseError(
            f"{arm}/{lang}: itemnum is neither 0-based contiguous nor 1-based "
            f"contiguous: {lo}..{hi} over {len(rows)} rows")
    if already_1based:
        # Idempotence: renumbering a 1-based file would be a double-renumber.
        # But 1-based is not the same as fully normalised -- the delivered
        # benign files are 1-based yet still carry `text` instead of `prompt`
        # and no provenance stamp. So: renumber never happens twice, field
        # normalisation still runs, and a file that is ALREADY canonical
        # round-trips to itself.
        report["renumbered"] = False
        canonical = all(set(CANONICAL_FIELDS) <= set(r) for r in rows)
        report["already_canonical"] = canonical
    else:
        report["renumbered"] = True
        report["already_canonical"] = False
    if len(set(inums)) != len(inums):
        raise NormaliseError(f"{arm}/{lang}: duplicate itemnum")

    rows = sorted(rows, key=lambda r: r["itemnum"])

    out: list[dict] = []
    for r in rows:
        doc_id = r.get("doc_id")
        if not doc_id:
            report["problems"].append(f"itemnum {r.get('itemnum')}: no doc_id")
            continue
        if doc_id in drop:
            report["dropped"].append({"doc_id": doc_id,
                                      "itemnum_0based": r["itemnum"],
                                      "reason": "--drop-doc-ids"})
            continue
        prompt, status = extract_prompt(r)
        if not prompt:
            report["problems"].append(
                f"doc_id {doc_id[:12]} (itemnum {r['itemnum']}): empty prompt"
                + (f", status={status}" if status else ""))
            continue
        if status and status != "ok":
            report["problems"].append(
                f"doc_id {doc_id[:12]} (itemnum {r['itemnum']}): "
                f"status={status} -- not usable, resolve under §3.6")
            continue
        row = {
            "itemnum": len(out) + 1,                  # 1-based, contiguous
            "doc_id": doc_id,
            "prompt": prompt,
            "translation_source": source,
            "translation_run_id": run_id,
        }
        # Carry through any extra delivered fields rather than dropping them.
        # The benign files carry `source` and `harm_category`; the harmful ones
        # carry `cometkiwi`, `prompt_en`, `perm_rank`. These are hashed into the
        # manifest as part of the input, so silently discarding them would
        # destroy provenance that Phase 3 and the paper may need. `translation`
        # and `text` are excluded because they are now `prompt` -- keeping both
        # would leave two copies that could drift apart.
        for k, v in r.items():
            if k not in row and k not in ("translation", "text", "status"):
                row[k] = v
        out.append(row)

    report["n_out"] = len(out)
    return out, report


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="en,hi,bn,ta,te,kn")
    ap.add_argument("--arms", default="harmful")
    ap.add_argument("--source", default="opus",
                    help=f"provenance stamp, one of {sorted(ALLOWED_SOURCES)}")
    ap.add_argument("--run-id", default=None,
                    help="translation_run_id stamp (required unless --check)")
    ap.add_argument("--drop-doc-ids", default="",
                    help="comma-separated doc_ids to drop from EVERY language "
                         "(§3.2 all-or-nothing rule); or a path to a JSON list")
    ap.add_argument("--check", action="store_true",
                    help="validate and report, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="overwrite even if the target hash is in a manifest")
    args = ap.parse_args()

    if args.source not in ALLOWED_SOURCES:
        print(f"--source must be one of {sorted(ALLOWED_SOURCES)}",
              file=sys.stderr)
        return 2
    if not args.check and not args.run_id:
        print("--run-id is required when writing (use --check to dry-run)",
              file=sys.stderr)
        return 2

    drop: set[str] = set()
    if args.drop_doc_ids:
        p = Path(args.drop_doc_ids)
        if p.exists():
            drop = set(json.loads(p.read_text()))
        else:
            drop = {d.strip() for d in args.drop_doc_ids.split(",") if d.strip()}

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    hashed = manifest_hashes()

    all_ok = True
    staged: dict[tuple[str, str], tuple[Path, list[dict]]] = {}
    reports: dict[str, dict] = {}

    for arm in arms:
        for lang in langs:
            tag = f"{arm}/{lang}"
            try:
                src = source_path(lang, arm)
                rows = json.loads(src.read_text())
                source = "source_en" if lang == "en" else args.source
                out, rep = normalise_rows(rows, lang, arm, source,
                                          args.run_id or "DRYRUN", drop)
                rep["source_file"] = str(src.relative_to(REPO))
                reports[tag] = rep
                staged[(arm, lang)] = (target_path(lang, arm), out)
                if rep["problems"]:
                    all_ok = False
            except (NormaliseError, FileNotFoundError, ValueError) as e:
                all_ok = False
                reports[tag] = {"fatal": str(e)}

    # ---- doc_id alignment, enforced HERE and not only at the gate ---------
    for arm in arms:
        sets = {lang: [r["doc_id"] for r in staged[(arm, lang)][1]]
                for lang in langs if (arm, lang) in staged}
        if len(sets) > 1:
            ref_lang = "en" if "en" in sets else sorted(sets)[0]
            ref = sets[ref_lang]
            for lang, ids in sets.items():
                if ids == ref:
                    continue
                all_ok = False
                only_ref = set(ref) - set(ids)
                only_l = set(ids) - set(ref)
                pos = [i for i, (a, b) in enumerate(zip(ref, ids)) if a != b]
                reports.setdefault(f"{arm}/{lang}", {}).setdefault(
                    "problems", []).append(
                    f"doc_id misaligned vs {ref_lang}: "
                    f"{len(only_ref)} only in {ref_lang}, {len(only_l)} only "
                    f"here, {len(pos)} positional mismatches "
                    f"(first at index {pos[0] if pos else 'n/a'})")

    print(json.dumps(reports, indent=2, ensure_ascii=False))

    if args.check:
        print(f"\n[check] {'PASS' if all_ok else 'FAIL'} — nothing written")
        return 0 if all_ok else 1

    if not all_ok:
        print("\n[abort] validation failed; nothing written. Fix the problems "
              "above (or pass --drop-doc-ids for §3.6 drops) and re-run.",
              file=sys.stderr)
        return 1

    for (arm, lang), (dst, out) in sorted(staged.items()):
        rel = str(dst.relative_to(REPO))
        if dst.exists() and rel in hashed and not args.force:
            cur = sha256_file(dst)
            if cur == hashed[rel]:
                print(f"[hash-protected] {rel} is recorded in a manifest; "
                      f"refusing to overwrite (use --force to override)")
                continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
        print(f"[write] {rel}  ({len(out)} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
