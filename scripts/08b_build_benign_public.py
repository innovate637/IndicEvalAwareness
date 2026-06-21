#!/usr/bin/env python3
"""
Build the BENIGN / "deployment" control set from PUBLIC data only (no API, no MT).

Source: ai4bharat/indic-align  config 'Dolly_T'  (CC-BY-4.0)
  = Dolly-15k benign instructions, professionally TRANSLATED & PARALLEL across languages,
    with the SAME column structure as our harmful Toxic_Matrix arm
    (eng_Latn / hin_Deva / ben_Beng / tam_Taml / tel_Telu / ory_Orya).
  => a CONTENT-MATCHED benign control mirroring the matched harmful set. Same dataset
     family, same provenance, publication-hygiene clean, no API spend.

Output: data/safety_prompts/benign/{lang}.json (schema mirrors data/safety_prompts/):
  [{"text","harm_category":"benign","source":"ai4bharat/indic-align/Dolly_T",
    "id":"{lang}_benign_0000","lang"}], n matched items, aligned by index across languages.

Usage:
  python scripts/08b_build_benign_public.py            # all 6 langs, n=100 matched
  python scripts/08b_build_benign_public.py --n 100 --max-scan 4000
"""
import sys
import ast
import re
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# language code -> Dolly_T native-script column (same mapping as Toxic_Matrix)
COL = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng",
       "ta": "tam_Taml", "te": "tel_Telu", "or": "ory_Orya"}
LANGS = list(COL)

OUT_DIR = config.SAFETY_DIR / "benign"

SCRIPT_RANGE = {"hi": (0x0900, 0x097F), "bn": (0x0980, 0x09FF), "ta": (0x0B80, 0x0BFF),
                "te": (0x0C00, 0x0C7F), "or": (0x0B00, 0x0B7F)}


def first_string(x):
    """Dig to the first non-empty string in a nested list/tuple."""
    if isinstance(x, str):
        return x.strip() or None
    if isinstance(x, (list, tuple)):
        for e in x:
            s = first_string(e)
            if s:
                return s
    return None


def parse_instruction(cell) -> str | None:
    """Cells look like "[['instruction', 'response'], ...]" — return the instruction
    (the user-facing prompt = first string), mirroring load_toxic_matrix's tolerance."""
    if cell is None:
        return None
    s = str(cell).strip()
    if not s:
        return None
    try:
        return first_string(ast.literal_eval(s))
    except (ValueError, SyntaxError):
        m = re.search(r"'([^']{3,}?)'", s) or re.search(r'"([^"]{3,}?)"', s)
        return m.group(1).strip() if m else (s if len(s) < 500 else None)


CONTEXT_REF_RE = re.compile(
    r"this paragraph|the passage|reference text|given this|using example|"
    r"based on the|according to the|following passage|provided text|given information|"
    r"the article|this article|the excerpt|the text (above|below|provided|given)|"
    r"the above|the following text|the story",
    re.IGNORECASE,
)


def has_context_ref(text: str) -> bool:
    """Return True if the instruction references a missing context passage."""
    return bool(CONTEXT_REF_RE.search(text or ""))


def in_script_frac(text: str, lang: str) -> float:
    rng = SCRIPT_RANGE.get(lang)
    if not rng:
        return 1.0
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(rng[0] <= ord(c) <= rng[1] for c in letters) / len(letters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="matched benign items per language")
    ap.add_argument("--max-scan", type=int, default=6000, help="max Dolly_T rows to scan")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset("ai4bharat/indic-align", "Dolly_T", split="train", streaming=True)

    matched = []          # list of {lang: instruction} dicts, all 6 langs present
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > args.max_scan or len(matched) >= args.n:
            break
        # single-turn only (mirror our single-turn harmful prompts)
        nt = str(row.get("num_turns", "1")).strip()
        if nt not in ("1", "1.0"):
            continue
        rec = {}
        ok = True
        for lang in LANGS:
            instr = parse_instruction(row.get(COL[lang]))
            if not instr or len(instr) < 3 or len(instr) > 2000:
                ok = False
                break
            rec[lang] = instr
        # Drop closed_qa / info-extraction items that reference a missing context passage.
        # Check English (the base language) — all other language columns are translations
        # of the same underlying item, so one check is sufficient.
        if ok and has_context_ref(rec.get("en", "")):
            ok = False
        if ok:
            matched.append(rec)

    print(f"scanned {scanned} Dolly_T rows -> {len(matched)} fully-matched benign items\n")

    for lang in LANGS:
        records = [{"text": matched[i][lang], "harm_category": "benign",
                    "source": "ai4bharat/indic-align/Dolly_T",
                    "id": f"{lang}_benign_{i:04d}", "lang": lang}
                   for i in range(len(matched))]
        out = OUT_DIR / f"{lang}.json"
        with open(out, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        fr = (sum(in_script_frac(r["text"], lang) for r in records) / len(records)
              if records else 0.0)
        sample = records[0]["text"][:70] if records else "(none)"
        flag = "" if len(records) >= args.n else "  ⚠ SHORT"
        print(f"  {lang}: {len(records)} items  in-script={fr:.2f}  e.g. {sample!r}{flag}")
        print(f"     → {out}")


if __name__ == "__main__":
    main()
