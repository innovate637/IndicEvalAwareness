#!/usr/bin/env python3
"""Gate J4 completeness audit (plan 7.4).

Reads the six per-language judgment files, verifies the whole 47,880-row grid
was judged exactly once, and on success concatenates them into a single
phase3/j4_results.jsonl for the section 8 analysis.

Plan 7.4 is non-optional and blocking: "The phase does not proceed to section 8
until this passes."

Takes no arguments.  python phase3/j4_audit.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Single source of truth for where the generations live and what the grid is.
from judge import (  # noqa: E402
    EXPECTED_ROWS,
    EXPECTED_ROWS_PER_LANG,
    GEN_ROOTS,
    VALID_LABELS,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = ["en", "hi", "bn", "ta", "te", "kn"]
MODELS = list(GEN_ROOTS)
RESULT_TMPL = os.path.join(REPO, "phase3", "j4_results_{lang}.jsonl")
MERGED_PATH = os.path.join(REPO, "phase3", "j4_results.jsonl")


def load_generation_index():
    """record_id -> (model_slug, lang) for all 47,880 generations."""
    index = {}
    for model, rel in GEN_ROOTS.items():
        root = os.path.join(REPO, rel)
        if not os.path.isdir(root):
            sys.exit(f"FATAL: missing generation root: {root}")
        for dirpath, _, filenames in os.walk(root):
            for fn in sorted(filenames):
                if not fn.endswith(".jsonl"):
                    continue
                with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        index[d["record_id"]] = (d.get("model_slug", model), d["lang"])
    return index


def load_results():
    """Returns (rows_by_lang, missing_files, malformed).

    rows_by_lang maps language -> list of judgment dicts, in file order.
    """
    rows_by_lang, missing, malformed = {}, [], []
    for lang in LANGS:
        path = RESULT_TMPL.format(lang=lang)
        if not os.path.exists(path):
            missing.append(path)
            continue
        rows = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    malformed.append(f"{os.path.basename(path)}:{lineno}: {e}")
        rows_by_lang[lang] = rows
    return rows_by_lang, missing, malformed


def report(check_no, name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {check_no}. {name}" + (f" — {detail}" if detail else ""))
    return ok


def print_distribution(rows_by_lang):
    all_rows = [r for rows in rows_by_lang.values() for r in rows]
    if not all_rows:
        return
    order = ["REFUSAL", "COMPLY", "UNUSABLE", "PARSE_ERROR"]

    def line(label, rows):
        c = Counter(r.get("label") for r in rows)
        n = len(rows)
        cells = "".join(f"{c.get(k, 0):>12}" for k in order)
        rr = c.get("REFUSAL", 0) / n if n else 0
        return f"{label:<26}{n:>8}{cells}{rr:>10.1%}"

    hdr = f"{'':<26}{'n':>8}" + "".join(f"{k:>12}" for k in order) + f"{'refusal':>10}"
    print("\n--- label distribution ---")
    print(hdr)
    print("-" * len(hdr))
    print(line("TOTAL", all_rows))

    print("\nby model:")
    for m in MODELS:
        print(line("  " + m, [r for r in all_rows if r.get("model_slug") == m]))

    print("\nby language:")
    for l in LANGS:
        print(line("  " + l, rows_by_lang.get(l, [])))

    print("\nby model x language:")
    for m in MODELS:
        for l in LANGS:
            sub = [r for r in rows_by_lang.get(l, []) if r.get("model_slug") == m]
            print(line(f"  {m} / {l}", sub))


def print_span_summary(rows_by_lang):
    """Check 8, informational only. Plan 5.4 re-judging happens after section 8."""
    all_rows = [r for rows in rows_by_lang.values() for r in rows]
    bad = [r for r in all_rows if r.get("evidence_span_in_window") is False]
    print("\n--- evidence_span not a substring of its window (informational) ---")
    if not all_rows:
        print("  no rows")
        return len(bad)
    print(f"  total: {len(bad)} / {len(all_rows)} = {len(bad) / len(all_rows):.2%}")
    print(f"\n  {'language':<12}{'n':>8}{'fails':>8}{'rate':>9}")
    for l in LANGS:
        rows = rows_by_lang.get(l, [])
        b = sum(1 for r in rows if r.get("evidence_span_in_window") is False)
        rate = b / len(rows) if rows else 0
        print(f"  {l:<12}{len(rows):>8}{b:>8}{rate:>9.2%}")
    print(f"\n  {'model':<26}{'n':>8}{'fails':>8}{'rate':>9}")
    for m in MODELS:
        rows = [r for r in all_rows if r.get("model_slug") == m]
        b = sum(1 for r in rows if r.get("evidence_span_in_window") is False)
        rate = b / len(rows) if rows else 0
        print(f"  {m:<26}{len(rows):>8}{b:>8}{rate:>9.2%}")
    print("\n  Plan 5.4 requires re-judging these once. Not a gate failure.")
    return len(bad)


def main():
    print("=" * 78)
    print("GATE J4 COMPLETENESS AUDIT (plan 7.4)")
    print("=" * 78)

    rows_by_lang, missing, malformed = load_results()

    if missing:
        print("\nFATAL: result files not found — J4 has not finished:")
        for p in missing:
            print(f"  missing: {p}")
        print("\nGATE J4-AUDIT: FAIL")
        return 1
    if malformed:
        print(f"\nFATAL: {len(malformed)} malformed JSON line(s):")
        for m in malformed[:10]:
            print(f"  {m}")
        print("\nGATE J4-AUDIT: FAIL")
        return 1

    for lang in LANGS:
        n = len(rows_by_lang[lang])
        flag = "" if n == EXPECTED_ROWS_PER_LANG else f"  <-- expected {EXPECTED_ROWS_PER_LANG}"
        print(f"  loaded j4_results_{lang}.jsonl: {n} rows{flag}")

    print("\nloading generation index...")
    gen_index = load_generation_index()
    print(f"  {len(gen_index)} generation record_ids")

    all_rows = [r for rows in rows_by_lang.values() for r in rows]
    checks = []

    print("\n--- checks ---")

    # 1. total rows
    total = len(all_rows)
    checks.append(report(1, "TOTAL ROWS", total == EXPECTED_ROWS,
                         f"{total} of {EXPECTED_ROWS}"))

    # 3. disjoint (computed before 2, since 2 depends on the id set)
    id_counts = Counter(r["record_id"] for r in all_rows)
    dupes = {k: v for k, v in id_counts.items() if v > 1}
    checks.append(report(3, "DISJOINT (no record_id twice)", not dupes,
                         "no duplicates" if not dupes
                         else f"{len(dupes)} duplicated, e.g. {list(dupes)[:3]}"))

    # 2. exhaustive
    judged, expected = set(id_counts), set(gen_index)
    unjudged = expected - judged
    exhaustive = not unjudged and not dupes
    checks.append(report(2, "EXHAUSTIVE (every generation judged exactly once)",
                         exhaustive,
                         "all judged once" if exhaustive
                         else f"{len(unjudged)} unjudged, e.g. {sorted(unjudged)[:3]}"))

    # 5. no unknown ids
    unknown = judged - expected
    checks.append(report(5, "NO UNKNOWN IDS", not unknown,
                         "none" if not unknown
                         else f"{len(unknown)} unknown, e.g. {sorted(unknown)[:3]}"))

    # 4. language match — row lang matches its file, and matches the generation
    lang_mismatch, gen_lang_mismatch = [], []
    for lang, rows in rows_by_lang.items():
        for r in rows:
            if r.get("lang") != lang:
                lang_mismatch.append((r["record_id"], r.get("lang"), lang))
            src = gen_index.get(r["record_id"])
            if src and src[1] != r.get("lang"):
                gen_lang_mismatch.append((r["record_id"], r.get("lang"), src[1]))
    lang_ok = not lang_mismatch and not gen_lang_mismatch
    detail = "all rows match their file and their generation"
    if lang_mismatch:
        detail = f"{len(lang_mismatch)} wrong file, e.g. {lang_mismatch[:2]}"
    elif gen_lang_mismatch:
        detail = f"{len(gen_lang_mismatch)} disagree with generation, e.g. {gen_lang_mismatch[:2]}"
    checks.append(report(4, "LANGUAGE MATCH", lang_ok, detail))

    # 6. no parse errors
    parse_errors = [r for r in all_rows if r.get("label") == "PARSE_ERROR"]
    checks.append(report(6, "NO PARSE ERRORS", not parse_errors,
                         "zero" if not parse_errors
                         else f"{len(parse_errors)} rows, e.g. "
                              f"{[r['record_id'] for r in parse_errors[:3]]}"))

    # 7. valid labels
    bad_labels = Counter(r.get("label") for r in all_rows
                         if r.get("label") not in VALID_LABELS)
    checks.append(report(7, "VALID LABELS", not bad_labels,
                         "all in {REFUSAL, COMPLY, UNUSABLE}" if not bad_labels
                         else f"{sum(bad_labels.values())} invalid: {dict(bad_labels)}"))

    # 8. evidence_span — informational, never gates
    print("  [INFO] 8. EVIDENCE SPAN CHECK — reported below, does not gate")

    print_distribution(rows_by_lang)
    print_span_summary(rows_by_lang)

    passed = all(checks)
    print("\n" + "=" * 78)
    print(f"GATE J4-AUDIT: {'PASS' if passed else 'FAIL'}")
    print("=" * 78)

    if not passed:
        print("\nSection 8 must not run. Fix the failures above and re-audit.")
        return 1

    with open(MERGED_PATH, "w", encoding="utf-8") as out:
        n = 0
        for lang in LANGS:
            for r in rows_by_lang[lang]:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"\nwrote {MERGED_PATH} ({n} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
