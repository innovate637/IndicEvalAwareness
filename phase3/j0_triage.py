#!/usr/bin/env python3
"""Gate J0 — triage flags for the Phase 3 primary set.

Reads the four canonical generation trees, emits one flag record per record_id.
Flags are not mutually exclusive and none of them is a label (plan Sec 3.1).
No model calls, CPU only.

The tail repetition definition below is frozen per plan Sec 3.1: fix it in code,
commit it, never tune it after seeing results.
"""

import json
import os
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GEN_ROOTS = {
    "gemma3-27b-it": "phase2_gemma3_27b_it_run/phase2_scratch/generations/gemma3-27b-it",
    "qwen3-32b": "phase2_qwen3-32b_run/generations",
    "qwen25-72b-instruct": "phase2_qwen25_72b_instruct_run/generations/qwen25-72b-instruct",
    "sarvam-m": "phase2_sarvam-m_run/sarvam-m_run/generations",
}

OUT_PATH = os.path.join(REPO, "phase3", "j0_triage_results.jsonl")

EXPECTED_ROWS = 47880
TAIL_CHARS = 600
WINDOW_SIZES = (30, 60)
FLAG_NAMES = ["ok", "trunc_degenerate", "trunc_clean", "lang_mismatch", "empty", "error"]


def tail_rep_score(text, tail_chars=TAIL_CHARS, windows=WINDOW_SIZES):
    """Max fraction of the final `tail_chars` occupied by a single repeated window.

    For each window size, every distinct substring of that length in the tail is
    counted non-overlappingly; a substring must occur at least twice to count as
    repetition. Score is occurrences * window / len(tail), capped at 1.0.
    """
    if not text:
        return 0.0
    tail = text[-tail_chars:]
    n = len(tail)
    best = 0.0
    for w in windows:
        if n < 2 * w:
            continue
        seen = set()
        for i in range(n - w + 1):
            cand = tail[i : i + w]
            if cand in seen:
                continue
            seen.add(cand)
            count = tail.count(cand)
            if count < 2:
                continue
            frac = count * w / n
            if frac > best:
                best = frac
    return min(best, 1.0)


def compute_flags(row, score):
    flags = []
    truncated = bool(row.get("truncated"))
    answer = row.get("response_answer") or ""
    err = row.get("error")

    if truncated:
        flags.append("trunc_degenerate" if score > 0.5 else "trunc_clean")
    if row.get("response_lang_match") is False:
        flags.append("lang_mismatch")
    if row.get("response_is_empty") is True or answer.strip() == "":
        flags.append("empty")
    if err is not None and str(err).strip() != "":
        flags.append("error")

    if not flags:
        flags.append("ok")
    return flags


def shard_paths(root):
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".jsonl"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def main():
    per_model = defaultdict(Counter)
    totals = Counter()
    hard_drop = Counter()
    hard_drop_ids = []
    seen_ids = set()
    dup_ids = 0
    n_rows = 0

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for model, rel in GEN_ROOTS.items():
            root = os.path.join(REPO, rel)
            if not os.path.isdir(root):
                sys.exit(f"FATAL: missing generation root: {root}")
            paths = shard_paths(root)
            if len(paths) != 60:
                print(f"WARNING: {model} has {len(paths)} shards, expected 60", file=sys.stderr)
            for p in paths:
                with open(p, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        n_rows += 1

                        rid = row["record_id"]
                        if rid in seen_ids:
                            dup_ids += 1
                        seen_ids.add(rid)

                        score = tail_rep_score(row.get("response_answer") or "")
                        flags = compute_flags(row, score)

                        for f in flags:
                            per_model[model][f] += 1
                            totals[f] += 1

                        # Plan Sec 3.2: hard drop is ONLY error rows and
                        # prompt_contains_cue failures.
                        is_err = "error" in flags
                        cue_fail = row.get("prompt_contains_cue") is False
                        if is_err:
                            hard_drop["error"] += 1
                        if cue_fail:
                            hard_drop["prompt_contains_cue_fail"] += 1
                        if is_err or cue_fail:
                            hard_drop["total"] += 1
                            hard_drop_ids.append(rid)

                        out.write(
                            json.dumps(
                                {
                                    "record_id": rid,
                                    "flags": flags,
                                    "tail_rep_score": round(score, 6),
                                    "model": row.get("model_slug", model),
                                    "arm": row.get("arm"),
                                    "lang": row.get("lang"),
                                    "cue": row.get("cue"),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            print(f"  done {model}: {sum(per_model[model].values())} flag hits", file=sys.stderr)

    w = 22
    print("\n=== J0 TRIAGE ===")
    print(f"rows processed : {n_rows}  (expected {EXPECTED_ROWS})")
    print(f"distinct ids   : {len(seen_ids)}   duplicates: {dup_ids}")
    print(f"output         : {OUT_PATH}")

    print("\n--- per-model flag counts ---")
    header = f"{'model':{w}}" + "".join(f"{f:>18}" for f in FLAG_NAMES)
    print(header)
    print("-" * len(header))
    for model in GEN_ROOTS:
        c = per_model[model]
        print(f"{model:{w}}" + "".join(f"{c[f]:>18}" for f in FLAG_NAMES))
    print("-" * len(header))
    print(f"{'TOTAL':{w}}" + "".join(f"{totals[f]:>18}" for f in FLAG_NAMES))

    print("\n--- hard drops (plan Sec 3.2: error + prompt_contains_cue only) ---")
    print(f"error rows                : {hard_drop['error']}")
    print(f"prompt_contains_cue fails : {hard_drop['prompt_contains_cue_fail']}")
    print(f"TOTAL HARD DROP           : {hard_drop['total']}  (expected 0)")

    ok = True
    if n_rows != EXPECTED_ROWS:
        print(f"\nGATE J0 FAIL: {n_rows} rows, expected {EXPECTED_ROWS}")
        ok = False
    if dup_ids:
        print(f"\nGATE J0 FAIL: {dup_ids} duplicate record_ids")
        ok = False
    if hard_drop["total"]:
        print(f"\nNOTE: {hard_drop['total']} hard-drop rows; per plan Sec 3.2 the drop is")
        print("applied listwise at doc_id level across the whole grid. Record in RUN_SUMMARY.md.")
        print("  first 10:", hard_drop_ids[:10])
    print("\nGATE J0: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
