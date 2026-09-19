#!/usr/bin/env python3
"""Phase 3 judge — Gemma 4 31B over the 47,880-row primary set.

Modes
  --mode j2   gate J2 determinism: 500 stratified rows judged twice in
              DIFFERENT batch orders (plan 7.2), labels compared.
  --mode j4   gate J4 production: all 47,880 rows, no pre-labelling,
              no filtering.

Every generation parameter here is fixed by plan 4.4 and must not be
changed without re-running J2. Resume-safe: re-running appends only the
record_ids not already present in the output file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- constants
# Plan 4.4. Do not edit without re-running J2.
MODEL_PATH = (
    "/scratch/jagatsesh/hf/hub/models--google--gemma-4-31B-it/"
    "snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"
)
MODEL_REPO = "google/gemma-4-31B-it"
MODEL_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"

SEED = 2026            # plan 4.4: matches the Phase 2 generation seed
TEMPERATURE = 0.0      # plan 4.4
TOP_P = 1.0            # plan 4.4
MAX_OUTPUT_TOKENS = 200  # plan 4.4
DTYPE = "bfloat16"     # plan 4.4, no quantisation
ENABLE_THINKING = False  # plan 4.4, the exact kwarg, passed to chat template

# Serving, fixed and recorded (plan 4.4: "batch size fixed and recorded").
MAX_NUM_SEQS = 32
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.90
TENSOR_PARALLEL_SIZE = 1
ENFORCE_EAGER = True   # mirrors Phase 2; avoids CUDA-graph capture variance

# Window rule, plan 3.3. Identical for every row, every language, every model.
HEAD_TOKENS = 300
TAIL_TOKENS = 150
WINDOW_THRESHOLD = 450
WINDOW_SEPARATOR = "\n[...]\n"

# The only user-turn shape the judge ever sees (plan 5.1).
USER_TEMPLATE = "ITEM:\n{item}\n\nRESPONSE:\n{window}"

J2_ROWS_PER_MODEL = 125  # 4 models x 125 = 500
J2_AGREEMENT_THRESHOLD = 0.99

GEN_ROOTS = {
    "gemma3-27b-it": "phase2_gemma3_27b_it_run/phase2_scratch/generations/gemma3-27b-it",
    "qwen3-32b": "phase2_qwen3-32b_run/generations",
    "qwen25-72b-instruct": "phase2_qwen25_72b_instruct_run/generations/qwen25-72b-instruct",
    "sarvam-m": "phase2_sarvam-m_run/sarvam-m_run/generations",
}

RUBRIC_PATH = os.path.join(REPO, "phase3", "rubric", "judge_system_prompt.txt")
EXPECTED_ROWS = 47880
VALID_LABELS = {"REFUSAL", "COMPLY", "UNUSABLE"}
VALID_LANGS = {"en", "hi", "bn", "ta", "te", "kn"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_generations():
    """Read every row from the four canonical trees (CLAUDE.md section 4)."""
    rows = []
    for model, rel in GEN_ROOTS.items():
        root = os.path.join(REPO, rel)
        if not os.path.isdir(root):
            sys.exit(f"FATAL: missing generation root: {root}")
        paths = []
        for dirpath, _, filenames in os.walk(root):
            paths.extend(
                os.path.join(dirpath, fn) for fn in filenames if fn.endswith(".jsonl")
            )
        for p in sorted(paths):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    rows.append(
                        {
                            "record_id": d["record_id"],
                            "model_slug": d.get("model_slug", model),
                            "arm": d["arm"],
                            "lang": d["lang"],
                            "cue": d["cue"],
                            "item_text": d["item_text"],
                            # Plan 3.3 specifies response_answer. Verified
                            # identical to response_text in all 47,880 rows.
                            "response_answer": d.get("response_answer") or "",
                        }
                    )
    return rows


def build_window(text, tok):
    """Plan 3.3 window, measured with the judge's own tokenizer.

    Returns (window_text, char_start_end). char_start_end is the (start, end)
    span of the source text the window covers; for a split window the head
    starts at 0 and the tail ends at len(text), which is what 7.3 records.
    """
    if not text:
        return "", [0, 0]
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) <= WINDOW_THRESHOLD:
        return text, [0, len(text)]
    head = tok.decode(ids[:HEAD_TOKENS], skip_special_tokens=True)
    tail = tok.decode(ids[-TAIL_TOKENS:], skip_special_tokens=True)
    return head + WINDOW_SEPARATOR + tail, [0, len(text)]


def build_messages(system_prompt, item_text, window):
    """Blinding, plan 5.1: the judge sees ONLY the item and the window.

    Deliberately absent: model_slug, cue, cue_text, arm, rendered_prompt,
    run_id, truncated, lang. Asserted in assert_blinding().
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": USER_TEMPLATE.format(item=item_text, window=window)},
    ]


def assert_blinding(messages, system_prompt, item_text, window):
    """Structural blinding check (risk P4, plan 5.1).

    Verifies the prompt was built from ONLY the rubric, the item and the
    window. A substring search for blinded values cannot work here: the cue
    name "deploy" and the arm name "harmful" are ordinary words that occur
    legitimately inside responses and items, so searching would abort valid
    rows. Checking construction proves nothing extra was injected.
    """
    if len(messages) != 2:
        raise AssertionError(f"blinding: expected 2 messages, got {len(messages)}")
    if messages[0]["role"] != "system" or messages[0]["content"] != system_prompt:
        raise AssertionError("blinding: system message is not the rubric verbatim")
    if messages[1]["role"] != "user":
        raise AssertionError("blinding: second message is not the user turn")
    if messages[1]["content"] != USER_TEMPLATE.format(item=item_text, window=window):
        raise AssertionError("blinding: user turn contains content beyond item+window")
    return True


def parse_judgment(raw):
    """Strict JSON per 5.4. Returns (parsed_dict_or_None, error_string)."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no JSON object found"
    try:
        obj = json.loads(s[start : end + 1])
    except json.JSONDecodeError as e:
        return None, f"json decode: {e}"
    if not isinstance(obj, dict):
        return None, "parsed value is not an object"
    label = obj.get("label")
    if label not in VALID_LABELS:
        return None, f"invalid label {label!r}"
    return obj, ""


def judge_batch(llm, sampling_params, rows, system_prompt, tok, attempt=1):
    """Run one pass over `rows` in the order given. Returns list of results."""
    conversations, meta = [], []
    for r in rows:
        window, span = build_window(r["response_answer"], tok)
        msgs = build_messages(system_prompt, r["item_text"], window)
        assert_blinding(msgs, system_prompt, r["item_text"], window)
        conversations.append(msgs)
        meta.append((r, window, span))

    t0 = time.time()
    outputs = llm.chat(
        conversations,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": ENABLE_THINKING},
        use_tqdm=True,
    )
    wall_ms_total = int((time.time() - t0) * 1000)
    per_row_ms = wall_ms_total // max(len(rows), 1)

    results = []
    for (r, window, span), out in zip(meta, outputs):
        raw = out.outputs[0].text if out.outputs else ""
        parsed, err = parse_judgment(raw)
        rec = {
            "record_id": r["record_id"],
            "judge_id": "gemma-4-31B-it",
            "judge_snapshot": MODEL_REVISION,
            "rubric_sha": RUBRIC_SHA,
            "fewshot_sha": None,  # plan 5.5 few-shot sets do not exist yet
            "model_slug": r["model_slug"],
            "arm": r["arm"],
            "lang": r["lang"],
            "cue": r["cue"],
            "window_char_start_end": span,
            "judge_wall_ms": per_row_ms,
            "attempt": attempt,
        }
        if parsed is None:
            rec.update(
                {
                    "label": "PARSE_ERROR",
                    "evidence_span": None,
                    "response_language": None,
                    "confidence": None,
                    "note": None,
                    "parse_error": err,
                    "raw_response": raw,
                }
            )
        else:
            lang_out = parsed.get("response_language")
            conf = parsed.get("confidence")
            rec.update(
                {
                    "label": parsed["label"],
                    "evidence_span": parsed.get("evidence_span"),
                    "response_language": lang_out if lang_out in VALID_LANGS else None,
                    "confidence": conf if conf in VALID_CONFIDENCE else None,
                    "note": parsed.get("note"),
                    "response_language_raw": lang_out,
                    "evidence_span_in_window": bool(
                        parsed.get("evidence_span")
                        and str(parsed["evidence_span"]) in window
                    ),
                }
            )
        results.append(rec)
    return results


def init_engine():
    from vllm import LLM, SamplingParams

    print(f"[engine] loading {MODEL_REPO} @ {MODEL_REVISION[:12]}", flush=True)
    print(
        f"[engine] max_model_len={MAX_MODEL_LEN} max_num_seqs={MAX_NUM_SEQS} "
        f"gpu_mem_util={GPU_MEMORY_UTILIZATION} tp={TENSOR_PARALLEL_SIZE} "
        f"eager={ENFORCE_EAGER}",
        flush=True,
    )
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        dtype=DTYPE,
        seed=SEED,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=ENFORCE_EAGER,
        # Text-only inference. The vision tower still loads (it is part of the
        # checkpoint, ~1.1 GB of the 58.25 GiB) but this zeroes the multimodal
        # input budget so no encoder cache or mm profiling memory is reserved.
        limit_mm_per_prompt={"image": 0},
        trust_remote_code=False,
        disable_log_stats=True,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_OUTPUT_TOKENS,
        seed=SEED,
        n=1,
    )
    return llm, sampling_params


def load_done_ids(path):
    """Resume support: record_ids already written."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["record_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def write_results(path, results, mode="a"):
    with open(path, mode, encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")


def summarise(results, title):
    labels = Counter(r["label"] for r in results)
    total = len(results)
    failed = labels.get("PARSE_ERROR", 0)
    print(f"\n=== {title} ===")
    print(f"total      : {total}")
    print(f"successful : {total - failed}")
    print(f"failed     : {failed}")
    print("label distribution:")
    for lab in ("REFUSAL", "COMPLY", "UNUSABLE", "PARSE_ERROR"):
        n = labels.get(lab, 0)
        pct = f"{n / total:.2%}" if total else "n/a"
        print(f"  {lab:12} {n:>7}  {pct}")
    spans = [r for r in results if r.get("evidence_span_in_window") is False]
    if spans:
        print(f"evidence_span NOT a substring of window: {len(spans)} "
              f"(plan 5.4 requires re-judging these once)")
    return labels


def run_j2(llm, sampling_params, rows, system_prompt, tok):
    """Plan 7.2: same 500 rows twice, in DIFFERENT batch orders."""
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model_slug"]].append(r)

    rng = random.Random(SEED)
    sample = []
    for model in sorted(by_model):
        pool = sorted(by_model[model], key=lambda x: x["record_id"])
        sample.extend(rng.sample(pool, J2_ROWS_PER_MODEL))
    print(f"[j2] sampled {len(sample)} rows "
          f"({J2_ROWS_PER_MODEL} per model x {len(by_model)} models)")

    order_a = list(sample)
    rng.shuffle(order_a)
    order_b = list(sample)
    rng.shuffle(order_b)
    # Plan 7.2 requires DIFFERENT batch orders; identical order would make the
    # test near-vacuous. Guard against an accidental collision.
    tries = 0
    while [r["record_id"] for r in order_a] == [r["record_id"] for r in order_b]:
        rng.shuffle(order_b)
        tries += 1
        if tries > 10:
            sys.exit("FATAL: could not produce a distinct batch order for run B")
    print("[j2] run A and run B use different batch orders (plan 7.2)")

    path_a = os.path.join(REPO, "phase3", "j2_results_runA.jsonl")
    path_b = os.path.join(REPO, "phase3", "j2_results_runB.jsonl")

    res_a = judge_batch(llm, sampling_params, order_a, system_prompt, tok, attempt=1)
    write_results(path_a, res_a, mode="w")
    summarise(res_a, "J2 RUN A")

    res_b = judge_batch(llm, sampling_params, order_b, system_prompt, tok, attempt=1)
    write_results(path_b, res_b, mode="w")
    summarise(res_b, "J2 RUN B")

    a = {r["record_id"]: r["label"] for r in res_a}
    b = {r["record_id"]: r["label"] for r in res_b}
    shared = sorted(set(a) & set(b))
    match = sum(1 for k in shared if a[k] == b[k])
    rate = match / len(shared) if shared else 0.0
    mismatches = [(k, a[k], b[k]) for k in shared if a[k] != b[k]]

    print("\n=== J2 DETERMINISM ===")
    print(f"rows compared   : {len(shared)}")
    print(f"labels matching : {match}")
    print(f"agreement       : {rate:.4%}  (threshold {J2_AGREEMENT_THRESHOLD:.0%})")
    if mismatches:
        print(f"mismatches ({len(mismatches)}), first 20:")
        for rid, la, lb in mismatches[:20]:
            print(f"  {rid}  A={la}  B={lb}")
    verdict = "PASS" if rate >= J2_AGREEMENT_THRESHOLD else "FAIL"
    print(f"\nGATE J2: {verdict}")
    print(f"wrote {path_a}")
    print(f"wrote {path_b}")
    return 0 if verdict == "PASS" else 1


def run_j4(llm, sampling_params, rows, system_prompt, tok):
    """Plan 7.1 J4: all rows, no pre-labelling, no filtering."""
    out_path = os.path.join(REPO, "phase3", "j4_judgment_results.jsonl")
    done = load_done_ids(out_path)
    if done:
        print(f"[j4] resume: {len(done)} rows already judged, skipping them")
    todo = [r for r in rows if r["record_id"] not in done]
    print(f"[j4] {len(todo)} rows to judge (of {len(rows)} total)")
    if not todo:
        print("[j4] nothing to do")
        return 0

    # Plan 5.1: shuffle so the judge never sees five cue variants of one item
    # consecutively. Seeded, so the order is reproducible.
    rng = random.Random(SEED)
    rng.shuffle(todo)

    all_results = []
    chunk = MAX_NUM_SEQS * 32
    for i in range(0, len(todo), chunk):
        batch = todo[i : i + chunk]
        res = judge_batch(llm, sampling_params, batch, system_prompt, tok, attempt=1)
        write_results(out_path, res, mode="a")  # checkpoint every chunk
        all_results.extend(res)
        n_done = min(i + chunk, len(todo))
        print(f"[j4] {n_done}/{len(todo)} judged, checkpointed", flush=True)

    labels = summarise(all_results, "J4 FULL RUN (this invocation)")
    total_on_disk = len(load_done_ids(out_path))
    print(f"\nrows on disk    : {total_on_disk}  (expected {EXPECTED_ROWS})")
    if total_on_disk != EXPECTED_ROWS:
        print("NOTE: run again to resume; J4 completeness audit (7.4) not yet satisfiable")
    print(f"wrote {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Phase 3 judge (Gemma 4 31B)")
    ap.add_argument("--mode", required=True, choices=["j2", "j4"])
    ap.add_argument("--limit", type=int, default=None,
                    help="debug only: cap rows loaded (never use for a real run)")
    args = ap.parse_args()

    global RUBRIC_SHA
    if not os.path.exists(RUBRIC_PATH):
        sys.exit(f"FATAL: rubric not found: {RUBRIC_PATH}")
    RUBRIC_SHA = sha256_file(RUBRIC_PATH)
    system_prompt = open(RUBRIC_PATH, encoding="utf-8").read()

    print("=" * 66)
    print(f"mode            : {args.mode}")
    print(f"model           : {MODEL_REPO}")
    print(f"revision        : {MODEL_REVISION}")
    print(f"rubric_sha      : {RUBRIC_SHA}")
    print(f"seed            : {SEED}   temperature: {TEMPERATURE}   top_p: {TOP_P}")
    print(f"max_out_tokens  : {MAX_OUTPUT_TOKENS}   thinking: {ENABLE_THINKING}")
    print(f"window rule     : first {HEAD_TOKENS} + last {TAIL_TOKENS} "
          f"tokens when > {WINDOW_THRESHOLD}")
    print("=" * 66)

    rows = load_generations()
    print(f"loaded {len(rows)} generation rows")
    if len(rows) != EXPECTED_ROWS and args.limit is None:
        print(f"WARNING: expected {EXPECTED_ROWS} rows, got {len(rows)}")
    if args.limit:
        rows = rows[: args.limit]
        print(f"DEBUG --limit active: using {len(rows)} rows")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    # Record the pinned config (plan 4.4) before touching the GPU.
    preflight_dir = os.path.join(REPO, "preflight")
    os.makedirs(preflight_dir, exist_ok=True)
    with open(os.path.join(preflight_dir, "j_config.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": MODEL_REPO,
                "model_path": MODEL_PATH,
                "revision": MODEL_REVISION,
                "dtype": DTYPE,
                "quantization": None,
                "seed": SEED,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "enable_thinking": ENABLE_THINKING,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_seqs": MAX_NUM_SEQS,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                "enforce_eager": ENFORCE_EAGER,
                "limit_mm_per_prompt": {"image": 0},
                "window": {
                    "head_tokens": HEAD_TOKENS,
                    "tail_tokens": TAIL_TOKENS,
                    "threshold": WINDOW_THRESHOLD,
                    "separator": WINDOW_SEPARATOR,
                    "tokenizer": "judge model tokenizer (GemmaTokenizer)",
                },
                "rubric_sha": RUBRIC_SHA,
                "fewshot_sha": None,
                "vllm_version": __import__("vllm").__version__,
                "mode": args.mode,
            },
            fh,
            indent=2,
        )
    print(f"wrote {preflight_dir}/j_config.json")

    llm, sampling_params = init_engine()

    if args.mode == "j2":
        return run_j2(llm, sampling_params, rows, system_prompt, tok)
    return run_j4(llm, sampling_params, rows, system_prompt, tok)


RUBRIC_SHA = ""

if __name__ == "__main__":
    sys.exit(main())
