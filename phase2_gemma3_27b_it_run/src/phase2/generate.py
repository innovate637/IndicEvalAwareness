"""Phase-2 generation runner.

One process = one (model, language). Loads the model once, then sweeps
arm x cue. Resumable, idempotent, SIGTERM-safe, and it records failures
as rows rather than dropping cells.

  python -m phase2.generate --model sarvam-m --lang ta --run-id <RUN_ID>
  python -m phase2.generate --model sarvam-m --lang ta --arm harmful --cue eval_log
  python -m phase2.generate --model sarvam-m --lang en --dry-run
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, re, signal, sys, time
from pathlib import Path

from phase2.config import (load_models, load_run, load_languages,
                           max_tokens_table)
from phase2.assemble import Assembler, AssemblyError
from phase2.io_jsonl import ShardWriter
from phase2 import script_lid

_STOP = {"flag": False}
_THINK = re.compile(r"<think>(.*?)</think>", re.S)


def _on_term(signum, _frame):
    _STOP["flag"] = True
    print(f"[signal] {signum} -- will stop after the current shard", flush=True)


signal.signal(signal.SIGTERM, _on_term)
signal.signal(signal.SIGINT, _on_term)


def record_id(run_id, model, arm, lang, cue, itemnum) -> str:
    return hashlib.sha1(
        f"{run_id}|{model}|{arm}|{lang}|{cue}|{itemnum}".encode()).hexdigest()


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Offline reasoning separation (N4). vLLM's reasoning parser is a
    fragile/server-side path, so split explicitly on <think> instead."""
    m = _THINK.search(text)
    if not m:
        return None, text
    return m.group(1).strip(), _THINK.sub("", text, count=1).lstrip()


def top_logprobs(logprobs, k_positions: int, top_n: int) -> list[dict]:
    """Rank-ordered top-n per position (N3).

    vLLM returns dict[token_id -> Logprob]; dict order is NOT rank order, and
    the dict may hold top_n+1 entries because the sampled token is always
    included. Logprob.rank is the authority.
    """
    out = []
    if not logprobs:
        return out
    for pos, d in enumerate(list(logprobs)[:k_positions]):
        items = sorted(d.items(),
                       key=lambda kv: (getattr(kv[1], "rank", None) or 10**6))
        out.append({"pos": pos, "top": [
            {"tok_id": int(tid),
             "tok": getattr(v, "decoded_token", None),
             "lp": float(v.logprob),
             "rank": getattr(v, "rank", None)}
            for tid, v in items[:top_n]]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", required=True)
    ap.add_argument("--run-id", default=os.environ.get("P2_RUN_ID", "dev"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--cue", default=None)
    ap.add_argument("--items", type=int, default=None, help="first N (gates)")
    ap.add_argument("--placement", default="user_prepend",
                    choices=["user_prepend", "system"])
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--max-tokens-override", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run, models = load_run(), load_models()
    if args.model not in models:
        print(f"unknown model {args.model}", file=sys.stderr)
        return 2
    m = models[args.model]
    langs_cfg = load_languages()
    out_root = Path(args.out_root or run.paths["generations"])
    arms = [args.arm] if args.arm else run.arms
    cues = [args.cue] if args.cue else run.cues

    mpath = Path(run.paths["preflight"]) / "manifest.json"
    manifest_sha = (json.loads(mpath.read_text())["manifest_sha"]
                    if mpath.exists() else "unpinned")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        m.repo, revision=m.revision, trust_remote_code=m.trust_remote_code)
    asm = Assembler(m, tok)
    budget = args.max_tokens_override or max_tokens_table()[m.slug][args.lang]

    # ---------------- dry run: prompts only (G3) ---------------------------
    if args.dry_run:
        dump = []
        for arm in arms:
            for cue in cues:
                sh = asm.shard(arm, cue, args.lang, args.placement)
                for p in (sh[: args.items] if args.items else sh):
                    dump.append({"arm": arm, "cue": cue, "lang": args.lang,
                                 "itemnum": p.itemnum, "doc_id": p.doc_id,
                                 "n_prompt_tokens": len(p.token_ids),
                                 "cue_in_prompt": p.cue_present,
                                 "rendered": p.rendered})
        out = Path(run.paths["preflight"]) / f"dryrun_{m.slug}_{args.lang}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dump, ensure_ascii=False, indent=2))
        print(f"[dry-run] {len(dump)} prompts -> {out}")
        return 0

    # ---------------- engine ------------------------------------------------
    batch_inv = bool(run.determinism.get("batch_invariant"))
    if batch_inv:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"     # must precede LLM()
    import torch, transformers, vllm
    from vllm import LLM, SamplingParams

    engine_extra = {}
    _ab = run.determinism.get("attention_backend")
    if _ab:
        engine_extra["attention_backend"] = _ab
    llm = LLM(model=m.repo, revision=m.revision, tokenizer_revision=m.revision,
              **engine_extra,
              dtype=m.dtype, tensor_parallel_size=m.tensor_parallel,
              gpu_memory_utilization=m.gpu_memory_utilization,
              max_model_len=m.max_model_len, max_num_seqs=m.max_num_seqs,
              max_num_batched_tokens=m.max_num_batched_tokens,
              enforce_eager=m.enforce_eager, seed=run.seed,
              trust_remote_code=m.trust_remote_code)

    gpu_name = torch.cuda.get_device_name(0)
    K = run.sampling["logprob_prefix_tokens"]
    TOPN = run.sampling["logprobs"]
    exp_script = langs_cfg[args.lang]["script"]
    sp = SamplingParams(temperature=run.sampling["temperature"],
                        top_p=run.sampling["top_p"],
                        top_k=run.sampling["top_k"],
                        seed=run.seed, logprobs=TOPN, n=1, stop=None,
                        skip_special_tokens=False, max_tokens=budget)

    def base_row(p, arm, cue) -> dict:
        return {
            "record_id": record_id(args.run_id, m.slug, arm, args.lang, cue,
                                   p.itemnum),
            "run_id": args.run_id, "manifest_sha": manifest_sha,
            "model_slug": m.slug, "arm": arm, "lang": args.lang, "cue": cue,
            "itemnum": p.itemnum, "doc_id": p.doc_id,
            "model_repo": m.repo, "model_revision": m.revision,
            "model_kind": m.kind, "dtype": m.dtype,
            "tensor_parallel": m.tensor_parallel, "gpu_model": gpu_name,
            "vllm_version": vllm.__version__, "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "batch_invariant": batch_inv, "enforce_eager": m.enforce_eager,
            "max_model_len": m.max_model_len, "max_num_seqs": m.max_num_seqs,
            "cue_text": p.cue_text, "item_text": p.item_text,
            "cue_placement": args.placement, "thinking_enabled": False,
            "rendered_prompt": p.rendered, "prompt_token_ids": p.token_ids,
            "prompt_sha": p.sha, "n_prompt_tokens": len(p.token_ids),
            "prompt_contains_cue": p.cue_present,
            "temperature": run.sampling["temperature"],
            "top_p": run.sampling["top_p"], "top_k": run.sampling["top_k"],
            "seed": run.seed, "max_tokens": budget,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "attempt": 1,
        }

    for arm in arms:
        for cue in cues:
            if _STOP["flag"]:
                print("[stop] exiting before next shard", flush=True)
                return 1

            path = Path(out_root) / m.slug / arm / args.lang / f"{cue}.jsonl"
            try:
                prompts = asm.shard(arm, cue, args.lang, args.placement)
            except AssemblyError as e:
                print(f"[FATAL] assembly failed: {e}", file=sys.stderr)
                return 3                       # never generate from a bad prompt
            if args.items:
                prompts = prompts[: args.items]

            with ShardWriter(path) as w:
                todo = [p for p in prompts
                        if not w.has(record_id(args.run_id, m.slug, arm,
                                               args.lang, cue, p.itemnum))]
                if not todo:
                    print(f"[skip] {m.slug}/{arm}/{args.lang}/{cue} complete")
                    continue
                print(f"[gen ] {m.slug}/{arm}/{args.lang}/{cue} "
                      f"{len(todo)}/{len(prompts)} max_tokens={budget}", flush=True)

                t0 = time.time()
                try:
                    outs = llm.generate(
                        [{"prompt_token_ids": p.token_ids} for p in todo], sp)
                except Exception as e:                              # I4
                    per = int((time.time() - t0) * 1000) // max(1, len(todo))
                    ecls = type(e).__name__
                    ecls = ("OOM" if "OutOfMemory" in ecls or "CUDA" in str(e)
                            else ecls)
                    for p in todo:
                        r = base_row(p, arm, cue)
                        r.update({"response_text": None, "response_answer": None,
                                  "reasoning_text": None,
                                  "response_token_ids": [],
                                  "n_completion_tokens": 0,
                                  "finish_reason": "error", "truncated": False,
                                  "first_token_logprobs": [],
                                  "cumulative_logprob": None,
                                  "response_script": None,
                                  "response_lang_match": None,
                                  "response_char_len": 0,
                                  "response_is_empty": True,
                                  "gen_wall_ms": per,
                                  "error": str(e)[:2000], "error_class": ecls})
                        w.write(r)
                    print(f"[ERROR] {path}: {ecls}: {e}", file=sys.stderr)
                    continue

                wall_ms = int((time.time() - t0) * 1000)
                per = wall_ms // max(1, len(todo))
                for p, o in zip(todo, outs):
                    c = o.outputs[0]
                    text = c.text or ""
                    reasoning, answer = split_reasoning(text)
                    script = script_lid.dominant_script(text)
                    r = base_row(p, arm, cue)
                    r.update({
                        "response_text": text, "response_answer": answer,
                        "reasoning_text": reasoning,
                        "response_token_ids": list(c.token_ids),
                        "n_completion_tokens": len(c.token_ids),
                        "finish_reason": c.finish_reason,
                        "truncated": c.finish_reason == "length",
                        "first_token_logprobs": top_logprobs(c.logprobs, K, TOPN),
                        "cumulative_logprob": (float(c.cumulative_logprob)
                                               if c.cumulative_logprob is not None
                                               else None),
                        "response_script": script,
                        "response_lang_match": script == exp_script,
                        "response_char_len": len(text),
                        "response_is_empty": not text.strip(),
                        "gen_wall_ms": per, "error": None, "error_class": None})
                    w.write(r)
            print(f"[done] {path}  ({wall_ms / 1000:.1f}s)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
