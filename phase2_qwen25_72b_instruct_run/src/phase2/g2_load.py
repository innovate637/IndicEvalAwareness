"""G2 — does this model load, template and tokenize? (plan §9)

Pass criteria, verbatim from §9:
  * loads at the pinned SHA in bf16
  * `apply_chat_template` works
  * **no double BOS**
  * `enable_thinking=False` produces **no `<think>` block**
  * Gemma 3 loads without a vision-processor error
  * (sarvam-30b only) works natively under the pinned vLLM

Needs a GPU. Run via cluster/sbatch/g2_<slug>.sbatch, never on the login node.
"""
from __future__ import annotations
import argparse, json, os, sys, traceback
from pathlib import Path

from phase2.config import load_models, load_run, max_tokens_table
from phase2.assemble import Assembler, AssemblyError


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--items", type=int, default=4)
    args = ap.parse_args()

    run, models = load_run(), load_models()
    m = models[args.model]
    res: dict = {"gate": "G2", "model": m.slug, "repo": m.repo,
                 "revision": m.revision, "checks": {}, "pass": True}

    def fail(k, v):
        res["checks"][k] = v
        res["pass"] = False

    from transformers import AutoTokenizer
    import torch, transformers, vllm

    res["versions"] = {"vllm": vllm.__version__, "torch": torch.__version__,
                       "transformers": transformers.__version__}
    res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    res["cuda_capability"] = list(torch.cuda.get_device_capability(0)) \
        if torch.cuda.is_available() else None

    # ---- tokenizer + chat template ---------------------------------------
    try:
        tok = AutoTokenizer.from_pretrained(
            m.repo, revision=m.revision, trust_remote_code=m.trust_remote_code)
        res["checks"]["tokenizer_loads"] = "ok"
    except Exception as e:
        fail("tokenizer_loads", f"{type(e).__name__}: {e}")
        print(json.dumps(res, indent=2)); return 1

    try:
        asm = Assembler(m, tok)
        prompts = asm.shard("harmful", "eval_log", args.lang)[: args.items]
        res["checks"]["apply_chat_template"] = "ok"
    except AssemblyError as e:
        fail("apply_chat_template", f"AssemblyError: {e}")
        print(json.dumps(res, indent=2)); return 1

    # ---- no double BOS ----------------------------------------------------
    bos = tok.bos_token_id
    counts = [p.token_ids.count(bos) for p in prompts] if bos is not None else []
    res["checks"]["bos_token_id"] = bos
    res["checks"]["bos_per_prompt"] = sorted(set(counts))
    if bos is not None and any(c > 1 for c in counts):
        fail("no_double_bos", f"BOS appears {max(counts)}x in a prompt")
    else:
        res["checks"]["no_double_bos"] = "ok"

    # ---- ids really are ints (the transformers-5.x BatchEncoding trap) ----
    ok_ids = all(isinstance(i, int) for p in prompts for i in p.token_ids)
    res["checks"]["token_ids_are_ints"] = "ok" if ok_ids else "CORRUPT"
    res["checks"]["n_prompt_tokens"] = [len(p.token_ids) for p in prompts]
    if not ok_ids:
        fail("token_ids_are_ints", "token_ids are not ints — see assemble._as_ids")

    # ---- engine load ------------------------------------------------------
    if run.determinism.get("batch_invariant"):
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    from vllm import LLM, SamplingParams
    try:
        llm = LLM(model=m.repo, revision=m.revision, tokenizer_revision=m.revision,
                  dtype=m.dtype, tensor_parallel_size=m.tensor_parallel,
                  gpu_memory_utilization=m.gpu_memory_utilization,
                  max_model_len=m.max_model_len, max_num_seqs=m.max_num_seqs,
                  max_num_batched_tokens=m.max_num_batched_tokens,
                  enforce_eager=m.enforce_eager, seed=run.seed,
                  trust_remote_code=m.trust_remote_code)
        res["checks"]["engine_loads_bf16"] = "ok"
    except Exception as e:
        fail("engine_loads_bf16", f"{type(e).__name__}: {str(e)[:1500]}")
        res["traceback"] = traceback.format_exc()[-2500:]
        print(json.dumps(res, indent=2, ensure_ascii=False)); return 1

    # ---- a few real generations ------------------------------------------
    budget = max_tokens_table()[m.slug][args.lang]
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, seed=run.seed,
                        max_tokens=min(128, budget), n=1, stop=None,
                        skip_special_tokens=False)
    try:
        outs = llm.generate([{"prompt_token_ids": p.token_ids} for p in prompts], sp)
        res["checks"]["generate"] = "ok"
    except Exception as e:
        fail("generate", f"{type(e).__name__}: {str(e)[:1500]}")
        res["traceback"] = traceback.format_exc()[-2500:]
        print(json.dumps(res, indent=2, ensure_ascii=False)); return 1

    texts = [o.outputs[0].text or "" for o in outs]
    res["samples"] = [t[:300] for t in texts]
    res["checks"]["finish_reasons"] = [o.outputs[0].finish_reason for o in outs]

    # ---- no <think> block -------------------------------------------------
    thinking = [t for t in texts if "<think>" in t]
    if thinking:
        fail("no_think_block", f"{len(thinking)}/{len(texts)} outputs contain <think>")
    else:
        res["checks"]["no_think_block"] = "ok"

    # ---- gemma vision-processor sanity ------------------------------------
    res["checks"]["gemma_loads_without_vision_error"] = \
        "ok" if "gemma" in m.slug else "n/a"

    print(json.dumps(res, indent=2, ensure_ascii=False))
    out = Path(run.paths["preflight"]) / f"g2_{m.slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nG2 {m.slug}: {'PASS' if res['pass'] else 'FAIL'}  -> {out}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
