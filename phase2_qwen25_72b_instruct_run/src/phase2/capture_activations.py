"""Phase-2 activation capture — Qwen2.5-72B-Instruct.

SEPARATE from generate.py, and run AFTER it. vLLM's LLM.generate() does not
expose residual-stream activations, so this re-runs each prompt through plain
HuggingFace transformers with output_hidden_states=True. It must not share a
job with vLLM: vLLM holds ~90% of VRAM for its KV pool and does not release it
until the process exits, so Phase A must fully terminate first.

  python -m phase2.capture_activations --model qwen25-72b-instruct
  python -m phase2.capture_activations --model qwen25-72b-instruct --arm harmful --lang en
  python -m phase2.capture_activations --model qwen25-72b-instruct --limit 4 --dry-run

WHAT IS CAPTURED
  For every generation row, the prompt is re-tokenised-free: we reuse the exact
  `prompt_token_ids` recorded in the JSONL, so the tensor fed here is
  byte-identical to what vLLM decoded from. No re-templating, no drift.

  Four position variants, each over all 81 hidden_states:
    last            hidden_states[i][0, -1, :]
    last_content    hidden_states[i][0, content_end, :]
    mean8           mean over hidden_states[i][0, -8:, :]
    mean8_content   mean over the last <=8 CONTENT tokens, clamped to the
                    content start (never reaching into the template prefix)

  hidden_states has 81 entries: index 0 is the EMBEDDING output, indices 1..80
  are the 80 decoder layers. Each saved tensor is (n_rows, 81, 8192) bf16.

THE CONTENT BOUNDARY
  Qwen2.5's chat template renders:

    <|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {cue}\n\n{item}<|im_end|>
    <|im_start|>assistant

  so the prompt ends with a FIXED 5-token generation suffix
  (`<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`) and
  content_end = n_tokens - 6.

  That constant is NOT hardcoded. It is derived from the tokenizer at runtime
  and then asserted against every row, so a template change breaks loudly here
  rather than silently mislabelling which position is "the last content token".

  Note the system block above is Qwen2.5's own injected default, not something
  this pipeline asked for. It is accepted deliberately — see
  docs/PLAN_DEVIATIONS.md D-72B-1. It sits in the PREFIX, so it shifts absolute
  indices but never intrudes on the content window.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

from phase2.config import load_models, load_run
from phase2.io_jsonl import read_shard

MEAN_WINDOW = 8
N_LAYERS_EXPECTED = 81          # 1 embedding + 80 decoder layers
HIDDEN_EXPECTED = 8192
VARIANTS = ("last", "last_content", "mean8", "mean8_content")


class CaptureError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# template geometry, derived once from the tokenizer and asserted per row
# ---------------------------------------------------------------------------
def template_geometry(tok) -> tuple[list[int], int]:
    """Return (generation_suffix_ids, content_start).

    suffix       = the fixed tail after the user content ends
    content_start = index of the first content token in a rendered prompt

    Both are measured from the tokenizer rather than assumed, because the whole
    point of `last_content` is that it is NOT `last`.
    """
    suffix = tok("<|im_end|>\n<|im_start|>assistant\n",
                 add_special_tokens=False)["input_ids"]
    if not suffix:
        raise CaptureError("empty generation suffix from tokenizer")

    empty = tok.apply_chat_template([{"role": "user", "content": ""}],
                                    add_generation_prompt=True, tokenize=True,
                                    return_dict=False)
    empty = list(getattr(empty, "input_ids", empty))
    if empty and isinstance(empty[0], (list, tuple)):
        empty = list(empty[0])
    if list(empty[-len(suffix):]) != list(suffix):
        raise CaptureError(
            "empty-content render does not end with the derived suffix; "
            "the chat template has changed and the content boundary is unsafe")
    return list(suffix), len(empty) - len(suffix)


def row_positions(ids: list[int], suffix: list[int], content_start: int) -> dict:
    """Per-row index arithmetic. Every value is recorded in the sidecar."""
    n = len(ids)
    if list(ids[-len(suffix):]) != list(suffix):
        raise CaptureError(
            f"prompt does not end with the generation suffix (n={n}); refusing "
            f"to guess the content boundary")
    content_end = n - len(suffix) - 1
    if content_end < content_start:
        raise CaptureError(
            f"content_end {content_end} < content_start {content_start}: "
            f"empty content after templating")

    # mean8 over the raw tail, including template tokens (clamped for short seqs)
    mean_lo = max(0, n - MEAN_WINDOW)
    # mean8_content clamped to the content start, so the window never reaches
    # back into the system/user template prefix (decision 7).
    cmean_lo = max(content_start, content_end - (MEAN_WINDOW - 1))
    return {
        "n_prompt_tokens": n,
        "content_end": content_end,
        "content_mean_lo": cmean_lo,
        "content_mean_n": content_end - cmean_lo + 1,
        "mean_window_lo": mean_lo,
        "mean_window_n": n - mean_lo,
        "n_trailing_template": len(suffix),
    }


# ---------------------------------------------------------------------------
def load_rows(path: Path) -> list[dict]:
    """Generation rows for one shard, deterministically ordered by itemnum."""
    rows = [r for r in read_shard(path) if r.get("prompt_token_ids")]
    rows.sort(key=lambda r: r["itemnum"])
    return rows


def capture_shard(model, tok, rows, suffix, content_start, device) -> tuple:
    import torch
    n = len(rows)
    bufs = {v: torch.empty((n, N_LAYERS_EXPECTED, HIDDEN_EXPECTED),
                           dtype=torch.bfloat16) for v in VARIANTS}
    meta_rows = []

    for i, r in enumerate(rows):
        ids = list(r["prompt_token_ids"])
        pos = row_positions(ids, suffix, content_start)

        inp = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(inp, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        if len(hs) != N_LAYERS_EXPECTED:
            raise CaptureError(
                f"expected {N_LAYERS_EXPECTED} hidden_states, got {len(hs)}")
        if hs[0].shape[-1] != HIDDEN_EXPECTED:
            raise CaptureError(
                f"expected hidden size {HIDDEN_EXPECTED}, got {hs[0].shape[-1]}")

        ce, cl = pos["content_end"], pos["content_mean_lo"]
        ml = pos["mean_window_lo"]
        # Stack across layers once per variant, then move to CPU immediately so
        # the 81 x seq_len x 8192 activation graph can be freed before the next
        # row (that tensor is ~10 GB at seq_len 8192).
        bufs["last"][i] = torch.stack(
            [h[0, -1, :] for h in hs]).to("cpu", torch.bfloat16)
        bufs["last_content"][i] = torch.stack(
            [h[0, ce, :] for h in hs]).to("cpu", torch.bfloat16)
        bufs["mean8"][i] = torch.stack(
            [h[0, ml:, :].float().mean(0) for h in hs]).to("cpu", torch.bfloat16)
        bufs["mean8_content"][i] = torch.stack(
            [h[0, cl:ce + 1, :].float().mean(0) for h in hs]).to("cpu", torch.bfloat16)

        del out, hs
        meta_rows.append({
            "record_id": r.get("record_id"),
            "arm": r.get("arm"), "lang": r.get("lang"), "cue": r.get("cue"),
            "itemnum": r.get("itemnum"), "doc_id": r.get("doc_id"),
            "gen_error": r.get("error_class"),
            **pos,
        })
    return bufs, meta_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-id", default=os.environ.get("P2_RUN_ID", "dev"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--cue", default=None)
    ap.add_argument("--gen-root", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N rows per shard (smoke test only)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-capture shards that already have output "
                         "(CLAUDE.md 14: off by default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve shards and index arithmetic, load no model")
    args = ap.parse_args()

    run, models = load_run(), load_models()
    if args.model not in models:
        print(f"unknown model {args.model}", file=sys.stderr)
        return 2
    m = models[args.model]
    gen_root = Path(args.gen_root or run.paths["generations"]) / m.slug
    out_root = Path(args.out_root or run.paths["activations"]) / m.slug
    arms = [args.arm] if args.arm else run.arms
    langs = [args.lang] if args.lang else run.langs
    cues = [args.cue] if args.cue else run.cues

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        m.repo, revision=m.revision, trust_remote_code=m.trust_remote_code)
    suffix, content_start = template_geometry(tok)
    print(f"[geom] suffix={suffix} ({len(suffix)} tok)  "
          f"content_start={content_start}  bos={tok.bos_token_id}", flush=True)

    todo = []
    for arm in arms:
        for lang in langs:
            for cue in cues:
                src = gen_root / arm / lang / f"{cue}.jsonl"
                dst = out_root / arm / lang / f"{cue}.safetensors"
                if not src.exists():
                    print(f"[miss] no generations: {src}", file=sys.stderr)
                    continue
                if dst.exists() and not args.overwrite:
                    print(f"[skip] exists: {dst}")
                    continue
                todo.append((arm, lang, cue, src, dst))

    if args.dry_run:
        for arm, lang, cue, src, dst in todo:
            rows = load_rows(src)[: args.limit] if args.limit else load_rows(src)
            if not rows:
                print(f"[dry ] {arm}/{lang}/{cue}: 0 rows")
                continue
            p0 = row_positions(list(rows[0]["prompt_token_ids"]),
                               suffix, content_start)
            print(f"[dry ] {arm}/{lang}/{cue}: {len(rows)} rows, row0 {p0}")
        print(f"[dry ] {len(todo)} shards would be captured")
        return 0

    import torch
    from transformers import AutoModelForCausalLM
    from safetensors.torch import save_file

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        m.repo, revision=m.revision, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=m.trust_remote_code)
    model.eval()
    device = next(model.parameters()).device
    print(f"[load] model ready in {time.time() - t0:.0f}s (device={device})",
          flush=True)

    for arm, lang, cue, src, dst in todo:
        rows = load_rows(src)
        if args.limit:
            rows = rows[: args.limit]
        if not rows:
            print(f"[warn] {src} has no usable rows", file=sys.stderr)
            continue

        t1 = time.time()
        bufs, meta_rows = capture_shard(model, tok, rows, suffix,
                                        content_start, device)
        dst.parent.mkdir(parents=True, exist_ok=True)
        save_file({v: bufs[v].contiguous() for v in VARIANTS}, str(dst))

        index = {
            "record_ids": [r["record_id"] for r in meta_rows],
            "rows": meta_rows,
            "file": {
                "n_rows": len(meta_rows),
                "layers": N_LAYERS_EXPECTED,
                "hidden": HIDDEN_EXPECTED,
                "dtype": "bfloat16",
                "variants": list(VARIANTS),
                "model_repo": m.repo,
                "model_revision": m.revision,
                "mean_window": MEAN_WINDOW,
                # Qwen2.5 has no BOS token at all (tok.bos_token_id is None), so
                # there is none to exclude; recorded explicitly rather than left
                # to inference.
                "bos_excluded": False,
                "bos_token_id": tok.bos_token_id,
                "generation_suffix_ids": suffix,
                "content_start": content_start,
                "run_id": args.run_id,
                "source_shard": str(src),
            },
        }
        dst.with_suffix(".index.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2))
        print(f"[done] {dst}  {len(meta_rows)} rows  "
              f"({time.time() - t1:.0f}s)", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
