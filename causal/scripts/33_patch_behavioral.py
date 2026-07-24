#!/usr/bin/env python3
"""
33 — BEHAVIORAL activation patching, WINDOWED localization (sound version).

Patching the residual at ALL 40 layers during generation CORRUPTS decoding (degenerate repetition) —
so, per the literature (ROME causal tracing; Zhang & Nanda 2309.16042: single-layer first, then small
windows), we patch a SMALL MID-LAYER WINDOW (default 4 consecutive layers) and SWEEP it across the
network.  Source activations are captured LIVE (same tokenization) so the identity control is a true
no-op.

Each window reports BOTH, in one run, so it is self-validating:
  * identity (eval->eval @ window): MUST stay coherent (deg≈0) and ≈ clean eval refusal.
  * suppression (deploy->eval @ window): does refusal DROP toward deploy?  frac=(r-rE)/(rD-rE).
Windows that degenerate are auto-flagged and discarded; the coherent window with the largest
suppression effect is S*.

PATCH OP: left-padding aligns every prompt's last token at index -1; replace hs[:, -1, :] (cloned, not
in-place) at the window layers, PREFILL only (seq>1).

  # self-test one window first:
  CUDA_VISIBLE_DEVICES=1 python scripts/33_patch_behavioral.py --lang en --n 4 --windows 16-19 --max-new 200
  # full sweep, sharded:
  CUDA_VISIBLE_DEVICES=1 ... --n 18 --shard-id 0 --nshard 3 ; (shards 1,2 on GPU 2,3) ; then --merge
Outputs: results/patching/behav_win_{lang}_s{k}.csv -> behav_win_{lang}.csv
"""
import os, sys, json, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import config
from behavioral_common import load_model_and_tokenizer, classify_refusal

def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / fn))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
s09 = _load("s09", "09_cue_battery_grid.py")
s14 = _load("s14", "14_steer_patch.py")

OUT_DIR = config.RESULTS_DIR / "patching"
MAX_PROMPT_TOKENS = 768
BATCH = 6
DEFAULT_WINDOWS = "8-11,12-15,16-19,20-23,24-27,28-31"
NO_THINK = False                       # set from CLI; no-think mode = answer is immediate (no <think>)


def parse_windows(s):
    out = []
    for part in s.split(","):
        a, b = part.split("-"); out.append(list(range(int(a), int(b) + 1)))
    return out


def build_inputs(tok, cue, texts):
    formatted = []
    for t in texts:
        pids = tok(t, add_special_tokens=False)["input_ids"]
        if len(pids) > MAX_PROMPT_TOKENS:
            t = tok.decode(pids[:MAX_PROMPT_TOKENS])
        formatted.append(tok.apply_chat_template(
            [{"role": "system", "content": cue}, {"role": "user", "content": t}],
            tokenize=False, add_generation_prompt=True, enable_thinking=not NO_THINK))
    tok.padding_side = "left"
    return tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")


def capture_resid(hf, tok, inputs, layers):
    """Capture via the SAME generation-prefill path (generate max_new=1) so the value matches what the
    real generation has at that position -> identity patch is a true no-op."""
    grabbed = {}
    handles = []
    for L in layers:
        def mk(L):
            def g(_m, _i, out):
                hs = out[0] if isinstance(out, tuple) else out
                if hs.shape[1] > 1:                      # prefill only
                    grabbed[L] = hs[:, -1, :].detach().float().cpu().numpy()
            return g
        handles.append(hf.model.layers[L].register_forward_hook(mk(L)))
    try:
        with torch.no_grad():
            hf.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        max_new_tokens=1, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        for h in handles:
            h.remove()
    return grabbed


def patched_generate(hf, tok, inputs, layer_vecs, max_new):
    n_in = inputs["input_ids"].shape[1]
    handles = []
    if layer_vecs:
        for L, V in layer_vecs.items():
            vt = torch.as_tensor(V, dtype=torch.float32, device="cuda:0")
            def mk(vt):
                def hook(_m, _i, out):
                    is_t = isinstance(out, tuple); hs = out[0] if is_t else out
                    if hs.shape[1] > 1:
                        hs = hs.clone()
                        hs[:, -1, :] = vt.to(hs.dtype)
                    return (hs,) + tuple(out[1:]) if is_t else hs
                return hook
            handles.append(hf.model.layers[L].register_forward_hook(mk(vt)))
    try:
        with torch.no_grad():
            out = hf.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                              max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        for h in handles:
            h.remove()
    return [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]


def degenerate(resp):
    # catch space-separated repetition AND no-space concatenated repetition ("OkayOkayOkay...")
    w = resp.split()
    if len(w) >= 10 and (len(set(w)) / len(w)) < 0.25:
        return True
    s = resp[:120]
    for unit in (3, 4, 5, 6, 8):                  # short repeated substring
        if len(s) >= unit * 5 and len(set(s[i:i+unit] for i in range(0, len(s) - unit, unit))) <= 2:
            return True
    return False


def run_one(hf, tok, lang, items, tgt_cue, src_cue, layers, kind, rng, max_new):
    recs = []
    for b0 in range(0, len(items), BATCH):
        batch = items[b0:b0 + BATCH]; texts = [it["text"] for it in batch]
        tgt_in = build_inputs(tok, tgt_cue, texts)
        lv = None
        if src_cue is not None and layers:
            cap = capture_resid(hf, tok, build_inputs(tok, src_cue, texts), layers)
            if kind == "mismatched":
                perm = rng.permutation(len(batch)); cap = {L: V[perm] for L, V in cap.items()}
            lv = cap
        resps = patched_generate(hf, tok, tgt_in, lv, max_new)
        for it, r in zip(batch, resps):
            recs.append(dict(item_id=it["id"], refusal=classify_refusal(r, lang),
                             deg=int(degenerate(r)), chars=len(r), resp=r[:160]))
    return recs


def run_shard(a):
    global NO_THINK; NO_THINK = a.no_think
    rng = np.random.default_rng(1000 + a.shard_id)
    windows = parse_windows(a.windows)
    battery, _, tr = s09.load_cue_sources()
    cues = {c: s09.get_cue(a.lang, c, battery, tr) for c in ["deploy", "eval_explicit"]}
    items = s14.load_harmful_subset(a.lang, a.n)[a.shard_id::a.nshard]
    model, tok = load_model_and_tokenizer(int8=True); hf = model._model
    print(f"[shard {a.shard_id}/{a.nshard}] {len(items)} items | windows={a.windows} | max_new={a.max_new}", flush=True)

    rows = []
    def record(label, recs):
        for r in recs: r["label"] = label; rows.append(r)
        rr = 100*np.mean([r["refusal"] for r in recs]); dg = 100*np.mean([r["deg"] for r in recs])
        print(f"  {label:<16s} refuse%={rr:5.0f}  deg%={dg:4.0f}", flush=True)

    record("clean_deploy", run_one(hf, tok, a.lang, items, cues["deploy"], None, None, "clean", rng, a.max_new))
    record("clean_eval",   run_one(hf, tok, a.lang, items, cues["eval_explicit"], None, None, "clean", rng, a.max_new))
    for w in windows:
        tag = f"{w[0]}-{w[-1]}"
        record(f"id_{tag}",  run_one(hf, tok, a.lang, items, cues["eval_explicit"], cues["eval_explicit"], w, "normal", rng, a.max_new))
        record(f"pos_{tag}", run_one(hf, tok, a.lang, items, cues["eval_explicit"], cues["deploy"], w, "normal", rng, a.max_new))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"behav_win_{a.lang}_s{a.shard_id}.csv", index=False)
    print(f"shard {a.shard_id} DONE -> behav_win_{a.lang}_s{a.shard_id}.csv", flush=True)


def merge(lang):
    import glob
    dfs = [pd.read_csv(f) for f in sorted(glob.glob(str(OUT_DIR / f"behav_win_{lang}_s*.csv")))]
    if not dfs: print("no shard files"); return
    df = pd.concat(dfs, ignore_index=True)
    rate = df.groupby("label")["refusal"].mean().mul(100)
    deg = df.groupby("label")["deg"].mean().mul(100); n = df.groupby("label").size()
    rD, rE = rate.get("clean_deploy", np.nan), rate.get("clean_eval", np.nan)
    print(f"clean: deploy={rD:.0f}%  eval={rE:.0f}%  gap={rE-rD:+.0f}pp  (n={int(n.get('clean_eval',0))})\n")
    rows = []
    for label in sorted([l for l in rate.index if l.startswith("id_")]):
        tag = label[3:]
        rid, did = rate[label], deg[label]
        rpos, dpos = rate.get(f"pos_{tag}", np.nan), deg.get(f"pos_{tag}", np.nan)
        coherent = did < 20 and dpos < 20
        frac = (rpos - rE) / (rD - rE) if (rD != rE and coherent) else np.nan
        rows.append(dict(window=tag, id_refuse=round(rid), id_deg=round(did), pos_refuse=round(rpos),
                         pos_deg=round(dpos), coherent=coherent,
                         frac_to_deploy=round(frac, 2) if frac == frac else ""))
    out = pd.DataFrame(rows); out.to_csv(OUT_DIR / f"behav_win_{lang}.csv", index=False)
    print(out.to_string(index=False))
    print("\nVALID window: id_deg≈0 AND id_refuse≈clean_eval (coherent no-op). Among valid windows, the "
          "one with the largest frac_to_deploy (refusal dropped toward deploy) is S*.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--windows", default=DEFAULT_WINDOWS)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--no-think", action="store_true", help="disable the <think> block (immediate answer)")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge(a.lang) if a.merge else run_shard(a)


if __name__ == "__main__":
    main()
