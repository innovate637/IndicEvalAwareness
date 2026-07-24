#!/usr/bin/env python3
"""
36 — BEHAVIORAL activation patching in NO-THINK mode.

The reasoning block is the ONLY reason patching was blind (single-forward next-token readout) or
corrupting (generation paste -> OkayOkay). In NO-THINK mode the refusal decision IS the immediate
answer, so we do a real activation PATCH (replace layer-L residual, eval->deploy) and read the ACTUAL
refusal behavior (generate short answer + dual judge).

Honest caveat: the no-think eval gap is smaller (~+5pp vs +25pp think) — most of the effect is
reasoning-mediated — so this tests the NON-reasoning slice. Clean behavioral causal result; doubles as
the reasoning-mediation (H6) measurement.

Conditions (single layer L, last prompt-token, within-item):
  deploy clean | eval clean | deploy<-deploy (IDENTITY, must stay coherent) | deploy<-EVAL (the patch) |
  deploy<-EVAL(mismatched item = in-distribution null) | harmless<-EVAL
Source residual captured in the SAME regime (generate max_new=1) so identity is a true no-op.

  CUDA_VISIBLE_DEVICES=1 python scripts/36_nothink_patch.py --src en --tgt hi --L 14 --n 24
Outputs: results/causal/nothink_patch_{src}2{tgt}_L{L}.csv (+ _items.csv, checkpointed)
"""
import os, sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import config
from behavioral_common import load_model_and_tokenizer
P = importlib.util.spec_from_file_location("p34", str(ROOT / "scripts" / "34_ablate_eval.py"))
p34 = importlib.util.module_from_spec(P); P.loader.exec_module(p34)

OUT_DIR = config.RESULTS_DIR / "causal"
BATCH = 8
MAX_NEW = 120


def nt_inputs(tok, cue, texts):
    formatted = []
    for t in texts:
        pids = tok(t, add_special_tokens=False)["input_ids"]
        if len(pids) > 768: t = tok.decode(pids[:768])
        formatted.append(tok.apply_chat_template(
            [{"role": "system", "content": cue}, {"role": "user", "content": t}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))   # NO-THINK
    tok.padding_side = "left"
    return tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")


def capture_L(hf, tok, cue, texts, L):
    out = []
    for b0 in range(0, len(texts), BATCH):
        inp = nt_inputs(tok, cue, texts[b0:b0+BATCH]); g = {}
        def hook(_m, _i, o):
            hs = o[0] if isinstance(o, tuple) else o
            if hs.shape[1] > 1: g["v"] = hs[:, -1, :].detach().clone()
        h = hf.model.layers[L].register_forward_hook(hook)
        with torch.no_grad():
            hf.generate(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                        max_new_tokens=1, do_sample=False, pad_token_id=tok.eos_token_id)
        h.remove(); out.append(g["v"].float().cpu().numpy())
    return np.concatenate(out, 0)


def gen_patch(hf, tok, cue, texts, L, patch_vecs):
    resps = []
    for b0 in range(0, len(texts), BATCH):
        inp = nt_inputs(tok, cue, texts[b0:b0+BATCH]); n_in = inp["input_ids"].shape[1]; handles = []
        if patch_vecs is not None:
            vt = torch.as_tensor(patch_vecs[b0:b0+BATCH], dtype=torch.float32, device="cuda:0")
            def hook(_m, _i, o):
                is_t = isinstance(o, tuple); hs = o[0] if is_t else o
                if hs.shape[1] > 1:
                    hs = hs.clone(); hs[:, -1, :] = vt.to(hs.dtype)
                return (hs,)+tuple(o[1:]) if is_t else hs
            handles.append(hf.model.layers[L].register_forward_hook(hook))
        try:
            with torch.no_grad():
                out = hf.generate(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                                  max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
        finally:
            for hh in handles: hh.remove()
        resps += [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]
    return resps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="en"); ap.add_argument("--tgt", default="hi")
    ap.add_argument("--L", type=int, default=14); ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    battery, _, tr = p34.s09.load_cue_sources(); L2 = {a.src, a.tgt}
    cdep = {l: p34.s09.get_cue(l, "deploy", battery, tr) for l in L2}
    cev  = {l: p34.s09.get_cue(l, "eval_explicit", battery, tr) for l in L2}
    key = p34.s30.or_key(); p34.sv.KEY = p34.sv.sarvam_key()
    tgt_h = p34.s14.load_harmful_subset(a.tgt, a.n + 8); test = tgt_h[8:8+a.n]
    harmless = p34.load_harmless(a.tgt, a.n)
    model, tok = load_model_and_tokenizer(int8=True); hf = model._model
    rng = np.random.default_rng(0)
    txt = [it["text"] for it in test]
    print(f"[nothink-patch {a.src}->{a.tgt}] L={a.L} n={len(test)} | capturing residuals…", flush=True)
    ev_resid = capture_L(hf, tok, cev[a.tgt], txt, a.L)
    dp_resid = capture_L(hf, tok, cdep[a.tgt], txt, a.L)
    mm = ev_resid[rng.permutation(len(ev_resid))]
    ev_h = capture_L(hf, tok, cev[a.tgt], [it["text"] for it in harmless], a.L)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"nothink_patch_{a.src}2{a.tgt}_L{a.L}"; ip = OUT_DIR / f"{tag}_items.csv"
    done = {}
    if ip.exists():
        for lab, g in pd.read_csv(ip).groupby("label"): done[lab] = g
        print(f"  [resume] done: {sorted(done)}", flush=True)
    def jr(label, items, resps):
        if label in done:
            g = done[label]; rg = 100*g["refuse_gemma"].dropna().mean() if len(g["refuse_gemma"].dropna()) else float("nan")
            print(f"  [resume] {label:<22s} gemma={rg:.0f}%", flush=True); return rg
        gl = p34.judge_gemma(items, resps, a.tgt, key); sl = p34.judge_sarvam(items, resps, a.tgt)
        rg, rs, dgp = p34._rate(gl), p34._rate(sl), p34.deg(resps)
        recs = [dict(tgt=a.tgt, L=a.L, label=label, item_id=it["id"], refuse_gemma=gg, refuse_sarvam=ss,
                     deg=int(p34.degenerate(r)), resp=r[:300]) for it, r, gg, ss in zip(items, resps, gl, sl)]
        pd.DataFrame(recs).to_csv(ip, mode="a", header=not ip.exists(), index=False)
        print(f"  {label:<22s} gemma={rg:5.0f}% sarvam={rs:5.0f}% deg={dgp:3.0f}%", flush=True); return rg
    rD = jr("deploy clean",         test, gen_patch(hf, tok, cdep[a.tgt], txt, a.L, None))
    rE = jr("eval clean",           test, gen_patch(hf, tok, cev[a.tgt],  txt, a.L, None))
    rI = jr("deploy<-deploy ident", test, gen_patch(hf, tok, cdep[a.tgt], txt, a.L, dp_resid))
    rP = jr("deploy<-EVAL patch",   test, gen_patch(hf, tok, cdep[a.tgt], txt, a.L, ev_resid))
    rM = jr("deploy<-EVAL mismatch",test, gen_patch(hf, tok, cdep[a.tgt], txt, a.L, mm))
    jr("harmless<-EVAL patch",      harmless, gen_patch(hf, tok, cdep[a.tgt], [it["text"] for it in harmless], a.L, ev_h))
    allit = pd.read_csv(ip)
    allit.groupby("label").agg(gemma=("refuse_gemma", lambda s: round(100*s.dropna().mean(),1)),
        sarvam=("refuse_sarvam", lambda s: round(100*s.dropna().mean(),1)),
        deg=("deg", lambda s: round(100*s.mean(),1)), n=("item_id","size")).to_csv(OUT_DIR / f"{tag}.csv")
    print(f"\n[{a.src}->{a.tgt} L={a.L}] deploy={rD:.0f} eval={rE:.0f} | identity={rI:.0f} | "
          f"PATCH eval={rP:.0f} | mismatch-null={rM:.0f}", flush=True)
    print("  BEHAVIORAL PATCH WORKS iff identity coherent (deg≈0, ≈deploy) AND patch-eval raises refusal "
          "toward eval beating mismatch-null.", flush=True)


if __name__ == "__main__":
    main()
