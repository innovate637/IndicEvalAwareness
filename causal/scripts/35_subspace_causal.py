#!/usr/bin/env python3
"""
35 — SUBSPACE causal test (H3/H4): rank-1 ablation was null; if eval-awareness is a DISTRIBUTED feature
(a rank-k subspace, not one direction), removing the whole subspace should move refusal where one
direction did not. We project out the rank-k eval subspace at every layer/step (gentle, generation-safe)
and compare to a RANK-MATCHED in-distribution null subspace.

  eval subspace V_eval[L] = top-k right singular vecs of per-item (eval-cue − deploy-cue) acts at layer L.
  null  subspace V_null[L] = same on a label-shuffled split (in-distribution, no eval signal).
NECESSITY: ablate V_eval from the EVAL run -> refusal drops toward deploy?  vs ablate V_null.
H4 cross-lingual: build V_eval on EN (src), ablate it from hi/bn (tgt) eval runs.

  CUDA_VISIBLE_DEVICES=1 python scripts/35_subspace_causal.py --src en --tgt hi --k 8 --n 24
Outputs: results/causal/subspace_{src}2{tgt}_k{k}.csv
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
NL = config.NUM_LAYERS
BATCH = 8


def subspaces(hf, tok, cev, cdep, texts, k):
    """V_eval[L],V_null[L] : [k,D] orthonormal rows. eval=SVD of (eval-deploy) per-item diffs;
    null=SVD of a label-shuffled split (in-distribution, rank-matched)."""
    ev = p34.capture_acts(hf, tok, cev, texts)        # [N,NL,D]
    dp = p34.capture_acts(hf, tok, cdep, texts)
    diff = ev - dp                                     # [N,NL,D]
    pool = np.concatenate([ev, dp], 0)
    idx = np.random.default_rng(0).permutation(len(pool)); h = len(pool) // 2
    ndiff = pool[idx[:h]] - pool[idx[h:]]              # [N,NL,D] null diffs
    Ve, Vn = {}, {}
    for L in range(NL):
        for D3, store in ((diff, Ve), (ndiff, Vn)):
            _, _, Vt = np.linalg.svd(D3[:, L, :], full_matrices=False)   # Vt:[min(N,D),D]
            store[L] = Vt[:k].astype(np.float32)       # top-k right singular vecs
    return Ve, Vn


def gen_ablate_subspace(hf, tok, cue, items, Vby, layers, max_new=400):
    """project out the rank-k subspace at each layer in `layers`, ALL positions, EVERY step (gentle)."""
    resps = []
    for b0 in range(0, len(items), BATCH):
        texts = [it["text"] for it in items[b0:b0+BATCH]]
        inp = p34.build_inputs(tok, cue, texts); n_in = inp["input_ids"].shape[1]; handles = []
        for L in layers:
            Vt = torch.as_tensor(Vby[L], dtype=torch.float32, device="cuda:0")   # [k,D]
            def mk(Vt):
                def hook(_m, _i, o):
                    is_t = isinstance(o, tuple); hs = o[0] if is_t else o
                    proj = (hs.to(Vt.dtype) @ Vt.T) @ Vt          # [B,seq,D] component in subspace
                    hs = hs - proj.to(hs.dtype)
                    return (hs,)+tuple(o[1:]) if is_t else hs
                return hook
            handles.append(hf.model.layers[L].register_forward_hook(mk(Vt)))
        try:
            with torch.no_grad():
                out = hf.generate(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                                  max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        finally:
            for hh in handles: hh.remove()
        resps += [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]
    return resps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="en"); ap.add_argument("--tgt", default="hi")
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--win", default="8-32")
    a = ap.parse_args()
    lo, hi = a.win.split("-"); window = list(range(int(lo), int(hi)+1))
    battery, _, tr = p34.s09.load_cue_sources(); L2 = {a.src, a.tgt}
    cdep = {l: p34.s09.get_cue(l, "deploy", battery, tr) for l in L2}
    cev  = {l: p34.s09.get_cue(l, "eval_explicit", battery, tr) for l in L2}
    key = p34.s30.or_key(); p34.sv.KEY = p34.sv.sarvam_key()
    src_h = p34.s14.load_harmful_subset(a.src, 80)
    tgt_h = p34.s14.load_harmful_subset(a.tgt, a.n + 8); test = tgt_h[8:8+a.n]
    harmless = p34.load_harmless(a.tgt, a.n)
    model, tok = load_model_and_tokenizer(int8=True); hf = model._model
    print(f"[subspace {a.src}->{a.tgt}] k={a.k} window L{window[0]}-{window[-1]} | build subspaces…", flush=True)
    Ve, Vn = subspaces(hf, tok, cev[a.src], cdep[a.src], [h["text"] for h in src_h[:64]], a.k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"subspace_{a.src}2{a.tgt}_k{a.k}"; items_path = OUT_DIR / f"{tag}_items.csv"
    # RESUME: load any conditions already finished; skip their (expensive) generation.
    done = {}
    if items_path.exists():
        prev = pd.read_csv(items_path)
        for lab, g in prev.groupby("label"): done[lab] = g
        print(f"  [resume] {len(done)} condition(s) already done: {sorted(done)}", flush=True)
    def jr(label, items, gen_thunk):
        if label in done:
            g = done[label]; rg = 100 * g["refuse_gemma"].dropna().mean() if len(g["refuse_gemma"].dropna()) else float("nan")
            print(f"  [resume] {label:<26s} gemma={rg:5.0f}%", flush=True); return rg
        resps = gen_thunk()
        gl = p34.judge_gemma(items, resps, a.tgt, key); sl = p34.judge_sarvam(items, resps, a.tgt)
        rg, rs, dgp = p34._rate(gl), p34._rate(sl), p34.deg(resps)
        recs = [dict(src=a.src, tgt=a.tgt, k=a.k, label=label, item_id=it["id"], refuse_gemma=g, refuse_sarvam=s, resp=r)
                for it, r, g, s in zip(items, resps, gl, sl)]
        pd.DataFrame(recs).to_csv(items_path, mode="a", header=not items_path.exists(), index=False)  # checkpoint NOW
        print(f"  {label:<28s} gemma={rg:5.0f}% sarvam={rs:5.0f}% deg={dgp:3.0f}%", flush=True); return rg
    rD = jr("deploy baseline",            test, lambda: p34.gen_all(hf, tok, cdep[a.tgt], test, a.tgt, "none"))
    rE = jr("eval baseline",              test, lambda: p34.gen_all(hf, tok, cev[a.tgt],  test, a.tgt, "none"))
    rAe = jr("eval - ABL eval-subspace",  test, lambda: gen_ablate_subspace(hf, tok, cev[a.tgt], test, Ve, window))
    rAn = jr("eval - ABL null-subspace",  test, lambda: gen_ablate_subspace(hf, tok, cev[a.tgt], test, Vn, window))
    jr("harmless - ABL eval-subspace",    harmless, lambda: gen_ablate_subspace(hf, tok, cdep[a.tgt], harmless, Ve, window))
    # rebuild summary from the (checkpointed) item rows
    allit = pd.read_csv(items_path)
    summ = (allit.groupby("label").agg(refusal_gemma=("refuse_gemma", lambda s: round(100*s.dropna().mean(),1)),
            refusal_sarvam=("refuse_sarvam", lambda s: round(100*s.dropna().mean(),1)), n=("item_id","size")).reset_index())
    summ.to_csv(OUT_DIR / f"{tag}.csv", index=False)
    nec_eval = (rE - rAe) / (rE - rD) if rE != rD else float("nan")
    nec_null = (rE - rAn) / (rE - rD) if rE != rD else float("nan")
    print(f"\n[{a.src}->{a.tgt} k={a.k}] deploy={rD:.0f} eval={rE:.0f} | ABL eval-subspace={rAe:.0f} (nec {nec_eval:+.2f}) "
          f"| ABL null-subspace={rAn:.0f} (nec {nec_null:+.2f})", flush=True)
    print("  CAUSAL (H3) iff ABL eval-subspace drops eval refusal toward deploy AND beats ABL null-subspace; harmless stays LOW (specificity).", flush=True)


if __name__ == "__main__":
    main()
