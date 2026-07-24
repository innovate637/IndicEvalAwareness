#!/usr/bin/env python3
"""
38 — NON-REASONING model contrast (H6), done carefully: coefficient sweep + IN-DISTRIBUTION null +
harmless specificity + per-condition checkpoint. On sarvam-m (reasoning) single-direction eval causality
is null; here we test whether it WORKS — specifically — on a non-reasoning model, and at which coeff.

Model default = Qwen2.5-7B-Instruct (non-reasoning, multilingual). Build per-layer eval direction
(eval-cue − deploy-cue diff-of-means) and a per-layer NORM-MATCHED label-shuffle null. Multi-layer
add/ablate every step over a window. Judge = gemma (sweep); sarvam added at confirmatory n via --sarvam.

  CUDA_VISIBLE_DEVICES=1 python scripts/38_nonreasoning_contrast.py --model qwen --n 30 \
      --coeffs 0.1 0.25 0.5 1 2 --win 6-22
Outputs: results/causal/nonreasoning_{tag}_items.csv -> nonreasoning_{tag}.csv
"""
import os, sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import config
P = importlib.util.spec_from_file_location("p34", str(ROOT / "scripts" / "34_ablate_eval.py"))
p34 = importlib.util.module_from_spec(P); P.loader.exec_module(p34)

MODELS = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "hermes": "NousResearch/Hermes-3-Llama-3.1-8B"}
OUT_DIR = config.RESULTS_DIR / "causal"
BATCH = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--coeffs", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--win", default="6-22"); ap.add_argument("--lang", default="en")
    ap.add_argument("--sarvam", action="store_true", help="also sarvam-105b judge (confirmatory; real-money)")
    a = ap.parse_args()
    name = MODELS[a.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda:0").eval()
    NL = model.config.num_hidden_layers; D = model.config.hidden_size; layers = model.model.layers
    lo, hi = a.win.split("-"); window = list(range(int(lo), min(int(hi)+1, NL)))
    battery, _, tr = p34.s09.load_cue_sources()
    cdep = p34.s09.get_cue(a.lang, "deploy", battery, tr); cev = p34.s09.get_cue(a.lang, "eval_explicit", battery, tr)
    items = p34.s14.load_harmful_subset(a.lang, a.n + 64)
    dirtxt = [it["text"] for it in items[a.n:a.n+64]]; test = items[:a.n]
    harmless = p34.load_harmless(a.lang, a.n)
    key = p34.s30.or_key()
    if a.sarvam: p34.sv.KEY = p34.sv.sarvam_key()
    print(f"[{a.model}] NL={NL} D={D} window {window[0]}-{window[-1]} coeffs={a.coeffs}", flush=True)

    def inputs(cue, texts):
        f = [tok.apply_chat_template([{"role":"system","content":cue},{"role":"user","content":t}],
             tokenize=False, add_generation_prompt=True) for t in texts]
        return tok(f, return_tensors="pt", padding=True).to("cuda:0")

    def capture_pp(cue, texts):                          # per-prompt last-token resid, all layers -> [N,NL,D]
        out = []
        for b0 in range(0, len(texts), BATCH):
            inp = inputs(cue, texts[b0:b0+BATCH]); g = {}
            hh = [layers[L].register_forward_hook((lambda L: lambda _m,_i,o: g.__setitem__(L,(o[0] if isinstance(o,tuple) else o)[:,-1,:].detach().float().cpu().numpy()))(L)) for L in range(NL)]
            with torch.no_grad(): model(**inp)
            for h in hh: h.remove()
            out.append(np.stack([g[L] for L in range(NL)], 1))
        return np.concatenate(out, 0)

    print("building eval direction + in-distribution null…", flush=True)
    ev = capture_pp(cev, dirtxt); dp = capture_pp(cdep, dirtxt)
    r_eval = (ev.mean(0) - dp.mean(0)).astype(np.float32)                 # [NL,D]
    pool = np.concatenate([ev, dp], 0); idx = np.random.default_rng(0).permutation(len(pool)); hlf = len(pool)//2
    r_null0 = (pool[idx[:hlf]].mean(0) - pool[idx[hlf:]].mean(0)).astype(np.float32)
    r_null = {L: r_null0[L] / (np.linalg.norm(r_null0[L])+1e-8) * np.linalg.norm(r_eval[L]) for L in window}  # norm-matched
    reval = {L: r_eval[L] for L in window}

    def gen(cue, texts, dirs, mode, coeff):
        resps = []
        for b0 in range(0, len(texts), BATCH):
            inp = inputs(cue, texts[b0:b0+BATCH]); n_in = inp["input_ids"].shape[1]; handles = []
            if dirs:
                for L in window:
                    if mode == "add":
                        vt = torch.as_tensor(coeff*dirs[L], dtype=torch.float16, device="cuda:0")
                        handles.append(layers[L].register_forward_hook((lambda vt: lambda _m,_i,o: ((o[0]+vt,)+tuple(o[1:])) if isinstance(o,tuple) else o+vt)(vt)))
                    else:
                        rt = torch.as_tensor(dirs[L]/(np.linalg.norm(dirs[L])+1e-8), dtype=torch.float16, device="cuda:0")
                        def mk(rt):
                            def hook(_m,_i,o):
                                is_t=isinstance(o,tuple); hs=o[0] if is_t else o
                                hs = hs - (hs@rt).unsqueeze(-1)*rt
                                return (hs,)+tuple(o[1:]) if is_t else hs
                            return hook
                        handles.append(layers[L].register_forward_hook(mk(rt)))
            try:
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
            finally:
                for h in handles: h.remove()
            resps += [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]
        return resps

    tag = f"{a.model}_{a.lang}"; ip = OUT_DIR / f"nonreasoning_{tag}_items.csv"; OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if ip.exists():
        done = set(pd.read_csv(ip)["label"].unique()); print(f"  [resume] done: {sorted(done)}", flush=True)
    def jr(label, its, resps):
        if label in done: print(f"  [resume] {label}", flush=True); return
        gl = p34.judge_gemma(its, resps, a.lang, key); rg = p34._rate(gl)
        sl = p34.judge_sarvam(its, resps, a.lang) if a.sarvam else [None]*len(its)
        rs = p34._rate(sl) if a.sarvam else float("nan")
        recs = [dict(model=a.model, label=label, item_id=it["id"], refuse_gemma=g, refuse_sarvam=s, deg=int(p34.degenerate(r)))
                for it, r, g, s in zip(its, resps, gl, sl)]
        pd.DataFrame(recs).to_csv(ip, mode="a", header=not ip.exists(), index=False)
        print(f"  {label:<22s} gemma={rg:5.0f}% sarvam={('%.0f%%'%rs) if a.sarvam else '  -- '} deg={100*np.mean([p34.degenerate(r) for r in resps]):3.0f}%", flush=True)
    jr("deploy baseline", test, gen(cdep, [it["text"] for it in test], None, None, 0))
    jr("eval baseline",   test, gen(cev,  [it["text"] for it in test], None, None, 0))
    for c in a.coeffs:
        jr(f"ADD eval c{c}",     test,     gen(cdep, [it["text"] for it in test], reval,  "add", c))
        jr(f"ADD null c{c}",     test,     gen(cdep, [it["text"] for it in test], r_null, "add", c))
        jr(f"harmless ADD c{c}", harmless, gen(cdep, [it["text"] for it in harmless], reval, "add", c))
    jr("eval ABL", test, gen(cev, [it["text"] for it in test], reval, "ablate", 1.0))
    # summary
    df = pd.read_csv(ip)
    s = df.groupby("label").agg(gemma=("refuse_gemma", lambda x: round(100*x.dropna().mean(),1)),
        n=("item_id","size")).reset_index()
    s.to_csv(OUT_DIR / f"nonreasoning_{tag}.csv", index=False)
    print("\nSWEEP SUMMARY (gemma refusal%):\n" + s.to_string(index=False))
    print("\nSPECIFIC iff some coeff has ADD-eval >> ADD-null AND harmless-ADD stays LOW (near benign baseline).", flush=True)


if __name__ == "__main__":
    main()
