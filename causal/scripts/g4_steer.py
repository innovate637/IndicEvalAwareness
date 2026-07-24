#!/usr/bin/env python3
"""
38 — NON-REASONING model contrast (H6), done carefully: coefficient sweep + IN-DISTRIBUTION null +
harmless specificity + per-condition checkpoint. On sarvam-m (reasoning) single-direction eval causality
is null; here we test whether it WORKS — specifically — on a non-reasoning model, and at which coeff.

Model default = Qwen2.5-7B-Instruct (non-reasoning, multilingual). Build per-layer eval direction
(eval-cue − deploy-cue diff-of-means) and a per-layer NORM-MATCHED label-shuffle null. Multi-layer
add/ablate every step over a window. Judge = gemma (sweep); sarvam added at confirmatory n via --sarvam.

FIX (2026-07-05): the harmful-prompt pool is capped at 100/language (see data/safety_prompts/*.json).
The original single-split design requested `items[a.n : a.n+64]` as a held-out direction-building set,
which silently returned EMPTY once `a.n + 64 > 100` (e.g. the overnight `--n 100` run) -> capture_pp got
zero batches -> `np.concatenate([], 0)` crashed with "need at least one array to concatenate". Fixed by
switching to the same 5-fold cross-validation design already validated on Hermes (`_hermes_cv.py`): build
the direction on 80% of the 100-item pool, test on the held-out 20%, rotate 5 ways -> full n=100 tested,
direction always held out from the item it's applied to. Also added the 768-token Indic prompt cap used
elsewhere in the project (Odia especially) to avoid an O(seq^2) OOM under the VM's memory cap.

  CUDA_VISIBLE_DEVICES=2 python scripts/g4_steer.py --n 100 --coeffs 0.1 0.25 --win 20-31 --lang en
  CUDA_VISIBLE_DEVICES=3 python scripts/g4_steer.py --n 100 --coeffs 0.1 0.25 --win 20-31 --lang hi
Outputs: results/causal/nonreasoning_{tag}_items.csv -> nonreasoning_{tag}.csv
"""
import os, sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import config
# --- gemma4 shim: .venv_gemma4 has no nnsight; behavioral_common only needs the import to resolve.
import types as _t
if "nnsight" not in sys.modules:
    _nn = _t.ModuleType("nnsight"); _nn.LanguageModel = object; sys.modules["nnsight"] = _nn
G4_PATH = str(ROOT / "models" / "gemma-4-31B-it-int8-foem")
P = importlib.util.spec_from_file_location("p34", str(ROOT / "scripts" / "34_ablate_eval.py"))
p34 = importlib.util.module_from_spec(P); P.loader.exec_module(p34)

MODELS = {"g4": G4_PATH}
OUT_DIR = config.RESULTS_DIR / "causal"
BATCH = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="g4", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--coeffs", nargs="+", type=float, default=[0.1, 0.25])
    ap.add_argument("--win", default="6-22"); ap.add_argument("--lang", default="en")
    ap.add_argument("--folds", type=int, default=5, help="k-fold CV so direction is always held out from its test item")
    ap.add_argument("--sarvam", action="store_true", help="also sarvam-105b judge (confirmatory; real-money)")
    a = ap.parse_args()
    name = MODELS[a.model]; MODEL = name
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, device_map="auto", dtype=torch.float16, low_cpu_mem_usage=True).eval()
    NL = model.config.text_config.num_hidden_layers; D = model.config.text_config.hidden_size; layers = model.model.language_model.layers
    lo, hi = a.win.split("-"); window = list(range(int(lo), min(int(hi)+1, NL)))
    battery, _, tr = p34.s09.load_cue_sources()
    cdep = p34.s09.get_cue(a.lang, "deploy", battery, tr); cev = p34.s09.get_cue(a.lang, "eval_explicit", battery, tr)
    items = p34.s14.load_harmful_subset(a.lang, a.n)                 # pool is capped at 100/lang; test == full pool
    harmless = p34.load_harmless(a.lang, a.n)
    key = p34.s30.or_key()
    if a.sarvam: p34.sv.KEY = p34.sv.sarvam_key()
    print(f"[{a.model}] lang={a.lang} NL={NL} D={D} window {window[0]}-{window[-1]} coeffs={a.coeffs} "
          f"folds={a.folds} pool={len(items)}", flush=True)

    def inputs(cue, texts):
        f = []
        for t in texts:                                  # 768-tok cap: long Indic (esp. Odia) prompts -> O(seq^2) OOM guard
            pids = tok(t, add_special_tokens=False)["input_ids"]
            if len(pids) > 768: t = tok.decode(pids[:768])
            f.append(tok.apply_chat_template([{"role":"system","content":cue},{"role":"user","content":t}],
                     tokenize=False, add_generation_prompt=True))
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

    def build_dirs(idx):                                  # direction from items at `idx` only (held out from test fold)
        txt = [items[i]["text"] for i in idx]
        ev = capture_pp(cev, txt); dp = capture_pp(cdep, txt)
        r = (ev.mean(0) - dp.mean(0)).astype(np.float32)                 # [NL,D]
        pool = np.concatenate([ev, dp], 0); pidx = np.random.default_rng(0).permutation(len(pool)); hlf = len(pool)//2
        n0 = (pool[pidx[:hlf]].mean(0) - pool[pidx[hlf:]].mean(0)).astype(np.float32)
        reval = {L: r[L] for L in window}
        rnull = {L: n0[L] / (np.linalg.norm(n0[L])+1e-8) * np.linalg.norm(r[L]) for L in window}   # norm-matched
        return reval, rnull

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

    _langtag = "" if a.lang == "en" else f"_{a.lang}"
    tag = f"{a.model}{_langtag}_n{a.n}"; ip = OUT_DIR / f"nonreasoning_{tag}_items.csv"; OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if ip.exists():
        d0 = pd.read_csv(ip); done = set(zip(d0.fold.astype(str), d0.label)); print(f"  [resume] {len(done)} (fold,label) done", flush=True)
    def jr(fold, label, its, resps):
        if (str(fold), label) in done: print(f"  [resume] fold{fold} {label}", flush=True); return
        gl = p34.judge_gemma(its, resps, a.lang, key); rg = p34._rate(gl)
        sl = p34.judge_sarvam(its, resps, a.lang) if a.sarvam else [None]*len(its)
        rs = p34._rate(sl) if a.sarvam else float("nan")
        recs = [dict(model=a.model, fold=fold, label=label, item_id=it["id"], refuse_gemma=g, refuse_sarvam=s, deg=int(p34.degenerate(r)))
                for it, r, g, s in zip(its, resps, gl, sl)]
        pd.DataFrame(recs).to_csv(ip, mode="a", header=not ip.exists(), index=False)
        print(f"  fold{fold} {label:<22s} gemma={rg:5.0f}% sarvam={('%.0f%%'%rs) if a.sarvam else '  -- '} deg={100*np.mean([p34.degenerate(r) for r in resps]):3.0f}% n={len(its)}", flush=True)

    order = np.random.default_rng(0).permutation(len(items))
    for f in range(a.folds):
        te = order[f::a.folds]; trn = np.array([i for i in order if i not in set(te)])
        reval, r_null = build_dirs(trn)
        tt = [items[i] for i in te]; txt = [items[i]["text"] for i in te]
        jr(f, "deploy baseline", tt, gen(cdep, txt, None, None, 0))
        jr(f, "eval baseline",   tt, gen(cev,  txt, None, None, 0))
        for c in a.coeffs:
            jr(f, f"ADD eval c{c}", tt, gen(cdep, txt, reval,  "add", c))
            jr(f, f"ADD null c{c}", tt, gen(cdep, txt, r_null, "add", c))
        jr(f, "eval ABL", tt, gen(cev, txt, reval, "ablate", 1.0))

    # harmless collateral: full-pool direction (no held-out item to leak into, since harmless != harmful pool)
    reval_full, _ = build_dirs(np.arange(len(items))); htxt = [h["text"] for h in harmless]
    for c in a.coeffs:
        jr("H", f"harmless ADD c{c}", harmless, gen(cdep, htxt, reval_full, "add", c))

    # summary
    df = pd.read_csv(ip)
    s = df.groupby("label").agg(gemma=("refuse_gemma", lambda x: round(100*x.dropna().mean(),1)),
        n=("item_id","size")).reset_index()
    s.to_csv(OUT_DIR / f"nonreasoning_{tag}.csv", index=False)
    print("\nSWEEP SUMMARY (gemma refusal%):\n" + s.to_string(index=False))
    print("\nSPECIFIC iff some coeff has ADD-eval >> ADD-null AND harmless-ADD stays LOW (near benign baseline).", flush=True)


if __name__ == "__main__":
    main()
