#!/usr/bin/env python3
"""
39 — REAL activation patching (interchange) on Hermes-3-Llama-3.1-**70B** (4-bit NF4, 80 layers, NON-reasoning).

This is PATCHING, not steering: we do NOT add a vector. We capture the eval-cue residual at layer L,
last prompt-token, and REPLACE the deploy-cue run's residual at that exact site with it, then generate
and judge the actual refusal behavior. The cue interchange (deploy<-eval) at a single layer/position.

Why this should not corrupt (it did on sarvam): sarvam was int8 (re-injecting fp32 into int8 matmuls
is not bit-reproducible) + reasoning (<think> desync). Here 4-bit is WEIGHT-only; the residual stream is
bfloat16, so capture (bf16→float32) and re-inject (float32→bf16) are exact and the IDENTITY patch
(deploy<-deploy) must be a near-exact no-op. That identity condition is the built-in corruption test: if
identity stays coherent and ~= deploy clean, patching is valid.

Conditions per layer L (single layer, last prompt-token, prefill only):
  deploy clean | eval clean | deploy<-deploy IDENTITY (no-op test) | deploy<-EVAL (the patch) |
  deploy<-EVAL mismatched-item (in-distribution magnitude-matched null) | harmless<-EVAL (collateral)
Dual judge gemma+sarvam; per-(L,label) checkpoint; saves response text + degeneration flag.

  python scripts/39_patch_hermes.py --layers 15 25 35 45 55 --n 100   # single layers rescaled 32->80
Outputs: results/causal/patch_hermes_items.csv  ->  patch_hermes.csv (summary)
"""
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import config
P = importlib.util.spec_from_file_location("p34", str(ROOT / "scripts" / "34_ablate_eval.py"))
p34 = importlib.util.module_from_spec(P); P.loader.exec_module(p34)

MODEL = "unsloth/Hermes-3-Llama-3.1-70B-bnb-4bit"; BATCH = 8; MAX_NEW = 200; LANG = "en"
OUT = config.RESULTS_DIR / "causal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=[15, 25, 35, 45, 55])
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cuda:0").eval()
    NL = model.config.num_hidden_layers; layers = model.model.layers
    b, _, tr = p34.s09.load_cue_sources()
    cdep = p34.s09.get_cue(LANG, "deploy", b, tr); cev = p34.s09.get_cue(LANG, "eval_explicit", b, tr)
    items = p34.s14.load_harmful_subset(LANG, a.n); test = items
    harmless = p34.load_harmless(LANG, a.n)
    key = p34.s30.or_key()
    SP = importlib.util.spec_from_file_location("_sarvam_pool", str(ROOT / "scripts" / "_sarvam_pool.py")); _sp = importlib.util.module_from_spec(SP); SP.loader.exec_module(_sp); _sp.install(p34.sv)  # 3-key rotation (key 1 dead)
    txt = [it["text"] for it in test]; htxt = [it["text"] for it in harmless]
    rng = np.random.default_rng(0)
    print(f"[patch hermes] NL={NL} layers={a.layers} n={len(test)}", flush=True)

    def inp(cue, texts):
        f = [tok.apply_chat_template([{"role":"system","content":cue},{"role":"user","content":t}],
             tokenize=False, add_generation_prompt=True) for t in texts]
        return tok(f, return_tensors="pt", padding=True).to("cuda:0")

    def cap_L(cue, texts, L):                       # last prompt-token residual at layer L, [N,D]
        out = []
        for i in range(0, len(texts), BATCH):
            x = inp(cue, texts[i:i+BATCH]); g = {}
            def hook(_m, _i, o):
                hs = o[0] if isinstance(o, tuple) else o
                g["v"] = hs[:, -1, :].detach().float().cpu().numpy()
            h = layers[L].register_forward_hook(hook)
            with torch.no_grad(): model(**x)
            h.remove(); out.append(g["v"])
        return np.concatenate(out, 0)

    def gen_patch(cue, texts, L, donor):            # donor [N,D] or None; patch prefill last-pos only
        R = []
        for i in range(0, len(texts), BATCH):
            x = inp(cue, texts[i:i+BATCH]); n_in = x["input_ids"].shape[1]; H = []
            if donor is not None:
                vt = torch.as_tensor(donor[i:i+BATCH], dtype=torch.bfloat16, device="cuda:0")
                def hook(_m, _i, o):
                    is_t = isinstance(o, tuple); hs = o[0] if is_t else o
                    if hs.shape[1] > 1:             # prefill only; leave decode steps untouched
                        hs = hs.clone(); hs[:, -1, :] = vt.to(hs.dtype)
                    return (hs,)+tuple(o[1:]) if is_t else hs
                H.append(layers[L].register_forward_hook(hook))
            try:
                with torch.no_grad():
                    o = model.generate(**x, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
            finally:
                for h in H: h.remove()
            R += [tok.decode(t[n_in:], skip_special_tokens=True).strip() for t in o]
        return R

    OUT.mkdir(parents=True, exist_ok=True); ip = OUT / "patch_hermes_items.csv"
    done = set()
    if ip.exists():
        d0 = pd.read_csv(ip); done = set(zip(d0.L, d0.label)); print(f"  [resume] {len(done)} (L,label) done", flush=True)

    def jr(L, label, its, resps):
        if (L, label) in done: print(f"  [resume] L{L} {label}", flush=True); return
        gl = p34.judge_gemma(its, resps, LANG, key); sl = p34.judge_sarvam(its, resps, LANG)
        recs = [dict(L=L, label=label, iid=it["id"], gemma=g, sarvam=s, deg=int(p34.degenerate(r)), resp=str(r)[:400])
                for it, r, g, s in zip(its, resps, gl, sl)]
        pd.DataFrame(recs).to_csv(ip, mode="a", header=not ip.exists(), index=False)
        rg = p34._rate(gl); rs = p34._rate(sl); dg = 100*np.mean([p34.degenerate(r) for r in resps])
        print(f"  L{L:02d} {label:<22s} gemma={rg:5.0f}% sarvam={rs:5.0f}% deg={dg:3.0f}%", flush=True)

    # clean baselines (layer-independent) stored under L=-1
    jr(-1, "deploy clean", test, gen_patch(cdep, txt, 0, None))
    jr(-1, "eval clean",   test, gen_patch(cev,  txt, 0, None))
    for L in a.layers:
        dep_resid = cap_L(cdep, txt, L)             # for identity no-op
        ev_resid  = cap_L(cev,  txt, L)             # donor: eval-cue residual, same items
        mm        = ev_resid[rng.permutation(len(ev_resid))]   # in-dist magnitude-matched null
        ev_h      = cap_L(cev, htxt, L)             # harmless donor
        jr(L, "deploy<-deploy IDENT", test,     gen_patch(cdep, txt,  L, dep_resid))
        jr(L, "deploy<-EVAL patch",   test,     gen_patch(cdep, txt,  L, ev_resid))
        jr(L, "deploy<-EVAL mismatch",test,     gen_patch(cdep, txt,  L, mm))
        jr(L, "harmless<-EVAL patch", harmless, gen_patch(cdep, htxt, L, ev_h))

    df = pd.read_csv(ip)
    s = df.groupby(["L", "label"]).agg(
        gemma=("gemma", lambda x: round(100*x.dropna().mean(), 1)),
        sarvam=("sarvam", lambda x: round(100*x.dropna().mean(), 1)),
        deg=("deg", lambda x: round(100*x.mean(), 1)), n=("iid", "size")).reset_index()
    s.to_csv(OUT / "patch_hermes.csv", index=False)
    print("\n=== PATCH SUMMARY ===\n" + s.to_string(index=False), flush=True)
    print("\nVALID iff IDENTITY ~= deploy clean & deg~0. POSITIVE iff deploy<-EVAL raises refusal toward "
          "eval, beating mismatch-null, with harmless staying low.", flush=True)


if __name__ == "__main__":
    main()
