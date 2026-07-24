#!/usr/bin/env python3
"""
37 — H5: CLEAN single-forward interchange patch + REFUSAL-PROBE readout (no generation -> no corruption,
no blind next-token). Patch eval's residual at layer L into a DEPLOY forward pass; read how much the
downstream REFUSAL representation moves. Localizes the cue->refusal-representation pathway.

  probe = projection of last-token residual at L_read onto an INDEPENDENT refusal direction
          (refused−complied diff-of-means; built from cached labels, independent of the eval/deploy cue).
  frac(L) = (probe[deploy + patch eval@L] − probe[deploy]) / (probe[eval] − probe[deploy]).
Control = mismatched-item patch (carries the SAME eval-component magnitude -> isolates item-specific
causal propagation beyond additive carry). Sweep patch layer L (< L_read).

  CUDA_VISIBLE_DEVICES=1 python scripts/37_singlefwd_patch.py --tgt hi --n 40
Outputs: results/causal/h5_singlefwd_{tgt}.csv
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
NL = config.NUM_LAYERS; BATCH = 8


def patch_capture(hf, tok, cue, texts, L, source, L_read):
    """forward `cue` prompts; replace last-token residual at L with source[i]; grab last-token residual
    at L_read. source=None -> clean. returns [N,D] at L_read."""
    out = []
    for b0 in range(0, len(texts), BATCH):
        inp = p34.build_inputs(tok, cue, texts[b0:b0+BATCH]); grabbed = {}; handles = []
        if source is not None:
            vt = torch.as_tensor(source[b0:b0+BATCH], dtype=torch.float32, device="cuda:0")
            def ph(_m, _i, o):
                is_t = isinstance(o, tuple); hs = o[0] if is_t else o
                hs = hs.clone(); hs[:, -1, :] = vt.to(hs.dtype)
                return (hs,)+tuple(o[1:]) if is_t else hs
            handles.append(hf.model.layers[L].register_forward_hook(ph))
        def gh(_m, _i, o):
            hs = o[0] if isinstance(o, tuple) else o
            grabbed["x"] = hs[:, -1, :].detach().float().cpu().numpy()
        handles.append(hf.model.layers[L_read].register_forward_hook(gh))
        try:
            with torch.no_grad():
                hf(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        finally:
            for h in handles: h.remove()
        out.append(grabbed["x"])
    return np.concatenate(out, 0)


def boot_specific(pm, pn, dep, ev, nboot=2000, seed=0):
    """bootstrap CI for specific = mean(pm-pn)/mean(ev-dep), resampling items (paired)."""
    pm, pn, dep, ev = map(np.asarray, (pm, pn, dep, ev)); n = len(pm); rng = np.random.default_rng(seed)
    g = ev.mean() - dep.mean()
    pe = (pm.mean() - pn.mean()) / g if g else float("nan")
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, n, n); gg = ev[i].mean() - dep[i].mean()
        bs.append((pm[i].mean() - pn[i].mean()) / gg if gg else np.nan)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return pe, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tgt", default="hi"); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    battery, _, tr = p34.s09.load_cue_sources()
    cdep = p34.s09.get_cue(a.tgt, "deploy", battery, tr); cev = p34.s09.get_cue(a.tgt, "eval_explicit", battery, tr)
    items = p34.s14.load_harmful_subset(a.tgt, a.n); txt = [it["text"] for it in items]
    rd = p34.s14.build_refusal_direction(a.tgt)            # independent readout axis (refused−complied)
    L_read = int(rd["layer"]); rhat = rd["vector"].astype(np.float32)
    model, tok = load_model_and_tokenizer(int8=True); hf = model._model
    rng = np.random.default_rng(0)
    print(f"[H5 {a.tgt}] refusal readout @L{L_read} | capturing eval/deploy per-layer residuals…", flush=True)
    ev = p34.capture_acts(hf, tok, cev, txt)              # [N,NL,D]
    dp = p34.capture_acts(hf, tok, cdep, txt)
    pin = lambda M: float((M @ rhat).mean())               # probe = mean projection onto refusal axis
    probe_dep = pin(dp[:, L_read, :]); probe_ev = pin(ev[:, L_read, :])
    gap = probe_ev - probe_dep
    print(f"  clean refusal-probe: deploy={probe_dep:+.3f} eval={probe_ev:+.3f} gap={gap:+.3f}", flush=True)
    perm = rng.permutation(len(items))
    dep_proj = dp[:, L_read, :] @ rhat; ev_proj = ev[:, L_read, :] @ rhat     # per-item refs
    OUT_DIR.mkdir(parents=True, exist_ok=True); rows = []
    for L in range(0, L_read):                             # every patch layer upstream of readout
        pm = patch_capture(hf, tok, cdep, txt, L, ev[:, L, :], L_read) @ rhat          # [N]
        pn = patch_capture(hf, tok, cdep, txt, L, ev[perm][:, L, :], L_read) @ rhat    # mismatched null [N]
        spec, lo, hi = boot_specific(pm, pn, dep_proj, ev_proj)
        fm = (pm.mean() - probe_dep) / gap if gap else float("nan")
        sig = int(lo > 0 or hi < 0)
        rows.append(dict(tgt=a.tgt, L=L, L_read=L_read, frac_matched=round(fm,3),
                         specific=round(spec,3), lo=round(lo,3), hi=round(hi,3), sig=sig))
        print(f"  patch L{L:02d}: frac_matched={fm:+.2f}  specific={spec:+.3f} [{lo:+.3f},{hi:+.3f}]{' *' if sig else ''}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / f"h5_singlefwd_{a.tgt}.csv", index=False)
    pos = df[(df.specific > 0) & (df.sig == 1)]
    best = df.loc[df["specific"].idxmax()]
    print(f"\n[H5 {a.tgt}] peak specific={best['specific']:+.3f} [{best['lo']:+.3f},{best['hi']:+.3f}] at L{int(best['L'])}->L{L_read} "
          f"| {len(pos)} layer(s) with specific>0 AND CI excludes 0", flush=True)
    print("  POSITIVE iff specific CI>0 at some layer -> eval activation causally propagates to the refusal "
          "representation (beyond additive carry). (representational, not behavioral.)", flush=True)


if __name__ == "__main__":
    main()
