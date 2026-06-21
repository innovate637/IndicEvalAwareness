#!/usr/bin/env python3
"""
26 — CAUSAL MANIPULATION CHECK (readout A): does injecting the EVAL-AWARENESS direction into a
     DEPLOY forward pass push the model into the eval-representation — cross-lingually?

Counterfactual sufficiency test. Clean = deploy context. We ADD alpha*Δ at a MID layer L_steer
(Δ = unit eval_log−neutral_log, orthogonalized vs refusal) during a single forward pass over a
DEPLOY-framed harmful prompt, then capture the residual at L_read = L_steer+2 and score a
CLEAN-TRAINED eval-probe (logistic + diff-of-means projection, trained on eval_log-vs-neutral_log).

Cross-lingual: inject the ENGLISH Δ into Indic forward passes (and Indic Δ into English), read with
the EN-trained probe. If the probe score RISES with +alpha (falls with −alpha, flat for a random
matched-norm direction), the eval direction CAUSALLY induces the eval-state — and across languages.

Forward-pass ONLY (no generation) → ~1-2s/item batched, so all 100 items × full α-sweep is cheap.
Reading at L_read = L_steer+2 (downstream of injection) tests PROPAGATION, not tautology.

  python scripts/26_steer_probe_readout.py --L 9 --n 100 --strengths -0.5 -0.25 0 0.25 0.5 --resume
Outputs: results/steering/probe_readout/readoutA_{kind}_{src}2{tgt}_L{L}.csv  (resume-safe)
"""
import os, sys, json, pickle, argparse, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import config
from behavioral_common import load_model_and_tokenizer


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

s25 = _load("s25", "25_steer_xlang.py")   # eval_dir_orth, build_random_direction, refusal_dir_at, layer_norm_scale
s14 = _load("s14", "14_steer_patch.py")   # load_harmful_subset
s09 = _load("s09", "09_cue_battery_grid.py")  # get_cue, load_cue_sources

OUT_DIR = config.STEER_DIR / "probe_readout"
MAX_PROMPT_TOKENS = 768
BATCH = 16


def acts(lang, cond):
    p = config.ACT_DIR / f"{lang}_{cond}.pkl"
    return np.asarray(pickle.load(open(p, "rb"))["acts"], dtype=np.float32) if p.exists() else None


def build_eval_probe(train_lang, L_read):
    """Clean eval-probe at L_read, trained on eval_log vs neutral_log (anti-trivial framing contrast).
    Returns (unit diff-of-means direction, fitted logistic pipeline)."""
    e = acts(train_lang, "eval_log")[:, L_read, :]
    n = acts(train_lang, "neutral_log")[:, L_read, :]
    X = np.vstack([e, n]); y = np.r_[np.ones(len(e)), np.zeros(len(n))]
    dom = e.mean(0) - n.mean(0); dom = dom / (np.linalg.norm(dom) + 1e-9)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X, y)
    return dom.astype(np.float32), clf


def capture_steered(hf, tokenizer, formatted, L_steer, add_vec, L_read):
    """One forward pass over `formatted` (list of chat-templated strings); add `add_vec` at L_steer
    (if not None) and capture last-token resid at L_read. Returns [B, D] numpy."""
    import torch
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(formatted, return_tensors="pt", padding=True).to("cuda:0")
    grabbed = {}
    handles = []
    if add_vec is not None:
        av = add_vec.to("cuda:0")

        def steer(_m, _i, out):
            is_t = isinstance(out, tuple); hs = out[0] if is_t else out
            hs = hs + av.to(hs.dtype)
            return (hs,) + tuple(out[1:]) if is_t else hs
        handles.append(hf.model.layers[L_steer].register_forward_hook(steer))

    def grab(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        grabbed["x"] = hs[:, -1, :].detach().float().cpu().numpy()
    handles.append(hf.model.layers[L_read].register_forward_hook(grab))
    try:
        with torch.no_grad():
            hf(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    finally:
        for h in handles:
            h.remove()
    return grabbed["x"]


def fmt(tokenizer, cue, text):
    pids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(pids) > MAX_PROMPT_TOKENS:
        text = tokenizer.decode(pids[:MAX_PROMPT_TOKENS])
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": cue}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True)


def run_cell(model, tokenizer, dom, clf, src, tgt, probe_lang, kind, vec_t, L_steer, L_read,
             scale, alphas, items, cue):
    import torch
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp = OUT_DIR / f"readoutA_{kind}_{src}2{tgt}_L{L_steer}.csv"
    done = set()
    if sp.exists():
        for _, r in pd.read_csv(sp).iterrows():
            done.add((float(r["alpha"]), str(r["item_id"])))
    header = not sp.exists()
    hf = model._model
    sweep = [0.0] + sorted(a for a in set(float(x) for x in alphas) if a != 0.0)
    vt = torch.tensor(vec_t, dtype=torch.float32)
    for alpha in sweep:
        todo = [it for it in items if (alpha, str(it.get("id", ""))) not in done]
        add_vec = None if alpha == 0.0 else (alpha * scale * vt)
        for b0 in tqdm(range(0, len(todo), BATCH), desc=f"{kind} {src}2{tgt} a={alpha}", leave=False):
            batch = todo[b0:b0 + BATCH]
            formatted = [fmt(tokenizer, cue, it["text"]) for it in batch]
            X = capture_steered(hf, tokenizer, formatted, L_steer, add_vec, L_read)   # [B,D]
            proj = X @ dom                                  # diff-of-means projection
            prob = clf.predict_proba(X)[:, 1]               # logistic P(eval)
            rows = [{"src": src, "tgt": tgt, "probe_lang": probe_lang, "kind": kind,
                     "L_steer": L_steer, "L_read": L_read, "alpha": alpha,
                     "item_id": str(it.get("id", "")), "proj_score": float(p), "prob_score": float(q)}
                    for it, p, q in zip(batch, proj, prob)]
            pd.DataFrame(rows).to_csv(sp, mode="a", header=header, index=False); header = False
            done.update((alpha, str(it.get("id", ""))) for it in batch)
    return sp


def build_jobs(langs):
    """(src_lang, tgt_lang, probe_lang, kind). Eval dir from src injected into tgt's forward pass,
    read with probe_lang's probe. en is the transfer source."""
    jobs = []
    jobs.append(("en", "en", "en", "eval"))           # within-lang sufficiency / smoke
    jobs.append(("en", "en", "en", "random"))         # negative control
    for tgt in [l for l in langs if l != "en"]:
        jobs.append(("en", tgt, "en", "eval"))        # EN -> Indic (cross-lingual headline)
    for src in [l for l in langs if l != "en"]:
        jobs.append((src, "en", "en", "eval"))        # Indic -> EN (reverse)
    jobs.append(("en", "bn", "en", "random"))         # negative control in an Indic target
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=9, help="L_steer (inject here; read at L+2)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--strengths", nargs="+", type=float, default=[-0.5, -0.25, 0.0, 0.25, 0.5])
    ap.add_argument("--langs", nargs="+", default=["en", "hi", "bn", "ta", "te"])
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    L_steer = a.L; L_read = a.L + 2

    probe = build_eval_probe("en", L_read)           # EN-trained clean eval-probe at L_read
    print(f"eval-probe trained on EN @ L_read={L_read}")
    battery, framings, translations = s09.load_cue_sources()
    model, tokenizer = load_model_and_tokenizer(int8=True)
    print(f"model loaded. L_steer={L_steer} L_read={L_read} n={a.n} strengths={a.strengths}")

    jobs = build_jobs(a.langs)
    print("jobs: " + ", ".join(f"{k}:{s}2{t}" for s, t, _, k in jobs))
    for src, tgt, probe_lang, kind in jobs:
        # direction at the STEER layer
        if kind == "eval":
            vec = s25.eval_dir_orth(src, L_steer)
        elif kind == "random":
            vec = s25.build_random_direction(L_steer)["vector"]
        elif kind == "refusal":
            vec = s25.refusal_dir_at(src, L_steer)
        if vec is None:
            print(f"  [skip] {kind} {src} (no direction at L{L_steer})"); continue
        scale = s25.layer_norm_scale(tgt, L_steer) or 1.0
        cue = s09.get_cue(tgt, "deploy", battery, translations)
        items = s14.load_harmful_subset(tgt, a.n)
        if cue is None or not items:
            print(f"  [skip] {kind} {src}2{tgt} (cue={'ok' if cue else 'MISSING'}, items={len(items) if items else 0})"); continue
        dom, clf = probe
        sp = run_cell(model, tokenizer, dom, clf, src, tgt, probe_lang, kind, vec,
                      L_steer, L_read, scale, a.strengths, items, cue)
        d = pd.read_csv(sp)
        g = d.groupby("alpha")["prob_score"].mean()
        print(f"  [{kind} {src}2{tgt}] P(eval) by alpha: " + "  ".join(f"{al:+.2f}:{v:.2f}" for al, v in g.items()))
    print("\nDone -> results/steering/probe_readout/")


if __name__ == "__main__":
    main()
