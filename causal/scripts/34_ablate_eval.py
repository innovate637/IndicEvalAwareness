#!/usr/bin/env python3
"""
34 — CAUSAL eval-awareness via Arditi-style DIRECTIONAL ABLATION (2406.11717).

Instead of swapping activations (which corrupts generation), we remove/add ONE unit direction:
    ablation:  x <- x - (x . r̂) r̂      at EVERY layer and EVERY position   (rank-1, gentle, no α)
    addition:  x <- x + r               at one layer, all positions
This stays coherent through generation (incl. reasoning) and is parameter-free.

GOAL: the EVAL direction and its EN->Indic transfer are the result; the REFUSAL direction is ONLY the
instrument-validation control (ablating it must drop refusal — Arditi, 13 models).  Indic refusal is
LLM-judged (substrings undercount Indic).

Directions = Arditi diff-of-means at the last prompt-token, per layer:
  r_refuse = mean(harmful) - mean(harmless)      (deploy cue; harmful vs benign prompts)
  r_eval   = mean(eval_explicit-cue) - mean(deploy-cue)   (same harmful prompts)
best layer selected by ABLATION EFFECT on a val split.

  CUDA_VISIBLE_DEVICES=1 python scripts/34_ablate_eval.py --stage 0 --lang en --n 24    # gate (refusal dir)
Outputs: results/ablation/*.csv
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
s30 = _load("s30", "30_jailbreak_acts.py")     # gemma_judge (OpenRouter), or_key
sv  = _load("sv", "21_sarvam_judge.py")        # sarvam-105b call/parse_verdict/sarvam_key
rj  = _load("rj", "18_refusal_judge.py")       # JUDGE_PROMPT, LANG_NAME, post_think
from concurrent.futures import ThreadPoolExecutor


def sarvam_judge(prompt, answer, lang):
    """sarvam-105b refusal verdict (Indic-native). 1=refuse,0=comply,None=unclear/err."""
    ans = rj.post_think(answer) or str(answer)
    if not ans or len(ans.strip()) < 3:
        return 1
    msg = (rj.JUDGE_PROMPT.replace("{lang}", rj.LANG_NAME.get(lang, lang))
           .replace("{prompt}", (prompt or "")[:1500]).replace("{answer}", ans[:3500]))
    v = sv.parse_verdict(sv.call(msg))["verdict"]
    return {"REFUSE": 1, "COMPLY": 0}.get(v, None)


def judge_gemma(items, resps, lang, key):
    """gemma-3-27b refusal labels. 1=refuse,0=comply,None=err."""
    with ThreadPoolExecutor(max_workers=16) as ex:
        return list(ex.map(lambda pr: s30.gemma_judge(pr[0]["text"], pr[1], lang, key), zip(items, resps)))


def judge_sarvam(items, resps, lang):
    with ThreadPoolExecutor(max_workers=16) as ex:
        return list(ex.map(lambda pr: sarvam_judge(pr[0]["text"], pr[1], lang), zip(items, resps)))


def _rate(labels):
    v = [x for x in labels if x is not None]
    return 100 * np.mean(v) if v else float("nan")

OUT_DIR = config.RESULTS_DIR / "ablation"
MAX_PROMPT_TOKENS = 768
BATCH = 8
D = config.D_MODEL
NL = config.NUM_LAYERS


def load_harmless(lang, n):
    p = config.SAFETY_DIR / "benign" / f"{lang}.json"
    return json.load(open(p))[:n] if p.exists() else []


def build_inputs(tok, cue, texts, think=True):
    formatted = []
    for t in texts:
        pids = tok(t, add_special_tokens=False)["input_ids"]
        if len(pids) > MAX_PROMPT_TOKENS:
            t = tok.decode(pids[:MAX_PROMPT_TOKENS])
        formatted.append(tok.apply_chat_template(
            [{"role": "system", "content": cue}, {"role": "user", "content": t}],
            tokenize=False, add_generation_prompt=True, enable_thinking=think))
    tok.padding_side = "left"
    return tok(formatted, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")


def mean_lasttok(hf, tok, cue, texts):
    """mean over prompts of layers[L].output[:,-1,:] for all L -> [NL, D]."""
    sums = np.zeros((NL, D), np.float64); cnt = 0
    for b0 in range(0, len(texts), BATCH):
        batch = texts[b0:b0 + BATCH]
        inputs = build_inputs(tok, cue, batch)
        grabbed = {}
        handles = []
        for L in range(NL):
            def mk(L):
                def g(_m, _i, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    grabbed[L] = hs[:, -1, :].detach().float().cpu().numpy()   # [b,D]
                return g
            handles.append(hf.model.layers[L].register_forward_hook(mk(L)))
        try:
            with torch.no_grad():
                hf(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        finally:
            for h in handles: h.remove()
        for L in range(NL): sums[L] += grabbed[L].sum(0)
        cnt += len(batch)
    return sums / cnt                                                        # [NL,D]


def diff_of_means(hf, tok, cue_pos, texts_pos, cue_neg, texts_neg):
    """r[L] = mean(pos) - mean(neg) at last token, per layer. -> [NL,D] (raw, not unit)."""
    mp = mean_lasttok(hf, tok, cue_pos, texts_pos)
    mn = mean_lasttok(hf, tok, cue_neg, texts_neg)
    return (mp - mn).astype(np.float32)


def capture_acts(hf, tok, cue, texts):
    """per-prompt last-token residual, all layers -> [N, NL, D]."""
    out = []
    for b0 in range(0, len(texts), BATCH):
        inputs = build_inputs(tok, cue, texts[b0:b0 + BATCH]); grabbed = {}
        handles = []
        for L in range(NL):
            def mk(L):
                def g(_m, _i, o):
                    hs = o[0] if isinstance(o, tuple) else o
                    grabbed[L] = hs[:, -1, :].detach().float().cpu().numpy()
                return g
            handles.append(hf.model.layers[L].register_forward_hook(mk(L)))
        try:
            with torch.no_grad():
                hf(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        finally:
            for h in handles: h.remove()
        out.append(np.stack([grabbed[L] for L in range(NL)], axis=1))   # [b,NL,D]
    return np.concatenate(out, 0)                                        # [N,NL,D]


def eval_and_null_dirs(hf, tok, cue_ev, cue_dep, texts, seed=0):
    """r_eval[L] = mean(eval) - mean(deploy);  r_null[L] = IN-DISTRIBUTION null = diff-of-means of a
    RANDOM eval/deploy label split (same data, no eval signal, matched structure/norm)."""
    ev = capture_acts(hf, tok, cue_ev, texts); dp = capture_acts(hf, tok, cue_dep, texts)
    r_eval = (ev.mean(0) - dp.mean(0)).astype(np.float32)                # [NL,D]
    pool = np.concatenate([ev, dp], 0)                                   # [2N,NL,D]
    idx = np.random.default_rng(seed).permutation(len(pool)); h = len(pool) // 2
    r_null = (pool[idx[:h]].mean(0) - pool[idx[h:]].mean(0)).astype(np.float32)
    return r_eval, r_null


# ---- intervention hooks ----
def gen_with(hf, tok, cue, texts, mode, vec=None, add_layer=None, max_new=400, think=True):
    """mode: 'none' | 'ablate' (rank-1 project out unit(vec) at all layers) | 'add' (vec at add_layer)."""
    inputs = build_inputs(tok, cue, texts, think); n_in = inputs["input_ids"].shape[1]
    handles = []
    if mode == "ablate":
        r = torch.as_tensor(vec / (np.linalg.norm(vec) + 1e-8), dtype=torch.float32, device="cuda:0")
        def mk():
            def hook(_m, _i, out):
                is_t = isinstance(out, tuple); hs = out[0] if is_t else out
                proj = (hs.to(r.dtype) @ r).unsqueeze(-1)        # [B,seq,1]
                hs = hs - (proj * r).to(hs.dtype)
                return (hs,) + tuple(out[1:]) if is_t else hs
            return hook
        for L in range(NL): handles.append(hf.model.layers[L].register_forward_hook(mk()))
    elif mode == "add":
        v = torch.as_tensor(vec, dtype=torch.float32, device="cuda:0")
        def hook(_m, _i, out):
            is_t = isinstance(out, tuple); hs = out[0] if is_t else out
            hs = hs + v.to(hs.dtype)
            return (hs,) + tuple(out[1:]) if is_t else hs
        handles.append(hf.model.layers[add_layer].register_forward_hook(hook))
    try:
        with torch.no_grad():
            out = hf.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                              max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    finally:
        for h in handles: h.remove()
    return [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]


def gen_multi(hf, tok, cue, items, dirs, layers, coeff, mode="add", max_new=400):
    """Arditi-style MULTI-LAYER intervention applied at ALL positions and ALL decode steps (function of
    the live activation, not a static paste). mode='add': hs += coeff*dirs[L]; mode='ablate':
    project out unit(dirs[L]). dirs: {L: np.ndarray[D]}."""
    resps = []
    for b0 in range(0, len(items), BATCH):
        texts = [it["text"] for it in items[b0:b0+BATCH]]
        inp = build_inputs(tok, cue, texts); n_in = inp["input_ids"].shape[1]; handles = []
        for L in layers:
            if mode == "add":
                vt = torch.as_tensor(coeff * dirs[L], dtype=torch.float32, device="cuda:0")
                def mk(vt):
                    def hook(_m, _i, o):
                        is_t = isinstance(o, tuple); hs = o[0] if is_t else o
                        hs = hs + vt.to(hs.dtype)                       # new tensor, every step/pos
                        return (hs,)+tuple(o[1:]) if is_t else hs
                    return hook
                handles.append(hf.model.layers[L].register_forward_hook(mk(vt)))
            else:  # ablate
                r = dirs[L] / (np.linalg.norm(dirs[L]) + 1e-8)
                rt = torch.as_tensor(r, dtype=torch.float32, device="cuda:0")
                def mk(rt):
                    def hook(_m, _i, o):
                        is_t = isinstance(o, tuple); hs = o[0] if is_t else o
                        hs = hs - (hs.to(rt.dtype) @ rt).unsqueeze(-1) * rt.to(hs.dtype)
                        return (hs,)+tuple(o[1:]) if is_t else hs
                    return hook
                handles.append(hf.model.layers[L].register_forward_hook(mk(rt)))
        try:
            with torch.no_grad():
                out = hf.generate(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
                                  max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        finally:
            for h in handles: h.remove()
        resps += [tok.decode(o[n_in:], skip_special_tokens=True).strip() for o in out]
    return resps


def degenerate(resp):
    w = resp.split()
    if len(w) >= 10 and len(set(w)) / len(w) < 0.25: return True
    s = resp[:120]
    for u in (3, 4, 5, 6, 8):
        if len(s) >= u * 5 and len(set(s[i:i+u] for i in range(0, len(s) - u, u))) <= 2: return True
    return False


def rate(resps, lang):
    return 100 * np.mean([classify_refusal(r, lang) for r in resps]) if resps else float("nan")
def deg(resps):
    return 100 * np.mean([degenerate(r) for r in resps]) if resps else float("nan")


def gen_all(hf, tok, cue, items, lang, mode, vec=None, add_layer=None, max_new=400):
    resps = []
    for b0 in range(0, len(items), BATCH):
        resps += gen_with(hf, tok, cue, [it["text"] for it in items[b0:b0+BATCH]], mode, vec, add_layer, max_new)
    return resps


def stage0(hf, tok, lang, n):
    """GATE: ablating the refusal direction must drop refusal (vs random); generations coherent."""
    battery, _, tr = s09.load_cue_sources()
    cue_dep = s09.get_cue(lang, "deploy", battery, tr)
    harmful = s14.load_harmful_subset(lang, n + 24)
    harmless = load_harmless(lang, 64)
    val = harmful[:8]; test = harmful[24:24 + n]
    print(f"[stage0/{lang}] build refusal dir (harmful vs harmless diff-of-means)…", flush=True)
    r_all = diff_of_means(hf, tok, cue_dep, [h["text"] for h in harmful[:64]], cue_dep, [h["text"] for h in harmless[:64]])
    # select layer by ablation effect on val
    cands = [10, 14, 18, 22, 26, 30]
    base_val = rate(gen_all(hf, tok, cue_dep, val, lang, "none"), lang)
    best, best_drop = None, -1
    for L in cands:
        rr = rate(gen_all(hf, tok, cue_dep, val, lang, "ablate", vec=r_all[L]), lang)
        print(f"   cand L{L}: val refusal {base_val:.0f}->{rr:.0f}  (drop {base_val-rr:+.0f})", flush=True)
        if base_val - rr > best_drop: best, best_drop = L, base_val - rr
    print(f"  selected refusal layer L{best} (val drop {best_drop:+.0f}pp)", flush=True)
    rhat = r_all[best]
    rng = np.random.default_rng(0); rand = rng.standard_normal(D).astype(np.float32); rand /= np.linalg.norm(rand)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    def report(label, cue, items, mode, vec):
        resps = gen_all(hf, tok, cue, items, lang, mode, vec=vec)
        rr, dg = rate(resps, lang), deg(resps)
        rows.append(dict(stage=0, label=label, refusal_pct=round(rr, 1), deg_pct=round(dg, 1), n=len(items)))
        print(f"  {label:<26s} refusal={rr:5.0f}%  deg={dg:4.0f}%", flush=True)
        return rr
    print("[stage0] test conditions:", flush=True)
    rb = report("harmful baseline",        cue_dep, test, "none",   None)
    ra = report("harmful ablate-REFUSAL",  cue_dep, test, "ablate", rhat)
    rr = report("harmful ablate-random",   cue_dep, test, "ablate", rand)
    report("harmless baseline",            cue_dep, harmless[:n], "none",   None)
    report("harmless ablate-REFUSAL",      cue_dep, harmless[:n], "ablate", rhat)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"stage0_{lang}.csv", index=False)
    drop_ref, drop_rand = rb - ra, rb - rr
    print(f"\nGATE: refusal-dir ablation drop={drop_ref:+.0f}pp  vs  random drop={drop_rand:+.0f}pp", flush=True)
    print(f"  -> {'PASS' if drop_ref >= 25 and drop_ref - drop_rand >= 15 else 'CHECK'}: "
          "want refusal-dir ablation to drop refusal sharply and >> random; generations coherent (deg≈0).", flush=True)


def build_eval_dir(hf, tok, lang, cue_dep, cue_ev, harmful_texts):
    """r_eval[L] = mean(eval-cue) - mean(deploy-cue) over the SAME harmful prompts. -> [NL,D]."""
    return diff_of_means(hf, tok, cue_ev, harmful_texts, cue_dep, harmful_texts)


def select_eval_layer(hf, tok, lang, cue_dep, cue_ev, r_all, val, key):
    """Pick layer where ADDING r_eval to DEPLOY runs most RAISES refusal (the headline operation).
    Judged by gemma (valid metric)."""
    base = 100 * np.mean([x for x in judge_gemma(val, gen_all(hf, tok, cue_dep, val, lang, "none"), lang, key) if x is not None])
    best, best_rise = None, -1e9
    for L in [10, 14, 18, 22, 26, 30]:
        lab = judge_gemma(val, gen_all(hf, tok, cue_dep, val, lang, "add", vec=r_all[L], add_layer=L), lang, key)
        rr = 100 * np.mean([x for x in lab if x is not None]) if any(x is not None for x in lab) else float("nan")
        print(f"   cand L{L}: deploy+ADD refusal {base:.0f}->{rr:.0f} (rise {rr-base:+.0f})", flush=True)
        if rr - base > best_rise: best, best_rise = L, rr - base
    print(f"  selected eval layer L{best} (val ADD rise {best_rise:+.0f}pp, gemma-judged)", flush=True)
    return best


def stage_eval(hf, tok, src, tgt, n):
    """Build src eval direction; ADD it to tgt DEPLOY (-> refusal up?) and ABLATE it from tgt EVAL
    (-> refusal down?). src==tgt = within-lang; src!=tgt = CROSS-LINGUAL headline."""
    battery, _, tr = s09.load_cue_sources()
    cdep = {l: s09.get_cue(l, "deploy", battery, tr) for l in {src, tgt}}
    cev  = {l: s09.get_cue(l, "eval_explicit", battery, tr) for l in {src, tgt}}
    src_harm = s14.load_harmful_subset(src, 80)
    tgt_harm = s14.load_harmful_subset(tgt, n + 8)
    val_t, test_t = tgt_harm[:8], tgt_harm[8:8 + n]
    harmless_t = load_harmless(tgt, n)
    key = s30.or_key(); sv.KEY = sv.sarvam_key()
    print(f"[eval {src}->{tgt}] build {src} eval direction + in-distribution null…", flush=True)
    r_all, null_all = eval_and_null_dirs(hf, tok, cev[src], cdep[src], [h["text"] for h in src_harm[:64]])
    L = 14                                      # fixed mid layer (selection found no better; saves the slow sweep)
    reval = r_all[L]; rnull = null_all[L]
    rnull = rnull / (np.linalg.norm(rnull) + 1e-8) * np.linalg.norm(reval)   # NORM-MATCH null to ||r_eval||
    print(f"  L{L}: ||r_eval||={np.linalg.norm(reval):.2f}  ||r_null(norm-matched)||={np.linalg.norm(rnull):.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True); rows = []; item_rows = []
    def rep(label, cue, items, mode, vec, add_layer=None):
        resps = gen_all(hf, tok, cue, items, tgt, mode, vec=vec, add_layer=add_layer)
        gl = judge_gemma(items, resps, tgt, key)        # gemma (OpenRouter, 16 parallel)
        sl = judge_sarvam(items, resps, tgt)            # sarvam-105b (Indic-native, 16 parallel)
        rg, rs, kw, dg = _rate(gl), _rate(sl), rate(resps, tgt), deg(resps)
        for it, r, g, s in zip(items, resps, gl, sl):
            item_rows.append(dict(src=src, tgt=tgt, layer=L, label=label, item_id=it["id"],
                                  prompt=it["text"], refuse_gemma=g, refuse_sarvam=s,
                                  refuse_kw=classify_refusal(r, tgt), resp=r))
        rows.append(dict(src=src, tgt=tgt, layer=L, label=label, refusal_gemma=round(rg,1),
                         refusal_sarvam=round(rs,1), refusal_kw=round(kw,1), deg_pct=round(dg,1), n=len(items)))
        print(f"  {label:<26s} gemma={rg:5.0f}%  sarvam={rs:5.0f}%  kw={kw:4.0f}%  deg={dg:3.0f}%", flush=True)
        return rg                                        # gemma primary for frac; sarvam reported alongside
    print(f"[eval {src}->{tgt}] conditions (n={len(test_t)}):", flush=True)
    rD = rep("deploy baseline",        cdep[tgt], test_t, "none", None)
    rE = rep("eval baseline",          cev[tgt],  test_t, "none", None)
    rAdd = rep(f"deploy + ADD {src}eval", cdep[tgt], test_t, "add", reval, add_layer=L)   # inject eval-awareness
    rAbl = rep(f"eval  - ABL {src}eval",  cev[tgt],  test_t, "ablate", reval)               # remove eval-awareness
    rep(f"deploy + ADD null",          cdep[tgt], test_t, "add", rnull, add_layer=L)         # IN-DISTRIBUTION null control
    rep("harmless deploy + ADD",       cdep[tgt], harmless_t, "add", reval, add_layer=L)    # specificity
    pd.DataFrame(rows).to_csv(OUT_DIR / f"eval_{src}2{tgt}.csv", index=False)
    pd.DataFrame(item_rows).to_csv(OUT_DIR / f"eval_{src}2{tgt}_items.csv", index=False)   # full resps for re-judge
    suf = (rAdd - rD) / (rE - rD) if rE != rD else float("nan")     # sufficiency: deploy+add -> eval level
    nec = (rAbl - rE) / (rD - rE) if rD != rE else float("nan")     # necessity: eval-abl -> deploy level
    print(f"\n[{src}->{tgt}] deploy={rD:.0f}% eval={rE:.0f}% | ADD eval->deploy: {rD:.0f}->{rAdd:.0f} "
          f"(suff frac {suf:+.2f}) | ABL eval->eval: {rE:.0f}->{rAbl:.0f} (nec frac {nec:+.2f})", flush=True)
    print("  CAUSAL TRANSFER if ADD raises refusal toward eval and/or ABL lowers it toward deploy, "
          "with random≈0 and harmless unchanged.", flush=True)


def stage_multi(hf, tok, src, tgt, n, coeff, window):
    """MULTI-LAYER (window) Arditi-style intervention every step, norm-matched null. The strongest,
    repo-faithful causal test we hadn't run (prior add was single-layer)."""
    battery, _, tr = s09.load_cue_sources(); L2 = {src, tgt}
    cdep = {l: s09.get_cue(l, "deploy", battery, tr) for l in L2}
    cev  = {l: s09.get_cue(l, "eval_explicit", battery, tr) for l in L2}
    key = s30.or_key(); sv.KEY = sv.sarvam_key()
    src_harm = s14.load_harmful_subset(src, 80); tgt_harm = s14.load_harmful_subset(tgt, n + 8)
    test = tgt_harm[8:8 + n]; harmless = load_harmless(tgt, n)
    print(f"[multi {src}->{tgt}] dirs + norm-matched null | window L{window[0]}-{window[-1]} coeff={coeff}", flush=True)
    r_all, null_all = eval_and_null_dirs(hf, tok, cev[src], cdep[src], [h["text"] for h in src_harm[:64]])
    reval, nmnull = {}, {}
    for L in window:
        rn = float(np.linalg.norm(r_all[L])); nn = float(np.linalg.norm(null_all[L])) + 1e-8
        reval[L] = r_all[L]; nmnull[L] = null_all[L] / nn * rn         # per-layer norm-matched null
    mid = window[len(window)//2]
    print(f"  ||r_eval[L{mid}]||={np.linalg.norm(reval[mid]):.2f}  ||null[L{mid}]||={np.linalg.norm(nmnull[mid]):.2f}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True); rows, item_rows = [], []
    def jr(label, items, resps):
        gl = judge_gemma(items, resps, tgt, key); sl = judge_sarvam(items, resps, tgt)
        rg, rs, dgp = _rate(gl), _rate(sl), deg(resps)
        for it, r, g, s in zip(items, resps, gl, sl):
            item_rows.append(dict(src=src, tgt=tgt, label=label, item_id=it["id"], refuse_gemma=g, refuse_sarvam=s, resp=r))
        rows.append(dict(src=src, tgt=tgt, coeff=coeff, window=f"{window[0]}-{window[-1]}", label=label,
                         refusal_gemma=round(rg,1), refusal_sarvam=round(rs,1), deg=round(dgp,1), n=len(items)))
        print(f"  {label:<26s} gemma={rg:5.0f}% sarvam={rs:5.0f}% deg={dgp:3.0f}%", flush=True); return rg
    rD = jr("deploy baseline",       test, gen_all(hf, tok, cdep[tgt], test, tgt, "none"))
    rE = jr("eval baseline",         test, gen_all(hf, tok, cev[tgt],  test, tgt, "none"))
    rAdd = jr("deploy+ADDmulti eval", test, gen_multi(hf, tok, cdep[tgt], test, reval,  window, coeff, "add"))
    rNul = jr("deploy+ADDmulti null", test, gen_multi(hf, tok, cdep[tgt], test, nmnull, window, coeff, "add"))
    rAbl = float("nan")
    if not LEAN:
        rAbl = jr("eval-ABLmulti eval",   test, gen_multi(hf, tok, cev[tgt],  test, reval,  window, 1.0,   "ablate"))
        jr("harmless+ADDmulti eval",     harmless, gen_multi(hf, tok, cdep[tgt], harmless, reval, window, coeff, "add"))
    tag = f"multi_{src}2{tgt}_c{coeff}"
    pd.DataFrame(rows).to_csv(OUT_DIR / f"{tag}.csv", index=False)
    pd.DataFrame(item_rows).to_csv(OUT_DIR / f"{tag}_items.csv", index=False)
    print(f"\n[{src}->{tgt} MULTI c={coeff}] deploy={rD:.0f} eval={rE:.0f} | ADD eval={rAdd:.0f} ADD null={rNul:.0f} | ABL eval={rAbl:.0f}", flush=True)
    print("  CAUSAL iff ADD eval >> ADD null and toward eval (and/or ABL << eval toward deploy), harmless low.", flush=True)


LEAN = False

def main():
    global LEAN
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--coeff", type=float, default=1.0)
    ap.add_argument("--win", default="8-32")
    ap.add_argument("--lean", action="store_true", help="sweep mode: only deploy/eval/ADD-eval/ADD-null")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--src", default="en")
    ap.add_argument("--tgt", default="hi")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args(); LEAN = a.lean
    model, tok = load_model_and_tokenizer(int8=True); hf = model._model
    print(f"model loaded. ablation engine | stage={a.stage} src={a.src} tgt={a.tgt} n={a.n}", flush=True)
    if a.stage == 0:
        stage0(hf, tok, a.lang, a.n)
    elif a.stage in (1, 2):
        stage_eval(hf, tok, a.src, a.tgt, a.n)
    elif a.stage == 3:
        lo, hi = a.win.split("-"); stage_multi(hf, tok, a.src, a.tgt, a.n, a.coeff, list(range(int(lo), int(hi) + 1)))


if __name__ == "__main__":
    main()
