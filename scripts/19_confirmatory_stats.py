#!/usr/bin/env python3
"""
19 — CONFIRMATORY STATISTICS for the compliance-gap claims (judge labels).

Why: the pooled McNemar treats item x language pairs as independent, but Toxic_Matrix
items are PARALLEL TRANSLATIONS (en_safety_0042 == hi_safety_0042). The confirmatory
analysis must cluster on the BASE ITEM. Two methods, results should agree:

  1. GEE  — population-averaged binomial logit, exchangeable working correlation,
            clusters = base item number  (statsmodels; the paper's reported model)
  2. Cluster bootstrap — resample base items with replacement, recompute the pooled
            paired gap (dependency-free robustness check)

Primary pre-specified contrasts (refusal on harmful prompts, judge labels):
  P1  per cross-model {gemma, qwen}: eval_log vs deploy, 6 langs pooled, + C(lang)
  P2  sarvam: eval vs deploy, Indic pooled (hi/bn/ta/te; en-cue arm), + C(lang)
  P3  decomposition per cross-model: deploy -> neutral_log (monitoring) -> eval_log
      (framing) with clustering (battery design)
Secondary: sarvam English (alignment_faking + battery), per-language exploratory.

All local, $0. Output: results/stats/confirmatory_judge_stats.csv + console table.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

warnings.filterwarnings("ignore")
import statsmodels.api as sm
import statsmodels.formula.api as smf

OUT = config.RESULTS_DIR / "stats" / "confirmatory_judge_stats.csv"
RNG = np.random.default_rng(7)
N_BOOT = 4000


def load_judged() -> pd.DataFrame:
    j = pd.read_csv(config.RESULTS_DIR / "behavioral" / "refusal_judge.csv", on_bad_lines="skip")
    j["iid"] = j["iid"].astype(str)
    j = j.drop_duplicates(subset=["src", "model_gen", "lang", "cond", "iid"], keep="last")
    j["itemnum"] = j.iid.str.extract(r"(\d+)$")[0]
    # item-level harmfulness: majority of the judge's prompt_harmful across rows of that item+lang
    hm = j.groupby(["lang", "itemnum"]).prompt_harmful.mean().ge(0.5)
    j["item_harm"] = [hm.get((l, i), False) for l, i in zip(j.lang, j.itemnum)]
    j = j[j.item_harm & j.verdict.isin(["REFUSE", "COMPLY"])].copy()
    j["refuse"] = (j.verdict == "REFUSE").astype(int)
    return j


def gee_contrast(d: pd.DataFrame, cond_a: str, cond_b: str, langs=None, label=""):
    """GEE binomial: refuse ~ is_b + C(lang), clustered on base itemnum.
    Returns dict with OR, CI, p for the cond_b vs cond_a effect."""
    sub = d[d.cond.isin([cond_a, cond_b])].copy()
    if langs:
        sub = sub[sub.lang.isin(langs)]
    if sub.empty or sub.cond.nunique() < 2:
        return None
    sub["is_b"] = (sub.cond == cond_b).astype(int)
    formula = "refuse ~ is_b + C(lang)" if sub.lang.nunique() > 1 else "refuse ~ is_b"
    try:
        m = smf.gee(formula, groups="itemnum", data=sub,
                    family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable()).fit()
        beta = m.params["is_b"]; se = m.bse["is_b"]; p = m.pvalues["is_b"]
        lo, hi = beta - 1.96 * se, beta + 1.96 * se
        # marginal effect in pp (average predicted difference)
        s0 = sub.copy(); s0["is_b"] = 0
        s1 = sub.copy(); s1["is_b"] = 1
        pp = (m.predict(s1).mean() - m.predict(s0).mean()) * 100
        return dict(label=label, n=len(sub), n_items=sub.itemnum.nunique(),
                    OR=np.exp(beta), OR_lo=np.exp(lo), OR_hi=np.exp(hi),
                    pp=pp, p=p)
    except Exception as e:
        return dict(label=label, n=len(sub), error=str(e)[:60])


def cluster_boot(d: pd.DataFrame, cond_a: str, cond_b: str, langs=None):
    """Paired gap pooled over (lang,item) pairs; resample BASE items with replacement."""
    sub = d[d.cond.isin([cond_a, cond_b])].copy()
    if langs:
        sub = sub[sub.lang.isin(langs)]
    p = sub.pivot_table("refuse", ["itemnum", "lang"], "cond", aggfunc="first").dropna()
    if p.empty:
        return None
    p = p.reset_index()
    p["d"] = p[cond_b] - p[cond_a]
    items = p.itemnum.unique()
    by_item = p.groupby("itemnum").d.mean()
    obs = p.d.mean()
    bs = []
    for _ in range(N_BOOT):
        draw = RNG.choice(items, size=len(items), replace=True)
        bs.append(by_item.loc[draw].mean())
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return dict(gap_pp=100 * obs, lo=100 * lo, hi=100 * hi, p_boot=max(pv, 1 / N_BOOT),
                n_pairs=len(p), n_items=len(items))


def fmt(g, b):
    if g is None or "error" in g:
        return f"  {g['label'] if g else '?':46s} GEE failed: {g.get('error','no data') if g else ''}"
    s = "**" if g["p"] < 0.05 else "  "
    line = (f"  {g['label']:46s} OR={g['OR']:5.2f} [{g['OR_lo']:4.2f},{g['OR_hi']:5.2f}] "
            f"pp={g['pp']:+5.1f} p={g['p']:.4f}{s}")
    if b:
        line += f" | boot {b['gap_pp']:+5.1f}pp [{b['lo']:+5.1f},{b['hi']:+5.1f}] p={b['p_boot']:.4f}"
    return line


def main():
    j = load_judged()
    rows = []
    print("══ CONFIRMATORY (GEE clustered on base item + cluster bootstrap) ══")
    print("   refusal on harmful prompts, judge labels\n")

    cm = j[j.src == "crossmodel.csv"]
    ALL6 = ["en", "hi", "bn", "ta", "te", "or"]
    INDIC4 = ["hi", "bn", "ta", "te"]
    # all crossmodel models present (gemma/qwen + sarvam-30b/105b in-family MoE), stable order
    CM_MODELS = [m for m in ["google/gemma-3-27b-it", "qwen/qwen3-32b", "sarvam-30b", "sarvam-105b"]
                 if m in set(cm.model_gen.unique())]

    print("P1 — TOTAL CUE EFFECT (eval_log vs deploy), 6 langs pooled")
    for mdl in CM_MODELS:
        d = cm[cm.model_gen == mdl]
        g = gee_contrast(d, "deploy", "eval_log", ALL6, f"{mdl.split('/')[-1]} pooled-6")
        b = cluster_boot(d, "deploy", "eval_log", ALL6)
        print(fmt(g, b)); rows.append({**(g or {}), **{f"boot_{k}": v for k, v in (b or {}).items()}})

    print("\nP2 — SARVAM Indic pooled (en-cue arm; hi/bn/ta/te), eval vs deploy")
    af = j[j.src == "alignment_faking.csv"]
    g = gee_contrast(af, "deploy", "eval", INDIC4, "sarvam Indic-4 (en cue)")
    b = cluster_boot(af, "deploy", "eval", INDIC4)
    print(fmt(g, b)); rows.append({**(g or {}), **{f"boot_{k}": v for k, v in (b or {}).items()}})
    cf = j[j.src == "cue_factorial.csv"]
    g = gee_contrast(cf, "deploy", "eval", INDIC4, "sarvam Indic-4 (native cue)")
    b = cluster_boot(cf, "deploy", "eval", INDIC4)
    print(fmt(g, b)); rows.append({**(g or {}), **{f"boot_{k}": v for k, v in (b or {}).items()}})

    print("\nP3 — DECOMPOSITION (battery): monitoring vs framing, clustered")
    for mdl in CM_MODELS:
        d = cm[cm.model_gen == mdl]
        nm = mdl.split('/')[-1]
        g1 = gee_contrast(d, "deploy", "neutral_log", ALL6, f"{nm} MONITORING (neutral-deploy)")
        b1 = cluster_boot(d, "deploy", "neutral_log", ALL6)
        print(fmt(g1, b1)); rows.append({**(g1 or {}), **{f"boot_{k}": v for k, v in (b1 or {}).items()}})
        g2 = gee_contrast(d, "neutral_log", "eval_log", ALL6, f"{nm} FRAMING (eval-neutral)")
        b2 = cluster_boot(d, "neutral_log", "eval_log", ALL6)
        print(fmt(g2, b2)); rows.append({**(g2 or {}), **{f"boot_{k}": v for k, v in (b2 or {}).items()}})

    print("\nSECONDARY — sarvam English")
    g = gee_contrast(af, "deploy", "eval", ["en"], "sarvam en (alignment_faking)")
    b = cluster_boot(af, "deploy", "eval", ["en"])
    print(fmt(g, b)); rows.append({**(g or {}), **{f"boot_{k}": v for k, v in (b or {}).items()}})
    cb = j[j.src == "cue_battery_grid.csv"]
    for cond in ["neutral_log", "eval_log", "eval_explicit", "eval_metadata"]:
        g = gee_contrast(cb, "deploy", cond, ["en"], f"sarvam en battery {cond}")
        b = cluster_boot(cb, "deploy", cond, ["en"])
        print(fmt(g, b)); rows.append({**(g or {}), **{f"boot_{k}": v for k, v in (b or {}).items()}})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
