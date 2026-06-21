#!/usr/bin/env python3
"""
21 — Residual analyses (F1): closes OPEN-3 and OPEN-4 on JUDGE labels. Local, $0.

OPEN-3  Cue-language interaction: does a NATIVE-language cue change the eval-cue compliance
        gap in Indic, vs the English cue? Per-arm we saw en-cue Indic +4.7pp (sig) vs native-cue
        +2.6pp (ns); here we test the formal condition × cue_lang INTERACTION (GEE, clustered on
        base item). ns interaction ⇒ the eval cue fires regardless of its language (Ahmed-consistent;
        refutes the "eval cue is an English-token artifact" reading).

OPEN-4  </think>-closure censoring: (a) does the eval cue change the rate the model closes
        </think> (non-random censoring)? (b) does excluding unclosed traces bias the gap?
        Sensitivity: judge compliance gap on ALL vs CLOSED-ONLY rows.

Sources: results/behavioral/refusal_judge.csv (judge verdicts) + raw response CSVs for closure.
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
import statsmodels.api as sm, statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
BEH = config.BEHAV_DIR
INDIC = ["hi", "bn", "ta", "te", "or"]


def judged():
    j = pd.read_csv(BEH / "refusal_judge.csv", on_bad_lines="skip")
    j = j[j.model_gen == "sarvamai/sarvam-m"].copy()
    j["iid"] = j["iid"].astype(str)
    j = j.drop_duplicates(["src", "lang", "cond", "iid"], keep="last")
    j["itemnum"] = j.iid.str.extract(r"(\d+)$")[0]
    j = j[(j.prompt_harmful == 1) & j.verdict.isin(["REFUSE", "COMPLY"])].copy()
    j["refuse"] = (j.verdict == "REFUSE").astype(int)
    return j


def open3(j):
    print("\n══ OPEN-3 — cue-language × eval-cue interaction (sarvam Indic, judge labels) ══")
    # en cue = alignment_faking.csv ; native cue = cue_factorial.csv ; conditions deploy/eval
    d = j[j.lang.isin(INDIC) & j.src.isin(["alignment_faking.csv", "cue_factorial.csv"])
          & j.cond.isin(["deploy", "eval"])].copy()
    d["is_eval"] = (d.cond == "eval").astype(int)
    d["is_native"] = (d.src == "cue_factorial.csv").astype(int)
    print(f"  N={len(d)}  (en-cue {int((d.is_native==0).sum())}, native-cue {int((d.is_native==1).sum())})")
    try:
        m = smf.gee("refuse ~ is_eval * is_native + C(lang)", groups="itemnum", data=d,
                    family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        for term in ["is_eval", "is_native", "is_eval:is_native"]:
            if term in m.params:
                b, se, p = m.params[term], m.bse[term], m.pvalues[term]
                print(f"  {term:20s} OR={np.exp(b):5.2f} [{np.exp(b-1.96*se):4.2f},{np.exp(b+1.96*se):5.2f}]  p={p:.4f}")
        print("  → eval-cue effect (is_eval) under EN cue; INTERACTION (is_eval:is_native) = "
              "how much the native cue CHANGES that effect. ns interaction ⇒ cue fires regardless of language.")
    except Exception as e:
        print("  GEE failed:", e)
    # plain per-arm gaps for context
    for nat, lab in [(0, "en-cue"), (1, "native-cue")]:
        s = d[d.is_native == nat]
        g = s[s.is_eval == 1].refuse.mean() - s[s.is_eval == 0].refuse.mean()
        print(f"    {lab:11s} eval−deploy gap = {100*g:+.1f}pp")


def open4(j):
    print("\n══ OPEN-4 — </think> closure censoring (sarvam) ══")
    # closure from raw response text across the sarvam behavioral sources
    frames = []
    for src in ["alignment_faking.csv", "cue_factorial.csv", "cue_battery_grid.csv"]:
        p = BEH / src
        if not p.exists():
            continue
        d = pd.read_csv(p)
        if not {"lang", "condition", "item_id", "response"} <= set(d.columns):
            continue
        d = d[["lang", "condition", "item_id", "response"]].copy()
        d["src"] = src
        d["closed"] = d.response.fillna("").astype(str).str.contains("</think>").astype(int)
        frames.append(d)
    c = pd.concat(frames, ignore_index=True)
    c["iid"] = c.item_id.astype(str)
    # (a) closure rate by condition, eval (eval/eval_log) vs deploy, paired McNemar per lang + pooled
    c["arm"] = np.where(c.condition.isin(["eval", "eval_log"]), "eval",
                        np.where(c.condition == "deploy", "deploy", "other"))
    cc = c[c.arm.isin(["eval", "deploy"])]
    print("  (a) closure rate by arm, and eval−deploy closure Δ (paired McNemar per item):")
    rows = []
    for lang in ["en"] + INDIC:
        s = cc[cc.lang == lang]
        piv = s.pivot_table("closed", ["iid"], "arm", aggfunc="max").dropna(subset=["deploy", "eval"])
        if len(piv) < 10:
            continue
        ev_col, dep_col = piv["eval"], piv["deploy"]
        b = int(((ev_col == 1) & (dep_col == 0)).sum()); cnt = int(((ev_col == 0) & (dep_col == 1)).sum())
        p = stats.binomtest(b, b + cnt, 0.5).pvalue if b + cnt else 1.0
        rows.append(dict(lang=lang, n=len(piv), dep=round(100*dep_col.mean(),1),
                         ev=round(100*ev_col.mean(),1), dClose=round(100*(ev_col.mean()-dep_col.mean()),1), p=round(p,3)))
    print(pd.DataFrame(rows).to_string(index=False))
    # (b) sensitivity: judge compliance gap on ALL vs CLOSED-ONLY (join closure into judged set)
    print("\n  (b) sensitivity — judge eval−deploy gap, ALL rows vs CLOSED-ONLY:")
    clo = c.groupby(["lang", "condition", "iid"]).closed.max().reset_index()
    clo["cond"] = clo.condition.replace({"eval": "eval", "eval_log": "eval"})
    jj = j[j.lang.isin(INDIC) & j.cond.isin(["deploy", "eval"])].merge(
        clo[["lang", "cond", "iid", "closed"]], on=["lang", "cond", "iid"], how="left")
    for subset, lab in [(jj, "ALL"), (jj[jj.closed == 1], "CLOSED-only")]:
        g = subset[subset.cond == "eval"].refuse.mean() - subset[subset.cond == "deploy"].refuse.mean()
        print(f"    Indic pooled {lab:12s} gap = {100*g:+.1f}pp   (n={len(subset)})")


def main():
    j = judged()
    open3(j)
    open4(j)
    print("\nF1 residual analyses complete (OPEN-3 + OPEN-4).")


if __name__ == "__main__":
    main()
