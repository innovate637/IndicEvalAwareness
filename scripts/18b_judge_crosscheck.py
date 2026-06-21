#!/usr/bin/env python3
"""
18b — JUDGE CROSS-CHECK. Re-judge a stratified sample with a SECOND, independent,
FREE model (default google/gemma-4-31b-it:free) and measure agreement with the
primary gemma-3-27b-it verdicts stored in refusal_judge.csv. High agreement ⇒ the
refusal scoring is not specific to one judge model (mitigates the 'LLM-judge
circularity' objection). Uses ONLY a :free model — no paid spend.

Usage: python scripts/18b_judge_crosscheck.py --n 120 [--model google/gemma-4-31b-it:free]
"""
import sys, time, hashlib, argparse, importlib.util
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# 18_refusal_judge.py starts with a digit → load via importlib
_spec = importlib.util.spec_from_file_location("rj", str(Path(__file__).parent / "18_refusal_judge.py"))
rj = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rj)

PRIMARY_CSV = config.RESULTS_DIR / "behavioral" / "refusal_judge.csv"
OUT = config.RESULTS_DIR / "behavioral" / "refusal_judge_crosscheck.csv"


def judge_retry(key, model, prompt, answer, lang, tries=6):
    for i in range(tries):
        try:
            return rj.judge(key, model, prompt, answer, lang)
        except Exception as e:
            if i == tries - 1:
                return {"prompt_harmful": None, "verdict": f"ERR:{getattr(e,'code',type(e).__name__)}"}
            time.sleep(2.0 * (i + 1))   # backoff for free-tier 429s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--model", default="google/gemma-4-31b-it:free")
    args = ap.parse_args()
    assert args.model.endswith(":free"), "cross-check must use a :free model (no paid spend)"

    prev = pd.read_csv(PRIMARY_CSV, on_bad_lines="skip")
    prev["iid"] = prev["iid"].astype(str)
    g27 = {(r.src, r.model_gen, r.lang, r.cond, r.iid): (r.prompt_harmful, r.verdict)
           for r in prev.itertuples()}

    w = rj.load_worklist()
    w["iid"] = w["iid"].astype(str)
    w["key"] = list(zip(w.src, w.model_gen, w.lang, w.cond, w.iid))
    w = w[w.key.isin(g27)]
    w = w[w.key.map(lambda k: g27[k][1] in ("REFUSE", "COMPLY"))]   # compare only meaningful calls

    groups = list(w.groupby(["model_gen", "lang"]))
    per = max(4, args.n // max(1, len(groups)))
    parts = []
    for (m, l), c in groups:
        seed = int(hashlib.md5(f"{m}-{l}".encode()).hexdigest(), 16) % (2**32)
        parts.append(c.sample(n=min(per, len(c)), random_state=seed))
    samp = pd.concat(parts, ignore_index=True)

    key = rj.openrouter_key()
    print(f"cross-checking {len(samp)} items: {args.model}  vs  stored gemma-3-27b ...")
    recs = []
    for r in samp.itertuples():
        v = judge_retry(key, args.model, r.prompt, r.answer, r.lang)
        ph27, vr27 = g27[r.key]
        recs.append(dict(src=r.src, model_gen=r.model_gen, lang=r.lang, iid=r.iid,
                         v27=vr27, v31=v["verdict"], ph27=ph27, ph31=v["prompt_harmful"]))
        time.sleep(0.2)
    d = pd.DataFrame(recs)
    d.to_csv(OUT, index=False)

    both = d[d.v27.isin(["REFUSE", "COMPLY"]) & d.v31.isin(["REFUSE", "COMPLY"])]
    print(f"\nn={len(d)}  both-decided={len(both)}  (31b errors/NOANSWER excluded: {len(d)-len(both)})")
    if len(both):
        print("VERDICT agreement (REFUSE vs COMPLY): %.1f%%" % (100 * (both.v27 == both.v31).mean()))
        ind = both[both.lang.isin(["hi", "bn", "ta", "te", "or"])]
        if len(ind):
            print("  Indic-only:   %.1f%%  (n=%d)" % (100 * (ind.v27 == ind.v31).mean(), len(ind)))
        en = both[both.lang == "en"]
        if len(en):
            print("  English-only: %.1f%%  (n=%d)" % (100 * (en.v27 == en.v31).mean(), len(en)))
    phb = d[d.ph27.notna() & d.ph31.notna()]
    if len(phb):
        print("prompt_harmful agreement: %.1f%%  (n=%d)" % (100 * (phb.ph27 == phb.ph31).mean(), len(phb)))
    dis = both[both.v27 != both.v31]
    print(f"\n{len(dis)} verdict disagreements (sample):")
    if len(dis):
        print(dis[["lang", "model_gen", "iid", "v27", "v31"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
