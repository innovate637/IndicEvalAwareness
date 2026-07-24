#!/usr/bin/env python3
"""Refill missing sarvam-105b judge labels on the cross-lingual results (API-only; responses already saved).
The sarvam Indic judge dropped many ta/te/or calls; re-judge the saved 400-char responses to restore
dual-judge coverage. Does NOT regenerate. Idempotent: only fills rows where sarvam is null."""
import sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).parent.parent; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
P=importlib.util.spec_from_file_location("p34","scripts/34_ablate_eval.py"); p34=importlib.util.module_from_spec(P); P.loader.exec_module(p34)
p34.sv.KEY=p34.sv.sarvam_key()
N=50
FILES=["xlingual_hermes_L10-17_tate_items.csv","xlingual_hermes_L10-17_or_items.csv",
       "xlingual_hermes_L10-17_items.csv","xlingual_hermes_L6-22_items.csv"]
for fn in FILES:
    fp=ROOT/"results"/"causal"/fn
    if not fp.exists(): print(f"skip {fn} (missing)"); continue
    df=pd.read_csv(fp)
    before=100*df.sarvam.notna().mean()
    for lang in sorted(df.lang.unique()):
        harmful={it["id"]:it["text"] for it in p34.s14.load_harmful_subset(lang,N)}
        harmless={it["id"]:it["text"] for it in p34.load_harmless(lang,N)}
        mask=(df.lang==lang)&(df.sarvam.isna())
        idx=df.index[mask].tolist()
        if not idx: continue
        its=[]; resps=[]
        for i in idx:
            r=df.at[i,"iid"]; lab=str(df.at[i,"label"]); txt=(harmless if "harmless" in lab else harmful).get(r,"")
            its.append({"id":r,"text":txt}); resps.append(str(df.at[i,"resp"]))
        labs=p34.judge_sarvam(its,resps,lang)
        for i,l in zip(idx,labs):
            if l is not None: df.at[i,"sarvam"]=l
        print(f"  {fn} {lang}: refilled {sum(l is not None for l in labs)}/{len(idx)} nulls",flush=True)
    df.to_csv(fp,index=False)
    print(f"{fn}: sarvam coverage {before:.0f}% -> {100*df.sarvam.notna().mean():.0f}%",flush=True)
print("done")
