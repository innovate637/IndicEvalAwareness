#!/usr/bin/env python3
"""SECOND-JUDGE pass: label the gemma-only Aya causal results with sarvam-105b (Indic-native reasoning
judge), so every eval-direction number gets DUAL-judge coverage + we get gemma-vs-sarvam agreement (κ) — a
direct answer to "your effect is one unvalidated judge." Judges the saved resp text (no GPU). Conditions
{deploy, eval, ADDnull_c0.25, ADDeval_c0.25}; checkpoints each row.

  python scripts/_sarvam_pass.py --langs en hi bn ta te or
Output: results/causal/sarvam_pass_{lang}.csv  + printed dual-judge recompute
"""
import sys, argparse, json, threading
from pathlib import Path
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.stats import binomtest, norm
import importlib.util
ROOT=Path(__file__).parent.parent; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import config
def L(n,f): s=importlib.util.spec_from_file_location(n,str(ROOT/"scripts"/f)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
p34=L("p34","34_ablate_eval.py")
CONDS=["deploy","eval","ADDnull_c0.25","ADDeval_c0.25"]

def kappa(a,b):
    a=np.array(a); b=np.array(b); m=~(np.isnan(a)|np.isnan(b)); a,b=a[m],b[m]
    if len(a)==0: return float("nan"),0
    po=(a==b).mean()
    pe=sum((a==k).mean()*(b==k).mean() for k in (0,1))
    return ((po-pe)/(1-pe) if pe<1 else 1.0), len(a)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--langs",nargs="+",default=["en","hi","bn","ta","te","or"])
    ap.add_argument("--workers",type=int,default=12); a=ap.parse_args()
    p34.sv.KEY=p34.sv.sarvam_key(); OUT=config.RESULTS_DIR/"causal"
    for lang in a.langs:
        suf="" if lang=="en" else f"_{lang}"; ip=OUT/f"hermes_cv{suf}_aya_items.csv"
        if not ip.exists(): print(f"{lang}: MISSING"); continue
        df=pd.read_csv(ip); df=df[df.label.isin(CONDS)].copy()
        items={it["id"]:it["text"] for it in json.load(open(config.SAFETY_DIR/f"{lang}.json"))}
        op=OUT/f"sarvam_pass_{lang}.csv"; done=set()
        if op.exists(): d0=pd.read_csv(op); done=set(zip(d0.label,d0.iid))
        todo=[(r.label,r.iid,r.resp) for r in df.itertuples() if (r.label,r.iid) not in done]
        print(f"{lang}: {len(todo)} to sarvam-judge ({len(done)} cached)",flush=True)
        def work(t):
            lab,iid,resp=t; return dict(label=lab,iid=iid,sarvam=p34.sarvam_judge(items.get(iid,""),resp,lang))
        if todo:
            lock=threading.Lock(); n=0
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
                for f in as_completed([ex.submit(work,t) for t in todo]):
                    with lock:
                        pd.DataFrame([f.result()]).to_csv(op,mode="a",header=not op.exists(),index=False); n+=1
                        if n%50==0: print(f"   {lang} {n}/{len(todo)}",flush=True)
        # recompute: sarvam causal effect + gemma-sarvam agreement
        sp=pd.read_csv(op); g=df.merge(sp,on=["label","iid"],suffixes=("_g",""))
        pg=g.pivot_table(index="iid",columns="label",values="gemma",aggfunc="first")
        ps=g.pivot_table(index="iid",columns="label",values="sarvam",aggfunc="first")
        k,kn=kappa(g.gemma.values,g.sarvam.values)
        print(f"\n===== {lang.upper()} dual-judge (gemma vs sarvam κ={k:.2f}, n={kn}) =====")
        for a2,b2,nm in [("eval","deploy","natural gap"),("ADDeval_c0.25","ADDnull_c0.25","causal c0.25")]:
            for jn,pv in [("gemma",pg),("sarvam",ps)]:
                if a2 not in pv or b2 not in pv: continue
                pr=pv[[a2,b2]].dropna(); x=int(((pr[b2]==0)&(pr[a2]==1)).sum()); y=int(((pr[b2]==1)&(pr[a2]==0)).sum())
                disc=x+y; p=binomtest(min(x,y),disc,0.5).pvalue if disc else 1.0; sig=norm.isf(p/2) if p<1 else 0
                print(f"  {nm:<12} [{jn:<6}] {100*pr[a2].mean():4.0f}% vs {100*pr[b2].mean():4.0f}%  Δ={100*(pr[a2].mean()-pr[b2].mean()):+5.1f}pp flips {x}:{y} p={p:.4f} ({sig:.2f}σ) n={len(pr)}")
    print("\nDONE",flush=True)

if __name__=="__main__": main()
