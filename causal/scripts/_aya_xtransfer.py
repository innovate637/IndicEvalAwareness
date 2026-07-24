#!/usr/bin/env python3
"""CROSS-LINGUAL eval-steering TRANSFER matrix on Aya-Expanse-8B (both directions EN<->HI).

Build the eval direction (eval-cue minus deploy-cue diff-of-means) in ONE language on a held-out split,
then ADD it at alpha across layers 6-22 into the OTHER language's deploy run and judge in that other
language. Symmetric 2x2:
  en test prompts : steer with d_en (NATIVE anchor, held-out) | d_hi (HI->EN transfer) | norm-matched null
  hi test prompts : steer with d_hi (NATIVE anchor, held-out) | d_en (EN->HI transfer) | norm-matched null
Transfer conditions are cross-language by construction (direction built on the other language's prompts),
so there is no train/test leakage. Native diagonal built on a disjoint held-out split. Dual baselines
(deploy clean, eval clean) per language give the natural gap. Harmless collateral with the transferred dir.
Saves resp[:600] so responses can be hand-adjudicated.

  CUDA_VISIBLE_DEVICES=3 python scripts/_aya_xtransfer.py --n 100
Output: results/causal/aya_xtransfer_items.csv -> aya_xtransfer.csv
"""
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd
from scipy.stats import binomtest, norm
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import config
P = importlib.util.spec_from_file_location("p34", str(ROOT/"scripts"/"34_ablate_eval.py")); p34=importlib.util.module_from_spec(P); P.loader.exec_module(p34)

ap=argparse.ArgumentParser()
ap.add_argument("--model", default="CohereForAI/aya-expanse-8b")
ap.add_argument("--coeffs", nargs="+", type=float, default=[0.1,0.25])
a=ap.parse_args()
MODEL=a.model; BATCH=8; WIN=list(range(6,23)); LANGS=["en","hi"]
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="cuda:0").eval()
NL=model.config.num_hidden_layers; layers=model.model.layers
b,_,tr=p34.s09.load_cue_sources(); key=p34.s30.or_key()

def inp(cue,texts):
    f=[]
    for t in texts:
        pids=tok(t,add_special_tokens=False)["input_ids"]
        if len(pids)>768: t=tok.decode(pids[:768])
        f.append(tok.apply_chat_template([{"role":"system","content":cue},{"role":"user","content":t}],tokenize=False,add_generation_prompt=True))
    return tok(f,return_tensors="pt",padding=True).to("cuda:0")
def cap(cue,texts):
    out=[]
    for i in range(0,len(texts),BATCH):
        x=inp(cue,texts[i:i+BATCH]); g={}
        hh=[layers[L].register_forward_hook((lambda L: lambda _m,_i,o: g.__setitem__(L,(o[0] if isinstance(o,tuple) else o)[:,-1,:].detach().float().cpu().numpy()))(L)) for L in range(NL)]
        with torch.no_grad(): model(**x)
        for h in hh: h.remove()
        out.append(np.stack([g[L] for L in range(NL)],1))
    return np.concatenate(out,0)
def mkdirs(lang, texts):
    cdep=p34.s09.get_cue(lang,"deploy",b,tr); cev=p34.s09.get_cue(lang,"eval_explicit",b,tr)
    ev=cap(cev,texts); dp=cap(cdep,texts); r=(ev.mean(0)-dp.mean(0)).astype(np.float32)
    pool=np.concatenate([ev,dp],0); p=np.random.default_rng(0).permutation(len(pool)); h=len(pool)//2
    n0=(pool[p[:h]].mean(0)-pool[p[h:]].mean(0)).astype(np.float32)
    reval={L:r[L] for L in WIN}
    rnull={L:(n0[L]/(np.linalg.norm(n0[L])+1e-8)*np.linalg.norm(r[L])) for L in WIN}
    return reval,rnull
def gen(cue,texts,dirs,coeff):
    R=[]
    for i in range(0,len(texts),BATCH):
        x=inp(cue,texts[i:i+BATCH]); n_in=x["input_ids"].shape[1]; H=[]
        if dirs:
            for L in WIN:
                vt=torch.as_tensor(coeff*dirs[L],dtype=torch.float16,device="cuda:0")
                H.append(layers[L].register_forward_hook((lambda vt: lambda _m,_i,o: ((o[0]+vt,)+tuple(o[1:])) if isinstance(o,tuple) else o+vt)(vt)))
        try:
            with torch.no_grad(): o=model.generate(**x,max_new_tokens=200,do_sample=False,pad_token_id=tok.eos_token_id)
        finally:
            for hh in H: hh.remove()
        R+=[tok.decode(t[n_in:],skip_special_tokens=True).strip() for t in o]
    return R

# ---- TRANSFER is inherently cross-language, so no held-out needed: build each dir on that language's
# ---- FULL pool, and only ever apply it to the OTHER language's prompts (no test item sees a dir built
# ---- on itself). Native within-language anchor already exists from the 5-fold CV run -> not re-run here.
D={}
for lang in LANGS:
    pool=p34.s14.load_harmful_subset(lang, 1000)      # returns up to the ~100 cap
    reval,rnull=mkdirs(lang,[x["text"] for x in pool])
    D[lang]=dict(test=pool, reval=reval, rnull=rnull,
                 cdep=p34.s09.get_cue(lang,"deploy",b,tr), cev=p34.s09.get_cue(lang,"eval_explicit",b,tr),
                 harmless=p34.load_harmless(lang,len(pool)))
    print(f"[xtransfer] {lang} dir built on FULL pool n={len(pool)} (applied only to the other language)",flush=True)

OUT=config.RESULTS_DIR/"causal"; OUT.mkdir(parents=True,exist_ok=True)
ip=OUT/"aya_xtransfer_items.csv"; done=set()
if ip.exists():
    d0=pd.read_csv(ip); done=set(zip(d0.tgt,d0.label)); print(f"[resume] {len(done)} done",flush=True)
def rec(tgt,label,its,resps):
    if (tgt,label) in done: print(f"  [resume] {tgt} {label}",flush=True); return
    gl=p34.judge_gemma(its,resps,tgt,key)
    rows=[dict(tgt=tgt,label=label,iid=it["id"],gemma=g,deg=int(p34.degenerate(r)),resp=str(r)[:600]) for it,r,g in zip(its,resps,gl)]
    pd.DataFrame(rows).to_csv(ip,mode="a",header=not ip.exists(),index=False)
    print(f"  {tgt} {label:<20} gemma={p34._rate(gl):3.0f}% deg={100*np.mean([p34.degenerate(r) for r in resps]):3.0f}% n={len(its)}",flush=True)

for tgt in LANGS:
    src=[l for l in LANGS if l!=tgt][0]         # the OTHER language = transfer donor
    d=D[tgt]; txt=[it["text"] for it in d["test"]]; htxt=[h["text"] for h in d["harmless"]]
    rec(tgt,"deploy",d["test"],gen(d["cdep"],txt,None,0))
    rec(tgt,"eval",d["test"],gen(d["cev"],txt,None,0))
    for c in a.coeffs:
        rec(tgt,f"{src}2{tgt}_c{c}",d["test"],gen(d["cdep"],txt,D[src]["reval"],c)) # TRANSFER: donor real eval dir
        rec(tgt,f"{src}null_c{c}",d["test"],gen(d["cdep"],txt,D[src]["rnull"],c))    # specificity: donor norm-matched null
    rec(tgt,f"harmless_{src}2{tgt}_c0.25",d["harmless"],gen(d["cdep"],htxt,D[src]["reval"],0.25))

# ---- summary + paired McNemar (transfer vs null, and native vs null) ----
df=pd.read_csv(ip)
def block(x): return x.groupby(["tgt","label"]).agg(gemma=("gemma",lambda v:round(100*v.dropna().mean(),1)),deg=("deg",lambda v:round(100*v.mean(),1)),n=("iid","size"))
print("\n=== ALL ITEMS ===\n"+block(df).to_string())
print("\n=== PAIRED McNEMAR (gemma) — TRANSFER vs donor-null, and natural eval vs deploy ===")
for tgt in LANGS:
    src=[l for l in LANGS if l!=tgt][0]
    piv=df[df.tgt==tgt].pivot_table(index="iid",columns="label",values="gemma",aggfunc="first")
    pairs=[("eval","deploy")]+[(f"{src}2{tgt}_c{c}",f"{src}null_c{c}") for c in a.coeffs]
    for cond,base in pairs:
        if cond not in piv or base not in piv: continue
        pr=piv[[cond,base]].dropna(); x=int(((pr[base]==0)&(pr[cond]==1)).sum()); y=int(((pr[base]==1)&(pr[cond]==0)).sum())
        disc=x+y; p=binomtest(min(x,y),disc,0.5).pvalue if disc else 1.0; sig=norm.isf(p/2) if p<1 else 0
        print(f"  {tgt}: {cond:<14} vs {base:<12} {100*pr[cond].mean():4.0f}% vs {100*pr[base].mean():4.0f}%  Δ={100*(pr[cond].mean()-pr[base].mean()):+5.1f}pp flips {x}:{y} p={p:.4f} ({sig:.2f}σ)")
block(df).to_csv(OUT/"aya_xtransfer.csv")
print("\nsaved -> results/causal/aya_xtransfer_items.csv",flush=True)
