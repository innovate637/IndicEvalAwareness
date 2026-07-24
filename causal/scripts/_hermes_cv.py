#!/usr/bin/env python3
"""Confirmatory Hermes-3-8B eval-steering at alpha=0.1 (and 0.25) with BIGGER n via 5-fold CV so the
direction is held out from every test item. Builds r_eval (eval-cue minus deploy-cue diff-of-means)
on 80 prompts, steers+generates on the held-out 20, rotates -> n=100 held-out. Dual judge gemma+sarvam;
saves text + degeneration flag so we can filter (the Qwen lesson). Harmless collateral on all 100
harmless with the full-data direction.

  CUDA_VISIBLE_DEVICES=0 python scripts/_hermes_cv.py --lang en          # original English run
  CUDA_VISIBLE_DEVICES=0 python scripts/_hermes_cv.py --lang hi          # NATIVE Hindi (cue+prompts+judge in hi)
Native-language runs build the eval direction from that language's own eval-vs-deploy cue contrast on
that language's prompts (NOT the English direction — that is the separate transfer test, script 41).
Output: results/causal/hermes_cv_items.csv (en) or hermes_cv_{lang}_items.csv (native Indic)."""
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import config
P = importlib.util.spec_from_file_location("p34","scripts/34_ablate_eval.py"); p34=importlib.util.module_from_spec(P); P.loader.exec_module(p34)

_ap=argparse.ArgumentParser()
_ap.add_argument("--lang", default="en", help="language for cue+prompts+judge (en/hi/bn/ta/te/or)")
_ap.add_argument("--tag", default="", help="extra suffix on the output filename")
_ap.add_argument("--model", default="NousResearch/Hermes-3-Llama-3.1-8B",
                 help="HF model id; Llama-3.1-8B-Instruct is the controlled pair (same base, Meta safety tuning kept)")
_ap.add_argument("--gemma-only", action="store_true", help="skip the sarvam-105b 2nd judge (dead key); gemma is primary")
_A=_ap.parse_args()

MODEL=_A.model; BATCH=8; WIN=list(range(6,23)); LANG=_A.lang; COEFFS=[0.1,0.25]; FOLDS=5
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="cuda:0").eval()
NL=model.config.num_hidden_layers; layers=model.model.layers
b,_,tr=p34.s09.load_cue_sources(); cdep=p34.s09.get_cue(LANG,"deploy",b,tr); cev=p34.s09.get_cue(LANG,"eval_explicit",b,tr)
items=p34.s14.load_harmful_subset(LANG,1000); N=len(items); print(f"harmful pool={N}",flush=True)
harmless=p34.load_harmless(LANG,min(100,N))
key=p34.s30.or_key()
if not _A.gemma_only: p34.sv.KEY=p34.sv.sarvam_key()

def inp(cue,texts):
    f=[]
    for t in texts:                                   # 768-tok cap: long Indic (esp. Odia) prompts -> O(seq^2) OOM guard
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
def build_dirs(idx):
    txt=[items[i]["text"] for i in idx]
    ev=cap(cev,txt); dp=cap(cdep,txt); r=(ev.mean(0)-dp.mean(0)).astype(np.float32)
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

_suffix = "" if LANG=="en" else f"_{LANG}"
ip=config.RESULTS_DIR/"causal"/f"hermes_cv{_suffix}{_A.tag}_items.csv"
print(f"[hermes_cv] model={MODEL} lang={LANG} -> {ip.name}", flush=True)
done=set()
if ip.exists():
    d0=pd.read_csv(ip); done=set(zip(d0.fold.astype(str),d0.label)); print(f"[resume] {len(done)} done",flush=True)
def rec(fold,label,its,resps):
    if (str(fold),label) in done: print(f"  [resume] {fold} {label}",flush=True); return
    gl=p34.judge_gemma(its,resps,LANG,key)
    sl=[None]*len(its) if _A.gemma_only else p34.judge_sarvam(its,resps,LANG)
    rows=[dict(fold=fold,label=label,iid=it["id"],gemma=g,sarvam=s,deg=int(p34.degenerate(r)),resp=str(r)[:400]) for it,r,g,s in zip(its,resps,gl,sl)]
    pd.DataFrame(rows).to_csv(ip,mode="a",header=not ip.exists(),index=False)
    _srv = " -- " if _A.gemma_only else f"{p34._rate(sl):3.0f}%"
    print(f"  fold{fold} {label:<14} gemma={p34._rate(gl):3.0f}% sarvam={_srv} deg={100*np.mean([p34.degenerate(r) for r in resps]):3.0f}% n={len(its)}",flush=True)

order=np.random.default_rng(0).permutation(N)
for f in range(FOLDS):
    te=order[f::FOLDS]; trn=np.array([i for i in order if i not in set(te)])
    reval,rnull=build_dirs(trn)
    tt=[items[i] for i in te]; txt=[items[i]["text"] for i in te]
    rec(f,"deploy",tt,gen(cdep,txt,None,0))
    rec(f,"eval",tt,gen(cev,txt,None,0))
    for c in COEFFS:
        rec(f,f"ADDeval_c{c}",tt,gen(cdep,txt,reval,c))
        rec(f,f"ADDnull_c{c}",tt,gen(cdep,txt,rnull,c))

# harmless collateral: full-data direction
reval,_=build_dirs(np.arange(N)); htxt=[h["text"] for h in harmless]
for c in COEFFS:
    rec("H",f"harmless_c{c}",harmless,gen(cdep,htxt,reval,c))

df=pd.read_csv(ip)
def block(d):
    return d.groupby("label").agg(gemma=("gemma",lambda x:round(100*x.dropna().mean(),1)),
        sarvam=("sarvam",lambda x:round(100*x.dropna().mean(),1)),
        deg=("deg",lambda x:round(100*x.mean(),1)),n=("iid","size"))
print("\n=== ALL ITEMS ===\n"+block(df).to_string())
print("\n=== NON-DEGENERATE ONLY ===\n"+block(df[df.deg==0]).to_string())
print("\nsaved -> results/causal/hermes_cv_items.csv",flush=True)
