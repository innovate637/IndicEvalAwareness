#!/usr/bin/env python3
"""Adjudication: regenerate Hermes-3-8B c0.1 & c0.25 conditions WITH text (script 38 saved none),
dual-judge gemma+sarvam, save all texts, surface gemma!=sarvam cases. Reproduces the overnight
Hermes run (same seed/slices) so we can inspect the clean c0.1 positive instead of trusting labels."""
import sys, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import config
P = importlib.util.spec_from_file_location("p34","scripts/34_ablate_eval.py"); p34=importlib.util.module_from_spec(P); P.loader.exec_module(p34)

MODEL="NousResearch/Hermes-3-Llama-3.1-8B"; BATCH=8; WIN=list(range(6,23)); N=30; LANG="en"
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="cuda:0").eval()
NL=model.config.num_hidden_layers; layers=model.model.layers
b,_,tr=p34.s09.load_cue_sources(); cdep=p34.s09.get_cue(LANG,"deploy",b,tr); cev=p34.s09.get_cue(LANG,"eval_explicit",b,tr)
items=p34.s14.load_harmful_subset(LANG,N+64); dirtxt=[x["text"] for x in items[N:N+64]]; test=items[:N]
harmless=p34.load_harmless(LANG,N)
key=p34.s30.or_key(); p34.sv.KEY=p34.sv.sarvam_key()

def inp(cue,texts):
    f=[tok.apply_chat_template([{"role":"system","content":cue},{"role":"user","content":t}],tokenize=False,add_generation_prompt=True) for t in texts]
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
ev=cap(cev,dirtxt); dp=cap(cdep,dirtxt); r_eval=(ev.mean(0)-dp.mean(0)).astype(np.float32)
reval={L:r_eval[L] for L in WIN}
pool=np.concatenate([ev,dp],0); idx=np.random.default_rng(0).permutation(len(pool)); h=len(pool)//2
n0=(pool[idx[:h]].mean(0)-pool[idx[h:]].mean(0)).astype(np.float32)
rnull={L:(n0[L]/(np.linalg.norm(n0[L])+1e-8)*np.linalg.norm(r_eval[L])) for L in WIN}

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

conds=[("deploy",cdep,[t["text"] for t in test],None,0),
       ("eval",cev,[t["text"] for t in test],None,0)]
for c in (0.1,0.25):
    conds+=[(f"ADDeval_c{c}",cdep,[t["text"] for t in test],reval,c),
            (f"ADDnull_c{c}",cdep,[t["text"] for t in test],rnull,c),
            (f"harmless_c{c}",cdep,[t["text"] for t in harmless],reval,c)]
rows=[]
for label,cue,texts,dirs,coeff in conds:
    its = harmless if label.startswith("harmless") else test
    resps=gen(cue,texts,dirs,coeff)
    gl=p34.judge_gemma(its,resps,LANG,key); sl=p34.judge_sarvam(its,resps,LANG)
    for it,r,g,s in zip(its,resps,gl,sl): rows.append(dict(label=label,iid=it["id"],gemma=g,sarvam=s,deg=int(p34.degenerate(r)),resp=r))
    rg=100*np.mean([x for x in gl if x is not None]); rs=100*np.mean([x for x in sl if x is not None])
    print(f"{label:<14} gemma={rg:3.0f}% sarvam={rs:3.0f}% deg={100*np.mean([p34.degenerate(r) for r in resps]):3.0f}%",flush=True)
df=pd.DataFrame(rows); df.to_csv(config.RESULTS_DIR/"causal"/"adj_hermes_items.csv",index=False)
print("\n===== gemma != sarvam cases =====")
for _,r in df[df.gemma!=df.sarvam].iterrows():
    print(f"[{r.label} {r.iid}] g={'REF' if r.gemma==1 else 'CMP'} s={'REF' if r.sarvam==1 else 'CMP'} deg={r.deg} :: {str(r.resp)[:200].replace(chr(10),' ')}")
print("\nsaved -> results/causal/adj_hermes_items.csv")
