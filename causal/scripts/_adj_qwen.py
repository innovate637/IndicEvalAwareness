#!/usr/bin/env python3
"""Adjudication: regenerate Qwen2.5-7B c0.5 conditions WITH text, judge gemma+sarvam, surface the
gemma!=sarvam cases so we can read who's right."""
import sys, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import config
P = importlib.util.spec_from_file_location("p34","scripts/34_ablate_eval.py"); p34=importlib.util.module_from_spec(P); P.loader.exec_module(p34)

MODEL="Qwen/Qwen2.5-7B-Instruct"; BATCH=8; COEFF=0.5; WIN=list(range(6,23))
tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="cuda:0").eval()
NL=model.config.num_hidden_layers; D=model.config.hidden_size; layers=model.model.layers
b,_,tr=p34.s09.load_cue_sources(); cdep=p34.s09.get_cue("en","deploy",b,tr); cev=p34.s09.get_cue("en","eval_explicit",b,tr)
items=p34.s14.load_harmful_subset("en",30+64); dirtxt=[x["text"] for x in items[30:94]]; test=items[:30]
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
ev=cap(cev,dirtxt); dp=cap(cdep,dirtxt); reval={L:(ev.mean(0)-dp.mean(0))[L].astype(np.float32) for L in WIN}
pool=np.concatenate([ev,dp],0); idx=np.random.default_rng(0).permutation(len(pool)); h=len(pool)//2
n0=(pool[idx[:h]].mean(0)-pool[idx[h:]].mean(0)); rnull={L:(n0[L]/(np.linalg.norm(n0[L])+1e-8)*np.linalg.norm(reval[L])).astype(np.float32) for L in WIN}

def gen(cue,texts,dirs):
    R=[]
    for i in range(0,len(texts),BATCH):
        x=inp(cue,texts[i:i+BATCH]); n_in=x["input_ids"].shape[1]; H=[]
        if dirs:
            for L in WIN:
                vt=torch.as_tensor(COEFF*dirs[L],dtype=torch.float16,device="cuda:0")
                H.append(layers[L].register_forward_hook((lambda vt: lambda _m,_i,o: ((o[0]+vt,)+tuple(o[1:])) if isinstance(o,tuple) else o+vt)(vt)))
        try:
            with torch.no_grad(): o=model.generate(**x,max_new_tokens=200,do_sample=False,pad_token_id=tok.eos_token_id)
        finally:
            for hh in H: hh.remove()
        R+=[tok.decode(t[n_in:],skip_special_tokens=True).strip() for t in o]
    return R

rows=[]
for label,cue,dirs in [("deploy",cdep,None),("eval",cev,None),("ADDeval",cdep,reval),("ADDnull",cdep,rnull)]:
    resps=gen(cue,[t["text"] for t in test],dirs)
    gl=p34.judge_gemma(test,resps,"en",key); sl=p34.judge_sarvam(test,resps,"en")
    for it,r,g,s in zip(test,resps,gl,sl): rows.append(dict(label=label,iid=it["id"],gemma=g,sarvam=s,resp=r))
    print(f"{label:<8} gemma={100*np.mean([x for x in gl if x is not None]):.0f}% sarvam={100*np.mean([x for x in sl if x is not None]):.0f}%",flush=True)
df=pd.DataFrame(rows); df.to_csv(config.RESULTS_DIR/"causal"/"adj_qwen_c0.5_items.csv",index=False)
print("\n===== gemma != sarvam cases (ADDnull & ADDeval) =====")
for _,r in df[(df.label.isin(["ADDnull","ADDeval"])) & (df.gemma!=df.sarvam)].iterrows():
    print(f"\n[{r.label} {r.iid}] gemma={'REF' if r.gemma==1 else 'COMP'} sarvam={'REF' if r.sarvam==1 else 'COMP'}")
    ans=p34.rj.post_think(r.resp) or r.resp
    print("ANS:",ans[:280].replace(chr(10)," "))
print("\nsaved -> results/causal/adj_qwen_c0.5_items.csv")
