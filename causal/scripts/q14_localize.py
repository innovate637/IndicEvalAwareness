#!/usr/bin/env python3
"""
42 — LOCALIZE the Hermes eval-steering effect. Full window 6-22 (17 layers) gives +23pp specific; here we
hold alpha=0.1 fixed and sweep 8-layer windows to find the minimal band that carries the effect. Builds
the EN eval direction once (held-out split), adds it only within each window, dual-judges ADD-eval vs
norm-matched ADD-null vs deploy/eval. Text + degeneration saved; checkpoint by label.

  CUDA_VISIBLE_DEVICES=2 python scripts/42_localize_steer.py --width 8 --alpha 0.1 --n 50
Outputs: results/causal/localize_steer_items.csv -> localize_steer.csv
"""
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import config
P = importlib.util.spec_from_file_location("p34", str(ROOT/"scripts"/"34_ablate_eval.py")); p34=importlib.util.module_from_spec(P); P.loader.exec_module(p34)

MODEL="Qwen/Qwen2.5-14B-Instruct"; BATCH=8; LANG="en"; OUT=config.RESULTS_DIR/"causal"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--width",type=int,default=8); ap.add_argument("--alpha",type=float,default=0.1)
    ap.add_argument("--n",type=int,default=50); ap.add_argument("--starts",nargs="+",type=int,default=[6,10,14,18]); ap.add_argument("--tag",default="")
    a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="cuda:0").eval()
    NL=model.config.num_hidden_layers; layers=model.model.layers
    b,_,tr=p34.s09.load_cue_sources(); cdep=p34.s09.get_cue(LANG,"deploy",b,tr); cev=p34.s09.get_cue(LANG,"eval_explicit",b,tr)
    items=p34.s14.load_harmful_subset(LANG,a.n+50); test=items[:a.n]; dirtxt=[x["text"] for x in items[a.n:a.n+50]]
    harmless=p34.load_harmless(LANG,a.n)
    key=p34.s30.or_key(); p34.sv.KEY=p34.sv.sarvam_key()
    windows={f"L{s}-{min(s+a.width-1,NL-1)}":list(range(s,min(s+a.width,NL))) for s in a.starts}
    print(f"[localize] alpha={a.alpha} width={a.width} windows={list(windows)} n={len(test)}",flush=True)

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
    ev=cap(cev,dirtxt); dp=cap(cdep,dirtxt); r=(ev.mean(0)-dp.mean(0)).astype(np.float32)
    pool=np.concatenate([ev,dp],0); pp=np.random.default_rng(0).permutation(len(pool)); h=len(pool)//2
    n0=(pool[pp[:h]].mean(0)-pool[pp[h:]].mean(0)).astype(np.float32)
    reval={L:r[L] for L in range(NL)}; rnull={L:(n0[L]/(np.linalg.norm(n0[L])+1e-8)*np.linalg.norm(r[L])) for L in range(NL)}
    rng_r=np.random.default_rng(123); rrand={}                       # RANDOM-direction baseline (norm-matched)
    for L in range(NL):
        g=rng_r.standard_normal(r[L].shape).astype(np.float32); rrand[L]=g/(np.linalg.norm(g)+1e-8)*np.linalg.norm(r[L])

    def gen(cue,texts,dirs,Ls,coeff):
        R=[]
        for i in range(0,len(texts),BATCH):
            x=inp(cue,texts[i:i+BATCH]); n_in=x["input_ids"].shape[1]; H=[]
            if dirs is not None:
                for L in Ls:
                    vt=torch.as_tensor(coeff*dirs[L],dtype=torch.float16,device="cuda:0")
                    H.append(layers[L].register_forward_hook((lambda vt: lambda _m,_i,o: ((o[0]+vt,)+tuple(o[1:])) if isinstance(o,tuple) else o+vt)(vt)))
            try:
                with torch.no_grad(): o=model.generate(**x,max_new_tokens=200,do_sample=False,pad_token_id=tok.eos_token_id)
            finally:
                for hh in H: hh.remove()
            R+=[tok.decode(t[n_in:],skip_special_tokens=True).strip() for t in o]
        return R

    OUT.mkdir(parents=True,exist_ok=True); ip=OUT/f"localize_q14{a.tag}_items.csv"; done=set()
    if ip.exists():
        d0=pd.read_csv(ip); done=set(d0.label); print(f"  [resume] {len(done)} labels done",flush=True)
    txt=[it["text"] for it in test]; htxt=[h["text"] for h in harmless]
    def jr(label,its,resps):
        if label in done: print(f"  [resume] {label}",flush=True); return
        gl=p34.judge_gemma(its,resps,LANG,key); sl=p34.judge_sarvam(its,resps,LANG)
        rows=[dict(label=label,iid=it["id"],gemma=g,sarvam=s,deg=int(p34.degenerate(r)),resp=str(r)[:400]) for it,r,g,s in zip(its,resps,gl,sl)]
        pd.DataFrame(rows).to_csv(ip,mode="a",header=not ip.exists(),index=False)
        print(f"  {label:<22} gemma={p34._rate(gl):3.0f}% sarvam={p34._rate(sl):3.0f}% deg={100*np.mean([p34.degenerate(r) for r in resps]):3.0f}%",flush=True)
    jr("deploy",test,gen(cdep,txt,None,None,0))
    jr("eval",test,gen(cev,txt,None,None,0))
    for name,Ls in windows.items():
        jr(f"{name} eval",test,gen(cdep,txt,reval,Ls,a.alpha))
        jr(f"{name} null",test,gen(cdep,txt,rnull,Ls,a.alpha))
        jr(f"{name} rand",test,gen(cdep,txt,rrand,Ls,a.alpha))
    df=pd.read_csv(ip)
    s=df.groupby("label").agg(gemma=("gemma",lambda x:round(100*x.dropna().mean(),1)),
        sarvam=("sarvam",lambda x:round(100*x.dropna().mean(),1)),deg=("deg",lambda x:round(100*x.mean(),1)),n=("iid","size"))
    s.to_csv(OUT/f"localize_q14{a.tag}.csv")
    print("\n=== LOCALIZE SUMMARY (alpha=%.2f) ===\n"%a.alpha+s.to_string(),flush=True)
    print("\nspecific(window) = eval - null; find the smallest band whose specific ~ full6-22's.",flush=True)


if __name__=="__main__":
    main()
