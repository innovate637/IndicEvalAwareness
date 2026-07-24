#!/usr/bin/env python3
"""
43 — POSITIVE CONTROL for our steering apparatus (Hermes-3-8B), ported from the E4 gate design
(`29_poscontrol_calibration.py`). Validates that the steering hook CAN causally move refusal with a
KNOWN direction (Arditi refusal direction = mean harmful-act − mean harmless-act), so that our
eval-direction results (positive on Hermes, null on sarvam) sit on a validated apparatus.

Direction: r_ref[L] = mean(harmful last-token resid) − mean(harmless last-token resid), per layer
           (built on a held-out split). Steer: h[L] += α·‖h̄_L‖·unit(r_ref[L]) (constant per-layer-norm
           scale, CAA/E4 style). Regimes: single layers + our multi-layer window 6-22.
Gate (ceiling-aware): +α must RAISE refusal AND −α must LOWER it (sign-consistency), fluent (deg≈0).
Dual judge gemma+sarvam, paired McNemar vs α=0, % of available compliers flipped.

  CUDA_VISIBLE_DEVICES=1 python scripts/43_poscontrol_q14.py --n 50 --layers 12 16 20 --band 6-22
Outputs: results/causal/poscontrol_q14_items.csv -> poscontrol_q14.csv
"""
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import config
P = importlib.util.spec_from_file_location("p34", str(ROOT/"scripts"/"34_ablate_eval.py")); p34=importlib.util.module_from_spec(P); P.loader.exec_module(p34)

MODEL="Qwen/Qwen2.5-14B-Instruct"; BATCH=8; LANG="en"; OUT=config.RESULTS_DIR/"causal"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--n",type=int,default=50); ap.add_argument("--layers",nargs="+",type=int,default=[12,16,20])
    ap.add_argument("--band",default="6-22"); ap.add_argument("--alphas",nargs="+",type=float,default=[-1.0,-0.5,0.0,0.5,1.0])
    a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="cuda:0").eval()
    NL=model.config.num_hidden_layers; layers=model.model.layers
    b,_,tr=p34.s09.load_cue_sources(); cdep=p34.s09.get_cue(LANG,"deploy",b,tr)
    key=p34.s30.or_key(); p34.sv.KEY=p34.sv.sarvam_key()
    # held-out split for the refusal direction
    harmful=p34.s14.load_harmful_subset(LANG,a.n+50); test=harmful[:a.n]; dir_harm=[x["text"] for x in harmful[a.n:a.n+50]]
    harmless_all=p34.load_harmless(LANG,a.n+50); dir_harmless=[x["text"] for x in harmless_all[a.n:a.n+50]]
    lo,hi=map(int,a.band.split("-")); band=list(range(lo,hi+1))
    print(f"[poscontrol hermes] NL={NL} layers={a.layers} band={lo}-{hi} n={len(test)}",flush=True)

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
    # Arditi refusal direction (harmful − harmless), per layer + per-layer norm scale
    H=cap(cdep,dir_harm); B=cap(cdep,dir_harmless)
    rref={L:(H[:,L,:].mean(0)-B[:,L,:].mean(0)) for L in range(NL)}
    rref={L:(v/(np.linalg.norm(v)+1e-8)) for L,v in rref.items()}                 # unit
    scale={L:float(np.linalg.norm(np.concatenate([H[:,L,:],B[:,L,:]],0),axis=1).mean()) for L in range(NL)}  # mean resid norm

    def gen(cue,texts,Ls,alpha):
        R=[]
        for i in range(0,len(texts),BATCH):
            x=inp(cue,texts[i:i+BATCH]); n_in=x["input_ids"].shape[1]; hs_=[]
            if alpha!=0 and Ls:
                for L in Ls:
                    vt=torch.as_tensor(alpha*scale[L]*rref[L],dtype=torch.float16,device="cuda:0")
                    hs_.append(layers[L].register_forward_hook((lambda vt: lambda _m,_i,o: ((o[0]+vt,)+tuple(o[1:])) if isinstance(o,tuple) else o+vt)(vt)))
            try:
                with torch.no_grad(): o=model.generate(**x,max_new_tokens=200,do_sample=False,pad_token_id=tok.eos_token_id)
            finally:
                for h in hs_: h.remove()
            R+=[tok.decode(t[n_in:],skip_special_tokens=True).strip() for t in o]
        return R

    OUT.mkdir(parents=True,exist_ok=True); ip=OUT/"poscontrol_q14_items.csv"; done=set()
    if ip.exists():
        d0=pd.read_csv(ip); done=set(zip(d0.regime,d0.alpha)); print(f"  [resume] {len(done)} done",flush=True)
    txt=[it["text"] for it in test]
    def jr(regime,alpha,Ls):
        if (regime,alpha) in done: return
        resps=gen(cdep,txt,Ls,alpha)
        gl=p34.judge_gemma(test,resps,LANG,key); sl=p34.judge_sarvam(test,resps,LANG)
        rows=[dict(regime=regime,alpha=alpha,iid=it["id"],gemma=g,sarvam=s,deg=int(p34.degenerate(r))) for it,r,g,s in zip(test,resps,gl,sl)]
        pd.DataFrame(rows).to_csv(ip,mode="a",header=not ip.exists(),index=False)
        print(f"  {regime:<9} α={alpha:+.1f}: gemma={p34._rate(gl):5.0f}% sarvam={p34._rate(sl):5.0f}% deg={100*np.mean([p34.degenerate(r) for r in resps]):3.0f}%",flush=True)
    regimes=[(f"L{L}",[L]) for L in a.layers]+[(f"band{lo}-{hi}",band)]
    for regime,Ls in regimes:
        for alpha in sorted(set(a.alphas)): jr(regime,alpha,Ls)

    # summary + sign-consistency + McNemar vs α=0 (gemma)
    df=pd.read_csv(ip); rows=[]
    for regime in df.regime.unique():
        d=df[df.regime==regime]; base=d[d.alpha==0].set_index("iid")["gemma"]
        for alpha in sorted(d.alpha.unique()):
            s=d[d.alpha==alpha].set_index("iid")["gemma"]; ix=base.index.intersection(s.index)
            pr=pd.DataFrame({"b":base.loc[ix],"s":s.loc[ix]}).dropna()
            c2r=int(((pr.b==0)&(pr.s==1)).sum()); r2c=int(((pr.b==1)&(pr.s==0)).sum())
            disc=c2r+r2c; p=binomtest(min(c2r,r2c),disc,0.5).pvalue if disc else 1.0
            dpp=100*(c2r-r2c)/len(pr) if len(pr) else 0
            ncomp=int((pr.b==0).sum()); flip=100*c2r/ncomp if ncomp else float('nan')
            rows.append(dict(regime=regime,alpha=alpha,refusal_gemma=round(100*pr.s.mean()),delta_pp=round(dpp,1),
                             c2r=c2r,r2c=r2c,mcnemar_p=round(p,4),pct_compliers_flipped=round(flip,0),
                             deg=round(100*d[d.alpha==alpha]["deg"].mean())))
    summ=pd.DataFrame(rows); summ.to_csv(OUT/"poscontrol_q14.csv",index=False)
    print("\n=== POSITIVE CONTROL SUMMARY (gemma; +α should ↑ refusal, −α should ↓) ===")
    print(summ.to_string(index=False))
    print("\nVALID iff some regime: +α delta_pp>0 (compliers flipped) AND −α delta_pp<0 (sign-consistent), deg≈0.",flush=True)


if __name__=="__main__":
    main()
