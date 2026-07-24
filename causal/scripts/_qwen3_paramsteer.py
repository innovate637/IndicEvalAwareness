#!/usr/bin/env python3
"""PARAMETRIC test-awareness steering on Qwen3-32B (FP8) — a faithful port of Abdelnabi & Salem
(NeurIPS'25 Spotlight, 2505.14617, microsoft/Test_Awareness_Steering) to our Indic (hi/bn) setup.

THEIR method (modify_model.get_regions): take a probe direction v (per-layer), rank every FFN gate_proj
row by cosine-sim to v, pick the top-K=800 rows R, and edit the weights
        gate_proj.weight[R] += sign*alpha*v         (sign=+1 "aware", -1 "unaware").
Direct int8/fp8 weight writes are not possible on a quantised checkpoint, so we apply the MATHEMATICALLY
IDENTICAL functional edit as a gate_proj forward hook:
        for a selected row k,  out[:,:,k] += sign*alpha*(x . v)     (x = gate_proj input)
which is exactly (W[k]+sign*alpha*v).x = W[k].x + sign*alpha*(v.x).  Composes with FP8; nothing else in the
model changes. (Optional true-weight-edit is a bf16 variant, not needed here.)

PHASE 1 direction v = OUR eval diff-of-means  mean(eval_explicit-cue) - mean(deploy-cue)  at the last prompt
token (unit-normalised), so this isolates THEIR parametric edit vs OUR activation-add, holding the direction
fixed. Conditions: baseline | aware(+a) | unaware(-a) | randrows(+a on 800 random rows = specificity null)
| activadd (our residual add of a at the same layer). Data: our hi/bn(+en) harmful prompts, deploy cue.
Dual judge gemma+sarvam; save resp for strict/coherence re-judge + hand adjudication.

  CUDA_VISIBLE_DEVICES=2 python scripts/_qwen3_paramsteer.py --langs en hi bn --edit-layer 60 --alphas 4 8 --n 100
Output: results/causal/qwen3_paramsteer_{lang}_items.csv
"""
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch, pandas as pd
from scipy.stats import binomtest, norm
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).parent.parent; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"scripts"))
import config
P=importlib.util.spec_from_file_location("p34",str(ROOT/"scripts"/"34_ablate_eval.py")); p34=importlib.util.module_from_spec(P); P.loader.exec_module(p34)

MODEL="Qwen/Qwen3-32B-FP8"; BATCH=8; OUT=config.RESULTS_DIR/"causal"

def dequant_gate(gp):
    """Reconstruct a bf16 [inter,hidden] gate_proj weight from an FP8 block-scaled checkpoint (or return
    the plain weight if not quantised). Qwen3-FP8: weight fp8 e4m3 + weight_scale_inv blocks of 128x128."""
    W=gp.weight
    sc=getattr(gp,"weight_scale_inv",None)
    if sc is None or W.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return W.detach().to(torch.bfloat16)
    Wf=W.detach().to(torch.float32); o,i=Wf.shape; bo,bi=(o+sc.shape[0]-1)//sc.shape[0], (i+sc.shape[1]-1)//sc.shape[1]
    s=sc.detach().to(torch.float32)
    s_full=s.repeat_interleave(bo,0)[:o].repeat_interleave(bi,1)[:,:i]
    return (Wf*s_full).to(torch.bfloat16)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--langs",nargs="+",default=["en","hi","bn"])
    ap.add_argument("--edit-layer",type=int,default=60,help="late layer to build direction + edit (they use 62/63 of 64)")
    ap.add_argument("--alphas",nargs="+",type=float,default=[4.0,8.0],help="edit strength on unit direction")
    ap.add_argument("--topk",type=int,default=800)
    ap.add_argument("--n",type=int,default=100)
    ap.add_argument("--think",action="store_true",help="Qwen3 reasoning mode on (Phase-2 style); default off")
    ap.add_argument("--gemma-only",action="store_true")
    ap.add_argument("--smoke",action="store_true")
    a=ap.parse_args(); L=a.edit_layer; THINK=a.think
    tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype="auto",device_map="cuda:0").eval()
    NL=model.config.num_hidden_layers; layers=model.model.layers; H=model.config.hidden_size
    gp=layers[L].mlp.gate_proj
    b,_,tr=p34.s09.load_cue_sources(); key=p34.s30.or_key()
    if not a.gemma_only: p34.sv.KEY=p34.sv.sarvam_key()
    MAXTOK=2048 if THINK else 256
    print(f"[paramsteer] Qwen3-32B-FP8 NL={NL} H={H} edit-layer={L} think={THINK} topk={a.topk}",flush=True)

    def inp(cue,texts):
        f=[]
        for t in texts:
            pids=tok(t,add_special_tokens=False)["input_ids"]
            if len(pids)>768: t=tok.decode(pids[:768])
            f.append(tok.apply_chat_template([{"role":"system","content":cue},{"role":"user","content":t}],
                     tokenize=False,add_generation_prompt=True,enable_thinking=THINK))
        return tok(f,return_tensors="pt",padding=True).to("cuda:0")

    def cap_dir(cev,cdep,texts):
        """eval-minus-deploy diff-of-means at layer L last prompt token -> unit vector v (hidden,)."""
        def cap(cue):
            acc=[]
            for i in range(0,len(texts),BATCH):
                x=inp(cue,texts[i:i+BATCH]); g={}
                h=layers[L].register_forward_hook(lambda _m,_i,o: g.__setitem__("v",(o[0] if isinstance(o,tuple) else o)[:,-1,:].detach().float().cpu().numpy()))
                with torch.no_grad(): model(**x)
                h.remove(); acc.append(g["v"])
            return np.concatenate(acc,0)
        v=(cap(cev).mean(0)-cap(cdep).mean(0)).astype(np.float32)
        v=v/(np.linalg.norm(v)+1e-8)
        return torch.tensor(v,dtype=torch.bfloat16,device="cuda:0")

    def pick_rows(v):
        W=dequant_gate(gp).to("cuda:0")                       # [inter,hidden] bf16
        cos=(W.to(torch.float32)@v.to(torch.float32))/(W.to(torch.float32).norm(dim=1)+1e-8)   # v is unit
        idx=torch.topk(cos,a.topk).indices
        rng=np.random.default_rng(0); rnd=torch.tensor(rng.choice(W.shape[0],a.topk,replace=False),device="cuda:0")
        del W; torch.cuda.empty_cache()
        return idx, rnd

    def param_hook(v,R,coeff):
        Rl=R.tolist()
        def hook(_m,inp,out):
            x=inp[0]                                          # [b,s,hidden]
            proj=(x.to(v.dtype)@v)                            # [b,s]
            out=out.clone(); out[...,Rl]+=coeff*proj.unsqueeze(-1).to(out.dtype)
            return out
        return gp.register_forward_hook(hook)
    def act_hook(v,coeff):
        def hook(_m,_i,o):
            hs=o[0] if isinstance(o,tuple) else o
            hs=hs+coeff*v.to(hs.dtype)
            return (hs,)+tuple(o[1:]) if isinstance(o,tuple) else hs
        return layers[L].register_forward_hook(hook)

    def gen(cue,texts,hooks_fn):
        R=[]
        for i in range(0,len(texts),BATCH):
            x=inp(cue,texts[i:i+BATCH]); n_in=x["input_ids"].shape[1]; hs=hooks_fn()
            try:
                with torch.no_grad(): o=model.generate(**x,max_new_tokens=MAXTOK,do_sample=False,pad_token_id=tok.eos_token_id)
            finally:
                for h in hs: h.remove()
            R+=[tok.decode(t[n_in:],skip_special_tokens=True).strip() for t in o]
        return R

    for lang in a.langs:
        cdep=p34.s09.get_cue(lang,"deploy",b,tr); cev=p34.s09.get_cue(lang,"eval_explicit",b,tr)
        pool=p34.s14.load_harmful_subset(lang,a.n if not a.smoke else 6)
        harmless=p34.load_harmless(lang,a.n if not a.smoke else 6)
        dirtxt=[it["text"] for it in p34.s14.load_harmful_subset(lang,a.n)]      # direction on full pool (Phase 1: reused, not held-out; parametric edit is layer-global)
        v=cap_dir(cev,cdep,dirtxt); idx,rnd=pick_rows(v)
        txt=[it["text"] for it in pool]; htxt=[h["text"] for h in harmless]
        ip=OUT/f"qwen3_paramsteer_{lang}_items.csv"; done=set()
        if ip.exists(): d0=pd.read_csv(ip); done=set(zip(d0.cond.astype(str),d0.iid.astype(str)))
        print(f"[{lang}] dir built, {a.topk} rows picked; -> {ip.name}",flush=True)
        def run(cond,its,resps):
            if all((cond,str(it["id"])) in done for it in its): print(f"  [resume] {cond}",flush=True); return
            gl=p34.judge_gemma(its,resps,lang,key); sl=[None]*len(its) if a.gemma_only else p34.judge_sarvam(its,resps,lang)
            rows=[dict(cond=cond,iid=it["id"],gemma=g,sarvam=s,deg=int(p34.degenerate(r)),resp=str(r)[:600]) for it,r,g,s in zip(its,resps,gl,sl)]
            pd.DataFrame(rows).to_csv(ip,mode="a",header=not ip.exists(),index=False)
            _sv=" -- " if a.gemma_only else f"{p34._rate(sl):3.0f}%"
            print(f"  {cond:<18} gemma={p34._rate(gl):3.0f}% sarvam={_sv} deg={100*np.mean([p34.degenerate(r) for r in resps]):3.0f}% n={len(its)}",flush=True)
        run("baseline",pool,gen(cdep,txt,lambda:[]))
        for al in a.alphas:
            run(f"aware_a{al}",pool,gen(cdep,txt,lambda al=al:[param_hook(v,idx,al)]))
            run(f"unaware_a{al}",pool,gen(cdep,txt,lambda al=al:[param_hook(v,idx,-al)]))
            run(f"randrows_a{al}",pool,gen(cdep,txt,lambda al=al:[param_hook(v,rnd,al)]))
            run(f"activadd_a{al}",pool,gen(cdep,txt,lambda al=al:[act_hook(v,al)]))
        run(f"harmless_aware_a{a.alphas[0]}",harmless,gen(cdep,htxt,lambda:[param_hook(v,idx,a.alphas[0])]))

        # recompute (gemma): aware/unaware vs baseline, and aware vs randrows (specificity)
        df=pd.read_csv(ip); piv=df.pivot_table(index="iid",columns="cond",values="gemma",aggfunc="first")
        def mc(A,B):
            if A not in piv or B not in piv: return
            pr=piv[[A,B]].dropna(); x=int(((pr[B]==0)&(pr[A]==1)).sum()); y=int(((pr[B]==1)&(pr[A]==0)).sum())
            d=x+y; p=binomtest(min(x,y),d,0.5).pvalue if d else 1.0; sig=norm.isf(p/2) if p<1 else 0
            print(f"    {A:<16} vs {B:<14} {100*pr[A].mean():3.0f}% vs {100*pr[B].mean():3.0f}% Δ={100*(pr[A].mean()-pr[B].mean()):+5.1f}pp flips {x}:{y} ({sig:.2f}σ)")
        print(f"  === {lang} McNemar (gemma) ===")
        for al in a.alphas:
            mc(f"aware_a{al}","baseline"); mc(f"unaware_a{al}","baseline")
            mc(f"aware_a{al}",f"randrows_a{al}"); mc(f"activadd_a{al}","baseline")
    print("DONE",flush=True)

if __name__=="__main__": main()
