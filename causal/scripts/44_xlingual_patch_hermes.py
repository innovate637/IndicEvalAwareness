#!/usr/bin/env python3
"""
44 — CROSS-LINGUAL activation PATCHING on Hermes-3-Llama-3.1-8B (fp16).

The interchange (transplant) analogue of the cross-lingual STEERING test (script 41). Script 41 builds a
single averaged ENGLISH eval DIRECTION and *adds* it into Indic deploy runs. Here we do the stronger,
independent method: capture the ENGLISH eval-cue last-token residual and literally *replace* the
target-language (hi/bn) DEPLOY run's residual with it, then generate and judge in the target language.
Same model, so the English and Indic residuals live in one activation space (D=4096, NL=32) — the
transplant is well-defined. This closes the "cross-lingual claim rests on one causal method" gap.

Donor  : captured under the SOURCE (en) deploy/eval cues on SOURCE prompts -> [N,NL,D].
Recipient: TARGET-language deploy-cue harmful prompts (parallel item ids, so src item i <-> tgt item i).

Conditions per target T (src fixed = en), per layer window:
  deploy clean (T) | eval clean (T)                      -- no-patch baselines (deploy start, eval ceiling)
  deploy(T)<-deploy(T) IDENT                             -- WITHIN-target no-op: validity gate (must ~= deploy clean, deg 0)
  deploy(T)<-EVAL(src) patch      [same base item]       -- the cross-lingual transfer test
  deploy(T)<-EVAL(src) mismatch   [different src item]   -- cue-effect NUMERATOR
  deploy(T)<-deploy(src) mismatch [different src item]   -- cue-effect DENOMINATOR (the key in-dist null)
  harmless(T)<-EVAL(src) patch                           -- collateral (must stay low)

DECISIVE cue-effect = (deploy<-EVAL(src) mismatch) - (deploy<-deploy(src) mismatch): both transplant a
DIFFERENT src item's residual, differ ONLY in the donor cue (eval vs deploy) -> isolates the transported
English eval SIGNATURE from a generic English-residual swap. en->en is the within-language anchor and
should reproduce script 40's +16pp, validating the harness.

  CUDA_VISIBLE_DEVICES=2 python scripts/44_xlingual_patch_hermes.py --src en --tgts en hi bn --n 100
Outputs: results/causal/xlpatch_hermes_{src}2{tgt}_n{n}_items.csv -> xlpatch_hermes_{src}2{tgt}_n{n}.csv
"""
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import config
P = importlib.util.spec_from_file_location("p34", str(ROOT/"scripts"/"34_ablate_eval.py")); p34=importlib.util.module_from_spec(P); P.loader.exec_module(p34)

MODEL="NousResearch/Hermes-3-Llama-3.1-8B"; BATCH=8; OUT=config.RESULTS_DIR/"causal"
SETS={"win6-22":list(range(6,23)),"low6-13":list(range(6,14)),"up14-22":list(range(14,23))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--src",default="en",help="donor (source) language for the transplanted residual")
    ap.add_argument("--tgts",nargs="+",default=["en","hi","bn"],help="recipient (target) languages")
    ap.add_argument("--n",type=int,default=100)
    ap.add_argument("--gemma-only",action="store_true",help="skip the sarvam-105b second judge (e.g. dead sarvam key); gemma-3-27b is primary")
    a=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float16,device_map="cuda:0").eval()
    NL=model.config.num_hidden_layers; layers=model.model.layers
    b,_,tr=p34.s09.load_cue_sources()
    key=p34.s30.or_key()
    if not a.gemma_only: p34.sv.KEY=p34.sv.sarvam_key()
    rng=np.random.default_rng(0)

    def inp(cue,texts):
        f=[]
        for t in texts:                               # 768-tok cap: long Indic (esp. Odia) prompts -> O(seq^2) OOM guard
            pids=tok(t,add_special_tokens=False)["input_ids"]
            if len(pids)>768: t=tok.decode(pids[:768])
            f.append(tok.apply_chat_template([{"role":"system","content":cue},{"role":"user","content":t}],tokenize=False,add_generation_prompt=True))
        return tok(f,return_tensors="pt",padding=True).to("cuda:0")
    def cap_all(cue,texts):                           # [N,NL,D] last-token resid every layer
        out=[]
        for i in range(0,len(texts),BATCH):
            x=inp(cue,texts[i:i+BATCH]); g={}
            hh=[layers[L].register_forward_hook((lambda L: lambda _m,_i,o: g.__setitem__(L,(o[0] if isinstance(o,tuple) else o)[:,-1,:].detach().float().cpu().numpy()))(L)) for L in range(NL)]
            with torch.no_grad(): model(**x)
            for h in hh: h.remove()
            out.append(np.stack([g[L] for L in range(NL)],1))
        return np.concatenate(out,0)
    def gen_patch(cue,texts,Ls,donor):                # donor [N,NL,D] or None; patch set Ls at prefill last-pos
        R=[]
        for i in range(0,len(texts),BATCH):
            x=inp(cue,texts[i:i+BATCH]); n_in=x["input_ids"].shape[1]; H=[]
            if donor is not None:
                for L in Ls:
                    vt=torch.as_tensor(donor[i:i+BATCH,L,:],dtype=torch.float16,device="cuda:0")
                    def mk(vt):
                        def hook(_m,_i,o):
                            is_t=isinstance(o,tuple); hs=o[0] if is_t else o
                            if hs.shape[1]>1:
                                hs=hs.clone(); hs[:,-1,:]=vt.to(hs.dtype)
                            return (hs,)+tuple(o[1:]) if is_t else hs
                        return hook
                    H.append(layers[L].register_forward_hook(mk(vt)))
            try:
                with torch.no_grad(): o=model.generate(**x,max_new_tokens=200,do_sample=False,pad_token_id=tok.eos_token_id)
            finally:
                for h in H: h.remove()
            R+=[tok.decode(t[n_in:],skip_special_tokens=True).strip() for t in o]
        return R

    # ---- SOURCE donor residuals (captured once) ----
    SRC=a.src
    cdep_src=p34.s09.get_cue(SRC,"deploy",b,tr); cev_src=p34.s09.get_cue(SRC,"eval_explicit",b,tr)
    src_items=p34.s14.load_harmful_subset(SRC,a.n); src_txt=[it["text"] for it in src_items]
    print(f"[xlpatch] src={SRC} capturing donor residuals (deploy + eval) on {len(src_items)} prompts…",flush=True)
    dep_src=cap_all(cdep_src,src_txt); ev_src=cap_all(cev_src,src_txt)
    perm=rng.permutation(len(ev_src))                 # ONE shared permutation for the two mismatch donors
    ev_src_mm=ev_src[perm]; dep_src_mm=dep_src[perm]

    for TGT in a.tgts:
        cdep=p34.s09.get_cue(TGT,"deploy",b,tr); cev=p34.s09.get_cue(TGT,"eval_explicit",b,tr)
        items=p34.s14.load_harmful_subset(TGT,a.n); harmless=p34.load_harmless(TGT,a.n)
        txt=[it["text"] for it in items]; htxt=[h["text"] for h in harmless]
        dep_tgt=cap_all(cdep,txt)                     # target's OWN deploy residual, for the within-target identity gate
        tag=f"xlpatch_hermes_{SRC}2{TGT}_n{a.n}"; ip=OUT/f"{tag}_items.csv"; OUT.mkdir(parents=True,exist_ok=True)
        done=set()
        if ip.exists():
            d0=pd.read_csv(ip); done=set(zip(d0.set,d0.label)); print(f"  [resume] {SRC}2{TGT}: {len(done)} done",flush=True)
        print(f"[xlpatch] {SRC}->{TGT} -> {tag}.csv",flush=True)
        def jr(s,label,its,resps):
            if (s,label) in done: print(f"  [resume] {s} {label}",flush=True); return
            gl=p34.judge_gemma(its,resps,TGT,key)
            sl=[None]*len(its) if a.gemma_only else p34.judge_sarvam(its,resps,TGT)
            rows=[dict(set=s,label=label,iid=it["id"],gemma=g,sarvam=sx,deg=int(p34.degenerate(r)),resp=str(r)[:400]) for it,r,g,sx in zip(its,resps,gl,sl)]
            pd.DataFrame(rows).to_csv(ip,mode="a",header=not ip.exists(),index=False)
            _srv = "  -- " if a.gemma_only else f"{p34._rate(sl):5.0f}%"
            print(f"  {s:<8} {label:<26s} gemma={p34._rate(gl):5.0f}% sarvam={_srv} deg={100*np.mean([p34.degenerate(r) for r in resps]):3.0f}%",flush=True)
        jr("-","deploy clean",items,gen_patch(cdep,txt,[],None))
        jr("-","eval clean",items,gen_patch(cev,txt,[],None))
        for s,Ls in SETS.items():
            jr(s,"deploy<-deploy IDENT",items,gen_patch(cdep,txt,Ls,dep_tgt))              # within-target no-op gate
            jr(s,f"deploy<-EVAL({SRC}) patch",items,gen_patch(cdep,txt,Ls,ev_src))         # same base item transfer
            jr(s,f"deploy<-EVAL({SRC}) mismatch",items,gen_patch(cdep,txt,Ls,ev_src_mm))   # cue-effect numerator
            jr(s,f"deploy<-deploy({SRC}) mismatch",items,gen_patch(cdep,txt,Ls,dep_src_mm))# cue-effect denominator (null)
            jr(s,f"harmless<-EVAL({SRC}) patch",harmless,gen_patch(cdep,htxt,Ls,ev_src))   # collateral
        df=pd.read_csv(ip)
        su=df.groupby(["set","label"]).agg(gemma=("gemma",lambda x:round(100*x.dropna().mean(),1)),
            sarvam=("sarvam",lambda x:round(100*x.dropna().mean(),1)),deg=("deg",lambda x:round(100*x.mean(),1)),n=("iid","size")).reset_index()
        su.to_csv(OUT/f"{tag}.csv",index=False)
        print(f"\n=== XLINGUAL PATCH SUMMARY {SRC}->{TGT} ===\n"+su.to_string(index=False),flush=True)
        print(f"cue-effect(win6-22) = [deploy<-EVAL({SRC}) mismatch] - [deploy<-deploy({SRC}) mismatch]; VALID iff IDENT~=deploy clean & deg~0.\n",flush=True)


if __name__=="__main__":
    main()
