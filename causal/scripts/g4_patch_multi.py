#!/usr/bin/env python3
"""
40 — MULTI-LAYER activation patching on Hermes-3-8B (fp16): the interchange analog of multi-layer
steering. Single-site patch was null (script 39) -> eval->refusal is distributed. Here we transplant
the eval-cue last-token residual at EVERY layer in a set simultaneously (deploy<-eval), and also test
subsets to see how many layers are needed. Identity (deploy<-deploy, same set) is the no-op test.

Sets: full window 6-22; lower half 6-13; upper half 14-22.
Conditions per set: IDENT | EVAL patch | EVAL mismatch (in-dist null) | harmless<-EVAL.
Dual judge gemma+sarvam; text+deg saved; checkpointed by (set,label).

  CUDA_VISIBLE_DEVICES=1 python scripts/40_patch_multi_hermes.py --n 30
Outputs: results/causal/patch_multi_hermes_items.csv -> patch_multi_hermes.csv
"""
import sys, argparse, importlib.util
from pathlib import Path
import numpy as np, torch
import pandas as pd
from transformers import AutoModelForImageTextToText, AutoTokenizer
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts"))
import config
# --- gemma4 shim: .venv_gemma4 has no nnsight; behavioral_common only needs the import to resolve.
import types as _t
if "nnsight" not in sys.modules:
    _nn = _t.ModuleType("nnsight"); _nn.LanguageModel = object; sys.modules["nnsight"] = _nn
G4_PATH = str(ROOT / "models" / "gemma-4-31B-it-int8-foem")
P = importlib.util.spec_from_file_location("p34", str(ROOT/"scripts"/"34_ablate_eval.py")); p34=importlib.util.module_from_spec(P); P.loader.exec_module(p34)

MODEL=G4_PATH; BATCH=8; LANG="en"; OUT=config.RESULTS_DIR/"causal"


def main():
    global LANG
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=30); ap.add_argument("--lang",default="en"); ap.add_argument("--band",default="16-40"); a=ap.parse_args(); LANG=a.lang
    tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
    model=AutoModelForImageTextToText.from_pretrained(MODEL, device_map="auto", dtype=torch.float16, low_cpu_mem_usage=True).eval()
    NL=model.config.text_config.num_hidden_layers; layers=model.model.language_model.layers
    b,_,tr=p34.s09.load_cue_sources(); cdep=p34.s09.get_cue(LANG,"deploy",b,tr); cev=p34.s09.get_cue(LANG,"eval_explicit",b,tr)
    items=p34.s14.load_harmful_subset(LANG,a.n); harmless=p34.load_harmless(LANG,a.n)
    key=p34.s30.or_key(); p34.sv.KEY=p34.sv.sarvam_key()
    txt=[it["text"] for it in items]; htxt=[h["text"] for h in harmless]; rng=np.random.default_rng(0)
    blo,bhi=map(int,a.band.split("-")); SETS={a.band:list(range(blo,bhi+1))}
    print(f"[multi-patch] NL={NL} sets={list(SETS)} n={len(items)}",flush=True)

    def inp(cue,texts):
        f=[tok.apply_chat_template([{"role":"system","content":cue},{"role":"user","content":t}],tokenize=False,add_generation_prompt=True) for t in texts]
        return tok(f,return_tensors="pt",padding=True).to("cuda:0")
    def cap_all(cue,texts):                       # [N,NL,D] last-token resid every layer
        out=[]
        for i in range(0,len(texts),BATCH):
            x=inp(cue,texts[i:i+BATCH]); g={}
            hh=[layers[L].register_forward_hook((lambda L: lambda _m,_i,o: g.__setitem__(L,(o[0] if isinstance(o,tuple) else o)[:,-1,:].detach().float().cpu().numpy()))(L)) for L in range(NL)]
            with torch.no_grad(): model(**x)
            for h in hh: h.remove()
            out.append(np.stack([g[L] for L in range(NL)],1))
        return np.concatenate(out,0)
    def gen_patch(cue,texts,Ls,donor):            # donor [N,NL,D] or None; patch set Ls at prefill last-pos
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

    print("capturing residuals…",flush=True)
    dep=cap_all(cdep,txt); ev=cap_all(cev,txt); evh=cap_all(cev,htxt)
    mm=ev[rng.permutation(len(ev))]                 # eval-other-item (in-dist, same eval cue, diff content)
    dep_mm=dep[rng.permutation(len(dep))]           # deploy-other-item (in-dist generic residual swap)
    OUT.mkdir(parents=True,exist_ok=True); tag=f"patch_multi_g4_{a.lang}_n{a.n}"; ip=OUT/f"{tag}_items.csv"; done=set()
    if ip.exists():
        d0=pd.read_csv(ip); done=set(zip(d0.set,d0.label)); print(f"  [resume] {len(done)} done",flush=True)
    def jr(s,label,its,resps):
        if (s,label) in done: print(f"  [resume] {s} {label}",flush=True); return
        gl=p34.judge_gemma(its,resps,LANG,key); sl=p34.judge_sarvam(its,resps,LANG)
        rows=[dict(set=s,label=label,iid=it["id"],gemma=g,sarvam=sx,deg=int(p34.degenerate(r)),resp=str(r)[:400]) for it,r,g,sx in zip(its,resps,gl,sl)]
        pd.DataFrame(rows).to_csv(ip,mode="a",header=not ip.exists(),index=False)
        print(f"  {s:<8} {label:<22s} gemma={p34._rate(gl):5.0f}% sarvam={p34._rate(sl):5.0f}% deg={100*np.mean([p34.degenerate(r) for r in resps]):3.0f}%",flush=True)
    jr("-","deploy clean",items,gen_patch(cdep,txt,[],None))
    jr("-","eval clean",items,gen_patch(cev,txt,[],None))
    for s,Ls in SETS.items():
        jr(s,"deploy<-deploy IDENT",items,gen_patch(cdep,txt,Ls,dep))
        jr(s,"deploy<-deploy mismatch",items,gen_patch(cdep,txt,Ls,dep_mm))
        jr(s,"deploy<-EVAL patch",items,gen_patch(cdep,txt,Ls,ev))
        jr(s,"deploy<-EVAL mismatch",items,gen_patch(cdep,txt,Ls,mm))
        jr(s,"harmless<-EVAL patch",harmless,gen_patch(cdep,htxt,Ls,evh))
    df=pd.read_csv(ip)
    su=df.groupby(["set","label"]).agg(gemma=("gemma",lambda x:round(100*x.dropna().mean(),1)),
        sarvam=("sarvam",lambda x:round(100*x.dropna().mean(),1)),deg=("deg",lambda x:round(100*x.mean(),1)),n=("iid","size")).reset_index()
    su.to_csv(OUT/f"{tag}.csv",index=False)
    print("\n=== MULTI-LAYER PATCH SUMMARY ===\n"+su.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
