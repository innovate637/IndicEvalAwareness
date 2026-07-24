#!/usr/bin/env python3
"""
41 — CROSS-LINGUAL eval-steering on Hermes-3-8B. Does the eval-awareness direction built in ENGLISH
transfer to raise refusal on Indic-language (hi/bn) harmful prompts? And how does EN-transfer compare to
a NATIVE-language direction?

For target T in {hi,bn}: build EN direction (eval-cue minus deploy-cue diff-of-means on English prompts)
and a NATIVE-T direction (held-out T prompts), add at alpha across layers 6-22 to T harmful prompts under
the T deploy cue, judge in T (dual gemma+sarvam). Norm-matched label-shuffle null per direction. Saves
text + degeneration flag; checkpoint by (lang,label).

  CUDA_VISIBLE_DEVICES=2 python scripts/41_xlingual_hermes.py --langs hi bn --n 50
Outputs: results/causal/xlingual_g4_items.csv -> xlingual_hermes.csv
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

MODEL=G4_PATH; BATCH=8; WIN=list(range(6,23)); OUT=config.RESULTS_DIR/"causal"


def main():
    global WIN
    ap=argparse.ArgumentParser(); ap.add_argument("--langs",nargs="+",default=["hi","bn"]); ap.add_argument("--n",type=int,default=50)
    ap.add_argument("--win",default="6-22",help="layer band lo-hi to steer in")
    ap.add_argument("--tag",default="",help="suffix for output file (avoid concurrent-write races)"); a=ap.parse_args()
    lo,hi=map(int,a.win.split("-")); WIN=list(range(lo,hi+1)); wtag=f"L{lo}-{hi}_n{a.n}{a.tag}"
    print(f"[xlingual] steering window {wtag} ({len(WIN)} layers)",flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side="left"
    model=AutoModelForImageTextToText.from_pretrained(MODEL, device_map="auto", dtype=torch.float16, low_cpu_mem_usage=True).eval()
    NL=model.config.text_config.num_hidden_layers; layers=model.model.language_model.layers
    b,_,tr=p34.s09.load_cue_sources()
    key=p34.s30.or_key(); p34.sv.KEY=p34.sv.sarvam_key()

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
    def mkdirs(cev,cdep,texts):
        ev=cap(cev,texts); dp=cap(cdep,texts); r=(ev.mean(0)-dp.mean(0)).astype(np.float32)
        pool=np.concatenate([ev,dp],0); p=np.random.default_rng(0).permutation(len(pool)); h=len(pool)//2
        n0=(pool[p[:h]].mean(0)-pool[p[h:]].mean(0)).astype(np.float32)
        reval={L:r[L] for L in WIN}; rnull={L:(n0[L]/(np.linalg.norm(n0[L])+1e-8)*np.linalg.norm(r[L])) for L in WIN}
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

    # EN direction (built on English prompts/cues; transferred to T)
    cdep_en=p34.s09.get_cue("en","deploy",b,tr); cev_en=p34.s09.get_cue("en","eval_explicit",b,tr)
    en_items=p34.s14.load_harmful_subset("en",100); reval_en,rnull_en=mkdirs(cev_en,cdep_en,[x["text"] for x in en_items])
    print("[xlingual] EN direction built",flush=True)

    OUT.mkdir(parents=True,exist_ok=True); ip=OUT/f"xlingual_g4_{wtag}_items.csv"; done=set()
    if ip.exists():
        d0=pd.read_csv(ip); done=set(zip(d0.lang,d0.label)); print(f"  [resume] {len(done)} done",flush=True)
    def jr(lang,label,its,resps):
        if (lang,label) in done: print(f"  [resume] {lang} {label}",flush=True); return
        gl=p34.judge_gemma(its,resps,lang,key); sl=p34.judge_sarvam(its,resps,lang)
        rows=[dict(lang=lang,label=label,iid=it["id"],gemma=g,sarvam=s,deg=int(p34.degenerate(r)),resp=str(r)[:400]) for it,r,g,s in zip(its,resps,gl,sl)]
        pd.DataFrame(rows).to_csv(ip,mode="a",header=not ip.exists(),index=False)
        print(f"  {lang} {label:<18} gemma={p34._rate(gl):3.0f}% sarvam={p34._rate(sl):3.0f}% deg={100*np.mean([p34.degenerate(r) for r in resps]):3.0f}% n={len(its)}",flush=True)

    for T in a.langs:
        cdep=p34.s09.get_cue(T,"deploy",b,tr); cev=p34.s09.get_cue(T,"eval_explicit",b,tr)
        test=p34.s14.load_harmful_subset(T,a.n)            # EN dir is English-built -> all target prompts valid test
        extra=p34.s14.load_harmful_subset(T,a.n+64)        # native dir needs held-out; only if pool allows
        have_native=len(extra)>=a.n+32
        if have_native:
            reval_T,rnull_T=mkdirs(cev,cdep,[x["text"] for x in extra[a.n:a.n+64]])
        harmless=p34.load_harmless(T,a.n)
        txt=[it["text"] for it in test]; htxt=[h["text"] for h in harmless]
        jr(T,"deploy",test,gen(cdep,txt,None,0))
        jr(T,"eval",test,gen(cev,txt,None,0))
        jr(T,"EN->T eval c0.1",test,gen(cdep,txt,reval_en,0.1))      # CROSS-LINGUAL transfer
        jr(T,"EN->T eval c0.25",test,gen(cdep,txt,reval_en,0.25))
        jr(T,"EN->T null c0.1",test,gen(cdep,txt,rnull_en,0.1))      # specificity
        if have_native:
            jr(T,"native eval c0.1",test,gen(cdep,txt,reval_T,0.1))  # native-language direction
            jr(T,"native null c0.1",test,gen(cdep,txt,rnull_T,0.1))
        else:
            print(f"  [{T}] n={a.n} uses full pool -> native-dir comparison skipped (no held-out)",flush=True)
        jr(T,"harmless EN c0.1",harmless,gen(cdep,htxt,reval_en,0.1))

    df=pd.read_csv(ip)
    def block(d):
        return d.groupby(["lang","label"]).agg(gemma=("gemma",lambda x:round(100*x.dropna().mean(),1)),
            sarvam=("sarvam",lambda x:round(100*x.dropna().mean(),1)),deg=("deg",lambda x:round(100*x.mean(),1)),n=("iid","size"))
    print("\n=== ALL ITEMS ===\n"+block(df).to_string())
    print("\n=== NON-DEGENERATE ONLY ===\n"+block(df[df.deg==0]).to_string())
    block(df).to_csv(OUT/f"xlingual_g4_{wtag}.csv")
    print(f"\nsaved -> results/causal/xlingual_g4_{wtag}_items.csv",flush=True)


if __name__=="__main__":
    main()
