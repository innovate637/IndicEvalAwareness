#!/bin/bash
# FULL gemma-4-31B-it int8-GPTQ (FOEM 8-bit, Apache-2.0) causal replication — STEERING + PATCHING,
# n=100, DUAL JUDGE, 3 GPUs, GPU0 free. Mirrors the q14 overnight; geometry scaled 48->60 layers.
# gemma-4 runs in its DEFAULT NO-THINK mode (template pre-closes the thought channel) -> comparable
# to Hermes (non-reasoning positive) and Qwen-14B (non-reasoning null). KEY QUESTION: does the
# best-multilingual open model (a) show the causal eval-direction at all, (b) rescue ta/te/or transfer?
# Phase A: localize eval band (12-layer windows over 60L) + positive control + single-layer patch.
# Phase B (picked band): cross-lingual PATCHING (en/hi/bn/ta/te/or) + eval-steer sweep +
#          cross-lingual STEERING at the proportional full window L11-41 (Hermes lesson: bands under-power).
cd /home/sibayan_mitra_2024/mech_interp
PY=.venv_gemma4/bin/python
export CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8
export PATH="/home/sibayan_mitra_2024/mech_interp/.venv_gemma4/bin:/usr/local/cuda-12.8/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.9"
Q=results/causal/gemma4_overnight.log
echo "==== gemma4 FULL (steer+patch, dual judge) queued $(date) ====" >> "$Q"
wait_pid(){ while kill -0 "$1" 2>/dev/null; do sleep 30; done; }

# SMOKE (patch path; steer already smoked interactively)
echo "[$(date +%H:%M)] smoke patch…" >> "$Q"
CUDA_VISIBLE_DEVICES=1 $PY scripts/g4_patch_multi.py --n 4 --lang en --band 20-31 >> results/causal/g4_smoke.log 2>&1 || { echo "ABORT: patch smoke FAIL" >> "$Q"; exit 1; }
echo "[$(date +%H:%M)] smoke PASS" >> "$Q"

# ---------- PHASE A: localize (GPU1+GPU3, 12-layer windows over 60L) + poscontrol + single-patch (GPU2) ----------
echo "[$(date +%H:%M)] PHASE A" >> "$Q"
CUDA_VISIBLE_DEVICES=1 $PY scripts/g4_localize.py --n 40 --width 12 --alpha 0.1 --starts 4 28 --tag _A >> results/causal/gemma4_locA.log 2>&1 & LA=$!
CUDA_VISIBLE_DEVICES=3 $PY scripts/g4_localize.py --n 40 --width 12 --alpha 0.1 --starts 16 40 --tag _B >> results/causal/gemma4_locB.log 2>&1 & LB=$!
( CUDA_VISIBLE_DEVICES=2 $PY scripts/g4_poscontrol.py --n 50 --layers 15 25 35 45 --band 11-41 >> results/causal/gemma4_poscontrol.log 2>&1
  CUDA_VISIBLE_DEVICES=2 $PY scripts/g4_patch_single.py --layers 10 20 30 40 50 --n 50 >> results/causal/gemma4_patchsingle.log 2>&1
  echo "[$(date +%H:%M)] GPU2 phase-A done" >> "$Q" ) & PA2=$!
wait_pid "$LA"; wait_pid "$LB"; echo "[$(date +%H:%M)] localization done; picking band" >> "$Q"

BAND=$($PY - <<'PY'
import pandas as pd, glob
fs=[f for f in glob.glob("results/causal/localize_g4_*.csv") if not f.endswith("_items.csv")]
d=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True).set_index("label")
def avg(r): return (r.gemma+r.sarvam)/2
best,bs=None,-1e9
for w in sorted({l[:-5] for l in d.index if l.endswith(" eval") and l.startswith("L")}):
    try:
        e,n,r=d.loc[w+" eval"],d.loc[w+" null"],d.loc[w+" rand"]
        if (avg(e)-avg(r))>0 and float(e.deg)<20 and (avg(e)-avg(n))>bs: bs,best=avg(e)-avg(n),w
    except Exception: pass
print(best.replace("L","") if best else "20-31")
PY
)
echo "[$(date +%H:%M)] PICKED BAND = $BAND" >> "$Q"

# ---------- PHASE B at band: cross-lingual PATCHING (priority) + eval-steer + cross-lingual STEERING ----------
echo "[$(date +%H:%M)] PHASE B band=$BAND" >> "$Q"
( for L in en hi bn; do echo "[$(date +%H:%M)] GPU1 patch $L" >> "$Q"
    CUDA_VISIBLE_DEVICES=1 $PY scripts/g4_patch_multi.py --n 100 --lang $L --band "$BAND" >> results/causal/gemma4_patch_$L.log 2>&1; done
  echo "[$(date +%H:%M)] GPU1 patching done" >> "$Q" ) & G1=$!
( for L in ta te or; do echo "[$(date +%H:%M)] GPU3 patch $L" >> "$Q"
    CUDA_VISIBLE_DEVICES=3 $PY scripts/g4_patch_multi.py --n 100 --lang $L --band "$BAND" >> results/causal/gemma4_patch_$L.log 2>&1; done
  echo "[$(date +%H:%M)] GPU3 patching done" >> "$Q" ) & G3=$!
( wait_pid "$PA2"
  echo "[$(date +%H:%M)] GPU2 eval-steer sweep" >> "$Q"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/g4_steer.py --n 100 --coeffs 0.05 0.1 0.25 --win "$BAND" --sarvam >> results/causal/gemma4_steer.log 2>&1
  echo "[$(date +%H:%M)] GPU2 cross-lingual steering (full window 11-41)" >> "$Q"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/g4_xlingual.py --langs hi bn ta te or --n 100 --win 11-41 --tag _g4full >> results/causal/gemma4_xlingual.log 2>&1
  echo "[$(date +%H:%M)] GPU2 steering done" >> "$Q" ) & G2=$!
wait_pid "$G1"; wait_pid "$G3"; wait_pid "$G2"
echo "[$(date +%H:%M)] ==== GEMMA4 FULL DONE (band=$BAND) $(date) ====" >> "$Q"
