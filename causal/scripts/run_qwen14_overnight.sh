#!/bin/bash
# FULL Qwen2.5-14B-Instruct (Apache-2.0, fp16, 48 layers) causal replication — STEERING + PATCHING,
# n=100, DUAL JUDGE, 3 GPUs, GPU0 free. Mirrors the entire Hermes causal leg + cross-lingual PATCHING.
# Phase A: auto-localize eval band (12-layer windows over 48 layers) + positive control + single-layer patch.
# Phase B (at picked band): cross-lingual PATCHING (en/hi/bn/ta/te/or) + eval-steer sweep + cross-lingual STEERING.
# All checkpoint per-condition. Uses q14_*.py (originals untouched). 14B fp16 ~28GB → fits one 49GB GPU, no int8.
cd $PROJECT_ROOT
PY=.venv/bin/python
Q=results/causal/qwen14_overnight.log
DL_PID=${1:-2368847}
echo "==== qwen14 FULL (steer+patch, dual judge) queued $(date) ====" >> "$Q"
wait_pid(){ while kill -0 "$1" 2>/dev/null; do sleep 30; done; }

echo "[$(date +%H:%M)] waiting download($DL_PID)…" >> "$Q"; wait_pid "$DL_PID"; echo "[$(date +%H:%M)] download done" >> "$Q"
$PY -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen2.5-14B-Instruct'); print('config ok')" >> "$Q" 2>&1 || { echo "ABORT: config" >> "$Q"; exit 1; }

# SMOKE (steer + patch fp16 paths)
echo "[$(date +%H:%M)] smoke…" >> "$Q"
CUDA_VISIBLE_DEVICES=1 $PY scripts/q14_steer.py --model qwen14 --n 4 --coeffs 0.1 --win 16-28 >> results/causal/qwen14_smoke.log 2>&1 || { echo "ABORT: steer smoke FAIL" >> "$Q"; exit 1; }
CUDA_VISIBLE_DEVICES=1 $PY scripts/q14_patch_multi.py --n 4 --lang en --band 16-28 >> results/causal/qwen14_smoke.log 2>&1 || { echo "ABORT: patch smoke FAIL" >> "$Q"; exit 1; }
echo "[$(date +%H:%M)] smoke PASS" >> "$Q"

# ---------- PHASE A: localize (GPU1+GPU3, 12-layer windows over 48L) + poscontrol + single-patch (GPU2) ----------
echo "[$(date +%H:%M)] PHASE A" >> "$Q"
CUDA_VISIBLE_DEVICES=1 $PY scripts/q14_localize.py --n 40 --width 12 --alpha 0.1 --starts 4 28 --tag _A >> results/causal/qwen14_locA.log 2>&1 & LA=$!
CUDA_VISIBLE_DEVICES=3 $PY scripts/q14_localize.py --n 40 --width 12 --alpha 0.1 --starts 16 40 --tag _B >> results/causal/qwen14_locB.log 2>&1 & LB=$!
( CUDA_VISIBLE_DEVICES=2 $PY scripts/q14_poscontrol.py --n 100 --layers 12 20 28 36 --band 16-28 >> results/causal/qwen14_poscontrol.log 2>&1
  CUDA_VISIBLE_DEVICES=2 $PY scripts/q14_patch_single.py --layers 8 16 24 32 40 --n 50 >> results/causal/qwen14_patchsingle.log 2>&1
  echo "[$(date +%H:%M)] GPU2 phase-A done" >> "$Q" ) & PA2=$!
wait_pid "$LA"; wait_pid "$LB"; echo "[$(date +%H:%M)] localization done; picking band" >> "$Q"

BAND=$($PY - <<'PY'
import pandas as pd, glob
d=pd.concat([pd.read_csv(f) for f in glob.glob("results/causal/localize_q14_*.csv")],ignore_index=True).set_index("label")
def avg(r): return (r.gemma+r.sarvam)/2
best,bs=None,-1e9
for w in sorted({l[:-5] for l in d.index if l.endswith(" eval") and l.startswith("L")}):
    try:
        e,n,r=d.loc[w+" eval"],d.loc[w+" null"],d.loc[w+" rand"]
        if (avg(e)-avg(r))>0 and float(e.deg)<20 and (avg(e)-avg(n))>bs: bs,best=avg(e)-avg(n),w
    except Exception: pass
print(best.replace("L","") if best else "16-28")
PY
)
echo "[$(date +%H:%M)] PICKED BAND = $BAND" >> "$Q"

# ---------- PHASE B at band: cross-lingual PATCHING (priority) + eval-steer + cross-lingual STEERING ----------
echo "[$(date +%H:%M)] PHASE B band=$BAND" >> "$Q"
( for L in en hi bn; do echo "[$(date +%H:%M)] GPU1 patch $L" >> "$Q"
    CUDA_VISIBLE_DEVICES=1 $PY scripts/q14_patch_multi.py --n 100 --lang $L --band "$BAND" >> results/causal/qwen14_patch_$L.log 2>&1; done
  echo "[$(date +%H:%M)] GPU1 patching done" >> "$Q" ) & G1=$!
( for L in ta te or; do echo "[$(date +%H:%M)] GPU3 patch $L" >> "$Q"
    CUDA_VISIBLE_DEVICES=3 $PY scripts/q14_patch_multi.py --n 100 --lang $L --band "$BAND" >> results/causal/qwen14_patch_$L.log 2>&1; done
  echo "[$(date +%H:%M)] GPU3 patching done" >> "$Q" ) & G3=$!
( wait_pid "$PA2"
  echo "[$(date +%H:%M)] GPU2 eval-steer sweep" >> "$Q"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/q14_steer.py --model qwen14 --n 100 --coeffs 0.05 0.1 0.25 0.5 --win "$BAND" --sarvam >> results/causal/qwen14_steer.log 2>&1
  echo "[$(date +%H:%M)] GPU2 cross-lingual steering" >> "$Q"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/q14_xlingual.py --langs hi bn ta te or --n 100 --win "$BAND" --tag _q14 >> results/causal/qwen14_xlingual.log 2>&1
  echo "[$(date +%H:%M)] GPU2 steering done" >> "$Q" ) & G2=$!
wait_pid "$G1"; wait_pid "$G3"; wait_pid "$G2"
echo "[$(date +%H:%M)] ==== QWEN14 FULL DONE (band=$BAND) $(date) ====" >> "$Q"
