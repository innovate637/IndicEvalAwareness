#!/bin/bash
# FULL Qwen2.5-32B-Instruct (Apache-2.0) causal replication — STEERING + PATCHING, int8, n=100, 3 GPUs.
# Mirrors the entire Hermes causal leg. Cross-lingual PATCHING (our deliverable) is prioritized.
# Phase A: auto-localize eval band (12-layer windows) + positive control + single-layer patch (validity).
# Phase B (at picked band): cross-lingual PATCHING (en/hi/bn/ta/te/or) + eval-steer + cross-lingual STEERING.
# All runs checkpoint per-condition → safe to span >1 night / resume. Uses q32_*.py (originals untouched).
cd /home/sibayan_mitra_2024/mech_interp
PY=.venv/bin/python
Q=results/causal/qwen32_overnight.log
DL_PID=${1:-2359706}; POS_PID=${2:-2357295}
echo "==== qwen32 FULL (steer+patch) queued $(date) ====" >> "$Q"
wait_pid(){ while kill -0 "$1" 2>/dev/null; do sleep 60; done; }

echo "[$(date +%H:%M)] waiting download($DL_PID) + hermes poscontrol($POS_PID)…" >> "$Q"
wait_pid "$DL_PID"; wait_pid "$POS_PID"; echo "[$(date +%H:%M)] prereqs done" >> "$Q"
$PY -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen2.5-32B-Instruct'); print('config ok')" >> "$Q" 2>&1 || { echo "ABORT: config" >> "$Q"; exit 1; }

# SMOKE (steer + patch int8 paths) — abort the night if int8 path is broken
echo "[$(date +%H:%M)] smoke (steer+patch)…" >> "$Q"
CUDA_VISIBLE_DEVICES=1 $PY scripts/q32_steer.py --model qwen32 --n 4 --coeffs 0.1 --win 20-32 >> results/causal/qwen32_smoke.log 2>&1 || { echo "ABORT: steer smoke FAIL" >> "$Q"; exit 1; }
CUDA_VISIBLE_DEVICES=1 $PY scripts/q32_patch_multi.py --n 4 --lang en --band 20-32 >> results/causal/qwen32_smoke.log 2>&1 || { echo "ABORT: patch smoke FAIL" >> "$Q"; exit 1; }
echo "[$(date +%H:%M)] smoke PASS" >> "$Q"

# ---------- PHASE A ----------
echo "[$(date +%H:%M)] PHASE A: localize + poscontrol + single-layer patch" >> "$Q"
CUDA_VISIBLE_DEVICES=1 $PY scripts/q32_localize.py --n 40 --width 12 --alpha 0.1 --starts 4 20 36 52 --tag _A >> results/causal/qwen32_locA.log 2>&1 & LA=$!
CUDA_VISIBLE_DEVICES=3 $PY scripts/q32_localize.py --n 40 --width 12 --alpha 0.1 --starts 12 28 44 --tag _B >> results/causal/qwen32_locB.log 2>&1 & LB=$!
( CUDA_VISIBLE_DEVICES=2 $PY scripts/q32_poscontrol.py --n 100 --layers 16 24 32 40 --band 20-32 >> results/causal/qwen32_poscontrol.log 2>&1
  CUDA_VISIBLE_DEVICES=2 $PY scripts/q32_patch_single.py --layers 12 20 28 36 44 --n 50 >> results/causal/qwen32_patchsingle.log 2>&1
  echo "[$(date +%H:%M)] GPU2 phase-A (poscontrol+single-patch) done" >> "$Q" ) & PA2=$!
wait_pid "$LA"; wait_pid "$LB"; echo "[$(date +%H:%M)] localization done; picking band" >> "$Q"

BAND=$($PY - <<'PY'
import pandas as pd, glob
d=pd.concat([pd.read_csv(f) for f in glob.glob("results/causal/localize_q32_*.csv")],ignore_index=True).set_index("label")
def avg(r): return (r.gemma+r.sarvam)/2
best,bs=None,-1e9
for w in sorted({l[:-5] for l in d.index if l.endswith(" eval") and l.startswith("L")}):
    try:
        e,n,r=d.loc[w+" eval"],d.loc[w+" null"],d.loc[w+" rand"]
        spec=avg(e)-avg(n)
        if (avg(e)-avg(r))>0 and float(e.deg)<20 and spec>bs: bs,best=spec,w
    except Exception: pass
print(best.replace("L","") if best else "20-32")
PY
)
echo "[$(date +%H:%M)] PICKED BAND = $BAND" >> "$Q"

# ---------- PHASE B (at band): cross-lingual PATCHING (priority) + eval-steer + cross-lingual STEERING ----------
echo "[$(date +%H:%M)] PHASE B band=$BAND" >> "$Q"
# GPU1 + GPU3: cross-lingual PATCHING (our deliverable) — split langs
( for L in en hi bn; do
    echo "[$(date +%H:%M)] GPU1 patch_multi $L" >> "$Q"
    CUDA_VISIBLE_DEVICES=1 $PY scripts/q32_patch_multi.py --n 100 --lang $L --band "$BAND" >> results/causal/qwen32_patch_$L.log 2>&1
  done; echo "[$(date +%H:%M)] GPU1 patching done" >> "$Q" ) & G1=$!
( for L in ta te or; do
    echo "[$(date +%H:%M)] GPU3 patch_multi $L" >> "$Q"
    CUDA_VISIBLE_DEVICES=3 $PY scripts/q32_patch_multi.py --n 100 --lang $L --band "$BAND" >> results/causal/qwen32_patch_$L.log 2>&1
  done; echo "[$(date +%H:%M)] GPU3 patching done" >> "$Q" ) & G3=$!
# GPU2: eval-steer specificity + cross-lingual STEERING (after its phase-A finishes)
( wait_pid "$PA2"
  echo "[$(date +%H:%M)] GPU2 eval-steer sweep" >> "$Q"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/q32_steer.py --model qwen32 --n 100 --coeffs 0.05 0.1 0.25 0.5 --win "$BAND" --sarvam >> results/causal/qwen32_steer.log 2>&1
  echo "[$(date +%H:%M)] GPU2 cross-lingual steering" >> "$Q"
  CUDA_VISIBLE_DEVICES=2 $PY scripts/q32_xlingual.py --langs hi bn ta te or --n 100 --win "$BAND" --tag _q32 >> results/causal/qwen32_xlingual.log 2>&1
  echo "[$(date +%H:%M)] GPU2 steering done" >> "$Q" ) & G2=$!
wait_pid "$G1"; wait_pid "$G3"; wait_pid "$G2"
echo "[$(date +%H:%M)] ==== QWEN32 FULL DONE (band=$BAND) $(date) ====" >> "$Q"
