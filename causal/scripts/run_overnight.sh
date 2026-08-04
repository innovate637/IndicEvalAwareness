#!/bin/bash
# Overnight causal queue: 3 GPU pipelines (1/2/3) in parallel, each sequential, all checkpointed.
# GPU 0 left FREE for the co-tenant. Judge = gemma only (free; no real-money sarvam overnight).
cd $PROJECT_ROOT
PY=.venv/bin/python
mkdir -p results/causal
Q=results/causal/overnight.log
echo "==== overnight queue started $(date) ====" >> "$Q"

run_gpu () {                       # $1=gpu ; remaining args = commands to run sequentially
  local gpu=$1; shift
  for cmd in "$@"; do
    echo "[$(date +%H:%M)] GPU$gpu START: $cmd" >> "$Q"
    for try in 1 2; do            # one retry (covers the intermittent mmap load failure)
      CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" >> results/causal/overnight_gpu$gpu.log 2>&1 && break
      echo "[$(date +%H:%M)] GPU$gpu retry $try: $cmd" >> "$Q"; sleep 5
    done
    echo "[$(date +%H:%M)] GPU$gpu DONE: $cmd" >> "$Q"
  done
}

run_gpu 1 \
  "$PY scripts/38_nonreasoning_contrast.py --model qwen --n 48 --coeffs 0.1 0.25 0.5 1 2 --win 6-22 --sarvam" \
  "$PY scripts/37_singlefwd_patch.py --tgt en --n 100" \
  "$PY scripts/35_subspace_causal.py --src en --tgt en --k 16 --n 24" &
run_gpu 2 \
  "$PY scripts/38_nonreasoning_contrast.py --model hermes --n 30 --coeffs 0.1 0.25 0.5 1 2 --win 6-22 --sarvam" \
  "$PY scripts/37_singlefwd_patch.py --tgt hi --n 100" \
  "$PY scripts/35_subspace_causal.py --src en --tgt hi --k 16 --n 24" &
run_gpu 3 \
  "$PY scripts/37_singlefwd_patch.py --tgt bn --n 100" \
  "$PY scripts/35_subspace_causal.py --src en --tgt bn --k 16 --n 24" \
  "$PY scripts/38_nonreasoning_contrast.py --model qwen --lang hi --n 30 --coeffs 0.25 0.5 1 --win 6-22 --sarvam" &
wait
echo "[$(date +%H:%M)] ==== ALL OVERNIGHT JOBS DONE $(date) ====" >> "$Q"
