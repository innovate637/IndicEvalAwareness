#!/bin/bash
#SBATCH --job-name=p2-gemmapt-pipe
#SBATCH --partition=h100-full
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/logs/pipe_gemmapt_%j.out
#SBATCH --account=professors-limited
#SBATCH --qos=professor-fullgpu-limited
set -uo pipefail

# [LOCAL] BUNDLED PIPELINE. The five stages below were previously five separate
# sbatch jobs. On this cluster the QOS caps the user at ONE full GPU, and a
# 48-hour `gptoss_server` job from an unrelated project cycles on h100-full, so
# five independent submissions could mean five multi-day waits for what is one
# sequence. One allocation = one wait. Costs 6 extra model loads (~1 min each),
# which is nothing against a 48-hour queue.
# CLAUDE.md rule 4: --mem=32G, ONE gpu, no --exclusive. Unchanged.

R=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness
export HOME=$R/phase2_scratch/home
export XDG_CACHE_HOME=$R/phase2_scratch/xdg_cache
export VLLM_CACHE_ROOT=$R/phase2_scratch/vllm_cache
export HF_HOME=$R/phase2_scratch/hf_home
export HF_HUB_CACHE=$R/phase2_scratch/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export P2_RUN_ID=20260904T113400Z-gemmapt
export PYTHONPATH=$R/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
PY=/localstorage/home/dhruvkumar/IndicEvalAwareness/.venv-p2/bin/python
cd $R
echo "[pipe] run_id=$P2_RUN_ID node=$SLURMD_NODENAME job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "=== STAGE 1/6: G2 load gate ==="
$PY -m phase2.g2_load --model gemma3-27b-pt --lang en --items 4 || { echo "G2 FAILED"; exit 1; }

echo "=== STAGE 2/6: calibration probe (ceiling 3500; kn worst prompt 576 + 3500 = 4076 < 4096) ==="
for L in en hi bn ta te kn; do
  for CUE in deploy eval_explicit; do
    $PY -m phase2.generate --model gemma3-27b-pt --lang $L --cue $CUE --items 40 \
        --max-tokens-override 3500 --run-id probe_budget \
        --out-root $R/phase2_scratch/probe_gen_gemmapt || echo "PROBE WARN $L/$CUE"
  done
done

echo "=== STAGE 3/6: derive budgets from the probe (merges; does not overwrite others) ==="
$PY -m phase2.token_budget --from-probe $R/phase2_scratch/probe_gen_gemmapt --models gemma3-27b-pt
$PY -c "import json;print('budgets:',json.load(open('config/max_tokens.json')).get('gemma3-27b-pt'))"

echo "=== STAGE 4/6: G1 determinism (199 items, en, re-run twice) ==="
$PY -m phase2.generate --model gemma3-27b-pt --lang en --cue deploy --arm harmful \
    --run-id g1_gemmapt_A --out-root $R/phase2_scratch/g1_gemmapt || echo "G1-A WARN"
$PY -m phase2.generate --model gemma3-27b-pt --lang en --cue deploy --arm harmful \
    --run-id g1_gemmapt_B --out-root $R/phase2_scratch/g1_gemmapt || echo "G1-B WARN"

echo "=== STAGE 5/6: main grid, 6 languages x 5 cues x 2 arms ==="
for L in en hi bn ta te kn; do
  echo "--- lang $L ---"
  $PY -m phase2.generate --model gemma3-27b-pt --lang $L --run-id "$P2_RUN_ID" \
      || { echo "MAIN RUN FAILED lang=$L"; exit 1; }
done

echo "=== STAGE 6/6: activation capture (4 tensors/prompt, all layers, bf16) ==="
$PY scripts/extract_activations.py --model gemma3-27b-pt \
    --gen-root $R/phase2_scratch/generations \
    --out-root $R/gemma3-27b-pt-activations || echo "ACTIVATIONS WARN"

echo "PIPELINE COMPLETE run_id=$P2_RUN_ID"
