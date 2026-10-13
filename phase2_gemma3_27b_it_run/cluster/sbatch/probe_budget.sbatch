#!/bin/bash
#SBATCH --job-name=p2-probe-budget
#SBATCH --partition=full_gpu_partition
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=<account>
#SBATCH --qos=<qos-fullgpu>
#SBATCH --output=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/logs/probe_budget_%j.out
set -uo pipefail
export HOME=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/home
export XDG_CACHE_HOME=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/xdg_cache
export VLLM_CACHE_ROOT=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/vllm_cache
export HF_HOME=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/hf_home
export HF_HUB_CACHE=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=/path/to/Behavioral-IndicEvalAwareness/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
cd /path/to/Behavioral-IndicEvalAwareness

# §8.14: the calibration probe needs no new script -- it is generate.py with a
# large --max-tokens-override over a small item subset. 3500 (not the plan's
# 16384) because max_model_len is 4096 and the longest prompt is 584 tokens.
# deploy + eval_explicit are the widest contrast, so they bracket the length
# distribution; budgets are per (model, lang), not per cue.
for L in en hi bn ta te kn; do
  for CUE in deploy eval_explicit; do
    /path/to/IndicEvalAwareness/.venv-p2/bin/python -m phase2.generate --model gemma3-27b-it --lang $L \
        --cue $CUE --items 40 --max-tokens-override 3500 \
        --run-id probe_budget --out-root /path/to/Behavioral-IndicEvalAwareness/phase2_scratch/probe_gen
  done
done
echo "PROBE COMPLETE"
