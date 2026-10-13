#!/bin/bash
#SBATCH --job-name=p2-g1-determinism
#SBATCH --partition=full_gpu_partition
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=<account>
#SBATCH --qos=<qos-fullgpu>
#SBATCH --output=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/logs/g1_%j.out
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

# G1: regenerate ONE shard a second time, identical engine config and run_id,
# into a separate root. English chosen deliberately: longest completions in the
# set, so the most opportunity for batch-composition nondeterminism to show.
/path/to/IndicEvalAwareness/.venv-p2/bin/python -m phase2.generate --model gemma3-27b-it --lang en \
    --arm harmful --cue eval_log --run-id "20260820T071113Z-gemma3-budgeted" \
    --out-root /path/to/Behavioral-IndicEvalAwareness/phase2_scratch/g1_runB
echo "G1 RUN B COMPLETE"
