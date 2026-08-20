#!/bin/bash
#SBATCH --job-name=p2-g1-determinism
#SBATCH --partition=h100-full
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=professors-limited
#SBATCH --qos=professor-fullgpu-limited
#SBATCH --output=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/logs/g1_%j.out
set -uo pipefail
export HOME=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/home
export XDG_CACHE_HOME=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/xdg_cache
export VLLM_CACHE_ROOT=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/vllm_cache
export HF_HOME=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/hf_home
export HF_HUB_CACHE=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
cd /localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness

# G1: regenerate ONE shard a second time, identical engine config and run_id,
# into a separate root. English chosen deliberately: longest completions in the
# set, so the most opportunity for batch-composition nondeterminism to show.
/localstorage/home/dhruvkumar/IndicEvalAwareness/.venv-p2/bin/python -m phase2.generate --model gemma3-27b-it --lang en \
    --arm harmful --cue eval_log --run-id "20260820T071113Z-gemma3-budgeted" \
    --out-root /localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/g1_runB
echo "G1 RUN B COMPLETE"
