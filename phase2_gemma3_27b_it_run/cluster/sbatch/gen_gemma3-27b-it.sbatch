#!/bin/bash
#SBATCH --job-name=p2-gemma3-27b-it
#SBATCH --partition=h100-full
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%1
#SBATCH --output=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/logs/gemma3-27b-it_%A_%a.out
#SBATCH --requeue
#SBATCH --account=professors-limited
#SBATCH --qos=professor-fullgpu-limited
set -euo pipefail

LANGS=(en hi bn ta te kn)   # positional: array index 5 -> kn (was or)
LANG_CODE=${LANGS[$SLURM_ARRAY_TASK_ID]}

# [LOCAL] $HOME (/home/dhruvkumar) is NOT WRITABLE on the compute node --
# vLLM's model-info cache died there with PermissionError (job 440). Pin every
# cache root inside the project so no job ever reaches for /home.
export HOME=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/home
export XDG_CACHE_HOME=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/xdg_cache
export VLLM_CACHE_ROOT=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/vllm_cache
export HF_HOME=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/hf_home
export HF_HUB_CACHE=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export P2_RUN_ID=20260820T102630Z-gemma3-invariant
export PYTHONPATH=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

# [LOCAL] .venv-p2: torch 2.10.0+cu128, which RUNS on this node's driver
# 565.57.01 via CUDA minor-version compat (verified job 438). The miniforge
# 'vllm' env is cu130 and cannot initialise CUDA here at all.
cd /localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness

echo "[task] model=gemma3-27b-it lang=$LANG_CODE node=$SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# No srun: single node, single GPU, and srun can interfere with vLLM's
# worker spawning for no benefit here.
/localstorage/home/dhruvkumar/IndicEvalAwareness/.venv-p2/bin/python -m phase2.generate --model gemma3-27b-it --lang "$LANG_CODE" \
       --run-id "$P2_RUN_ID"
