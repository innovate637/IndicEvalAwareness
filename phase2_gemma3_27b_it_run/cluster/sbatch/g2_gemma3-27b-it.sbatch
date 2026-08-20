#!/bin/bash
#SBATCH --job-name=p2-g2-gemma3
#SBATCH --partition=h100-full
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --account=professors-limited
#SBATCH --qos=professor-fullgpu-limited
#SBATCH --output=/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness/phase2_scratch/logs/g2_gemma3-27b-it_%j.out
set -uo pipefail

# [LOCAL] .venv-p2 (torch 2.10.0+cu128) — verified working on this node's
# driver 565.57.01 via CUDA minor-version compat (job 438). The miniforge
# 'vllm' env is cu130 and CANNOT initialise CUDA here.
# [LOCAL] $HOME (/home/dhruvkumar) is NOT WRITABLE on the compute node --
# vLLM's model-info cache died with PermissionError there (job 440). The real
# home is /localstorage/home/dhruvkumar, but $HOME does not point at it. Pin
# every cache root inside the project so no job ever reaches for /home.
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

echo "[g2] node=$SLURMD_NODENAME job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
/localstorage/home/dhruvkumar/IndicEvalAwareness/.venv-p2/bin/python -m phase2.g2_load --model gemma3-27b-it --lang en --items 4
