#!/bin/bash
#SBATCH --job-name=p2-g1-bi
#SBATCH --partition=full_gpu_partition
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --account=<account>
#SBATCH --qos=<qos-fullgpu>
#SBATCH --output=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/logs/g1bi_%j.out
set -uo pipefail
export HOME=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/home
export XDG_CACHE_HOME=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/xdg_cache
export VLLM_CACHE_ROOT=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/vllm_cache
export HF_HOME=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/hf_home
export HF_HUB_CACHE=/path/to/Behavioral-IndicEvalAwareness/phase2_scratch/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# batch_invariant requires FLASH_ATTN/TRITON_ATTN; Gemma 3 multimodal-bidirectional
# attention otherwise resolves to a backend batch_invariant rejects (job 461).
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTHONPATH=/path/to/Behavioral-IndicEvalAwareness/src:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
cd /path/to/Behavioral-IndicEvalAwareness
# Two fresh runs of the SAME shard, both with batch-invariant kernels on.
for R in A B; do
  /path/to/IndicEvalAwareness/.venv-p2/bin/python -m phase2.generate --model gemma3-27b-it --lang en \
      --arm harmful --cue eval_log --run-id bi_test \
      --out-root /path/to/Behavioral-IndicEvalAwareness/phase2_scratch/g1_bi_$R
done
echo "G1-BI COMPLETE"
