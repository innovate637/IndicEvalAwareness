#!/bin/bash
#SBATCH --job-name=iea_judge
#SBATCH --account=bits
#SBATCH --partition=gpu_h200_8
#SBATCH --qos=qos_gpu_h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/home/jagatsesh/IEA-phase3/phase3/logs/judge_%x_%j.out
#SBATCH --error=/home/jagatsesh/IEA-phase3/phase3/logs/judge_%x_%j.err
#
# Phase 3 judge — Gemma 4 31B, gate J2 (determinism) or J4 (full run).
#
# Usage:   sbatch --time=02:00:00 --job-name=j2 phase3/run_judge.sbatch j2
#          sbatch --time=24:00:00 --job-name=j4 phase3/run_judge.sbatch j4
#
# Time is NOT hardcoded: gpu_h200_8 caps at 1-00:00:00, so J4 must be
# requested at exactly the cap and J2 well under it. Pass --time on the
# command line so the wall clock is visible in the submission, not buried.
#
# PARTITION CHOICE — gpu_h200_8 (1x NVIDIA H200 NVL, 141 GB), see run_log.md:
#   weights            58.25 GiB   (index total_size 62,546,177,752 B, bf16)
#   budget @ 0.90      118.2 GiB   on 131.3 GiB card
#   free for KV        ~56 GiB     after ~4 GiB activation/graph overhead
#   KV per token       <=0.9375 MiB (60 layers, worst case all-local sizing)
#   => ~61,000 tokens of KV pool, ~27 concurrent 2.2k-token sequences.
#
#   An A100 80 GB (gpu_a100_8) fits the weights but leaves only ~4.9 GiB for
#   KV after overhead, i.e. a batch of ~2 across 47,880 rows.
#   Plan 4.1 and 12.2 both specify H200 / qos_gpu_h200 explicitly.
#   CLAUDE.md section 2 still says gpu_a100_8 and understates the weights at
#   ~61.4 GB; the plan governs. Tradeoff accepted: H200 caps at 24h where
#   A100 allows 5 days, which is why judge.py is resume-safe.

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "j2" && "$MODE" != "j4" ]]; then
    echo "ERROR: mode must be 'j2' or 'j4' (got '${MODE}')" >&2
    exit 2
fi

REPO=/home/jagatsesh/IEA-phase3
mkdir -p "$REPO/phase3/logs"

set +u
source /home/jagatsesh/miniconda3/etc/profile.d/conda.sh
conda activate slaybench
set -u

export LD_LIBRARY_PATH="/home/jagatsesh/miniconda3/envs/slaybench/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/scratch/jagatsesh/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=INFO

echo "=================================================================="
echo "job          : ${SLURM_JOB_ID:-nojob}  mode=${MODE}"
echo "node         : $(hostname)"
echo "started      : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "time limit   : $(squeue -h -j "${SLURM_JOB_ID:-0}" -o '%l' 2>/dev/null || echo 'n/a')"
echo "=================================================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true
echo "=================================================================="

cd "$REPO"
srun python phase3/judge.py --mode "$MODE"

echo "=================================================================="
echo "finished     : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "=================================================================="
