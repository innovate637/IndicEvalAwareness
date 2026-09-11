#!/bin/bash
# =============================================================================
# JOB B — Phase-2 ACTIVATION CAPTURE (HF transformers) for Qwen2.5-72B-Instruct
#
# Runs ONLY after Job A succeeds. Submit via submit_both.sh, which sets
#   --dependency=afterok:<jobA>
# so this never starts against a partial generation set.
#
# Separate job, not a second phase of Job A, for two reasons:
#   1. vLLM does not release its ~90% VRAM KV pool until the process exits, so
#      transformers would OOM if both ran in one allocation.
#   2. Generation alone may approach the 24h gpu_h200_8 wall (the qwen3-32b
#      precedent took 8h31m at TP=1 on a 32B model). Sharing one 24h wall would
#      risk producing zero activations after a full day of GPU time.
#
# RESUMABLE. capture_activations.py skips any shard whose .safetensors already
# exists unless --overwrite is passed (CLAUDE.md 14: never silently overwrite
# run artifacts).
#
# OUTPUT LOCATION. run.yaml paths.activations is ./activations, which is a
# SYMLINK to /scratch/jagatsesh/IEA-qwen72b-activations. The full set is
# ~61 GiB and must never land in /home (40 GiB quota, CLAUDE.md 8).
# =============================================================================
#SBATCH --job-name=iea_q72b_act
#SBATCH --partition=gpu_h200_8
#SBATCH --account=bits
#SBATCH --qos=qos_gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jagatsesh/IEA-qwen72b/phase2_qwen25_72b_instruct_run/logs/act_%j.out
#SBATCH --error=/home/jagatsesh/IEA-qwen72b/phase2_qwen25_72b_instruct_run/logs/act_%j.err

set -euo pipefail

RUN_DIR=/home/jagatsesh/IEA-qwen72b/phase2_qwen25_72b_instruct_run
MODEL_SLUG=qwen25-72b-instruct

export HF_HOME=/scratch/jagatsesh/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# `set -u` MUST be lifted across conda activation. The env's cuda-nvcc hook
# (etc/conda/activate.d/~cuda-nvcc_activate.sh:48) reads NVCC_PREPEND_FLAGS with
# no default, which is a fatal "unbound variable" under -u. That killed probe
# job 334761 in 40s before it generated anything. Restore -u immediately after.
set +u
source /home/jagatsesh/miniconda3/etc/profile.d/conda.sh
conda activate slaybench
set -u

# GLIBCXX_3.4.32 fix (job 336549/336552, 2026-09-08): FlashInfer's JIT is built
# with the conda env's GCC 15.2.0, which emits code requiring GLIBCXX_3.4.32.
# The system's /lib64/libstdc++.so.6 only provides up to GLIBCXX_3.4.25, so the
# JIT .so fails to dlopen unless the conda env's own (newer) libstdc++.so.6 is
# found first. Symbol versioning is strictly additive, so this cannot break
# anything that worked against the older system copy. Verified: smoke test
# (2xH200 vLLM TP=2 init) printed literal SUCCESS with this line present.
export LD_LIBRARY_PATH="/home/jagatsesh/miniconda3/envs/slaybench/lib:${LD_LIBRARY_PATH:-}"

cd "$RUN_DIR"
export PYTHONPATH="$RUN_DIR/src"

mkdir -p logs

# Reuse Job A's run id purely for provenance in the sidecar index files; the
# record_ids themselves are read from the generation shards, so a mismatch here
# cannot corrupt the join.
if [[ -f preflight/RUN_ID ]]; then
  P2_RUN_ID="$(cat preflight/RUN_ID)"
else
  echo "[warn] preflight/RUN_ID missing — Job A may not have run here" >&2
  P2_RUN_ID="unknown"
fi
export P2_RUN_ID
echo "[run-id] $P2_RUN_ID"

echo "[env ] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# Verify the activations symlink still resolves to scratch before spending GPU
# time — a dangling symlink here would otherwise fail 145 GB into the job.
if [[ ! -d activations ]]; then
  echo "[FATAL] ./activations does not resolve; expected symlink to " \
       "/scratch/jagatsesh/IEA-qwen72b-activations" >&2
  exit 1
fi

python -m phase2.capture_activations \
    --model "$MODEL_SLUG" \
    --run-id "$P2_RUN_ID"

echo "[ok  ] activation capture complete for run $P2_RUN_ID"
