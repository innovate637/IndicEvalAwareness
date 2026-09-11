#!/bin/bash
# =============================================================================
# JOB A — Phase-2 GENERATION (vLLM) for Qwen2.5-72B-Instruct
#
# Submit Job A and Job B together with submit_both.sh, which wires the
# afterok dependency. Do NOT run Phase A and Phase B in one job: vLLM holds
# ~90% of VRAM for its KV pool and only releases it when the process exits,
# so the transformers pass in Job B would OOM.
#
# RESUMABLE. generate.py is idempotent per (run_id, model, arm, lang, cue,
# itemnum) via ShardWriter.has(), so if this job hits the 24h wall it can be
# resubmitted and will skip completed rows. That ONLY works if the run id is
# stable across submissions, which is why P2_RUN_ID is persisted to
# preflight/RUN_ID on first run and re-read afterwards. Never delete that file
# mid-campaign or every record_id changes and nothing resumes.
# =============================================================================
#SBATCH --job-name=iea_q72b_gen
#SBATCH --partition=gpu_h200_8
#SBATCH --account=bits
#SBATCH --qos=qos_gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jagatsesh/IEA-qwen72b/phase2_qwen25_72b_instruct_run/logs/gen_%j.out
#SBATCH --error=/home/jagatsesh/IEA-qwen72b/phase2_qwen25_72b_instruct_run/logs/gen_%j.err

set -euo pipefail

RUN_DIR=/home/jagatsesh/IEA-qwen72b/phase2_qwen25_72b_instruct_run
MODEL_SLUG=qwen25-72b-instruct

# Weights are pre-staged; OFFLINE forbids any network call, so a cache miss
# fails loudly instead of silently re-downloading 145 GB onto a compute node.
export HF_HOME=/scratch/jagatsesh/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Determinism. generate.py also sets this from run.yaml BEFORE importing vllm
# (which is the hard requirement); setting it here too makes it visible in the
# job environment. The attention backend is NOT set here — vLLM 0.19.1 dropped
# the VLLM_ATTENTION_BACKEND env var and made it an engine arg, so it comes
# from run.yaml determinism.attention_backend instead.
export VLLM_BATCH_INVARIANT=1

# conda activate is a shell function; a non-interactive batch shell has not
# sourced the hook, so `conda activate` alone would fail.
#
# `set -u` MUST be lifted across activation. The env's cuda-nvcc hook
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
# run.yaml paths are relative and resolve from CWD, hence the cd above.
export PYTHONPATH="$RUN_DIR/src"

mkdir -p logs preflight

# --- stable, resumable run id -------------------------------------------
if [[ -f preflight/RUN_ID ]]; then
  P2_RUN_ID="$(cat preflight/RUN_ID)"
  echo "[run-id] resuming existing run: $P2_RUN_ID"
else
  P2_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${MODEL_SLUG}-invariant"
  echo "$P2_RUN_ID" > preflight/RUN_ID
  echo "[run-id] new run: $P2_RUN_ID"
fi
export P2_RUN_ID

echo "[env ] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# One process per (model, language); each loads the model once and sweeps
# arm x cue internally (2 arms x 5 cues = 10 shards per language, 60 total).
# NOTE: the loop variable is L, not LANG — LANG is the shell's locale variable
# and overwriting it can change tokenisation-adjacent locale behaviour.
for L in en hi bn ta te kn; do
  echo "=== generating lang=$L ==="
  python -m phase2.generate \
      --model "$MODEL_SLUG" \
      --lang "$L" \
      --run-id "$P2_RUN_ID"
done

echo "[ok  ] generation complete for run $P2_RUN_ID"
