#!/bin/bash
# =============================================================================
# PROBE JOB — empirical max_tokens measurement (plan §5.2, option i)
#
# NOT PART OF THE MAIN GRID. Runs BEFORE Job A, writes to its own out-root and
# its own run id, and never touches ./generations. Its only product is a
# measured config/max_tokens.json.
#
# WHY: the heuristic table currently in config/max_tokens.json is unsafe —
#   en=512 is the formula's FLOOR (English is the reference language, so its
#   fertility ratio is 1.0 by construction and the heuristic ALWAYS returns
#   exactly BASE_EN); te=kn=3072 are the CEIL. Neither is a measurement.
#   §5.2: "Truncation must never correlate with language — that is the cheapest
#   way to manufacture the headline result." qwen3-32b already FAILED G6 on
#   te (6.93%) and kn (7.44%).
#
# WHAT IT DOES: generates a 40-item subset of every (arm, lang, cue) cell with
#   a deliberately huge completion cap, so completions end at EOS rather than
#   at the cap. token_budget.py --from-probe then sets each cell to
#   ceil32(p99_observed * 1.25).
#
# [72B] OVERRIDE IS 5824, NOT the docstring's 16384. vLLM rejects any request
#   where prompt_len + max_tokens > max_model_len. Longest measured prompt in
#   this dataset is 2343 tokens (te), so the ceiling is 8192 - 2343 = 5849;
#   5824 is the largest 32-aligned value under it. Using 16384 here would fail
#   every request.
#
# RESUMABLE via its own preflight/PROBE_RUN_ID.
# =============================================================================
#SBATCH --job-name=iea_q72b_probe
#SBATCH --partition=gpu_h200_8
#SBATCH --account=bits
#SBATCH --qos=qos_gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=/home/jagatsesh/IEA-qwen72b/phase2_qwen25_72b_instruct_run/logs/probe_%j.out
#SBATCH --error=/home/jagatsesh/IEA-qwen72b/phase2_qwen25_72b_instruct_run/logs/probe_%j.err

set -euo pipefail

RUN_DIR=/home/jagatsesh/IEA-qwen72b/phase2_qwen25_72b_instruct_run
MODEL_SLUG=qwen25-72b-instruct
OVERRIDE=5824          # 8192 max_model_len - 2343 longest prompt, 32-aligned
ITEMS=40

export HF_HOME=/scratch/jagatsesh/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_BATCH_INVARIANT=1

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
mkdir -p logs preflight probe_gen

if [[ -f preflight/PROBE_RUN_ID ]]; then
  PROBE_RUN_ID="$(cat preflight/PROBE_RUN_ID)"
  echo "[run-id] resuming probe: $PROBE_RUN_ID"
else
  PROBE_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${MODEL_SLUG}-probe"
  echo "$PROBE_RUN_ID" > preflight/PROBE_RUN_ID
  echo "[run-id] new probe: $PROBE_RUN_ID"
fi

echo "[env ] node=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"

for L in en hi bn ta te kn; do
  echo "=== probing lang=$L (${ITEMS} items, max_tokens=${OVERRIDE}) ==="
  python -m phase2.generate \
      --model "$MODEL_SLUG" \
      --lang "$L" \
      --run-id "$PROBE_RUN_ID" \
      --items "$ITEMS" \
      --max-tokens-override "$OVERRIDE" \
      --out-root ./probe_gen
done

# --- derive the measured budget -----------------------------------------
# CLAUDE.md 14: never silently overwrite. Keep the heuristic table alongside,
# matching the gemma run's own config/max_tokens.json.bak_heuristic.
if [[ -f config/max_tokens.json && ! -f config/max_tokens.json.bak_heuristic ]]; then
  cp config/max_tokens.json config/max_tokens.json.bak_heuristic
  echo "[bak ] heuristic table saved to config/max_tokens.json.bak_heuristic"
fi

python -m phase2.token_budget --from-probe probe_gen

echo
echo "[ok  ] probe complete. REVIEW config/max_tokens.json before running Job A."
echo "       Any cell reported as saturated means p99 is a floor, not an"
echo "       estimate, and must be re-probed with a larger override."
