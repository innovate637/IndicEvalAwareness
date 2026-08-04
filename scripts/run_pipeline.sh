#!/usr/bin/env bash
# Unattended pipeline: cue-language factorial -> recognition -> analysis.
# SINGLE GPU only (pinned below). No sudo. venv-only. All writes under mech_interp/.
# Each stage saves incrementally and supports --resume, so a crash never loses
# finished languages. Re-run this script to resume; add --resume inside if needed.

set +e
export CUDA_VISIBLE_DEVICES=3          # least-used GPU at launch (all 4 were idle)
cd $PROJECT_ROOT || exit 1
PY="$PROJECT_ROOT/.venv/bin/python"

echo "=== PIPELINE START $(date) | GPU=$CUDA_VISIBLE_DEVICES ==="

# Remove dry-run partials so the full run regenerates each language completely.
rm -f results/behavioral/cue_factorial_hi.csv \
      results/behavioral/cue_factorial.csv \
      results/behavioral/recognition_cue_en_hi.csv \
      results/behavioral/recognition_cue.csv
echo "cleaned dry-run partials"

echo "--- [1/3] cue-language factorial (hi,bn,ta,te,or; Odia last; ~1000 gens @400tok) $(date) ---"
$PY scripts/05_cue_factorial.py
echo "factorial exit=$? $(date)"

echo "--- [2/3] recognition transcript-classification (en+native cues; thinking off; ~2200 gens) $(date) ---"
$PY scripts/07_recognition.py --thinking off --max-new-tokens 64
echo "recognition exit=$? $(date)"

echo "--- [3/3] analysis: factorial 2x2 + McNemar + traces + VEA-in-CoT + recognition $(date) ---"
$PY scripts/06_cue_factorial_analysis.py
echo "analysis exit=$? $(date)"

echo "=== PIPELINE DONE $(date) ==="
