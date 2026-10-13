#!/bin/bash
# Submit Job A (generation) and Job B (activation capture), wiring the
# afterok dependency so B only starts if A succeeds.
#
# NOT RUN AUTOMATICALLY. Review both sbatch scripts, then run this by hand.
#
#   bash submit_both.sh
#
# To resume after a 24h timeout, just re-run Job A alone — it skips completed
# rows via the persisted preflight/RUN_ID:
#   sbatch run_qwen25_72b_gen.sbatch
set -euo pipefail
cd "$(dirname "$0")"

JOBA=$(sbatch --parsable run_qwen25_72b_gen.sbatch)
echo "Job A (generation)        : $JOBA"

JOBB=$(sbatch --parsable --dependency=afterok:"$JOBA" run_qwen25_72b_capture.sbatch)
echo "Job B (activation capture): $JOBB  [afterok:$JOBA]"

echo
squeue -u "$USER" -o "%.10i %.14j %.10P %.8T %.10M %.20R"
