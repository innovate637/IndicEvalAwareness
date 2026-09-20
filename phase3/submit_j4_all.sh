#!/bin/bash
# Submit all six J4 language jobs (gate J4, plan 7.1).
#
# Each job judges one language: ~7,980 rows (1,995 per model x 4 models),
# ~2.8h at the 1.2 s/row measured in J2 job 353752, inside a 4h wall.
# Together the six cover the full 47,880-row primary set.
#
# Splitting is a scheduling change only. Every job loads the same pinned
# revision with the same rubric_sha and the same config, so the whole grid is
# judged under one judge snapshot (12.3), and J2's determinism result carries.
#
# THIS SUBMITS GPU JOBS. Per CLAUDE.md section 5 rule 7, run it only with
# explicit sign-off from Arya.
#
# Usage:  ./phase3/submit_j4_all.sh
#         ./phase3/submit_j4_all.sh en hi     # subset, e.g. to re-run two

set -euo pipefail

REPO=/home/jagatsesh/IEA-phase3
cd "$REPO"

LANGS=("$@")
if [[ ${#LANGS[@]} -eq 0 ]]; then
    LANGS=(en hi bn ta te kn)
fi

for lang in "${LANGS[@]}"; do
    case "$lang" in
        en|hi|bn|ta|te|kn) ;;
        *) echo "ERROR: unknown language '$lang'" >&2; exit 2 ;;
    esac
done

echo "submitting ${#LANGS[@]} J4 job(s): ${LANGS[*]}"
echo

for lang in "${LANGS[@]}"; do
    cmd=(sbatch --time=04:00:00 --job-name="j4_${lang}"
         phase3/run_judge.sbatch j4 --lang "$lang")
    echo "+ ${cmd[*]}"
    "${cmd[@]}"
    echo
done

echo "Submitted ${#LANGS[@]} J4 jobs"
echo
squeue -u jagatsesh
