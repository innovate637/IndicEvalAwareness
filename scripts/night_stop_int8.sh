#!/bin/bash
# night_stop_int8.sh — graceful stop of the int8 E1 runs to hand GPUs 2/3 over.
#
# Usage:  bash scripts/night_stop_int8.sh all     # stop both GPUs (2 and 3)
#         bash scripts/night_stop_int8.sh en      # stop only the en->bn->or chain (GPU2)
#         bash scripts/night_stop_int8.sh hi      # stop only the hi->ta->te chain (GPU3)
#
# Kills the wrapper bash loop FIRST (so the next language never launches), then the python.
# NOTE: the en/hi processes launched 2026-06-12 ~09:00 run OLD code that saves only at
# language end — killing them mid-language loses that language's generations. Languages
# launched AFTER the checkpointing patch (bn/or/ta/te) lose at most one batch.

TARGET="${1:-all}"
cd "$(dirname "$0")/.." || exit 1

mapfile -t PIDS < <(pgrep -f "09_cue_battery_grid.py --int8")
if [ ${#PIDS[@]} -eq 0 ]; then
    echo "no int8 cue_battery python processes found."
else
    for p in "${PIDS[@]}"; do
        ARGS=$(ps -o args= -p "$p" 2>/dev/null)
        LANG_RUNNING=$(echo "$ARGS" | grep -oE -- "--lang [a-z]+" | awk '{print $2}')
        # only the python generator has an EXPANDED --lang (en/hi/...); the wrapper bash
        # carries the literal "--lang $L" → empty here → skip it (we kill it via ppid).
        if [ -z "$LANG_RUNNING" ]; then
            continue
        fi
        if [ "$TARGET" != "all" ] && [ "$TARGET" != "$LANG_RUNNING" ]; then
            echo "skip python $p (lang=$LANG_RUNNING, target=$TARGET)"
            continue
        fi
        WRAP=$(ps -o ppid= -p "$p" | tr -d ' ')
        echo "stopping: wrapper $WRAP then python $p (lang=$LANG_RUNNING)"
        kill "$WRAP" 2>/dev/null
        sleep 2
        kill "$p" 2>/dev/null
    done
    sleep 25   # give CUDA time to release
fi

echo ""
echo "=== GPU state after stop ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo ""
echo "=== int8 data on disk (safe) ==="
ls -la results/behavioral/cue_battery_*int8* 2>/dev/null || echo "(none yet)"
echo ""
echo "Post-weekend resume: rerun the same wrapper commands with --resume;"
echo "checkpointed languages continue from their last completed batch."
