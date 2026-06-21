#!/usr/bin/env bash
# E1 cue-battery grid, data-parallel across 2 GPUs (language-sharded). ~2x speed.
# GPU 3: en, hi, ta   |   GPU 2: bn, te, or  (Odia balanced against lighter langs).
# Per-language CSVs are written independently (no cross-process file conflict);
# a final authoritative merge rebuilds cue_battery_grid.csv from all of them.
# No sudo, venv-only, all writes under mech_interp/.
set +e
cd /home/sibayan_mitra_2024/mech_interp || exit 1
PY=/home/sibayan_mitra_2024/mech_interp/.venv/bin/python

echo "=== E1 2-GPU START $(date) ==="
# remove dry-run partials so the full run regenerates each language completely
rm -f results/behavioral/cue_battery_hi.csv results/behavioral/cue_battery_grid.csv

# GPU 3 shard
( for L in en hi ta; do echo "[GPU3] start $L $(date)"; \
    CUDA_VISIBLE_DEVICES=3 $PY scripts/09_cue_battery_grid.py --lang "$L" --resume; \
    echo "[GPU3] done $L exit=$? $(date)"; done; echo "[GPU3] SHARD DONE $(date)" ) \
  > results/behavioral/e1_gpu3.log 2>&1 &
P3=$!

# GPU 2 shard
( for L in bn te or; do echo "[GPU2] start $L $(date)"; \
    CUDA_VISIBLE_DEVICES=2 $PY scripts/09_cue_battery_grid.py --lang "$L" --resume; \
    echo "[GPU2] done $L exit=$? $(date)"; done; echo "[GPU2] SHARD DONE $(date)" ) \
  > results/behavioral/e1_gpu2.log 2>&1 &
P2=$!

wait $P3 $P2

# final authoritative merge of all per-language CSVs
$PY -c "
import pandas as pd, glob
fs=[f for f in sorted(glob.glob('results/behavioral/cue_battery_*.csv')) if not f.endswith('cue_battery_grid.csv')]
parts=[pd.read_csv(f) for f in fs]
m=pd.concat(parts, ignore_index=True)
m.to_csv('results/behavioral/cue_battery_grid.csv', index=False)
print('merged', len(m), 'rows from', len(fs), 'lang files ->', sorted(m.lang.unique()))
print(m.groupby(['lang','condition'])['refusal'].mean().round(2).to_string())
"
echo "=== E1 2-GPU DONE $(date) ==="
