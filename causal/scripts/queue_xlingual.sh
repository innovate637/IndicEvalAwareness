#!/bin/bash
# Queue: wait for the steering-localization run to finish, pick the carrying band, then run the
# cross-lingual EN->hi/bn steering transfer at that band (fall back to full 6-22 if no single 8-layer
# band captures >=70% of the full-window specific effect). GPU2 (freed by localization). GPU 0 free.
cd $PROJECT_ROOT
PY=.venv/bin/python
Q=results/causal/queue.log
echo "==== xlingual queue started $(date) ====" >> "$Q"
wait_pid(){ while kill -0 "$1" 2>/dev/null; do sleep 30; done; }

LOCALIZE_PID=${1:-2008886}
wait_pid "$LOCALIZE_PID"
echo "[$(date +%H:%M)] localization (pid $LOCALIZE_PID) finished" >> "$Q"

band=$($PY - <<'PY'
import pandas as pd
try:
    d=pd.read_csv("results/causal/localize_steer.csv").set_index("label")
    def spec(w):
        e=d.loc[w+" eval"]; n=d.loc[w+" null"]
        return ((e.gemma+e.sarvam)-(n.gemma+n.sarvam))/2
    full=spec("full6-22")
    wins=[l[:-5] for l in d.index if l.endswith(" eval") and l.startswith("L")]
    best=max(wins,key=spec) if wins else None
    use=best.replace("L","") if (best and full>0 and spec(best)>=0.7*full) else "6-22"
    print(use)
except Exception:
    print("6-22")
PY
)
echo "[$(date +%H:%M)] chosen band=$band -> cross-lingual EN->hi,bn on GPU2" >> "$Q"
CUDA_VISIBLE_DEVICES=2 $PY scripts/41_xlingual_hermes.py --langs hi bn --n 50 --win "$band" >> results/causal/xlingual_band.log 2>&1
echo "[$(date +%H:%M)] ==== xlingual (band=$band) DONE $(date) ====" >> "$Q"
