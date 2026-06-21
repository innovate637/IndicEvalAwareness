#!/usr/bin/env python3
"""
Regenerate specific (lang, condition) cue-battery cells whose cue text was CORRECTED
after the cell was already generated (native-speaker fixes to provisional cues).

Why this is needed: 09_cue_battery_grid.py / the top-up watcher load the cue JSONs ONCE
at process startup, so a cue edited mid-run or after-run does NOT propagate to data already
on disk. This script makes the corrected cells "missing" again (drops their stale rows from
results/behavioral/cue_battery_{lang}.csv), then calls 09 --lang {lang} --resume, whose
condition-aware resume regenerates exactly those now-missing conditions using the CURRENT
(corrected) cue JSON, and appends them. Finally it re-merges cue_battery_grid.csv.

Cells to regenerate are read from data/cues/review/rerun_needed.json:
  { "rerun": { "bn": ["eval_explicit"], "ta": ["neutral_log", "eval_explicit"] } }

Run this only AFTER E1 + the top-up have fully finished AND all collected corrections are
applied to data/cues/cue_battery.json (so a single pass regenerates everything at once).
Pin the GPU yourself: CUDA_VISIBLE_DEVICES selects the device; venv-only; no sudo.

Usage:
  CUDA_VISIBLE_DEVICES=3 python scripts/rerun_corrected_cells.py
  python scripts/rerun_corrected_cells.py --dry-run     # show plan, change nothing
"""
import sys
import json
import argparse
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import config

RERUN_JSON = ROOT / "data" / "cues" / "review" / "rerun_needed.json"
NINE = ROOT / "scripts" / "09_cue_battery_grid.py"


def per_lang_path(lang: str) -> Path:
    return config.BEHAV_DIR / f"cue_battery_{lang}.csv"


def load_rerun_spec() -> dict:
    if not RERUN_JSON.exists():
        print(f"No {RERUN_JSON} — nothing to regenerate.")
        return {}
    spec = json.load(open(RERUN_JSON)).get("rerun", {})
    return {l: list(conds) for l, conds in spec.items() if conds}


def drop_stale_rows(lang: str, conditions: list, dry_run: bool) -> bool:
    """Remove rows for the given conditions from the per-lang CSV so --resume regenerates
    them. Returns True if rows were (or would be) dropped."""
    p = per_lang_path(lang)
    if not p.exists():
        print(f"  [{lang}] {p.name} not on disk yet — --resume will generate all "
              f"conditions fresh (incl. the corrected ones). OK.")
        return True
    df = pd.read_csv(p)
    if "condition" not in df.columns:
        print(f"  [{lang}] {p.name} has no 'condition' column — skipping.")
        return False
    mask = df["condition"].isin(conditions)
    n = int(mask.sum())
    present = sorted(set(conditions) & set(df["condition"].unique()))
    print(f"  [{lang}] drop {n} stale rows for {present} "
          f"(conditions requested: {conditions})")
    if dry_run:
        return n > 0
    df[~mask].to_csv(p, index=False)
    return True


def regenerate(lang: str, dry_run: bool):
    cmd = [sys.executable, str(NINE), "--lang", lang, "--resume"]
    print(f"  [{lang}] regenerate: {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=False)


def merge_grid(dry_run: bool):
    if dry_run:
        print("  [merge] (dry-run) would rebuild cue_battery_grid.csv")
        return
    import glob
    fs = [f for f in sorted(glob.glob(str(config.BEHAV_DIR / "cue_battery_*.csv")))
          if not f.endswith("cue_battery_grid.csv")]
    parts = [pd.read_csv(f) for f in fs]
    if not parts:
        print("  [merge] no per-lang CSVs found.")
        return
    m = pd.concat(parts, ignore_index=True)
    out = config.BEHAV_DIR / "cue_battery_grid.csv"
    m.to_csv(out, index=False)
    print(f"  [merge] {len(m)} rows -> {out}")
    print(m.groupby(["lang", "condition"]).size().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="show plan, change nothing")
    args = ap.parse_args()

    spec = load_rerun_spec()
    if not spec:
        return
    print(f"Cells to regenerate (from {RERUN_JSON.name}): {spec}\n")

    affected = []
    for lang, conds in spec.items():
        if drop_stale_rows(lang, conds, args.dry_run):
            affected.append(lang)

    print()
    for lang in affected:
        regenerate(lang, args.dry_run)

    print()
    merge_grid(args.dry_run)

    if not args.dry_run:
        print("\nDone. Clear regenerated cells from rerun_needed.json so they are not "
              "re-dropped on a future run.")


if __name__ == "__main__":
    main()
