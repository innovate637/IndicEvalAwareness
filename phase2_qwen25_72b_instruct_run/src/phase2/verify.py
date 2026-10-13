"""The run is not finished until this passes, or the missing set is frozen
in config/exclusions.json and reported in the paper."""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from phase2.config import load_exclusions, load_models, load_run
from phase2.io_jsonl import read_shard

run, models = load_run(), load_models()


def audit(root: Path, expect_models: list[str] | None = None) -> dict:
    excluded = load_exclusions()                                    # I10
    slugs = expect_models or list(models)
    expected = {(s, arm, lang, cue, i)
                for s in slugs
                for arm in run.arms
                for lang in run.langs
                for cue in run.cues
                for i in range(1, run.n_items + 1)
                if (s, lang) not in excluded}

    mpath = Path(run.paths["preflight"]) / "manifest.json"
    manifest_sha = (json.loads(mpath.read_text())["manifest_sha"]
                    if mpath.exists() else None)

    seen, counts = set(), defaultdict(int)
    dupes, errors, mism = [], [], []
    for p in Path(root).rglob("*.jsonl"):
        for r in read_shard(p):
            key = (r["model_slug"], r["arm"], r["lang"], r["cue"], r["itemnum"])
            counts[key] += 1
            if counts[key] > 1:
                dupes.append(key)
            seen.add(key)
            if r.get("error"):
                errors.append(r.get("error_class") or "Other")
            if manifest_sha and r.get("manifest_sha") != manifest_sha:
                mism.append(key)

    missing = sorted(expected - seen)
    by_cell = defaultdict(int)
    for s, arm, lang, cue, _ in missing:
        by_cell[f"{s}/{arm}/{lang}/{cue}"] += 1

    return {
        "models_expected": slugs,
        "cells_excluded": sorted(f"{a}/{b}" for a, b in excluded),
        "expected": len(expected), "seen": len(seen),
        "missing_total": len(missing),
        "missing_by_cell": dict(sorted(by_cell.items())),
        "missing_examples": [list(k) for k in missing[:25]],
        "duplicates": len(dupes),
        "error_rows": len(errors),
        "error_breakdown": {k: errors.count(k) for k in sorted(set(errors))},
        "manifest_mismatches": len(mism),
        "pass": not missing and not dupes and not mism,
    }


if __name__ == "__main__":
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(run.paths["generations"])
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    rep = audit(root, only)
    out = Path(run.paths["preflight"]) / "verify.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items()
                      if k != "missing_examples"}, indent=2))
    raise SystemExit(0 if rep["pass"] else 2)
