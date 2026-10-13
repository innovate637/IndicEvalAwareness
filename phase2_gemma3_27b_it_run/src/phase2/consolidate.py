"""JSONL shards -> partitioned Parquet for the Phase-3 handoff."""
from __future__ import annotations
import json
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
from phase2.config import load_run
from phase2.io_jsonl import read_shard

run = load_run()
HEAVY = ["prompt_token_ids", "response_token_ids", "first_token_logprobs"]


def consolidate(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for model_dir in sorted(Path(root).iterdir()):
        if not model_dir.is_dir():
            continue
        light, heavy = [], []
        for p in sorted(model_dir.rglob("*.jsonl")):
            for r in read_shard(p):
                h = {"record_id": r["record_id"]}
                for k in HEAVY:
                    h[k] = json.dumps(r.pop(k, None))
                heavy.append(h)
                light.append(r)
        if not light:
            continue
        keys = sorted({k for r in light for k in r})     # uniform schema
        light = [{k: r.get(k) for k in keys} for r in light]
        pq.write_table(pa.Table.from_pylist(light),
                       out / f"{model_dir.name}.parquet", compression="zstd")
        pq.write_table(pa.Table.from_pylist(heavy),
                       out / f"{model_dir.name}.tokens.parquet",
                       compression="zstd")
        print(f"{model_dir.name}: {len(light)} rows")


if __name__ == "__main__":
    import sys
    consolidate(Path(sys.argv[1]) if len(sys.argv) > 1
                else Path(run.paths["generations"]),
                Path(sys.argv[2]) if len(sys.argv) > 2
                else Path(run.paths["home_out"]) / "generations_parquet")
