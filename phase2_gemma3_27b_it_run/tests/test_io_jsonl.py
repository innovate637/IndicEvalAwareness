"""C5 — ShardWriter resume semantics.

Closes: idempotent resume (writing the same record_id twice yields one row)
and torn-tail recovery (a SIGKILL mid-write must not crash the shard on
restart, and must not leave a half-row that read_shard silently mis-parses).
"""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2.io_jsonl import ShardWriter, read_shard   # noqa: E402


def _rec(i: int) -> dict:
    return {"record_id": f"r{i}", "payload": i}


def test_duplicate_key_writes_once():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cue.jsonl"
        with ShardWriter(p) as w:
            w.write(_rec(1))
            w.write(_rec(1))
            w.write(_rec(2))
        rows = list(read_shard(p))
        assert len(rows) == 2, f"expected 2 rows, got {len(rows)}"
        assert [r["record_id"] for r in rows] == ["r1", "r2"]


def test_resume_skips_completed():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cue.jsonl"
        with ShardWriter(p) as w:
            for i in range(5):
                w.write(_rec(i))
        # reopen: everything already present must be reported done
        with ShardWriter(p) as w:
            assert all(w.has(f"r{i}") for i in range(5))
            assert not w.has("r99")
            w.write(_rec(99))
        rows = list(read_shard(p))
        assert len(rows) == 6, f"expected 6 rows, got {len(rows)}"


def test_torn_final_line_is_repaired():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cue.jsonl"
        with ShardWriter(p) as w:
            w.write(_rec(1))
            w.write(_rec(2))
        # simulate SIGKILL mid-write: append a partial line, no trailing \n
        with open(p, "a", encoding="utf-8") as f:
            f.write('{"record_id": "r3", "payl')
        with ShardWriter(p) as w:          # must not raise
            assert w.has("r1") and w.has("r2")
            assert not w.has("r3"), "torn row must not count as complete"
            w.write(_rec(3))
        rows = list(read_shard(p))
        assert len(rows) == 3, f"expected 3 rows, got {len(rows)}"
        assert rows[-1]["record_id"] == "r3"
        # and the file must be valid JSONL throughout
        for line in p.read_text().splitlines():
            json.loads(line)


def test_intact_final_line_is_not_truncated():
    # The repair must trigger only on a MISSING trailing newline, never on a
    # complete file -- otherwise every clean restart would silently drop a row.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cue.jsonl"
        with ShardWriter(p) as w:
            w.write(_rec(1))
            w.write(_rec(2))
        with ShardWriter(p) as w:
            pass
        assert len(list(read_shard(p))) == 2


def test_missing_file_reads_empty():
    with tempfile.TemporaryDirectory() as d:
        assert list(read_shard(Path(d) / "absent.jsonl")) == []


if __name__ == "__main__":
    import _runner
    sys.exit(_runner.run(sys.modules[__name__]))
