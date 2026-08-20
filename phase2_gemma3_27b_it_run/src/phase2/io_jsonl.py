"""Append-only, idempotent, crash-tolerant shard writer.

One file per (model, arm, lang, cue) => single writer, no locking. Restart
re-reads the file, drops a torn final line, and skips completed keys.
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Iterator


class ShardWriter:
    def __init__(self, path: Path, fsync_every: int = 25):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync_every, self._n = fsync_every, 0
        self._done = self._load_completed()
        self._fh = open(self.path, "a", encoding="utf-8")

    def _repair_tail(self) -> None:
        """Truncate a partial final line left by SIGKILL."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        data = self.path.read_bytes()
        if data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n")
        self.path.write_bytes(data[: cut + 1] if cut >= 0 else b"")

    def _load_completed(self) -> set[str]:
        self._repair_tail()
        done: set[str] = set()
        if not self.path.exists():
            return done
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["record_id"])
                except Exception:
                    continue                      # tolerate a corrupt mid-file row
        return done

    def has(self, record_id: str) -> bool:
        return record_id in self._done

    def write(self, rec: dict) -> None:
        if rec["record_id"] in self._done:
            return
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._done.add(rec["record_id"])
        self._n += 1
        if self._n % self.fsync_every == 0:
            self.flush()

    def flush(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_shard(path: Path) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue
