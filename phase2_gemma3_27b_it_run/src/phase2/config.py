from __future__ import annotations
import os, yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def _load(name: str) -> dict:
    return _expand(yaml.safe_load((CONFIG / name).read_text()))


@dataclass(frozen=True)
class ModelCfg:
    slug: str
    repo: str
    revision: str | None
    kind: str                       # instruct | base
    thinking_kwarg: str | None
    true_system_role: bool
    tensor_parallel: int
    partition: str
    gres: str | None = None     # [LOCAL] explicit --gres string; this cluster
                                # names its GPUs gpu:nvidia_h100_nvl:1, so the
                                # plan's bare gpu:N would not resolve.
    gpus: int = 1
    mem: str = "32G"
    time: str = "08:00:00"
    approx_bf16_gb: int = 0
    conditional: bool = False
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = True
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 8192
    cpus_per_task: int = 8
    trust_remote_code: bool = False

    @property
    def is_base(self) -> bool:
        return self.kind == "base"


@dataclass(frozen=True)
class RunCfg:
    seed: int
    arms: list[str]
    cues: list[str]
    langs: list[str]
    n_items: int
    n_items_per_arm: dict
    sampling: dict
    determinism: dict
    paths: dict
    qos_gpu_caps: dict
    slurm: dict


def load_models() -> dict[str, ModelCfg]:
    raw = _load("models.yaml")
    defaults = raw.get("defaults", {})
    return {m["slug"]: ModelCfg(**{**defaults, **m}) for m in raw["models"]}


def load_run() -> RunCfg:
    raw = _load("run.yaml")
    mraw = _load("models.yaml")
    return RunCfg(seed=raw["seed"], arms=raw["arms"], cues=raw["cues"],
                  langs=raw["langs"], n_items=raw["n_items"],
                  n_items_per_arm=raw.get("n_items_per_arm") or {},
                  sampling=raw["sampling"], determinism=raw["determinism"],
                  paths=raw["paths"], qos_gpu_caps=mraw["qos_gpu_caps"],
                  slurm=mraw.get("slurm") or {})


def load_languages() -> dict[str, dict]:
    return _load("languages.yaml")["languages"]


def load_exclusions() -> set[tuple[str, str]]:
    import json
    f = CONFIG / "exclusions.json"
    if not f.exists():
        return set()
    return {(e["model_slug"], e["lang"])
            for e in json.loads(f.read_text()).get("excluded", [])}


def load_gate_markers() -> dict[str, list[str]]:
    """C4 (plan §8.14): the G7/G8 gate-only refusal lexicon.

    Lives in config/gate_refusal_markers.json rather than hardcoded in
    gates.py so that (a) a native reviewer never has to edit Python and
    (b) manifest.py hashes it, making the lexicon that produced a G7/G8
    verdict recoverable from the run record.

    Missing file or missing language => empty list, and gates.py reports
    "no gate lexicon" rather than silently scoring zero refusals.
    """
    import json
    f = CONFIG / "gate_refusal_markers.json"
    if not f.exists():
        return {}
    raw = json.loads(f.read_text())
    markers = raw.get("markers", raw)
    return {k: list(v) for k, v in markers.items() if not k.startswith("_")}


def max_tokens_table() -> dict[str, dict[str, int]]:
    import json
    f = CONFIG / "max_tokens.json"
    if not f.exists():
        raise FileNotFoundError("run `python -m phase2.token_budget` first")
    return json.loads(f.read_text())
