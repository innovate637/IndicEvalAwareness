"""Freeze and hash every input AND every config before generation."""
from __future__ import annotations
import datetime as dt, hashlib, json, subprocess, sys, uuid
from pathlib import Path
from phase2.config import CONFIG, REPO, load_models, load_run

# C8 (plan §8.14) — scripts the plan CALLS but did not WRITE, whose assumed
# properties Phase 2 depends on. Verify empirically before relying on them:
#   14_sample_final_set.py  -> prefix stability (generate 200 and 203, diff
#                              the first 200). Asserted throughout §3.2/§3.6/§10
#                              and never tested.
#   02_screen_harm.py       -> harm-retention screening on an arbitrary subset.
#   score_cometkiwi.py      -> re-runnable per language.
INHERITED_SCRIPTS = [
    "scripts/14_sample_final_set.py",
    "scripts/02_screen_harm.py",
    "scripts/score_cometkiwi.py",
]


def new_run_id() -> str:
    """Stdlib-only, sortable, unique. Replaces the ulid dependency (N8)."""
    return (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-" + uuid.uuid4().hex[:8])


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build(run_id: str) -> dict:
    run, models = load_run(), load_models()
    import torch, transformers, vllm
    from phase2.assemble import load_items

    files: dict[str, str] = {}
    for p in sorted((REPO / "data").rglob("*.json")):
        files[str(p.relative_to(REPO))] = sha256_file(p)
    for p in sorted(CONFIG.glob("*.yaml")) + sorted(CONFIG.glob("*.json")):
        files[str(p.relative_to(REPO))] = sha256_file(p)      # fixes I9

    # C8 (plan §8.14): three inherited Phase-1 scripts. Not Phase-2 code, but
    # their OUTPUT is Phase-2 input, and the plan assumes properties of them
    # (prefix stability of the seed-2026 permutation; re-runnability on an item
    # subset; per-language re-scoring). Hash them so the assumption is pinned.
    inherited: dict[str, str] = {}
    for rel in INHERITED_SCRIPTS:
        p = REPO / rel
        inherited[rel] = sha256_file(p) if p.exists() else "MISSING"

    docsets = {}
    for arm in run.arms:
        for lang in run.langs:
            try:
                rows = load_items(arm, lang)
            except FileNotFoundError:
                continue
            docsets[f"{arm}/{lang}"] = {
                "n": len(rows),
                "doc_id_sha": sha256_obj(sorted(r["doc_id"] for r in rows))}

    man = {
        "run_id": run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_driver": torch.version.cuda,
        "seed": run.seed, "arms": run.arms, "cues": run.cues,
        "langs": run.langs, "n_items": run.n_items,
        "sampling": run.sampling, "determinism": run.determinism,
        "models": {s: {"repo": m.repo, "revision": m.revision, "kind": m.kind,
                       "dtype": m.dtype, "tp": m.tensor_parallel,
                       "max_model_len": m.max_model_len,
                       "max_num_seqs": m.max_num_seqs,
                       "conditional": m.conditional}
                   for s, m in models.items()},
        "input_files": files, "doc_sets": docsets,
        "inherited_scripts": inherited,
    }
    man["manifest_sha"] = sha256_obj({k: v for k, v in man.items()
                                      if k != "manifest_sha"})
    return man


if __name__ == "__main__":
    rid = sys.argv[1] if len(sys.argv) > 1 else new_run_id()
    man = build(rid)
    out = Path(load_run().paths["preflight"]) / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    print(json.dumps({"run_id": rid, "manifest_sha": man["manifest_sha"],
                      "path": str(out)}, indent=2))
