"""Emit and optionally submit one sbatch per model, array over languages,
throttled to the per-user concurrent-GPU cap.

[LOCAL] Adapted from plan §8.12 for THIS cluster. Deviations, all forced:
  * --gres comes from models.yaml `gres` (gpu:nvidia_h100_nvl:1). The plan's
    bare `gpu:N` does not resolve against this node's Gres declaration.
  * --mem is whatever models.yaml says, and models.yaml says 32G everywhere.
    CLAUDE.md rule 4 is a hard rule: a job without a bounded --mem reserves the
    entire ~503 GB node RAM and blocks the node's other GPU for everyone else.
    An explicit --mem line is emitted ALWAYS -- see the assertion below.
  * Throttle is %1, not %3: this cluster has one full H100 NVL, so the
    concurrent cap is 1 (qos_gpu_caps in models.yaml).
  * HF_HOME points at ~/.cache/huggingface (no /scratch on this machine) and
    the conda env is `vllm`, not `p2`.
  * HF_HUB_OFFLINE=1 is kept: weights must be staged BEFORE submitting, so a
    compute node never reaches for the network mid-run.
"""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
from phase2.config import REPO, load_models, load_run

TPL = """#!/bin/bash
#SBATCH --job-name=p2-{slug}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --array=0-5%{throttle}
#SBATCH --output={logdir}/{slug}_%A_%a.out
#SBATCH --requeue
{extra}
set -euo pipefail

LANGS=(en hi bn ta te kn)   # positional: array index 5 -> kn (was or)
LANG_CODE=${{LANGS[$SLURM_ARRAY_TASK_ID]}}

# [LOCAL] $HOME (/home/dhruvkumar) is NOT WRITABLE on the compute node --
# vLLM's model-info cache died there with PermissionError (job 440). Pin every
# cache root inside the project so no job ever reaches for /home.
export HOME={repo}/phase2_scratch/home
export XDG_CACHE_HOME={repo}/phase2_scratch/xdg_cache
export VLLM_CACHE_ROOT={repo}/phase2_scratch/vllm_cache
export HF_HOME={repo}/phase2_scratch/hf_home
export HF_HUB_CACHE={repo}/phase2_scratch/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export P2_RUN_ID={run_id}
export PYTHONPATH={repo}/src:${{PYTHONPATH:-}}
export PYTHONUNBUFFERED=1

# [LOCAL] .venv-p2: torch 2.10.0+cu128, which RUNS on this node's driver
# 565.57.01 via CUDA minor-version compat (verified job 438). The miniforge
# 'vllm' env is cu130 and cannot initialise CUDA here at all.
cd {repo}

echo "[task] model={slug} lang=$LANG_CODE node=$SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# No srun: single node, single GPU, and srun can interfere with vLLM's
# worker spawning for no benefit here.
/localstorage/home/dhruvkumar/IndicEvalAwareness/.venv-p2/bin/python -m phase2.generate --model {slug} --lang "$LANG_CODE" \\
       --run-id "$P2_RUN_ID"
"""


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: submit.py <RUN_ID> [--submit] [slug ...]")
        sys.exit(2)
    run, models = load_run(), load_models()
    run_id, submit = sys.argv[1], "--submit" in sys.argv
    only = {a for a in sys.argv[2:] if not a.startswith("-")}

    # Absolute: Slurm resolves a relative --output against the SUBMIT cwd, and
    # a job whose log path does not resolve dies with no log at all (N10).
    logdir = Path(run.paths["logs"])
    if not logdir.is_absolute():
        logdir = (REPO / logdir).resolve()
    logdir.mkdir(parents=True, exist_ok=True)      # Slurm will NOT do this (N10)
    outdir = REPO / "cluster" / "sbatch"
    outdir.mkdir(parents=True, exist_ok=True)

    extra = []
    if run.slurm.get("account"):
        extra.append(f"#SBATCH --account={run.slurm['account']}")
    if run.slurm.get("qos"):
        extra.append(f"#SBATCH --qos={run.slurm['qos']}")

    for slug, m in models.items():
        if only and slug not in only:
            continue
        cap = run.qos_gpu_caps.get(m.partition)
        if not cap:
            print(f"[skip] {slug}: partition {m.partition} has concurrent cap "
                  f"{cap!r} -- refusing to emit a job that cannot schedule")
            continue
        throttle = max(1, cap // m.gpus)

        # CLAUDE.md rule 4, enforced rather than trusted. A missing or absurd
        # --mem silently reserves the whole node on this cluster.
        if not m.mem or not str(m.mem).strip():
            raise SystemExit(f"{slug}: no --mem in models.yaml; refusing to "
                             f"submit an unbounded-memory job")
        if not m.gres:
            raise SystemExit(f"{slug}: no gres in models.yaml; refusing to "
                             f"guess a GPU request on this cluster")

        text = TPL.format(slug=slug, partition=m.partition, gres=m.gres,
                          cpus=m.cpus_per_task, mem=m.mem, time=m.time,
                          throttle=throttle, run_id=run_id, repo=REPO,
                          logdir=logdir, extra="\n".join(extra))
        assert "--mem=" in text and "--gres=" in text, "bounded request missing"
        f = outdir / f"gen_{slug}.sbatch"
        f.write_text(text)
        flag = " [CONDITIONAL: run only if G2 passed]" if m.conditional else ""
        print(f"wrote {f}  (array 0-5%{throttle} on {m.partition}, "
              f"gres={m.gres}, mem={m.mem}){flag}")
        if submit:
            print(subprocess.check_output(["sbatch", str(f)], text=True).strip())


if __name__ == "__main__":
    main()
