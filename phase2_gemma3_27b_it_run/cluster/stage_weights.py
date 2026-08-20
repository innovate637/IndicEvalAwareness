"""Login-node weight staging and revision pinning.

Compute nodes may lack internet: download here, resolve commit SHAs, write
them back into config/models.yaml, then run jobs with HF_HUB_OFFLINE=1.
"""
from __future__ import annotations
import os, sys, yaml
from huggingface_hub import HfApi, snapshot_download
from phase2.config import CONFIG, load_models

# hf_transfer was REMOVED upstream and HF_HUB_ENABLE_HF_TRANSFER is a no-op (V2).
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

# *.py is MANDATORY: trust_remote_code models (sarvam-30b) load custom
# modelling code and will fail without it (N9). *.bin covers repos that ship
# no safetensors.
ALLOW = ["*.safetensors", "*.safetensors.index.json", "*.bin", "*.json",
         "*.model", "*.txt", "*.jinja", "*.py", "tokenizer*"]


def main() -> None:
    api, models = HfApi(), load_models()
    changed: list[tuple[str, str]] = []
    raw = yaml.safe_load((CONFIG / "models.yaml").read_text())
    only = {a for a in sys.argv[1:] if not a.startswith("-")}
    for slug, m in models.items():
        if only and slug not in only:
            continue
        sha = m.revision or api.model_info(m.repo).sha
        print(f"[stage] {slug}  {m.repo}@{sha[:12]}"
              f"{'  (CONDITIONAL)' if m.conditional else ''}")
        snapshot_download(repo_id=m.repo, revision=sha,
                          allow_patterns=ALLOW, max_workers=8)
        for e in raw["models"]:
            if e["slug"] == slug:
                if e.get("revision") in (None, "", sha):
                    if e.get("revision") != sha:
                        changed.append((slug, sha))
                    e["revision"] = sha
                elif e["revision"] != sha:
                    print(f"  [WARN] {slug}: models.yaml pins "
                          f"{e['revision'][:12]} but the hub resolved "
                          f"{sha[:12]}. NOT overwriting — investigate.")

    # [LOCAL] The original rewrote models.yaml unconditionally with
    # yaml.safe_dump, which silently DESTROYS every comment in the file. This
    # repo's models.yaml carries the [LOCAL] deviation notes (partition, gres,
    # mem=32G per CLAUDE.md rule 4, KV re-derivation, the mandatory QOS) and
    # losing them would lose the record of why the resource map differs from
    # the plan. So: only rewrite when a revision actually changed, and say so
    # loudly enough that the comments can be restored by hand.
    if changed:
        print("\n[!] revisions newly resolved: "
              + ", ".join(f"{s}={h[:12]}" for s, h in changed))
        print("[!] models.yaml NOT auto-rewritten (it would strip all comments).")
        print("[!] Paste these in by hand, or re-run with --write-yaml.")
        if "--write-yaml" in sys.argv:
            (CONFIG / "models.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
            print("[!] models.yaml REWRITTEN — comments lost, restore from git.")
    else:
        print("\nall requested revisions already pinned; models.yaml untouched")


if __name__ == "__main__":
    main()
