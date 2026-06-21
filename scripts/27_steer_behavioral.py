#!/usr/bin/env python3
"""
27 — BEHAVIORAL causal readout (readout B): does steering DEPLOY generation with the EVAL-AWARENESS
     direction RAISE refusal (reproduce the +5-9pp compliance gap), cross-lingually — and NOT on
     benign prompts (specificity)?

Reuses script 25's validated machinery (steer_generate_batch / run_shard / judge_one) but:
  - builds eval/refusal/random directions at a FIXED MID layer L_mid (NOT the probe-best or the last
    layer L39, whose final-RMSNorm washes steering out — the bug that killed the earlier run);
  - adds a BENIGN prompt set as a specificity control;
  - controls-first job order (positive=refusal@L_mid, negative=random, then within-lang anchor, then
    cross-lingual, then benign) so a partial overnight finish still has the diagnostic cells.

Refusal scored inline by the sarvam-105b LLM judge (NOT keyword). Sign-swept dose-response.

  python scripts/27_steer_behavioral.py --L 9 --n 20 --strengths -0.5 0 0.5 --resume
Outputs: results/steering/behavioral/steer_{kind}_{lang}.csv  (per-generation checkpoint, resume-safe)
"""
import os, sys, json, argparse, importlib.util
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import config
from behavioral_common import load_model_and_tokenizer


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

s25 = _load("s25", "25_steer_xlang.py")   # run_shard, steer_generate_batch, judge_one, eval_dir_orth, ...
s14 = _load("s14", "14_steer_patch.py")   # load_harmful_subset
s09 = _load("s09", "09_cue_battery_grid.py")  # get_cue, load_cue_sources

s25.OUT_DIR = config.STEER_DIR / "behavioral"   # default; main() namespaces it by steer layer


def load_benign_subset(lang, n):
    p = config.SAFETY_DIR / "benign" / f"{lang}.json"
    if not p.exists():
        return []
    return json.load(open(p))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=9, help="mid layer to steer at (NOT last layer)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--strengths", nargs="+", type=float, default=[-0.5, 0.0, 0.5])
    ap.add_argument("--langs", nargs="+", default=["en", "hi", "bn", "ta", "te"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    L = a.L
    s25.OUT_DIR = config.STEER_DIR / f"behavioral_L{L}"   # keep layers in separate dirs (resume-safe)

    # ---- build directions at L_mid ----
    def vec_entry(dir_kind, src):
        if dir_kind == "refusal":
            v = s25.refusal_dir_at(src, L)
        elif dir_kind == "random":
            v = s25.build_random_direction(L)["vector"]
        else:
            v = s25.eval_dir_orth(src, L)
        return None if v is None else {"layer": L, "vector": v}

    # ---- job matrix: (target_lang, kind, src, dir_kind, item_set) — CONTROLS FIRST ----
    indic = [l for l in a.langs if l != "en"]
    jobs = [("en", "refusal", "en", "refusal", "harmful")]        # POSITIVE control (gate; runs/resumes first)
    for t in indic:
        jobs.append((t, "xeval_from_en", "en", "eval", "harmful"))    # cross-lingual HEADLINE (the question) — prioritized
        jobs.append((t, "random_into",   "en", "random", "harmful"))  # matched NEGATIVE control in the same target
    jobs += [
        ("en", "random",      "en", "random", "harmful"),         # negative control (EN)
        ("en", "eval_within", "en", "eval",   "harmful"),         # within-lang anchor
    ]
    # specificity: eval steering on BENIGN must NOT induce generic over-refusal
    jobs.append(("en", "eval_within_benign", "en", "eval", "benign"))
    for t in indic[:2]:
        jobs.append((t, "xeval_from_en_benign", "en", "eval", "benign"))
    # reverse transfer: each Indic eval dir -> EN
    for s in indic:
        jobs.append(("en", f"xeval_from_{s}", s, "eval", "harmful"))

    s25.sv.KEY = s25.sv.sarvam_key()
    battery, framings, translations = s09.load_cue_sources()
    model, tokenizer = load_model_and_tokenizer(int8=True)
    print(f"model loaded. L_mid={L} n={a.n} strengths={a.strengths}")
    print("jobs: " + ", ".join(f"{k}:{t}" for t, k, *_ in jobs))

    s25.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tgt, kind, src, dir_kind, item_set in jobs:
        dir_entry = vec_entry(dir_kind, src)
        if dir_entry is None:
            print(f"  [skip] {kind} (no {dir_kind} dir for {src}@L{L})"); continue
        items = (load_benign_subset(tgt, a.n) if item_set == "benign"
                 else s14.load_harmful_subset(tgt, a.n))
        cue = s09.get_cue(tgt, "deploy", battery, translations)
        if not items or cue is None:
            print(f"  [skip] {kind} {tgt} (items={len(items) if items else 0}, cue={'ok' if cue else 'MISSING'})"); continue
        print(f"\n→ {kind} | {tgt} | src={src} dir={dir_kind} @L{L} | {item_set} n={len(items)}")
        df = s25.run_shard(model, tokenizer, tgt, kind, dir_entry, items, cue, a.strengths, a.batch_size)
        if not df.empty:
            g = df.groupby("strength")["refusal_steered"].mean()
            print("  refusal by alpha: " + "  ".join(f"{s_:+.2f}:{r:.2f}" for s_, r in g.items()))
    print("\nDone -> results/steering/behavioral/")


if __name__ == "__main__":
    main()
