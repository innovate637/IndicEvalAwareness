#!/usr/bin/env python3
"""
13b — Cross-WORDING generalization of the eval-framing steering vector (anti-lexical test, NO GPU).

The eval_framing direction (eval_log - neutral_log) hits AUROC 1.0 at an EARLY layer (6), which
could mean it encodes the literal eval_log WORDING, not an abstract "I'm being evaluated" state.
This test builds the diff-of-means direction  d = mean(eval_log) - mean(neutral_log)  on a TRAIN
half of items, then asks whether d still separates OTHER eval wordings from neutral_log on the
HELD-OUT half:
   in-wording   : eval_log[test]      vs neutral_log[test]   (baseline)
   cross-wording: eval_explicit[test] vs neutral_log[test]
   cross-wording: eval_metadata[test] vs neutral_log[test]
High cross-wording AUROC  -> direction captures eval-framing that generalizes (NOT lexical).
Collapse toward 0.5       -> direction is wording-specific (lexical artifact).

d is exactly the vector script 14 would steer, so this directly bears on the steering claim.
Langs: en/hi/bn/ta/te (Odia lacks eval_explicit/eval_metadata). CPU only.
"""
import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from sklearn.metrics import roc_auc_score

LANGS = ["en", "hi", "bn", "ta", "te"]
LAYERS = [3, 6, 9, 12, 16, 19, 25, 32]
SEED = 0


def load(lang, cond):
    d = pickle.load(open(config.ACT_DIR / f"{lang}_{cond}.pkl", "rb"))
    return {i: a for i, a in zip(d["item_id"], np.asarray(d["acts"], dtype=np.float32))}


def main():
    rng = np.random.RandomState(SEED)
    rows = []
    for lang in LANGS:
        conds = {c: load(lang, c) for c in
                 ["eval_log", "neutral_log", "eval_explicit", "eval_metadata"]}
        ids = sorted(set.intersection(*[set(conds[c]) for c in conds]))
        ids = np.array(ids)
        perm = rng.permutation(len(ids))
        tr = ids[perm[:len(perm) // 2]]
        te = ids[perm[len(perm) // 2:]]

        def mat(cond, idset, layer):
            return np.stack([conds[cond][i][layer] for i in idset])

        for L in LAYERS:
            e = mat("eval_log", tr, L)
            n = mat("neutral_log", tr, L)
            d = e.mean(0) - n.mean(0)
            d = d / (np.linalg.norm(d) + 1e-9)

            def auroc(pos):
                sp = mat(pos, te, L) @ d
                sn = mat("neutral_log", te, L) @ d
                y = np.r_[np.ones(len(sp)), np.zeros(len(sn))]
                return roc_auc_score(y, np.r_[sp, sn])

            rows.append({"lang": lang, "layer": L,
                         "in_eval_log": round(auroc("eval_log"), 3),
                         "cross_explicit": round(auroc("eval_explicit"), 3),
                         "cross_metadata": round(auroc("eval_metadata"), 3)})

    df = pd.DataFrame(rows)
    df["cross_mean"] = df[["cross_explicit", "cross_metadata"]].mean(axis=1)
    df.to_csv(config.PROBE_DIR / "cross_wording.csv", index=False)

    print("=== per-lang @ layer 6 (the eval_framing layer) ===")
    print(df[df.layer == 6][["lang", "in_eval_log", "cross_explicit", "cross_metadata"]]
          .to_string(index=False))
    print("\n=== pooled mean AUROC by layer (across en/hi/bn/ta/te) ===")
    print(df.groupby("layer")[["in_eval_log", "cross_explicit", "cross_metadata", "cross_mean"]]
          .mean().round(3).to_string())
    best = df.groupby("layer")["cross_mean"].mean().idxmax()
    print(f"\nbest cross-wording layer = {best}  "
          f"(cross_mean={df[df.layer==best].cross_mean.mean():.3f}, "
          f"in_wording={df[df.layer==best].in_eval_log.mean():.3f})")
    print(f"→ {config.PROBE_DIR / 'cross_wording.csv'}")


if __name__ == "__main__":
    main()
