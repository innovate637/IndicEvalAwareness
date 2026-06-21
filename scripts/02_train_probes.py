"""
Phase B, Step 2 — Train per-layer eval/deploy linear probes.

For each (language, script, layer) triple, trains a logistic regression probe
on residual-stream activations from 01_extract_activations.py and reports AUROC.

Output:
  results/probes/auroc_all.csv         — AUROC per (lang, script, layer)
  results/probes/{lang}_{script}.pkl   — trained probes for steering vector use

Usage:
  python scripts/02_train_probes.py --lang hi --script native
  python scripts/02_train_probes.py --lang all

Reuses pattern from: ~/user-representation-interp/train_probes.py
"""

import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_activations(lang: str, script: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: [N_samples, NUM_LAYERS, D_MODEL]
      y: [N_samples]  (1 = eval, 0 = deploy)
    """
    path = config.ACT_DIR / f"{lang}_{script}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No activations at {path}. Run 01_extract_activations.py first.")
    with open(path, "rb") as f:
        data = pickle.load(f)

    eval_acts   = np.stack(data["eval"],   axis=0)   # [N_eval, L, D]
    deploy_acts = np.stack(data["deploy"], axis=0)   # [N_deploy, L, D]

    X = np.concatenate([eval_acts, deploy_acts], axis=0)
    y = np.array(
        [config.LABEL_EVAL]   * len(data["eval"]) +
        [config.LABEL_DEPLOY] * len(data["deploy"])
    )
    return X, y


def train_probe_at_layer(
    X_layer: np.ndarray, y: np.ndarray
) -> dict:
    """
    X_layer: [N, D_MODEL] activations at one layer.
    y:       [N] binary labels.

    Returns dict with:
      auroc_mean, auroc_std: cross-validated AUROC
      probe: trained LogisticRegression on full data (for steering vector extraction)
      scaler, pca: fitted transformers
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_layer)
    pca = PCA(n_components=min(config.PCA_COMPONENTS, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # Cross-validated AUROC
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=42)
    aurocs = []
    for train_idx, test_idx in skf.split(X_pca, y):
        clf = LogisticRegression(
            C=config.REGULARIZATION_C,
            max_iter=config.MAX_ITER,
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(X_pca[train_idx], y[train_idx])
        proba = clf.predict_proba(X_pca[test_idx])[:, 1]
        aurocs.append(roc_auc_score(y[test_idx], proba))

    # Full-data probe (for steering direction)
    probe = LogisticRegression(
        C=config.REGULARIZATION_C,
        max_iter=config.MAX_ITER,
        solver="lbfgs",
        random_state=42,
    )
    probe.fit(X_pca, y)

    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std":  float(np.std(aurocs)),
        "probe": probe,
        "scaler": scaler,
        "pca":   pca,
    }


def run_probing(lang: str, script: str) -> pd.DataFrame:
    probe_path = config.PROBE_DIR / f"{lang}_{script}.pkl"
    if probe_path.exists():
        print(f"  {lang}/{script}: probes already trained, skipping.")
        with open(probe_path, "rb") as f:
            probes = pickle.load(f)
        rows = [
            {"lang": lang, "script": script, "layer": l,
             "auroc_mean": probes[l]["auroc_mean"], "auroc_std": probes[l]["auroc_std"]}
            for l in probes
        ]
        return pd.DataFrame(rows)

    X, y = load_activations(lang, script)   # [N, L, D]
    n_layers = X.shape[1]

    probes = {}
    rows   = []
    for layer in range(n_layers):
        result = train_probe_at_layer(X[:, layer, :], y)
        probes[layer] = result
        rows.append({
            "lang":       lang,
            "script":     script,
            "layer":      layer,
            "auroc_mean": result["auroc_mean"],
            "auroc_std":  result["auroc_std"],
        })
        if layer % 5 == 0:
            print(f"    layer {layer:3d}: AUROC = {result['auroc_mean']:.3f} ± {result['auroc_std']:.3f}")

    with open(probe_path, "wb") as f:
        pickle.dump(probes, f)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",   default="all")
    parser.add_argument("--script", default="all")
    args = parser.parse_args()

    config.PROBE_DIR.mkdir(parents=True, exist_ok=True)
    auroc_path = config.PROBE_DIR / "auroc_all.csv"
    all_rows = []

    if auroc_path.exists():
        all_rows = [pd.read_csv(auroc_path)]

    langs   = config.LANG_CODES if args.lang   == "all" else [args.lang]
    scripts = config.SCRIPTS    if args.script == "all" else [args.script]

    for lang in langs:
        for script in scripts:
            if lang == "en" and script == "roman":
                continue
            print(f"\n→ {lang} / {script}")
            df = run_probing(lang, script)
            all_rows.append(df)

    pd.concat(all_rows).to_csv(auroc_path, index=False)
    print(f"\nAUROC summary → {auroc_path}")


if __name__ == "__main__":
    main()
