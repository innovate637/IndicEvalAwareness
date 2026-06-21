"""
E4b — Per-layer eval-awareness probes + cross-lingual transfer (NO GPU).

Pure sklearn over the residual-stream activations cached by 12_* into
results/activations/{lang}_{condition}.pkl. No model load, no generation, no API.

Two contrasts are probed independently, per layer (0..NUM_LAYERS-1):

  (a) CUE-PRESENCE   : eval_log activations (class 1) vs deploy activations (class 0).
                       "Can a linear probe read off whether the eval cue was present?"
  (b) AWARENESS      : VEA-positive (class 1) vs VEA-negative (class 0), POOLING all
      (anti-trivial)   available conditions for the language and dropping samples whose
                       vea label is None. This is the anti-trivial contrast: it does not
                       hand the probe the cue itself, only whether the model verbalized
                       eval-awareness, so a high AUROC means an internal "I'm being
                       evaluated" representation rather than mere cue-token detection.

Per layer: StandardScaler -> PCA(128) -> LogisticRegression(C=1, max_iter=1000),
5-fold StratifiedKFold AUROC (held-out, never trained-on).

Cross-lingual TRANSFER: fit the full pipeline on EN activations for a contrast, then
score (AUROC) the EN-trained probe on each Indic language's held activations at the
EN best layer. Because en/hi/bn carry the richer 5-cue battery while ta/te/or carry
only deploy+eval_log, cross-cue-WORDING transfer is exercised implicitly wherever the
extra conditions exist; languages missing a contrast's classes are skipped + logged.

Outputs:
  results/probes/auroc_all.csv     — held-out AUROC per (contrast, lang, layer)
  results/probes/transfer.csv      — EN->lang transfer AUROC at EN best layer per contrast
  results/probes/directions.pkl    — diff-of-means direction per (lang, contrast, best_layer)
  results/probes/cosine.csv        — cosine matrix across per-lang AWARENESS eval directions

GATE printout (publishability checks):
  - EN held-out AUROC > 0.80 ?
  - EN -> Indic transfer AUROC > 0.65 ?

Cache contract (produced by 12_*; this script is robust to two layouts):
  results/activations/{lang}_{condition}.pkl  is a dict with EITHER
    {"acts": [np.ndarray[NUM_LAYERS, D_MODEL], ...],   # one per sample
     "vea":  [0|1|None, ...]}                          # aligned per-sample VEA label
  OR the 01_*-style minimal form
    {"acts": np.ndarray[N, NUM_LAYERS, D_MODEL]}       # vea then treated as all-None
  Extra keys (item_id, condition, ...) are ignored. Missing files are skipped + logged.

Usage:
  python scripts/13_probe_transfer.py                 # all contrasts, all langs (Odia last)
  python scripts/13_probe_transfer.py --lang hi       # restrict probing to one language
  python scripts/13_probe_transfer.py --limit 40      # cap samples per (lang,condition) class
  python scripts/13_probe_transfer.py --resume        # reuse auroc_all.csv rows already on disk

CPU-only: validate with
  python -c "import ast; ast.parse(open('scripts/13_probe_transfer.py').read())"
"""

import sys
import pickle
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── Experiment layout ───────────────────────────────────────────────────────────

# Per-language available cue conditions (see CLAUDE.md / cue JSONs):
#   en/hi/bn -> full 5-cue battery; ta/te/or -> deploy + eval_log only (rest untranslated).
FULL_CONDITIONS = ["deploy", "neutral_log", "eval_log", "eval_explicit", "eval_metadata"]
MIN_CONDITIONS  = ["deploy", "eval_log"]
CONDITIONS_BY_LANG = {
    "en": FULL_CONDITIONS, "hi": FULL_CONDITIONS, "bn": FULL_CONDITIONS,
    "ta": MIN_CONDITIONS,  "te": MIN_CONDITIONS,  "or": MIN_CONDITIONS,
}

# Contrasts, primary first. EVAL_FRAMING (eval_log vs neutral_log) is the anti-trivial eval
# direction — both arms carry a monitoring sentence, so a probe separating them reads
# "is this a safety eval" not "is a cue present". CUE_PRESENCE (eval_log vs deploy) is a
# cue-token-detection baseline. AWARENESS (VEA) is leaky/partial → exploratory only.
CONTRASTS = ["eval_framing", "cue_presence", "awareness"]

# Run order: en first (the transfer SOURCE + ceiling baseline), Odia ('or') LAST (slowest
# to extract upstream / lowest-confidence cues). Indic = everything except en.
LANG_ORDER  = ["en", "hi", "bn", "ta", "te", "or"]
INDIC_LANGS = [l for l in LANG_ORDER if l != "en"]

PCA_COMPONENTS   = config.PCA_COMPONENTS    # 128
REGULARIZATION_C = config.REGULARIZATION_C  # 1.0
MAX_ITER         = config.MAX_ITER          # 1000
CV_FOLDS         = config.CV_FOLDS          # 5
RANDOM_STATE     = 42

GATE_HELDOUT = 0.80   # EN held-out AUROC must exceed this
GATE_TRANSFER = 0.65  # EN->Indic transfer AUROC must exceed this


# ── Activation cache loading ─────────────────────────────────────────────────────

def _act_path(lang: str, condition: str) -> Path:
    return config.ACT_DIR / f"{lang}_{condition}.pkl"


def load_condition(lang: str, condition: str):
    """
    Load one cached (lang, condition) activation file produced by 12_*.

    Returns (acts, vea) where
      acts : np.ndarray [N, NUM_LAYERS, D_MODEL]
      vea  : np.ndarray [N], dtype=object, entries in {0, 1, None}
    or (None, None) if the file is missing / empty (caller skips + logs).
    """
    path = _act_path(lang, condition)
    if not path.exists():
        return None, None
    with open(path, "rb") as f:
        data = pickle.load(f)

    raw = data.get("acts", data) if isinstance(data, dict) else data
    if raw is None:
        return None, None

    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        acts = raw.astype(np.float32, copy=False)
    else:
        # list (or 1-D object array) of per-sample [NUM_LAYERS, D_MODEL] arrays
        raw = list(raw)
        if len(raw) == 0:
            return None, None
        acts = np.stack([np.asarray(a, dtype=np.float32) for a in raw], axis=0)

    n = acts.shape[0]
    if n == 0:
        return None, None

    vea_raw = data.get("vea") if isinstance(data, dict) else None
    if vea_raw is None:
        vea = np.array([None] * n, dtype=object)
    else:
        vea_list = list(vea_raw)
        if len(vea_list) != n:
            warnings.warn(
                f"{path.name}: vea length {len(vea_list)} != n acts {n}; "
                f"treating vea as all-None."
            )
            vea = np.array([None] * n, dtype=object)
        else:
            vea = np.array(
                [None if (v is None or (isinstance(v, float) and np.isnan(v))) else int(v)
                 for v in vea_list],
                dtype=object,
            )
    return acts, vea


def build_contrast(lang: str, contrast: str, limit: int = None):
    """
    Assemble (X, y) for one (lang, contrast).

    X : np.ndarray [N, NUM_LAYERS, D_MODEL]
    y : np.ndarray [N] in {0, 1}

    cue_presence : class 1 = eval_log, class 0 = deploy.
    awareness    : pool ALL available conditions for the lang, label by vea
                   (class 1 = vea==1, class 0 = vea==0), drop vea is None.

    Returns (None, None) if either class is missing (caller skips + logs why).
    """
    conditions = CONDITIONS_BY_LANG.get(lang, MIN_CONDITIONS)

    if contrast == "cue_presence":
        acts1, _ = load_condition(lang, "eval_log")
        acts0, _ = load_condition(lang, "deploy")
        if acts1 is None or acts0 is None:
            missing = [c for c, a in (("eval_log", acts1), ("deploy", acts0)) if a is None]
            print(f"    [{lang}/{contrast}] missing condition(s): {missing} — skipping.")
            return None, None
        if limit:
            acts1, acts0 = acts1[:limit], acts0[:limit]
        X = np.concatenate([acts1, acts0], axis=0)
        y = np.concatenate([np.ones(len(acts1), int), np.zeros(len(acts0), int)])
        return X, y

    if contrast == "eval_framing":
        # eval_log vs neutral_log: BOTH carry a monitoring sentence; they differ ONLY in
        # safety/compliance-eval vs service-quality-logging framing. A probe separating them
        # cannot be reading "is a cue present" → this is the anti-trivial eval-awareness
        # direction, and it mirrors the significant behavioral contrast (eval_log−neutral_log).
        acts1, _ = load_condition(lang, "eval_log")
        acts0, _ = load_condition(lang, "neutral_log")
        if acts1 is None or acts0 is None:
            missing = [c for c, a in (("eval_log", acts1), ("neutral_log", acts0)) if a is None]
            print(f"    [{lang}/{contrast}] missing condition(s): {missing} — skipping.")
            return None, None
        if limit:
            acts1, acts0 = acts1[:limit], acts0[:limit]
        X = np.concatenate([acts1, acts0], axis=0)
        y = np.concatenate([np.ones(len(acts1), int), np.zeros(len(acts0), int)])
        return X, y

    # awareness: pool conditions, drop vea None
    pos_acts, neg_acts = [], []
    used = []
    for cond in conditions:
        acts, vea = load_condition(lang, cond)
        if acts is None:
            continue
        used.append(cond)
        pos_mask = np.array([v == 1 for v in vea], dtype=bool)
        neg_mask = np.array([v == 0 for v in vea], dtype=bool)
        if pos_mask.any():
            pos_acts.append(acts[pos_mask])
        if neg_mask.any():
            neg_acts.append(acts[neg_mask])
    if not pos_acts or not neg_acts:
        print(f"    [{lang}/{contrast}] only one VEA class present across {used} "
              f"(pos={sum(len(a) for a in pos_acts)}, neg={sum(len(a) for a in neg_acts)}) "
              f"— skipping.")
        return None, None
    acts1 = np.concatenate(pos_acts, axis=0)
    acts0 = np.concatenate(neg_acts, axis=0)
    if limit:
        acts1, acts0 = acts1[:limit], acts0[:limit]
    X = np.concatenate([acts1, acts0], axis=0)
    y = np.concatenate([np.ones(len(acts1), int), np.zeros(len(acts0), int)])
    return X, y


# ── Probe pipeline ───────────────────────────────────────────────────────────────

def make_pipeline(n_features: int, n_samples: int = None) -> Pipeline:
    """StandardScaler -> PCA(<=128, capped by fit-set size) -> LogisticRegression(C=1)."""
    n_comp = min(PCA_COMPONENTS, n_features)
    if n_samples is not None:
        # PCA requires n_components <= n_samples (small-N contrasts, e.g. awareness/VEA)
        n_comp = max(2, min(n_comp, n_samples - 1))
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(
            C=REGULARIZATION_C, max_iter=MAX_ITER,
            solver="lbfgs", random_state=RANDOM_STATE)),
    ])


def cv_auroc_at_layer(X_layer: np.ndarray, y: np.ndarray) -> float:
    """5-fold StratifiedKFold held-out AUROC for one layer. NaN if not splittable."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float("nan")
    n_splits = int(min(CV_FOLDS, counts.min()))
    if n_splits < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aurocs = []
    for tr, te in skf.split(X_layer, y):
        pipe = make_pipeline(X_layer.shape[1], n_samples=len(tr))
        pipe.fit(X_layer[tr], y[tr])
        proba = pipe.predict_proba(X_layer[te])[:, 1]
        aurocs.append(roc_auc_score(y[te], proba))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def fit_full_pipeline(X_layer: np.ndarray, y: np.ndarray) -> Pipeline:
    """Fit the pipeline on ALL samples at one layer (for cross-lingual scoring)."""
    pipe = make_pipeline(X_layer.shape[1], n_samples=len(X_layer))
    pipe.fit(X_layer, y)
    return pipe


def score_pipeline(pipe: Pipeline, X_layer: np.ndarray, y: np.ndarray) -> float:
    """AUROC of a fitted pipeline on a held (lang) set at one layer."""
    if len(np.unique(y)) < 2:
        return float("nan")
    proba = pipe.predict_proba(X_layer)[:, 1]
    return float(roc_auc_score(y, proba))


def diff_of_means_direction(X_layer: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normalized mean(class1) - mean(class0) in raw d_model space. [D_MODEL]."""
    mu1 = X_layer[y == 1].mean(axis=0)
    mu0 = X_layer[y == 0].mean(axis=0)
    d = mu1 - mu0
    return d / (np.linalg.norm(d) + 1e-8)


# ── Main analysis ────────────────────────────────────────────────────────────────

def per_layer_auroc(lang: str, contrast: str, limit: int = None):
    """
    Probe every layer for one (lang, contrast).

    Returns (rows, cache) where rows is a list of per-layer AUROC dicts and cache
    is {"X": X, "y": y} (kept for direction extraction), or (None, None) if skipped.
    """
    X, y = build_contrast(lang, contrast, limit=limit)
    if X is None:
        return None, None
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    print(f"    [{lang}/{contrast}] N={len(y)} (pos={n1}, neg={n0}), "
          f"layers=0..{X.shape[1]-1}")
    rows = []
    for layer in range(X.shape[1]):
        auroc = cv_auroc_at_layer(X[:, layer, :], y)
        rows.append({
            "contrast": contrast, "lang": lang, "layer": layer,
            "auroc": auroc, "n_pos": n1, "n_neg": n0,
        })
        if layer % 5 == 0 or layer == X.shape[1] - 1:
            print(f"      layer {layer:3d}: AUROC = {auroc:.3f}"
                  if not np.isnan(auroc) else f"      layer {layer:3d}: AUROC = nan")
    return rows, {"X": X, "y": y}


def best_layer(rows: list) -> int:
    """Layer with max (non-NaN) AUROC among a contrast's per-layer rows."""
    valid = [r for r in rows if not np.isnan(r["auroc"])]
    if not valid:
        return 0
    return int(max(valid, key=lambda r: r["auroc"])["layer"])


def run(langs: list, limit: int = None, resume: bool = False):
    config.PROBE_DIR.mkdir(parents=True, exist_ok=True)
    auroc_path     = config.PROBE_DIR / "auroc_all.csv"
    transfer_path  = config.PROBE_DIR / "transfer.csv"
    directions_path = config.PROBE_DIR / "directions.pkl"
    cosine_path    = config.PROBE_DIR / "cosine.csv"

    # Resume: keep already-computed (contrast, lang) AUROC rows.
    done = set()
    prior_rows = []
    if resume and auroc_path.exists():
        prev = pd.read_csv(auroc_path)
        prior_rows = prev.to_dict("records")
        done = set(zip(prev["contrast"], prev["lang"]))
        print(f"--resume: {len(done)} (contrast,lang) cells already on disk.")

    auroc_rows = list(prior_rows)
    # caches[(contrast, lang)] = {"X","y","rows","best_layer"} for langs computed THIS run
    caches = {}

    # CRITICAL: always load existing directions first so --resume never loses prior work.
    # directions is merged: prior keys are kept; new keys overwrite (re-computed this run).
    directions = {}
    if directions_path.exists():
        with open(directions_path, "rb") as f:
            existing_dirs = pickle.load(f)
        if isinstance(existing_dirs, dict):
            directions.update(existing_dirs)
            if resume:
                print(f"--resume: loaded {len(directions)} existing directions from "
                      f"{directions_path}")

    # ── 1. Per-(contrast, lang) per-layer probing ──
    for contrast in CONTRASTS:
        print(f"\n══ contrast: {contrast} ══")
        for lang in langs:
            if resume and (contrast, lang) in done:
                print(f"  [{lang}] cached (--resume) — skipping probe fit.")
                continue
            print(f"  → {lang}")
            rows, cache = per_layer_auroc(lang, contrast, limit=limit)
            if rows is None:
                continue
            auroc_rows.extend(rows)
            bl = best_layer(rows)
            caches[(contrast, lang)] = {**cache, "rows": rows, "best_layer": bl}
            # diff-of-means direction at this lang's best layer
            X, y = cache["X"], cache["y"]
            directions[(lang, contrast, bl)] = diff_of_means_direction(X[:, bl, :], y)
            bl_auroc = next(r["auroc"] for r in rows if r["layer"] == bl)
            print(f"    [{lang}/{contrast}] best layer = {bl} (AUROC={bl_auroc:.3f})")

            # incremental save so a later language crashing never loses finished work
            pd.DataFrame(auroc_rows).to_csv(auroc_path, index=False)
            with open(directions_path, "wb") as f:
                pickle.dump(directions, f)

    pd.DataFrame(auroc_rows).to_csv(auroc_path, index=False)
    print(f"\nAUROC table → {auroc_path}")
    with open(directions_path, "wb") as f:
        pickle.dump(directions, f)
    print(f"Diff-of-means directions → {directions_path}")

    # ── 2. Cross-lingual transfer: train on EN, test on each Indic lang at EN best layer ──
    transfer_rows = []
    for contrast in CONTRASTS:
        en_cache = caches.get((contrast, "en"))
        if en_cache is None:
            print(f"\n[transfer/{contrast}] EN cache unavailable this run "
                  f"(skipped or --resume) — cannot fit transfer source. Skipping.")
            continue
        en_layer = en_cache["best_layer"]
        en_X, en_y = en_cache["X"], en_cache["y"]
        en_pipe = fit_full_pipeline(en_X[:, en_layer, :], en_y)
        # EN self held-out AUROC at its best layer (the GATE numerator)
        en_heldout = next(r["auroc"] for r in en_cache["rows"] if r["layer"] == en_layer)
        print(f"\n[transfer/{contrast}] EN best layer={en_layer}, "
              f"held-out AUROC={en_heldout:.3f}")
        for lang in INDIC_LANGS:
            tgt = caches.get((contrast, lang))
            if tgt is None:
                # need target activations even under --resume; rebuild if absent
                X, y = build_contrast(lang, contrast, limit=limit)
                if X is None:
                    transfer_rows.append({
                        "contrast": contrast, "src": "en", "tgt": lang,
                        "en_layer": en_layer, "transfer_auroc": float("nan"),
                        "en_heldout_auroc": en_heldout, "note": "target unavailable",
                    })
                    continue
            else:
                X, y = tgt["X"], tgt["y"]
            auroc = score_pipeline(en_pipe, X[:, en_layer, :], y)
            transfer_rows.append({
                "contrast": contrast, "src": "en", "tgt": lang,
                "en_layer": en_layer, "transfer_auroc": auroc,
                "en_heldout_auroc": en_heldout, "note": "",
            })
            print(f"  en -> {lang}: transfer AUROC = {auroc:.3f}"
                  if not np.isnan(auroc) else f"  en -> {lang}: transfer AUROC = nan")

    if transfer_rows:
        pd.DataFrame(transfer_rows).to_csv(transfer_path, index=False)
        print(f"\nTransfer AUROC → {transfer_path}")

    # ── 3. Geometry: cosine matrix across per-lang EVAL-FRAMING directions ──
    # (the "eval direction" = the eval_framing diff-of-means per language — the anti-trivial
    #  direction that feeds steering in 14. High cross-lingual cosine = a shared direction.)
    aware_dirs = {lang: vec for (lang, contrast, _), vec in directions.items()
                  if contrast == "eval_framing"}
    if len(aware_dirs) >= 2:
        keys = [l for l in LANG_ORDER if l in aware_dirs]
        M = np.stack([aware_dirs[k] for k in keys], axis=0)   # [L, D] (already unit-norm)
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        M_unit = M / (norms + 1e-8)
        cos = M_unit @ M_unit.T
        cos_df = pd.DataFrame(cos, index=keys, columns=keys)
        cos_df.to_csv(cosine_path)
        print(f"\nAwareness-direction cosine matrix → {cosine_path}")
        print(cos_df.round(3).to_string())
    else:
        print("\n[geometry] <2 awareness directions available — cosine matrix skipped.")

    # ── 4. GATE printout ──
    print("\n══ GATES ══")
    en_rows = [r for r in auroc_rows if r["lang"] == "en"]
    for contrast in CONTRASTS:
        cr = [r for r in en_rows if r["contrast"] == contrast and not np.isnan(r["auroc"])]
        if cr:
            best = max(cr, key=lambda r: r["auroc"])
            ok = best["auroc"] > GATE_HELDOUT
            print(f"  [{contrast}] EN held-out AUROC (best layer {best['layer']}) "
                  f"= {best['auroc']:.3f}  {'PASS' if ok else 'FAIL'} (>{GATE_HELDOUT})")
        else:
            print(f"  [{contrast}] EN held-out AUROC = n/a (no EN probe this run)")
    if transfer_rows:
        tdf = pd.DataFrame(transfer_rows)
        for contrast in CONTRASTS:
            sub = tdf[(tdf["contrast"] == contrast) & tdf["transfer_auroc"].notna()]
            if len(sub):
                mean_t = sub["transfer_auroc"].mean()
                min_t  = sub["transfer_auroc"].min()
                ok = min_t > GATE_TRANSFER
                print(f"  [{contrast}] EN->Indic transfer AUROC mean={mean_t:.3f} "
                      f"min={min_t:.3f}  {'PASS' if ok else 'FAIL'} (all >{GATE_TRANSFER})")
    else:
        print("  EN->Indic transfer = n/a (no transfer computed this run)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", default="all",
                        help="Restrict probing to one language code, or 'all' (Odia last).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap samples per class per (lang,condition) for a dry run.")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse (contrast,lang) AUROC rows already in auroc_all.csv.")
    args = parser.parse_args()

    if args.lang == "all":
        langs = LANG_ORDER
    else:
        if args.lang not in config.LANG_CODES:
            parser.error(f"--lang {args.lang} not in {config.LANG_CODES}")
        # keep en in the set when probing a single Indic lang so transfer has a source
        langs = ["en", args.lang] if args.lang != "en" else ["en"]

    run(langs, limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
