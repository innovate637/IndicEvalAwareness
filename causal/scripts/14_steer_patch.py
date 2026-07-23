"""
E4c — Steering / activation patching with a POSITIVE CONTROL.

=============================================================================
  !!! NEEDS-GPU-VALIDATION !!!
  The steering HOOK in this script (steer_generate) uses the nnsight-0.7
  generation-intervention API and has NOT been run on the 24B model during
  authoring (no GPU, per project authoring rules). The OPERATOR must
  GPU-validate the hook before trusting ANY steered numbers. Concretely,
  validate that:
    1. The intervention actually fires at EVERY generated token (not once).
       (Sanity check: steered output should DIVERGE from the base output for
        any alpha > 0; with alpha=0 it must be byte-identical to the base.)
    2. The POSITIVE CONTROL works: steering the REFUSAL direction with a
       large alpha RAISES refusal rate. If the positive control does not
       move refusal, the hook has no causal power and the EVAL-direction
       null result is UNINTERPRETABLE — debug the hook, do not report.
    3. The sign convention is right: we ADD +alpha*direction where direction
       points from deploy->eval (or complied->refused). Adding the refusal
       direction should push TOWARD refusal.
=============================================================================

What this script does
---------------------
Builds TWO unit-norm steering directions from CACHED activations and applies
them additively to the residual stream during DEPLOY-condition generation on a
subset of harmful prompts, sweeping strength. It then re-scores refusal with the
SHARED classify_refusal (cross-experiment comparability).

  (1) EVAL direction   = the EVAL-FRAMING diff-of-means (eval_log vs neutral_log) at the
                         best probe layer in results/probes/directions.pkl (script 13).
                         CORRECTED PREDICTION (2026-06-11): steering it during DEPLOY produces
                         a SMALL POSITIVE refusal change (~the +5-9pp behavioral compliance
                         gap) -- NOT null. The result is the EFFECT-SIZE ORDERING
                         refusal-direction >> eval-framing > 0 ("represented but weakly
                         coupled"). The old "null/dissociation" prediction died with the
                         keyword-label artifact (see E4_AUDIT.md / RESULTS.md).
  (2) REFUSAL direction = diff-of-means(refused_acts - complied_acts) at its own
                         best layer. This is the POSITIVE CONTROL: steering it
                         MUST raise refusal, proving the hook has causal power.

Inputs (CACHED — produced by upstream GPU steps; this script never extracts):
  results/probes/directions.pkl   (see EXPECTED-FORMAT below; built by an
                                    upstream step from 01's activations + probe
                                    AUROC sweep). If absent, this script tells
                                    the operator exactly what to produce.

EXPECTED-FORMAT of results/probes/directions.pkl (dict):
  {
    "eval": {
        "<lang>": {                # at minimum "en" and "hi"
            "layer": int,          # best probe layer for the eval direction
            "deploy_mean": np.ndarray[D_MODEL],   # mean act, deploy class
            "eval_mean":   np.ndarray[D_MODEL],   # mean act, eval_log class
            # OR a precomputed "vector": np.ndarray[D_MODEL] (deploy->eval)
        }, ...
    },
    "refusal": {
        "<lang>": {
            "layer": int,                            # best refusal layer
            "complied_mean": np.ndarray[D_MODEL],
            "refused_mean":  np.ndarray[D_MODEL],
            # OR a precomputed "vector": np.ndarray[D_MODEL] (complied->refused)
        }, ...
    },
  }
We accept either (mean_a, mean_b) pairs (diff-of-means computed here) or a
ready "vector"; both are normalized to unit L2 norm of shape [D_MODEL]=5120.

Output:
  results/steering/steer_results.csv  with columns:
    lang, direction(eval|refusal), strength, item_id,
    refusal_base, refusal_steered, response_steered
  (Incremental per-(lang,direction) CSVs are written first so a crash never
   loses finished work; the merged CSV is rewritten after each shard.)

Modes
-----
  --selftest   CPU-only, NO model: asserts both directions load to shape [5120]
               and are finite/unit-norm. Use this during authoring/CI.
  (default)    GPU run: loads the 24B model, sweeps strengths on DEPLOY prompts.

Usage
-----
  python scripts/14_steer_patch.py --selftest
  python scripts/14_steer_patch.py --langs en hi --n 40 --strengths 0 2 4 8
  python scripts/14_steer_patch.py --langs en --n 5 --resume     # quick GPU probe

Conventions: greedy/do_sample=False everywhere (reproducible); device 'cuda:0';
DEPLOY cue from cue_battery.json (en/hi present). Odia would run LAST but the
default subset is en+hi only (the langs with cached eval-direction means).
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from behavioral_common import classify_refusal

# config / D_MODEL is the single source of truth for the direction shape.
D_MODEL = config.D_MODEL          # 5120
DIRECTIONS_PATH = config.PROBE_DIR / "directions.pkl"

# Default experiment knobs (all overridable via argv).
DEFAULT_LANGS = ["en", "hi"]      # only these have cached eval-direction means
DEFAULT_N = 40                    # harmful prompts per language
DEFAULT_STRENGTHS = [0.0, 0.1, 0.25, 0.5]   # NORM-RELATIVE alpha = fraction of per-token residual norm
                                            # (replaces the old absolute [0,2,4,8] which ignored the
                                            #  growing residual norm across layers — the bug we hit).
DIRECTION_KINDS = ["refusal", "eval"]   # POSITIVE CONTROL (refusal) first, on purpose
MAX_NEW_TOKENS = 768              # raised from 400: thinking model truncates </think> for Indic
                                  # (Odia 52-64% NOANSWER at 400). Fast langs stop at EOS early,
                                  # so the cap only lets Indic generate the headroom it needs.


# ── Direction construction (diff-of-means from cached activations) ─────────────

def _unit(vec: np.ndarray) -> np.ndarray:
    """L2-normalize to a unit vector, guarding against a zero direction."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(vec)
    if not np.isfinite(n) or n < 1e-8:
        raise ValueError(f"Degenerate direction (||v||={n}); cannot normalize.")
    return vec / n


def _direction_from_entry(entry: dict, kind: str) -> np.ndarray:
    """
    Build a unit diff-of-means direction from a directions.pkl entry.

    Accepts either:
      - a precomputed "vector" (already pointing the desired way), or
      - a pair of class means, from which we form (positive_mean - negative_mean):
          eval:    eval_mean    - deploy_mean      (deploy -> eval)
          refusal: refused_mean - complied_mean    (complied -> refused)
    Returns float32 [D_MODEL], unit norm.
    """
    if "vector" in entry and entry["vector"] is not None:
        vec = np.asarray(entry["vector"], dtype=np.float32).reshape(-1)
    elif kind == "eval":
        vec = (np.asarray(entry["eval_mean"], dtype=np.float32)
               - np.asarray(entry["deploy_mean"], dtype=np.float32))
    elif kind == "refusal":
        vec = (np.asarray(entry["refused_mean"], dtype=np.float32)
               - np.asarray(entry["complied_mean"], dtype=np.float32))
    else:
        raise ValueError(f"Unknown direction kind: {kind!r}")

    if vec.shape[0] != D_MODEL:
        raise ValueError(
            f"Direction for kind={kind!r} has shape {vec.shape}, expected [{D_MODEL}]."
        )
    return _unit(vec)


def load_directions(path: Path = DIRECTIONS_PATH) -> dict:
    """
    Load results/probes/directions.pkl and return:
      { kind: { lang: {"layer": int, "vector": np.ndarray[D_MODEL]} } }
    for kind in {"eval","refusal"}.

    Accepts TWO storage formats produced by the upstream scripts:

    FORMAT A (script 13 — tuple-key flat dict):
      { (lang, contrast, best_layer): np.ndarray[D_MODEL], ... }
      Mapping: contrast='cue_presence' → kind='eval';
               contrast='awareness'   → kind='eval' (awareness is the anti-trivial direction;
                 takes precedence over cue_presence if both are present for the same lang).
      'refusal' is absent → computed inline by build_refusal_direction() below.

    FORMAT B (old nested dict, documented in file header):
      { "eval":    { lang: {"layer":int, "eval_mean":..., "deploy_mean":...} },
        "refusal": { lang: {"layer":int, "refused_mean":..., "complied_mean":...} } }

    Raises FileNotFoundError if the cache is absent; returns {} for missing kinds
    (caller must check and build the refusal direction separately).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached directions at {path}.\n"
            f"Run scripts/13_probe_transfer.py first (requires E4a activations from 12_*)."
        )
    with open(path, "rb") as f:
        raw = pickle.load(f)

    out = {"eval": {}, "refusal": {}}

    if isinstance(raw, dict) and raw:
        first_key = next(iter(raw))
        if isinstance(first_key, tuple):
            # Format A (script 13): {(lang, contrast, layer): ndarray}
            # Priority for the steering EVAL direction: eval_framing (anti-trivial: isolates
            # safety-eval framing from monitoring) > cue_presence (cue-token-detection fallback).
            # 'awareness' (VEA) is leaky → used as the eval direction only if nothing else
            # exists, and logged. (2026-06-11 redesign; see E4_AUDIT.md.)
            priority = {"eval_framing": 3, "cue_presence": 2, "awareness": 1}
            chosen = {}  # lang -> (rank, layer, vec_unit)
            for (lang, contrast, layer), vec in raw.items():
                rank = priority.get(contrast, 0)
                if rank == 0:
                    continue
                if lang not in chosen or rank > chosen[lang][0]:
                    chosen[lang] = (rank, layer, _unit(vec))
            for lang, (rank, layer, vec_unit) in chosen.items():
                out["eval"][lang] = {"layer": layer, "vector": vec_unit}
                if rank == 1:
                    print(f"[warn] eval direction for {lang} falls back to leaky 'awareness' "
                          f"(no eval_framing/cue_presence direction found).")
        else:
            # Format B (old nested dict).
            for kind in DIRECTION_KINDS:
                if kind not in raw:
                    continue
                for lang, entry in raw[kind].items():
                    if "layer" not in entry:
                        continue
                    try:
                        out[kind][lang] = {
                            "layer": int(entry["layer"]),
                            "vector": _direction_from_entry(entry, kind),
                        }
                    except (KeyError, ValueError) as e:
                        print(f"[warn] directions['{kind}']['{lang}']: {e} — skipping.")

    return out


# ── DEPLOY cue lookup (cue_battery.json: en/hi/bn present) ──────────────────────

def load_deploy_cue(lang: str) -> str | None:
    """Return the DEPLOY system prompt for a language, or None if unavailable."""
    battery_path = config.DATA_DIR / "cues" / "cue_battery.json"
    with open(battery_path) as f:
        battery = json.load(f)
    deploy = battery["conditions"]["deploy"]
    return deploy.get(lang)   # only en/hi/bn defined; skip+log otherwise


# ── Refusal-direction builder (POSITIVE CONTROL) ───────────────────────────────

def build_refusal_direction(lang: str) -> dict | None:
    """
    Build the REFUSAL steering direction from cached activation pkl files (E4a output).
    This is the POSITIVE CONTROL for E4c: steering it MUST raise refusal rate, proving
    the nnsight hook has causal power.  If this control fails, the eval-direction null
    result is uninterpretable — stop and debug the hook.

    Reads ALL available (lang, condition) pkl files, pools refused_acts vs complied_acts,
    picks the layer with the largest mean-difference norm, returns unit diff-of-means.
    Returns None if there are too few samples in either class (<5) to be meaningful.
    """
    refused_by_layer = None
    complied_by_layer = None
    n_r = n_c = 0

    conditions = ["deploy", "neutral_log", "eval_log", "eval_explicit", "eval_metadata"]
    for cond in conditions:
        pkl_path = config.ACT_DIR / f"{lang}_{cond}.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        acts = np.asarray(data.get("acts", []), dtype=np.float32)
        if acts.ndim != 3 or acts.shape[0] == 0:
            continue
        refusals = data.get("refusal") or [None] * acts.shape[0]
        for i, ref in enumerate(refusals):
            if ref is None:
                continue
            a = acts[i]  # [NUM_LAYERS, D_MODEL]
            if refused_by_layer is None:
                refused_by_layer = np.zeros((a.shape[0], a.shape[1]), dtype=np.float64)
                complied_by_layer = np.zeros_like(refused_by_layer)
            if int(ref) == 1:
                refused_by_layer += a.astype(np.float64)
                n_r += 1
            else:
                complied_by_layer += a.astype(np.float64)
                n_c += 1

    if n_r < 5 or n_c < 5:
        print(f"[refusal dir / {lang}] too few labeled samples "
              f"(refused={n_r}, complied={n_c}) — skipping.")
        return None

    # Per-layer mean diff-of-means; pick the layer with highest norm (most separating).
    diff = (refused_by_layer / n_r) - (complied_by_layer / n_c)  # [NUM_LAYERS, D_MODEL]
    norms = np.linalg.norm(diff, axis=1)  # [NUM_LAYERS]
    best_layer = int(np.argmax(norms))
    vec = _unit(diff[best_layer].astype(np.float32))
    print(f"[refusal dir / {lang}] best_layer={best_layer}  "
          f"||diff||={norms[best_layer]:.4f}  n_refused={n_r}  n_complied={n_c}")
    return {"layer": best_layer, "vector": vec}


# ── Harmful-prompt subset ──────────────────────────────────────────────────────

def load_harmful_subset(lang: str, n: int) -> list[dict]:
    """First n harmful prompts for a language (deterministic subset)."""
    path = config.SAFETY_DIR / f"{lang}.json"
    if not path.exists():
        return []
    with open(path) as f:
        items = json.load(f)
    return items[:n]


# ── nnsight-0.7 STEERED generation hook ─────────────────────────────────────────
# !!! NEEDS-GPU-VALIDATION (see file header) !!!
#
# nnsight-0.7 generate-intervention API (verified against the installed package
# docs/examples — nnsight 0.7.0):
#   * `model.generate(prompt, max_new_tokens=N, do_sample=False)` returns an
#     InterleavingTracer usable as a context manager.
#   * `with tracer.all():` applies the enclosed intervention body at EVERY
#     generation step (it is the recursive form of `tracer.iter[:]`).
#     NOTE: the older `model.model.layers[L].all()` form is DEPRECATED in 0.7
#     ("Use tracer.all() instead"), so we use the tracer form.
#   * Decoder-layer output is a TUPLE; the residual stream is element [0].
#     We mutate it in place: `model.model.layers[L].output[0][:] += alpha*vec`.
#   * `model.generator.output.save()` returns the full generated token ids
#     (prompt + new tokens), saved out of the trace.
#   * Layer path is `model.model.layers[L]` (NOT model.model.model.layers — that
#     was bug #2 in the old 03_build_steering.py). The Envoy `model.model` IS the
#     MistralModel, whose `.layers` are the decoder blocks.
#   * Model is loaded via behavioral_common.load_model_and_tokenizer(), which
#     already fixes bug #1 (uses `dtype=` not `torch_dtype=` for nnsight 0.7).


def steer_generate(model, tokenizer, prompt_text: str, system_cue: str,
                   layer: int, vec_t, alpha: float,
                   max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """
    Generate one DEPLOY-condition response with +alpha*vec added to the residual
    stream at `layer` on EVERY generated token. alpha == 0.0 is the unsteered
    baseline (no intervention registered -> must match plain generation exactly).

    Returns the decoded NEW tokens only (post-prompt), stripped.

    !!! NEEDS-GPU-VALIDATION: this hook has not been executed on the model. !!!
    """
    import torch  # local import so --selftest stays CPU/torch-light

    # Cap the user prompt so a single forward pass can't OOM on O(seq^2) attention. Odia ('or')
    # tokenizes into 3-5x more tokens than English; an uncapped Odia prompt OOMs an 8-bit
    # forward+steering pass (same wall hit in script 12). Capping the prompt (NOT the cue) keeps
    # the framing cue and the last-token boundary intact; no-op for the already-short other langs.
    MAX_PROMPT_TOKENS = 768
    _pids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(_pids) > MAX_PROMPT_TOKENS:
        prompt_text = tokenizer.decode(_pids[:MAX_PROMPT_TOKENS])

    messages = [
        {"role": "system", "content": system_cue},
        {"role": "user", "content": prompt_text},
    ]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to("cuda:0")
    n_in = inputs["input_ids"].shape[1]

    # Generate via the underlying HF model (behavioral_common uses model._model.generate too),
    # steering with a plain PyTorch forward hook on the decoder layer — robust and validated,
    # unlike the nnsight generation-trace path (which left out_ids unassigned). For alpha != 0 the
    # hook adds +alpha*vec to the layer's OUTPUT hidden states (resid_post — the exact space the
    # diff-of-means direction was extracted from in script 12) at EVERY generated token.
    hf = model._model
    handle = None
    if alpha != 0.0:
        unit = vec_t.to("cuda:0")   # unit direction; magnitude set per-token below

        def _steer_hook(_module, _inp, output):
            # NORM-RELATIVE steering: add  alpha * ||h_t|| * unit , where ||h_t|| is the
            # per-token residual-stream norm at THIS layer. A fixed ABSOLUTE alpha is
            # mis-calibrated because residual norms grow across layers (an absolute alpha
            # that dominates an early layer is negligible at a late one); scaling by the
            # local norm makes alpha a dimensionless FRACTION of the residual, directly
            # comparable across layers and token positions. (cf. CAA/RepE raw mean-diff,
            # whose magnitude is implicitly in the layer's activation scale.)
            is_tuple = isinstance(output, tuple)
            hs = output[0] if is_tuple else output
            norm = hs.norm(dim=-1, keepdim=True)                  # [batch, seq, 1]
            hs = hs + (alpha * norm) * unit.to(hs.dtype)
            return (hs,) + tuple(output[1:]) if is_tuple else hs

        handle = hf.model.layers[layer].register_forward_hook(_steer_hook)

    try:
        with torch.no_grad():
            out_ids = hf.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()

    return tokenizer.decode(out_ids[0][n_in:], skip_special_tokens=True).strip()


# ── Per-(lang, direction) run with incremental save ─────────────────────────────

def shard_path(lang: str, kind: str) -> Path:
    return config.STEER_DIR / f"steer_{kind}_{lang}.csv"


def run_shard(model, tokenizer, lang: str, kind: str, dir_entry: dict,
              items: list[dict], system_cue: str, strengths: list[float]) -> pd.DataFrame:
    """
    Sweep strengths for one (lang, direction) on the harmful subset.
    refusal_base is computed once per item from the alpha==0 generation and reused
    across strengths (it does not depend on alpha).
    """
    import torch
    layer = dir_entry["layer"]
    vec_t = torch.tensor(dir_entry["vector"], dtype=torch.float32)

    # Ensure 0.0 baseline is present and runs first so we can cache refusal_base.
    sweep = sorted(set([0.0] + [float(s) for s in strengths]))
    sp = shard_path(lang, kind)

    # PER-GENERATION CHECKPOINT + item-level resume: every row is appended to the shard CSV
    # the instant it is produced, so a kill/crash (or the 9am force-stop) loses <=1 generation.
    # On resume, reload what's on disk and skip those (strength, item); recover baselines from
    # the saved strength==0 rows. Baseline rows ARE saved now (alpha=0 is the deploy baseline).
    done, base_ref = set(), {}
    if sp.exists():
        prev = pd.read_csv(sp)
        for _, r in prev.iterrows():
            done.add((float(r["strength"]), str(r["item_id"])))
            if float(r["strength"]) == 0.0:
                base_ref[str(r["item_id"])] = r["refusal_steered"]
    header_needed = not sp.exists()

    for strength in sweep:
        desc = f"{lang}/{kind} a={strength}"
        for item in tqdm(items, desc=desc, leave=False):
            iid = str(item.get("id", ""))
            if (strength, iid) in done:
                continue   # already on disk (resume)
            resp = steer_generate(
                model, tokenizer, item["text"], system_cue, layer, vec_t, strength,
            )
            ref = classify_refusal(resp, lang)
            if strength == 0.0:
                base_ref[iid] = ref
            row = {
                "lang": lang,
                "direction": kind,
                "strength": strength,
                "item_id": iid,
                "refusal_base": base_ref.get(iid, ""),
                "refusal_steered": ref,
                "response_steered": resp,
            }
            pd.DataFrame([row]).to_csv(sp, mode="a", header=header_needed, index=False)
            header_needed = False
            done.add((strength, iid))
    return pd.read_csv(sp) if sp.exists() else pd.DataFrame()


def merge_shards(langs: list[str], strengths: list[float], kinds=None) -> pd.DataFrame:
    """Concatenate every existing per-shard CSV into the merged steer_results.csv."""
    if kinds is None:
        kinds = DIRECTION_KINDS
    parts = []
    for kind in kinds:
        for lang in langs:
            p = shard_path(lang, kind)
            if p.exists():
                parts.append(pd.read_csv(p))
    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out = config.STEER_DIR / "steer_results.csv"
    merged.to_csv(out, index=False)
    print(f"\n→ merged {len(merged)} rows → {out}")
    return merged


# ── Self-test (CPU, no model) ───────────────────────────────────────────────────

def selftest() -> int:
    """
    CPU-only validation: load both directions for every available lang and assert
    each is a finite, unit-norm vector of shape [D_MODEL]. No model is loaded.
    Returns process exit code (0 = pass).
    """
    print(f"[selftest] expecting directions of shape [{D_MODEL}] from {DIRECTIONS_PATH}")
    if not DIRECTIONS_PATH.exists():
        print(f"[selftest] FAIL: {DIRECTIONS_PATH} does not exist yet.")
        print("[selftest] Build it upstream (see EXPECTED-FORMAT in the file header).")
        return 1

    dirs = load_directions()
    ok = True
    for kind in DIRECTION_KINDS:
        if kind not in dirs or not dirs[kind]:
            if kind == "refusal":
                # Refusal direction is built INLINE at GPU-run time (from E4a activation pkls).
                # Its absence here is expected; selftest cannot validate it without the pkls.
                print(f"[selftest] WARN: no {kind!r} directions in pkl "
                      f"(will be built from activation pkls at run time — OK).")
            else:
                print(f"[selftest] FAIL: no {kind!r} directions found.")
                ok = False
            continue
        for lang, entry in dirs[kind].items():
            v = entry["vector"]
            shape_ok = (v.shape == (D_MODEL,))
            finite_ok = bool(np.all(np.isfinite(v)))
            norm = float(np.linalg.norm(v))
            unit_ok = abs(norm - 1.0) < 1e-4
            status = "ok" if (shape_ok and finite_ok and unit_ok) else "FAIL"
            print(f"[selftest] {kind:7s} {lang:3s} layer={entry['layer']:2d} "
                  f"shape={tuple(v.shape)} ||v||={norm:.5f} -> {status}")
            assert shape_ok, f"{kind}/{lang}: shape {v.shape} != ({D_MODEL},)"
            assert finite_ok, f"{kind}/{lang}: non-finite values in direction"
            assert unit_ok, f"{kind}/{lang}: not unit norm (||v||={norm})"
    if ok:
        print("[selftest] PASS — all directions are finite unit vectors of shape "
              f"[{D_MODEL}].")
    return 0 if ok else 1


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", nargs="+", default=DEFAULT_LANGS,
                        help="languages to steer (default: en hi)")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="harmful prompts per language (default: 40)")
    parser.add_argument("--strengths", nargs="+", type=float, default=DEFAULT_STRENGTHS,
                        help="norm-relative alpha sweep, fraction of residual norm (default: 0 0.1 0.25 0.5)")
    parser.add_argument("--resume", action="store_true",
                        help="skip (lang,direction) shards already on disk")
    parser.add_argument("--xlang", action="store_true",
                        help="CROSS-LINGUAL: also inject the ENGLISH eval direction into every "
                             "(incl. Indic) language's generation — the causal transfer test.")
    parser.add_argument("--selftest", action="store_true",
                        help="CPU-only: assert direction shapes are [5120]; no model")
    args = parser.parse_args()

    config.STEER_DIR.mkdir(parents=True, exist_ok=True)

    if args.selftest:
        sys.exit(selftest())

    # ── GPU run ──
    dirs = load_directions()

    # Build the REFUSAL direction (positive control) from cached E4a activation pkls
    # for any language that is missing it from directions.pkl.
    # This is the self-contained builder — no separate script required.
    if "refusal" not in dirs:
        dirs["refusal"] = {}
    for lang in args.langs:
        if lang not in dirs["refusal"]:
            print(f"\n[refusal direction / {lang}] not found in directions.pkl — building "
                  f"from E4a activation pkls (POSITIVE CONTROL)...")
            rd = build_refusal_direction(lang)
            if rd is not None:
                dirs["refusal"][lang] = rd
            else:
                print(f"  [warn] could not build refusal direction for {lang} — "
                      f"positive control will be skipped for this language.")

    # Lazy import so --selftest never touches the model stack.
    from behavioral_common import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(int8=True)   # match 8-bit direction extraction (12)
    print(f"Model loaded (8-bit): {config.MODEL_NAME}")
    print("!!! Steering hook is NEEDS-GPU-VALIDATION — verify the POSITIVE "
          "CONTROL (refusal direction raises refusal) before trusting results.")

    # Cross-lingual transfer: inject the ENGLISH eval direction into every language.
    run_kinds = list(DIRECTION_KINDS)   # ["refusal", "eval"]
    if args.xlang:
        en_eval = dirs.get("eval", {}).get("en")
        if en_eval is not None:
            dirs["xeval"] = {l: en_eval for l in args.langs if l != "en"}
            run_kinds.append("xeval")
            print(f"\n[xlang] cross-lingual eval = ENGLISH eval direction "
                  f"(layer={en_eval['layer']}) injected into: "
                  f"{[l for l in args.langs if l != 'en']}")
        else:
            print("\n[xlang] no English eval direction in directions.pkl — "
                  "cannot run the cross-lingual test.")

    # POSITIVE CONTROL (refusal) runs before the EVAL directions, by design.
    for kind in run_kinds:
        for lang in args.langs:
            # Skip + log unavailable combinations rather than crashing.
            if lang not in dirs.get(kind, {}):
                print(f"  [skip] no cached {kind} direction for lang={lang}.")
                continue
            cue = load_deploy_cue(lang)
            if cue is None:
                print(f"  [skip] no DEPLOY cue for lang={lang} in cue_battery.json.")
                continue
            items = load_harmful_subset(lang, args.n)
            if not items:
                print(f"  [skip] no harmful prompts for lang={lang}.")
                continue

            sp = shard_path(lang, kind)
            expected = len(sorted(set([0.0] + [float(s) for s in args.strengths]))) * len(items)
            if args.resume and sp.exists() and len(pd.read_csv(sp)) >= expected:
                print(f"  [resume] {sp.name} complete ({expected} rows) — skipping.")
                continue   # partial shards fall through; run_shard resumes them per-item

            print(f"\n→ {kind} direction | {lang} | layer={dirs[kind][lang]['layer']} "
                  f"| n={len(items)} | strengths={args.strengths}")
            df = run_shard(model, tokenizer, lang, kind, dirs[kind][lang],
                           items, cue, args.strengths)
            df.to_csv(sp, index=False)   # incremental: this shard is now safe
            print(f"  [{lang}/{kind}] → {sp}  ({len(df)} rows)")
            if not df.empty:
                rr = df.groupby("strength")["refusal_steered"].mean()
                base = df["refusal_base"].astype(float).mean()
                print(f"  [{lang}/{kind}] refusal_base≈{base:.2f} | steered by alpha:")
                for a, r in rr.items():
                    print(f"      alpha={a:>4}: refusal_steered={r:.2f}")
            merge_shards(args.langs, args.strengths, run_kinds)   # keep merged fresh

    merge_shards(args.langs, args.strengths, run_kinds)
    print("\nDone. REMINDER: the EVAL-direction effect is only interpretable if the "
          "REFUSAL positive control moved refusal. Check that first.")


if __name__ == "__main__":
    main()
