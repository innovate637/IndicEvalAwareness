"""
E4a — Residual-stream activation extraction adapted to OUR stimuli (cue × harmful prompt).

This is the activation-extraction step for the causal-mechanistic experiment (E4),
re-run on the Proposal-7 stimuli: for each language we take the two core cue conditions
{deploy, eval_log} and every harmful prompt, run ONE forward pass per (cue, prompt), and
save the LAST-token residual-stream activation at ALL 40 layers.

8-BIT NOTE (interp fidelity): unlike 01_extract_activations.py (INT4) and the behavioral
runs (INT4 via behavioral_common), this script loads the model in 8-BIT
(BitsAndBytesConfig load_in_8bit=True). 8-bit keeps the residual stream much closer to
the fp16/bf16 reference than nf4, which matters for probing/steering interpretability where
we read directions off these activations. Behavioral *labels* still come from the INT4
behavioral runs (we only re-derive them, never re-generate here), so quantization mismatch
between activations (8-bit) and labels (INT4) is intentional and noted.

Pipeline per (lang, condition, prompt):
  messages = [{system: cue}, {user: prompt}]
  formatted = apply_chat_template(..., add_generation_prompt=True)   # thinking ON (default)
  ONE forward pass → last-token residual at all 40 layers → [40, 5120]

Labels (joined from the INT4 behavioral results, never regenerated here):
  refusal : behavioral_common.classify_refusal(stored_response, lang)   (1 = refusal)
  vea     : detect_vea(split_think(stored_response)[0])["vea"] from 06_cue_factorial_analysis
  Responses are matched by (lang, condition, item_id) against:
    - eval_log  →  results/behavioral/cue_battery_grid.csv  (E1; condition 'eval_log'),
                   falling back to the 'eval' rows of alignment_faking.csv (en cue) /
                   cue_factorial.csv (native cue) — these are the same eval_log cue.
    - deploy    →  the 'deploy' rows of the same sources.
  If no stored response is found for a cell, refusal=None and vea=None.

Cue availability (mirrors the shared conventions):
  - en/hi/bn : deploy + eval_log come from data/cues/cue_battery.json.
  - ta/te/or : deploy + eval_log come from data/cues/cue_translations.json
               (its 'eval' == eval_log). The 5 new cue conditions are NOT translated yet.
  Either way, deploy+eval_log are available for ALL six languages, so both core-contrast
  conditions are extractable everywhere. Any genuinely missing (lang, condition) is
  SKIPPED + logged, never crashed on.

Output (incremental, one file per (lang, condition)):
  results/activations/{lang}_{condition}.pkl = {
    'acts'     : np.ndarray [N, 40, 5120] (float32),
    'item_id'  : [str, ...]   (length N),
    'refusal'  : [int|None, ...],
    'vea'      : [int|None, ...],
    'lang'     : str,
    'condition': str,
  }

Usage:
  python scripts/12_extract_activations_v2.py                      # all langs, Odia last
  python scripts/12_extract_activations_v2.py --lang hi            # one language
  python scripts/12_extract_activations_v2.py --limit 4 --lang hi  # dry-size run
  python scripts/12_extract_activations_v2.py --resume             # skip finished cells
"""

import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from nnsight import LanguageModel
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
# Labels come from the CALIBRATED JUDGE (results/behavioral/refusal_judge.csv), NOT the keyword
# classifier — keyword refusal labels were proven invalid this session (esp. Indic: ~1–13% kw
# vs ~80–95% judge), which would near-invert the refusal pools and destroy 14's positive
# control. VEA comes from the CoT-monitor's eval_aware where available. See E4_AUDIT.md.
import pandas as pd

# All five cue conditions so the awareness probe (script 13) has within-eval VEA variation.
# en/hi/bn carry all five; ta/te/or only have deploy+eval_log in cue_translations.json.
# resolve_cue() returns None for unavailable (lang, condition) pairs → run_cell skips gracefully.
CONDITIONS = ["deploy", "neutral_log", "eval_log", "eval_explicit", "eval_metadata"]

# Run order: Odia ('or') last (slowest tokenization). English first.
LANG_ORDER = ["en", "hi", "bn", "ta", "te", "or"]

CUE_BATTERY_PATH = config.DATA_DIR / "cues" / "cue_battery.json"
CUE_TRANS_PATH = config.DATA_DIR / "cues" / "cue_translations.json"

# Behavioral CSVs that hold stored responses, by cue-language.
# 'eval_log' in our naming == 'eval' condition in these files (same logged+reviewed cue).
# 'deploy' == 'deploy'. cue_battery_grid.csv (E1) is preferred if/when present.
CUE_BATTERY_GRID = config.BEHAV_DIR / "cue_battery_grid.csv"
ALIGNMENT_FAKING = config.BEHAV_DIR / "alignment_faking.csv"   # English-cue eval/deploy
CUE_FACTORIAL = config.BEHAV_DIR / "cue_factorial.csv"         # native-cue eval/deploy

# Map our condition name → the 'condition' value used in alignment_faking / cue_factorial CSVs.
# These older files only have 'eval' (= our eval_log) and 'deploy'.
# neutral_log / eval_explicit / eval_metadata only exist in cue_battery_grid.csv (E1), which
# is queried with the raw condition name, so no mapping is needed for that path.
CONDITION_TO_BEHAV = {
    "deploy": "deploy",
    "eval_log": "eval",
    "neutral_log": "neutral_log",
    "eval_explicit": "eval_explicit",
    "eval_metadata": "eval_metadata",
}


# ── model loading (8-bit, for interp fidelity) ─────────────────────────────────

def load_model_and_tokenizer_8bit():
    """
    Mirror behavioral_common.load_model_and_tokenizer, but 8-BIT instead of INT4.
    8-bit residuals are closer to the bf16 reference → cleaner probing/steering directions.
    """
    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = LanguageModel(
        config.MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="cuda:0",
        dtype=torch.bfloat16,         # nnsight 0.7: dtype= not torch_dtype=
        dispatch=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ── cue text resolution ────────────────────────────────────────────────────────

def load_cue_sources() -> tuple[dict, dict]:
    # cue_translations.json is no longer used: as of 2026-06-11 cue_battery.json carries all six
    # languages for every condition (ta/te/or deploy+eval_log were patched in from cue_factorial).
    with open(CUE_BATTERY_PATH) as f:
        battery = json.load(f)["conditions"]
    return battery, {}


def resolve_cue(lang: str, condition: str, battery: dict, trans: dict):
    """
    System-prompt cue text for (lang, condition), or None if unavailable (cell skipped).

    BUGFIX 2026-06-11: previously routed ta/te/or to cue_translations.json, which is now EMPTY
    (those cues were moved into cue_battery.json). Read cue_battery.json for ALL six languages.
    """
    cond = battery.get(condition)
    if cond is None:
        return None
    return cond.get(lang)


# ── label join (CALIBRATED JUDGE, not keyword) ──────────────────────────────────
# Our extraction condition name -> condition value(s) used in the results CSVs.
# alignment_faking / cue_factorial store eval_log as 'eval'; cue_battery_grid stores 'eval_log'.
JUDGE_COND_ALIASES = {
    "deploy": ["deploy"],
    "neutral_log": ["neutral_log"],
    "eval_log": ["eval_log", "eval"],
    "eval_explicit": ["eval_explicit"],
    "eval_metadata": ["eval_metadata"],
}

JUDGE_CSV = config.BEHAV_DIR / "refusal_judge.csv"
COT_CSV = config.RESULTS_DIR / "cot_monitor" / "cot_monitor.csv"


def _label_table(path: Path, value_col: str, mapper) -> dict:
    """{(lang, cond, item_id) -> label} for sarvam rows from a results CSV.
    Handles both naming schemes: refusal_judge (cond/iid/model_gen) and cot_monitor
    (condition/item_id, no model_gen → assumed sarvam)."""
    if not path.exists():
        return {}
    d = pd.read_csv(path, on_bad_lines="skip")
    if "lang" not in d.columns or value_col not in d.columns:
        return {}
    out = {}
    for rd in d.to_dict("records"):
        if rd.get("model_gen", "sarvamai/sarvam-m") != "sarvamai/sarvam-m":
            continue
        cond = str(rd.get("cond", rd.get("condition", "")))
        iid = str(rd.get("iid", rd.get("item_id", "")))
        out[(rd.get("lang"), cond, iid)] = mapper(rd.get(value_col))
    return out


def load_judge_labels(lang: str) -> dict:
    """{(cond, item_id) -> refusal 1/0/None} from the calibrated judge verdict."""
    tbl = _label_table(JUDGE_CSV, "verdict",
                       lambda v: 1 if v == "REFUSE" else (0 if v == "COMPLY" else None))
    return {(c, i): r for (l, c, i), r in tbl.items() if l == lang}


def load_vea_labels(lang: str) -> dict:
    """{(cond, item_id) -> eval_aware 1/0/None} from the CoT monitor (partial coverage)."""
    tbl = _label_table(COT_CSV, "eval_aware",
                       lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v))) else int(v))
    return {(c, i): r for (l, c, i), r in tbl.items() if l == lang}


def _lookup(table: dict, condition: str, iid: str):
    """Look up a label by condition, trying the result-CSV condition aliases."""
    for c in JUDGE_COND_ALIASES.get(condition, [condition]):
        if (c, iid) in table:
            return table[(c, iid)]
    return None


# ── activation extraction ──────────────────────────────────────────────────────

def load_safety_prompts(lang: str) -> list[dict]:
    path = config.SAFETY_DIR / f"{lang}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# Cap the USER prompt length so a single forward pass can't OOM. Odia ('or') tokenizes into
# 3-5x more tokens than English (low-resource script); an untruncated Odia prompt makes one
# all-layer trace allocate >45 GB and OOM. Capping the prompt (NOT the cue) keeps the
# condition cue intact (the contrast variable) AND the last-token generation boundary intact,
# losing only the tail of pathologically long prompts. en/hi/bn/ta/te are already short, so
# this is a no-op for them and only bounds Odia.
MAX_PROMPT_TOKENS = 768


def format_chat(tokenizer, cue: str, prompt: str) -> str:
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) > MAX_PROMPT_TOKENS:
        prompt = tokenizer.decode(ids[:MAX_PROMPT_TOKENS])
    messages = [
        {"role": "system", "content": cue},
        {"role": "user", "content": prompt},
    ]
    # Thinking ON by default (do NOT pass enable_thinking=False).
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_all_layers(model, tokenizer, formatted: str) -> np.ndarray:
    """
    One forward pass → last-token residual at EVERY layer. Returns [NUM_LAYERS, D_MODEL].

    EXACT mirror of 01_extract_activations.py's nnsight 0.7 trace pattern:
      - model.model.layers[i]  (NOT model.model.model.layers)
      - .output is [batch, seq, d_model]; take [:, -1, :] and .save()
      - all saves in ONE trace (one forward pass); tensors are plain after trace exits.
    """
    layer_saves = []
    with model.trace(formatted):
        for layer_idx in range(config.NUM_LAYERS):
            saved = model.model.layers[layer_idx].output[:, -1, :].save()
            layer_saves.append(saved)

    acts = np.stack(
        [s.detach().float().cpu().numpy()[0] for s in layer_saves],
        axis=0,
    )   # [NUM_LAYERS, D_MODEL]
    return acts


def out_path(lang: str, condition: str) -> Path:
    return config.ACT_DIR / f"{lang}_{condition}.pkl"


def run_cell(model, tokenizer, lang: str, condition: str,
             battery: dict, trans: dict, limit: int = None) -> bool:
    """
    Extract activations for one (lang, condition) cell. Returns True if a file was written,
    False if the cell was skipped (unavailable cue / no prompts).
    """
    cue = resolve_cue(lang, condition, battery, trans)
    if cue is None:
        print(f"  [SKIP] ({lang}, {condition}): no cue text available — logging and skipping.")
        return False

    items = load_safety_prompts(lang)
    if not items:
        print(f"  [SKIP] ({lang}, {condition}): no safety prompts at "
              f"{config.SAFETY_DIR / (lang + '.json')} — skipping.")
        return False
    if limit:
        items = items[:limit]

    judge_map = load_judge_labels(lang)   # calibrated judge: refusal 1/0/None
    vea_map = load_vea_labels(lang)        # CoT-monitor eval_aware (partial coverage)
    if not judge_map:
        print(f"  [warn] ({lang}, {condition}): no judge labels in refusal_judge.csv "
              f"— refusal labels will be None for this cell (re-judge needed).")

    acts_list, item_ids, refusals, veas = [], [], [], []
    for item in tqdm(items, desc=f"{lang}/{condition}"):
        iid = item.get("id", "")
        prompt = item["text"]
        formatted = format_chat(tokenizer, cue, prompt)
        acts = extract_all_layers(model, tokenizer, formatted)   # [40, 5120]

        refusal = _lookup(judge_map, condition, iid)
        vea = _lookup(vea_map, condition, iid)

        acts_list.append(acts.astype(np.float32))
        item_ids.append(iid)
        refusals.append(refusal)
        veas.append(vea)

    payload = {
        "acts": np.stack(acts_list, axis=0),   # [N, 40, 5120]
        "item_id": item_ids,
        "refusal": refusals,
        "vea": veas,
        "lang": lang,
        "condition": condition,
    }
    op = out_path(lang, condition)
    with open(op, "wb") as f:
        pickle.dump(payload, f)

    n_lab = sum(r is not None for r in refusals)
    print(f"  [{lang}/{condition}] → {op}  "
          f"({payload['acts'].shape[0]} items, {n_lab} labeled, "
          f"acts {payload['acts'].shape})")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="all", help="language code or 'all' (Odia last)")
    parser.add_argument("--limit", type=int, default=None,
                        help="max prompts per cell (dry-size run)")
    parser.add_argument("--resume", action="store_true",
                        help="skip (lang, condition) cells whose .pkl already exists")
    args = parser.parse_args()

    config.ACT_DIR.mkdir(parents=True, exist_ok=True)

    langs = LANG_ORDER if args.lang == "all" else [args.lang]
    battery, trans = load_cue_sources()

    model, tokenizer = load_model_and_tokenizer_8bit()
    print(f"Model loaded (8-BIT, interp fidelity): {config.MODEL_NAME}")

    skipped = []
    for lang in langs:
        for condition in CONDITIONS:
            if args.resume and out_path(lang, condition).exists():
                print(f"  [{lang}/{condition}] .pkl exists — skipping (--resume).")
                continue
            print(f"\n→ {lang} / {condition}")
            wrote = run_cell(model, tokenizer, lang, condition, battery, trans,
                             limit=args.limit)
            if not wrote:
                skipped.append((lang, condition))

    if skipped:
        print("\nSkipped cells (unavailable cue / no prompts):")
        for lang, condition in skipped:
            print(f"  - {lang} / {condition}")
    print("\nActivation extraction (v2, 8-bit) complete.")


if __name__ == "__main__":
    main()
