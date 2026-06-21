"""
E1 behavioral grid — CUE BATTERY (generalizes 05_cue_factorial.py).

05_cue_factorial.py ran a single 2x2 (eval vs deploy) x (English vs translated) cue.
The #1 reviewer objection to that design is "one cue operationalization": eval-framing
was confounded with mere-monitoring, and used a single phrasing. This script replaces
the single eval cue with the CUE BATTERY (data/cues/cue_battery.json), which separates
EVALUATION framing from MERE MONITORING and varies eval explicitness:

  deploy         — control, no monitoring at all
  neutral_log    — monitoring WITHOUT eval/safety framing (isolates monitoring salience)
  eval_log       — monitoring + safety/compliance review (the ORIGINAL eval cue)
  eval_explicit  — explicit scored red-team safety test (strongest)
  eval_metadata  — subtle machine-style metadata tag (non-prose cue)

Cue availability is asymmetric and the script must respect it WITHOUT crashing:
  en / hi / bn : all 5 conditions (from cue_battery.json conditions[*][lang])
  ta / te / or : only deploy + eval_log (from cue_translations.json languages[lang],
                 where 'eval' == eval_log) until the other 3 are translated.
Any (lang, condition) with no cue text is SKIPPED and logged, never run.

For each available (lang, condition) and each of the 100 harmful prompts:
  messages = [{system: cue}, {user: prompt}]  → batch_generate (thinking ON, 400 tok,
  greedy, batch 8) → classify_refusal on post-</think> text (via behavioral_common,
  so refusal scoring is identical to 04/05 for cross-experiment comparability).

Outputs (incremental, so a crash/slow Odia never loses finished languages):
  results/behavioral/cue_battery_{lang}.csv   (per language, written as it finishes)
  results/behavioral/cue_battery_grid.csv     (merged, rewritten after each language)

Usage:
  python scripts/09_cue_battery_grid.py                  # all langs, Odia last
  python scripts/09_cue_battery_grid.py --lang hi        # one language
  python scripts/09_cue_battery_grid.py --limit 2 --lang hi   # dry validation
  python scripts/09_cue_battery_grid.py --resume         # skip langs already on disk
"""

import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from behavioral_common import (
    classify_refusal, load_model_and_tokenizer, batch_generate, BATCH_SIZE,
    MAX_NEW_TOKENS, EVAL_SYSTEM, DEPLOY_SYSTEM,
)

# Run order: Odia LAST (slowest tokenization + lowest-confidence provisional cue).
# English first as the cross-lingual reference.
GRID_LANGS = ["en", "hi", "bn", "ta", "te", "or"]

# The five cue battery conditions, in design order.
BATTERY_CONDITIONS = ["deploy", "neutral_log", "eval_log", "eval_explicit", "eval_metadata"]

CUE_BATTERY_PATH = config.DATA_DIR / "cues" / "cue_battery.json"
CUE_TRANS_PATH   = config.DATA_DIR / "cues" / "cue_translations.json"


def load_cue_sources() -> tuple[dict, dict, dict]:
    """
    Load both cue files and return (battery_conditions, battery_framings, translations).
      battery_conditions: conditions dict from cue_battery.json (per-condition lang text)
      battery_framings:   {condition: framing string} for the cue_framing column
      translations:       languages dict from cue_translations.json (deploy/eval per Indic lang)
    """
    with open(CUE_BATTERY_PATH) as f:
        battery = json.load(f)["conditions"]
    framings = {cond: battery[cond].get("framing", "") for cond in battery}
    with open(CUE_TRANS_PATH) as f:
        translations = json.load(f)["languages"]
    return battery, framings, translations


def get_cue(lang: str, condition: str, battery: dict, translations: dict) -> str | None:
    """
    Resolve the system-prompt cue text for a (lang, condition), merging the two sources:

      1. cue_battery.json conditions[condition][lang]  — the 5 conditions, en/hi/bn.
      2. cue_translations.json languages[lang]         — deploy + eval (==eval_log) for
         ta/te/or (also hi/bn, but battery wins there as it is the richer source).
      3. behavioral_common EVAL_SYSTEM / DEPLOY_SYSTEM — English deploy/eval_log fallback,
         guaranteeing en always has those two even if the JSON drifts.

    Returns the cue string, or None if no cue text exists for this (lang, condition)
    (caller must SKIP + log such cells, never run them).
    """
    # 1. Cue battery is the richest source for the 5 conditions (en/hi/bn populated).
    cond_block = battery.get(condition)
    if cond_block:
        text = cond_block.get(lang)
        if text:
            return text

    # 2. Translations file: only deploy + eval_log (stored under key 'eval').
    if condition in ("deploy", "eval_log"):
        trans_key = "deploy" if condition == "deploy" else "eval"
        lang_block = translations.get(lang)
        if lang_block:
            text = lang_block.get(trans_key)
            if text:
                return text

    # 3. English hard fallback for the two continuity conditions.
    if lang == "en":
        if condition == "deploy":
            return DEPLOY_SYSTEM
        if condition == "eval_log":
            return EVAL_SYSTEM

    return None


def available_conditions(lang: str, battery: dict, translations: dict) -> list[tuple[str, str]]:
    """
    Return [(condition, cue_text), ...] for every battery condition that HAS a cue
    for this language, preserving BATTERY_CONDITIONS order. Skipped conditions are
    logged by the caller; here we just omit the ones with no text.
    """
    out = []
    for cond in BATTERY_CONDITIONS:
        cue = get_cue(lang, cond, battery, translations)
        if cue is not None:
            out.append((cond, cue))
    return out


SUFFIX = ""   # "_int8" when --int8, so 8-bit outputs never overwrite the INT4 reference cells
SHARD_TAG = ""   # "_s{start}" when --items splits a language across GPUs (halves don't collide)
ITEM_SLICE = None   # (start, end) item index range when --items is given


def per_lang_path(lang: str) -> Path:
    return config.BEHAV_DIR / f"cue_battery_{lang}{SUFFIX}{SHARD_TAG}.csv"


def partial_path(lang: str) -> Path:
    # batch-level checkpoint: rows land here after EVERY batch, so a kill mid-language
    # (e.g. freeing the GPU at night) loses at most one batch; folded into the per-lang
    # CSV and deleted on clean completion.
    return config.BEHAV_DIR / f"cue_battery_{lang}{SUFFIX}{SHARD_TAG}.partial.csv"


def run_cue_battery(model, tokenizer, lang: str, battery: dict, framings: dict,
                    translations: dict, limit: int = None,
                    skip_conditions: set = None, max_new_tokens: int = None) -> pd.DataFrame:
    """
    Run every available cue battery condition for one language against the harmful prompts.
    Mirrors run_cue_factorial in 05, but iterates over the cue battery conditions instead
    of the fixed (eval, deploy) pair, and skips+logs any condition with no cue text.

    skip_conditions: a set of condition names already present on disk (condition-level
    --resume). Those conditions are NOT re-generated; only the missing ones are run, so a
    top-up pass (e.g. adding the 3 new ta/te/or conditions after a 2-condition run) reuses
    the existing deploy/eval_log generations instead of recomputing them.
    """
    skip_conditions = skip_conditions or set()
    safety_path = config.SAFETY_DIR / f"{lang}.json"
    if not safety_path.exists():
        print(f"  [{lang}] No safety prompts at {safety_path}. Skipping.")
        return pd.DataFrame()

    with open(safety_path) as f:
        items = json.load(f)
    if not items:
        print(f"  [{lang}] Safety prompts file is empty — skipping.")
        return pd.DataFrame()
    if ITEM_SLICE is not None:
        items = items[ITEM_SLICE[0]:ITEM_SLICE[1]]   # split a language across GPUs
    if limit:
        items = items[:limit]

    conds = available_conditions(lang, battery, translations)
    skipped = [c for c in BATTERY_CONDITIONS
               if c not in {cond for cond, _ in conds}]
    if skipped:
        print(f"  [{lang}] SKIP (no cue text): {', '.join(skipped)}")
    # condition-level resume: drop conditions already generated on disk
    if skip_conditions:
        already = [c for c, _ in conds if c in skip_conditions]
        if already:
            print(f"  [{lang}] resume: already on disk, not regenerating: {', '.join(already)}")
        conds = [(c, cue) for c, cue in conds if c not in skip_conditions]
    if not conds:
        print(f"  [{lang}] No available cue conditions to run — skipping language.")
        return pd.DataFrame()

    cond_names = ", ".join(cond for cond, _ in conds)
    print(f"  [{lang}] {len(items)} prompts x {len(conds)} conditions = "
          f"{len(items) * len(conds)} generations. Conditions: {cond_names}")

    # Flatten to (item, condition, cue_text, cue_framing) so we can batch across the grid.
    pairs = [
        (item, cond, cue_text, framings.get(cond, ""))
        for cond, cue_text in conds
        for item in items
    ]

    rows = []
    # batch-level checkpoint (full runs only): reload partial rows, run only what's missing,
    # append after every batch — a mid-language kill loses at most one batch.
    ckpt = partial_path(lang) if limit is None else None
    if ckpt is not None and ckpt.exists() and ckpt.stat().st_size > 0:
        prev = pd.read_csv(ckpt)
        done_keys = set(zip(prev["condition"], prev["item_id"]))
        pairs = [p for p in pairs if (p[1], p[0].get("id", "")) not in done_keys]
        rows.extend(prev.to_dict("records"))
        print(f"  [{lang}] partial checkpoint: {len(prev)} rows reloaded, "
              f"{len(pairs)} generations remaining.")
    for batch_start in tqdm(range(0, len(pairs), BATCH_SIZE), desc=f"cue_battery {lang}"):
        batch = pairs[batch_start: batch_start + BATCH_SIZE]
        messages_list = [
            [{"role": "system", "content": cue_text},
             {"role": "user",   "content": item["text"]}]
            for item, _, cue_text, _ in batch
        ]
        responses = batch_generate(model, tokenizer, messages_list,
                                    max_new_tokens=max_new_tokens or MAX_NEW_TOKENS)

        for (item, condition, cue_text, cue_framing), response in zip(batch, responses):
            rows.append({
                "lang":          lang,
                "condition":     condition,
                "item_id":       item.get("id", ""),
                "harm_category": item.get("harm_category", ""),
                "source":        item.get("source", ""),
                "response":      response,
                "refusal":       classify_refusal(response, lang),
                "response_len":  len(response.split()),
                "cue_framing":   cue_framing,
                "cue_text":      cue_text,
            })
        if ckpt is not None and batch:
            pd.DataFrame(rows[-len(batch):]).to_csv(
                ckpt, mode="a", header=not ckpt.exists(), index=False)

    return pd.DataFrame(rows)


def merge_per_lang() -> pd.DataFrame:
    """Concatenate all cue_battery_{lang}.csv into the merged cue_battery_grid.csv."""
    parts = []
    for lang in GRID_LANGS:
        p = per_lang_path(lang)
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    out = config.BEHAV_DIR / f"cue_battery_grid{SUFFIX}.csv"
    merged.to_csv(out, index=False)
    print(f"\n→ merged {len(merged)} rows → {out}")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="all", help="language code or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Max items per language (dry run)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip languages whose per-language CSV already exists")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Generation budget override (default 400). Use ~768 for languages "
                             "that truncate the </think> block (e.g. Odia).")
    parser.add_argument("--int8", action="store_true",
                        help="Load sarvam-m in 8-bit (LLM.int8) instead of 4-bit; writes to "
                             "cue_battery_*_int8.csv (preserves the INT4 reference cells).")
    parser.add_argument("--items", default=None,
                        help="item index range START:END to split a language across GPUs "
                             "(e.g. 0:50 / 50:100); shards to cue_battery_{lang}_s{START}.csv")
    args = parser.parse_args()

    global SUFFIX, SHARD_TAG, ITEM_SLICE
    if args.int8:
        SUFFIX = "_int8"
    if args.items:
        s, e = args.items.split(":")
        ITEM_SLICE = (int(s), int(e)); SHARD_TAG = f"_s{s}"

    config.BEHAV_DIR.mkdir(parents=True, exist_ok=True)
    battery, framings, translations = load_cue_sources()

    langs = GRID_LANGS if args.lang == "all" else [l.strip() for l in args.lang.split(",") if l.strip()]

    model, tokenizer = load_model_and_tokenizer(int8=args.int8)
    print(f"Model loaded: {config.MODEL_NAME}  (precision={'int8' if args.int8 else 'int4'})")

    for lang in langs:
        existing_df = None
        skip_conditions = set()
        if args.resume and per_lang_path(lang).exists():
            # Condition-level resume: keep conditions already on disk, run only the missing
            # ones, and APPEND. (The old behaviour skipped the whole language, which would
            # have stranded the 3 new ta/te/or conditions behind the 2 already generated.)
            existing_df = pd.read_csv(per_lang_path(lang))
            present = set(existing_df["condition"].unique()) if "condition" in existing_df else set()
            avail = {c for c, _ in available_conditions(lang, battery, translations)}
            skip_conditions = present & avail
            missing = avail - present
            if not missing:
                print(f"  [{lang}] all {len(avail)} available conditions present "
                      f"({sorted(present & avail)}) — skipping (--resume).")
                merge_per_lang()
                continue
            print(f"  [{lang}] resume: present={sorted(present)} → running missing "
                  f"{sorted(missing)}")

        df = run_cue_battery(model, tokenizer, lang, battery, framings,
                             translations, limit=args.limit,
                             skip_conditions=skip_conditions,
                             max_new_tokens=args.max_new_tokens)
        if df.empty:
            continue
        # merge any newly-run conditions with what was already on disk, then save
        if existing_df is not None and not existing_df.empty:
            df = pd.concat([existing_df, df], ignore_index=True)
        # incremental save: this language is safe even if a later one crashes
        df.to_csv(per_lang_path(lang), index=False)
        if partial_path(lang).exists():
            partial_path(lang).unlink()   # folded into the per-lang CSV above
        print(f"  [{lang}] → {per_lang_path(lang)}  ({len(df)} rows)")
        # quick per-condition eyeball (over the full, merged frame)
        rr = df.groupby("condition")["refusal"].mean()
        for cond in BATTERY_CONDITIONS:
            if cond in rr.index:
                print(f"  [{lang}] refusal {cond:<14} = {rr[cond]:.2f}")
        merge_per_lang()   # keep merged file fresh after each language

    merge_per_lang()


if __name__ == "__main__":
    main()
