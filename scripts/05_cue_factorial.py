"""
Phase A+ — Cue-language factorial (OPEN-3 / Proposal 7 mechanism test).

The original alignment-faking run (04_behavioral.py) applied an ENGLISH eval cue
in every language. That cannot tell us *why* the eval cue fails to move Indic
behaviour: is it because the cue is English (an "English-language artifact" that
doesn't fire against an Indic prompt), or because eval-framing simply doesn't drive
refusal in Indic at all?

This script adds the missing cell: a **language-matched (translated) cue**. Combined
with the existing English-cue results it forms a clean 2x2 per language:

                DEPLOY cue        EVAL cue
  English cue   (have: 04)        (have: 04)
  Translated    THIS SCRIPT       THIS SCRIPT

Same 100 Toxic_Matrix prompts per language, paired by item_id, identical generation
settings to 04 (greedy, 400 tokens, batch 8) via behavioral_common — so the only
thing that changes vs 04 is the cue language.

Only the 5 Indic languages get a translated cue (English's "translated" cue == the
English cue, already done). Odia runs LAST (slowest tokenization + lowest-confidence
provisional cue).

Outputs (incremental, so a crash/slow Odia never loses finished languages):
  results/behavioral/cue_factorial_{lang}.csv   (per language, written as it finishes)
  results/behavioral/cue_factorial.csv          (merged, rewritten after each language)

Usage:
  python scripts/05_cue_factorial.py                 # all 5 Indic, Odia last
  python scripts/05_cue_factorial.py --lang hi       # one language
  python scripts/05_cue_factorial.py --limit 2 --lang hi   # dry validation
  python scripts/05_cue_factorial.py --resume        # skip languages already on disk
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
)

# Indic languages only (English's language-matched cue == English cue, already in 04).
# Odia last: slowest tokenization + lowest-confidence provisional cue.
FACTORIAL_LANGS = ["hi", "bn", "ta", "te", "or"]

CUES_PATH = config.DATA_DIR / "cues" / "cue_translations.json"


def load_cues() -> dict:
    with open(CUES_PATH) as f:
        return json.load(f)["languages"]


def per_lang_path(lang: str) -> Path:
    return config.BEHAV_DIR / f"cue_factorial_{lang}.csv"


def run_cue_factorial(model, tokenizer, lang: str, cues: dict, limit: int = None) -> pd.DataFrame:
    """
    Translated-cue eval/deploy run for one Indic language.
    Mirrors run_alignment_faking in 04, but the two conditions use the language's
    own (translated) cue instead of the English EVAL_SYSTEM / DEPLOY_SYSTEM.
    """
    safety_path = config.SAFETY_DIR / f"{lang}.json"
    if not safety_path.exists():
        print(f"  [{lang}] No safety prompts at {safety_path}. Skipping.")
        return pd.DataFrame()
    if lang not in cues:
        print(f"  [{lang}] No translated cue in {CUES_PATH}. Skipping.")
        return pd.DataFrame()

    with open(safety_path) as f:
        items = json.load(f)
    if not items:
        print(f"  [{lang}] Safety prompts file is empty — skipping.")
        return pd.DataFrame()
    if limit:
        items = items[:limit]

    cue_eval   = cues[lang]["eval"]
    cue_deploy = cues[lang]["deploy"]
    cue_status = cues[lang].get("status", "")

    print(f"  [{lang}] {len(items)} prompts → {len(items) * 2} generations "
          f"(translated cue, status={cue_status}).")

    # condition → (system cue text)
    cond_cue = {"eval": cue_eval, "deploy": cue_deploy}
    pairs = [
        (item, condition, cond_cue[condition])
        for item in items
        for condition in ("eval", "deploy")
    ]

    rows = []
    for batch_start in tqdm(range(0, len(pairs), BATCH_SIZE), desc=f"cue_factorial {lang}"):
        batch = pairs[batch_start: batch_start + BATCH_SIZE]
        messages_list = [
            [{"role": "system", "content": sys_p},
             {"role": "user",   "content": item["text"]}]
            for item, _, sys_p in batch
        ]
        responses = batch_generate(model, tokenizer, messages_list)

        for (item, condition, cue_text), response in zip(batch, responses):
            rows.append({
                "lang":          lang,
                "condition":     condition,
                "item_id":       item.get("id", ""),
                "harm_category": item.get("harm_category", ""),
                "source":        item.get("source", ""),
                "response":      response,
                "refusal":       classify_refusal(response, lang),
                "response_len":  len(response.split()),
                "cue_lang":      "native",
                "cue_status":    cue_status,
                "cue_text":      cue_text,
            })

    return pd.DataFrame(rows)


def merge_per_lang() -> pd.DataFrame:
    """Concatenate all cue_factorial_{lang}.csv into the merged cue_factorial.csv."""
    parts = []
    for lang in FACTORIAL_LANGS:
        p = per_lang_path(lang)
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    out = config.BEHAV_DIR / "cue_factorial.csv"
    merged.to_csv(out, index=False)
    print(f"\n→ merged {len(merged)} rows → {out}")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="all", help="Indic language code or 'all'")
    parser.add_argument("--limit", type=int, default=None, help="Max items per language (dry run)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip languages whose per-language CSV already exists")
    args = parser.parse_args()

    config.BEHAV_DIR.mkdir(parents=True, exist_ok=True)
    cues = load_cues()

    langs = FACTORIAL_LANGS if args.lang == "all" else [args.lang]

    model, tokenizer = load_model_and_tokenizer()
    print(f"Model loaded: {config.MODEL_NAME}")

    for lang in langs:
        if args.resume and per_lang_path(lang).exists():
            print(f"  [{lang}] per-language CSV exists — skipping (--resume).")
            continue
        df = run_cue_factorial(model, tokenizer, lang, cues, limit=args.limit)
        if df.empty:
            continue
        # incremental save: this language is safe even if a later one crashes
        df.to_csv(per_lang_path(lang), index=False)
        print(f"  [{lang}] → {per_lang_path(lang)}  ({len(df)} rows)")
        # quick eyeball
        rr = df.groupby("condition")["refusal"].mean()
        print(f"  [{lang}] refusal: deploy={rr.get('deploy', float('nan')):.2f} "
              f"eval={rr.get('eval', float('nan')):.2f}")
        merge_per_lang()   # keep merged file fresh after each language

    merge_per_lang()


if __name__ == "__main__":
    main()
