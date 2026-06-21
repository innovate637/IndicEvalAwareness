#!/usr/bin/env python3
"""
Generate human-review sheets for the cue translations, one Markdown file per language.

Reads the SAME two JSON files the experiment uses (so the sheet matches what runs):
  data/cues/cue_battery.json       (5 conditions; neutral_log/eval_explicit/eval_metadata)
  data/cues/cue_translations.json  (deploy + eval_log per Indic language)

Output: data/cues/review/{lang}.md  — each cue shown with English original + current
translation + a blank correction line, ordered with the highest-priority (newest) cues
flagged. Forward to a native speaker; they fill the correction lines and send back.

Usage:
  python scripts/make_cue_review.py            # bn, hi, ta (default)
  python scripts/make_cue_review.py te or      # any subset
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
BATTERY = ROOT / "data" / "cues" / "cue_battery.json"
TRANS = ROOT / "data" / "cues" / "cue_translations.json"
OUT_DIR = ROOT / "data" / "cues" / "review"

LANG_NAME = {"hi": "Hindi", "bn": "Bengali", "ta": "Tamil",
             "te": "Telugu", "or": "Odia"}

CONDITIONS = ["deploy", "neutral_log", "eval_log", "eval_explicit", "eval_metadata"]
# Conditions whose translations are NEW (added for design symmetry) → highest review priority.
NEW_CONDITIONS = {"neutral_log", "eval_explicit", "eval_metadata"}


def get_cue(lang, cond, battery, trans):
    """Same resolution order as scripts/09_cue_battery_grid.py:get_cue."""
    cb = battery.get(cond)
    if cb and cb.get(lang):
        return cb[lang]
    if cond in ("deploy", "eval_log"):
        key = "deploy" if cond == "deploy" else "eval"
        lb = trans.get(lang)
        if lb and lb.get(key):
            return lb[key]
    return None


def main():
    langs = sys.argv[1:] or ["bn", "hi", "ta"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bat = json.load(open(BATTERY))
    battery = bat["conditions"]
    trans = json.load(open(TRANS))["languages"]
    verified = set(bat["_meta"].get("verification", {}).get("author_verified", []))

    for lang in langs:
        name = LANG_NAME.get(lang, lang)
        is_verified_lang = lang in verified
        lines = []
        lines.append(f"# Cue translation review — {name} ({lang})\n")
        lines.append(
            "These are the **system-prompt cues** used in our evaluation-awareness "
            "experiment on an Indic LLM. Each is shown with the **English original**, the "
            f"**current {name} translation**, and a blank line for your correction.\n")
        lines.append(
            "Please check the translation **faithfully and naturally** conveys the English "
            "meaning. Notes:\n"
            "- **\"AI\" is intentionally kept in Latin script** (natural in Indic tech register) — that's fine.\n"
            "- The metadata tag `[session_type: ... ]` is **intentionally left in English** — leave it as-is.\n"
            "- Leave a correction line **blank if the translation is already correct**.\n")
        if is_verified_lang:
            lines.append(
                f"> Status: `deploy` and `eval_log` were previously author-verified for {name} "
                "(quick confirm). The other three (`neutral_log`, `eval_explicit`, "
                "`eval_metadata`) are **drafts — please check carefully**.\n")
        else:
            lines.append(
                f"> Status: all {name} cues are **provisional drafts pending native review**. "
                "The three marked ⭐ NEW are the newest and highest priority.\n")
        lines.append("\n---\n")

        for i, cond in enumerate(CONDITIONS, 1):
            en = (battery.get(cond) or {}).get("en", "(missing)")
            framing = (battery.get(cond) or {}).get("framing", "")
            tr = get_cue(lang, cond, battery, trans) or "(NONE — missing!)"
            new_flag = "  ⭐ NEW — please check" if cond in NEW_CONDITIONS else ""
            lines.append(f"## {i}. `{cond}` — _{framing}_{new_flag}\n")
            lines.append(f"**English:**\n> {en}\n")
            lines.append(f"**{name} (current):**\n> {tr}\n")
            lines.append("**✍️ Your correction (leave blank if correct):**\n>\n")
            lines.append("\n")

        lines.append("---\n")
        lines.append(
            "When done, send the file back (or just paste the corrected cues). "
            "Even a quick \"all correct\" is useful. Thank you! 🙏\n")

        out = OUT_DIR / f"{lang}.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"  wrote {out}")

    print(f"\nReview sheets in: {OUT_DIR}")


if __name__ == "__main__":
    main()
