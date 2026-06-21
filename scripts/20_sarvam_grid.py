#!/usr/bin/env python3
"""
20 — Sarvam-family in-family robustness grid (sarvam-30b, sarvam-105b) via Sarvam's API.

NOT a scaling law (these are MoE — 30B has 2.4B active — and 24B sarvam-m is a dense Mistral
fine-tune; architecture + active-params both vary). This is an IN-FAMILY robustness check:
does the corrected pattern (high refusal everywhere, no Indic gap, recognition transfers,
small compliance gap) hold across the Sarvam family including the from-scratch MoE models?

Reuses E3's EXACT cue resolution + recognition prompt + choice parser (imported from
11_crossmodel_openrouter.py) so the numbers are directly comparable to gemma/qwen. Only the
API call differs: Sarvam endpoint, Bearer auth, reasoning returned in a separate field
(content = the final answer; no inline </think> to strip).

Writes per-(model,lang) CSVs `crossmodel_{slug}_{lang}.csv` with the SAME schema as
crossmodel.csv, then merges ALL crossmodel_*.csv → crossmodel.csv so the rows flow through
18_refusal_judge.py + the cross-model analysis untouched.

Cost: sarvam-30b / sarvam-105b are FREE on Sarvam's first-party API (sarvam-m is deprecated
there). Greedy (temperature=0) for reproducibility.

Usage:
  python scripts/20_sarvam_grid.py --limit 2 --lang en --models sarvam-30b   # dry probe
  python scripts/20_sarvam_grid.py --resume                                   # full grid
  python scripts/20_sarvam_grid.py --merge-only                               # rebuild crossmodel.csv
"""
import os, sys, json, time, argparse, importlib.util, urllib.request, urllib.error, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# reuse E3 helpers verbatim (filename starts with a digit → importlib)
_spec = importlib.util.spec_from_file_location("cm11", str(Path(__file__).parent / "11_crossmodel_openrouter.py"))
cm = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(cm)

SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
DEFAULT_MODELS = ["sarvam-30b", "sarvam-105b"]
MAX_TOKENS = 4096            # = starter-tier hard cap. 1024 truncated reasoning-model output WITH a system cue
                             # (reasoning ate the whole budget → empty content / NOANSWER); 4096 is the max allowed.
CONCURRENCY = 8           # free tier 429'd at 16; 8 is the sustainable ceiling (+ resilient calls)
MAX_RETRIES = 8


def sarvam_key() -> str:
    for line in open(os.path.expanduser("~/CAISc/.env")):
        if line.startswith("SARVAM_API_KEY="):
            v = line.split("=", 1)[1].strip()
            if v:
                return v
    raise RuntimeError("SARVAM_API_KEY not found in ~/CAISc/.env")


KEY = None  # set in main


def call(model: str, messages: list) -> tuple:
    """Return (content, reasoning). content = final answer; reasoning = think trace."""
    body = json.dumps({"model": model, "messages": messages,
                       "max_tokens": MAX_TOKENS, "temperature": 0.0}).encode()
    last = None
    for i in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(SARVAM_URL, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            m = r["choices"][0]["message"]
            content = (m.get("content", "") or "").strip()
            reasoning = str(m.get("reasoning") or m.get("reasoning_content") or "").strip()
            return content, reasoning
        except urllib.error.HTTPError as e:
            last = f"HTTP{e.code}"
            if e.code in (429, 500, 502, 503, 529) and i < MAX_RETRIES - 1:
                time.sleep(min(60.0, 3.0 * (2 ** i))); continue   # exp backoff for rate limits
            return "", f"ERR:{last}"                               # resilient: one failure must NOT crash the grid
        except Exception as e:
            last = type(e).__name__
            if i < MAX_RETRIES - 1:
                time.sleep(min(30.0, 2.0 * (i + 1))); continue
            return "", f"ERR:{last}"
    return "", f"ERR:{last}"


def per_cell_path(model: str, lang: str) -> Path:
    return config.BEHAV_DIR / f"crossmodel_{cm.model_slug(model)}_{lang}.csv"


def run_cell(model: str, lang: str, conditions: dict, tasks: list, limit) -> pd.DataFrame:
    items = cm.load_items(lang, limit) if hasattr(cm, "load_items") else json.load(open(config.SAFETY_DIR / f"{lang}.json"))[:(limit or 10**9)]
    if not items:
        return pd.DataFrame()
    cue_by_cond = {}
    for c in cm.CONDITIONS:
        t = cm.cue_text_for(conditions, lang, c)
        if t is None:
            print(f"  [{model}/{lang}] no cue for '{c}' — skipping it.")
            continue
        cue_by_cond[c] = t
    if not cue_by_cond:
        return pd.DataFrame()

    rows = []
    if "behavioral" in tasks:
        work = [(it, c, cue) for it in items for c, cue in cue_by_cond.items()]
        def _b(t):
            it, c, cue = t
            content, reasoning = call(model, [{"role": "system", "content": cue},
                                              {"role": "user", "content": it["text"]}])
            return {"lang": lang, "condition": c, "model": model, "task": "behavioral",
                    "item_id": it.get("id", ""), "harm_category": it.get("harm_category", ""),
                    "response": content, "reasoning": reasoning,
                    "refusal": cm.classify_refusal(content, lang), "pred": "", "correct": ""}
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            rows += list(tqdm(ex.map(_b, work), total=len(work), desc=f"behav {model}/{lang}"))

    if "recognition" in tasks:
        work = [(it, c, cue) for it in items for c, cue in cue_by_cond.items()]
        def _r(t):
            it, c, cue = t
            content, reasoning = call(model, [{"role": "user",
                                               "content": cm.build_recognition_prompt(cue, it["text"])}])
            pred = cm.parse_choice(content)
            truth = "eval" if c in cm.EVAL_CONDITIONS else "deploy"
            return {"lang": lang, "condition": c, "model": model, "task": "recognition",
                    "item_id": it.get("id", ""), "harm_category": it.get("harm_category", ""),
                    "response": content, "reasoning": reasoning, "refusal": "",
                    "pred": pred, "correct": int(pred == truth) if pred != "unclear" else ""}
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            rows += list(tqdm(ex.map(_r, work), total=len(work), desc=f"recog {model}/{lang}"))
    return pd.DataFrame(rows)


def merge_all() -> pd.DataFrame:
    fs = [f for f in sorted(glob.glob(str(config.BEHAV_DIR / "crossmodel_*.csv")))
          if not f.endswith("crossmodel.csv")]
    parts = [pd.read_csv(f) for f in fs]
    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out = config.BEHAV_DIR / "crossmodel.csv"
    merged.to_csv(out, index=False)
    print(f"\n→ merged {len(merged)} rows from {len(fs)} per-cell files → {out}")
    if len(merged):
        print(merged.groupby(['model']).size().to_string())
    return merged


def main():
    global KEY
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--lang", default="all")
    ap.add_argument("--task", default="all", choices=["all", "behavioral", "recognition"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    if args.merge_only:
        merge_all(); return

    KEY = sarvam_key()
    conditions = cm.load_cue_battery()
    langs = cm.CROSSMODEL_LANGS if args.lang == "all" else [args.lang]
    tasks = ["behavioral", "recognition"] if args.task == "all" else [args.task]
    print(f"Models: {args.models}  Langs: {langs}  Tasks: {tasks}  limit={args.limit}")

    for model in args.models:
        for lang in langs:
            p = per_cell_path(model, lang)
            if args.resume and p.exists() and args.limit is None:
                print(f"  [{model}/{lang}] exists — skipping (--resume).")
                continue
            print(f"\n→ {model} / {lang}")
            df = run_cell(model, lang, conditions, tasks, args.limit)
            if df.empty:
                continue
            err = df["reasoning"].astype(str).str.startswith("ERR:").mean() if len(df) else 0.0
            if args.limit is None:
                if err > 0.10:
                    print(f"  [{model}/{lang}] {100*err:.0f}% calls failed (rate-limited) — NOT writing; --resume will retry this cell.")
                    continue
                df.to_csv(p, index=False)
                print(f"  wrote {len(df)} rows → {p.name}  ({100*err:.0f}% errors)")
            else:
                print(df[["task", "condition", "refusal", "pred"]].head(8).to_string())
            b = df[df.task == "behavioral"]
            if len(b):
                rr = pd.to_numeric(b["refusal"], errors="coerce").groupby(b["condition"]).mean().round(2)
                print(f"  keyword refusal by condition: {rr.to_dict()}")

    if args.limit is None:
        merge_all()
    print("\nDone.")


if __name__ == "__main__":
    main()
