#!/usr/bin/env python3
"""
28 — Judge the INT8 cue-battery generations for ALL languages with the sarvam-105b LLM judge
     (Indic-reliable), to get the CROSS-LINGUAL quantization-ceiling result: does INT8 reveal the
     deploy→eval compliance gap in Indic the way it did in English?

Reads cue_battery_{lang}_int8.csv (lang in en/hi/bn/ta/te/or), judges the post-</think> answer with
sarvam-105b (same JUDGE_PROMPT as 18/21). Parallel, checkpoint/resume. Then prints deploy-vs-eval_log
refusal rate per language.

  python scripts/28_judge_int8_all.py --workers 12 --run
  python scripts/28_judge_int8_all.py --compare
"""
import os, sys, json, time, argparse, importlib.util, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import config


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

rj = _load("rj18", "18_refusal_judge.py")   # JUDGE_PROMPT / LANG_NAME / post_think / load_prompts
sv = _load("sv21", "21_sarvam_judge.py")    # sarvam-105b call() / parse_verdict / sarvam_key

LANGS = ["en", "hi", "bn", "ta", "te", "or"]
OUT = config.BEHAV_DIR / "refusal_int8_all_sarvam.csv"
GEMMA_MODEL = "google/gemma-3-27b-it"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def openrouter_key():
    for line in open(os.path.expanduser("~/CAISc/.env")):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no OPENROUTER_API_KEY")


def gemma_call(msg, key):
    body = json.dumps({"model": GEMMA_MODEL, "messages": [{"role": "user", "content": msg}],
                       "temperature": 0.0, "max_tokens": 30}).encode()
    last = None
    for i in range(8):
        try:
            req = urllib.request.Request(OPENROUTER_URL, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost", "X-Title": "int8-xlang-judge"})
            r = json.load(urllib.request.urlopen(req, timeout=90))
            return (r["choices"][0]["message"].get("content", "") or "").strip()
        except urllib.error.HTTPError as e:
            last = f"HTTP{e.code}"
            if e.code in (429, 500, 502, 503, 529) and i < 7:
                time.sleep(min(60.0, 3.0 * (2 ** i))); continue
            return f"ERR:{last}"
        except Exception as e:
            last = type(e).__name__
            if i < 7:
                time.sleep(min(30.0, 2.0 * (i + 1))); continue
            return f"ERR:{last}"
    return f"ERR:{last}"


def worklist():
    prompts = rj.load_prompts()
    rows = []
    for lang in LANGS:
        p = config.BEHAV_DIR / f"cue_battery_{lang}_int8.csv"
        if not p.exists():
            print(f"  [skip] {lang}: {p.name} missing"); continue
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            iid = str(r["item_id"])
            rows.append({"lang": lang, "cond": r["condition"], "iid": iid,
                         "prompt": prompts.get((lang, iid), ""),
                         "answer": rj.post_think(r["response"])})
    return pd.DataFrame(rows)


def judge_one(r):
    ans = str(r["answer"])
    if not ans or len(ans.strip()) < 3:
        v = "NOANSWER"
    else:
        msg = (rj.JUDGE_PROMPT.replace("{lang}", rj.LANG_NAME.get(r["lang"], r["lang"]))
               .replace("{prompt}", (r["prompt"] or "")[:1500]).replace("{answer}", ans[:3500]))
        v = sv.parse_verdict(sv.call(msg))["verdict"]
    return {"lang": r["lang"], "cond": r["cond"], "iid": r["iid"], "verdict": v}


def run(workers):
    sv.KEY = sv.sarvam_key()
    w = worklist()
    done = set()
    if OUT.exists():
        prev = pd.read_csv(OUT); ok = prev[~prev.verdict.astype(str).str.startswith("ERR")]
        done = set(zip(ok.lang, ok.cond, ok.iid.astype(str)))
    todo = [r for _, r in w.iterrows() if (r["lang"], r["cond"], str(r["iid"])) not in done]
    print(f"{len(todo)}/{len(w)} to judge with sarvam-105b ({workers} workers)", flush=True)
    header = OUT.exists(); B = 96
    for i in range(0, len(todo), B):
        chunk = todo[i:i + B]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(judge_one, chunk))
        pd.DataFrame(recs).to_csv(OUT, mode="a", header=not header, index=False); header = True
        nerr = sum(str(x["verdict"]).startswith("ERR") for x in recs)
        print(f"  {min(i + B, len(todo))}/{len(todo)}  ({nerr} err)", flush=True)
    print(f"done -> {OUT}")
    compare()


def compare():
    if not OUT.exists():
        print("no verdicts yet"); return
    d = pd.read_csv(OUT)
    d = d[d.verdict.isin(["REFUSE", "COMPLY"])]
    print("\n=== INT8 deploy vs eval_log refusal % (sarvam-105b judge), per language ===")
    print(f"{'lang':<6}{'deploy':>8}{'eval_log':>10}{'gap':>7}{'  (n dep/eval)':>16}")
    for lang in LANGS:
        sub = d[d.lang == lang]
        dep = sub[sub.cond == "deploy"]; ev = sub[sub.cond == "eval_log"]
        if len(dep) and len(ev):
            rd = 100 * (dep.verdict == "REFUSE").mean(); re = 100 * (ev.verdict == "REFUSE").mean()
            print(f"{lang:<6}{rd:>7.1f}{re:>10.1f}{re - rd:>+7.1f}    {len(dep)}/{len(ev)}")
    print("\n=== full dose-response (refusal %, all conditions) ===")
    piv = (d.assign(r=(d.verdict == "REFUSE").astype(int))
           .pivot_table(index="lang", columns="cond", values="r", aggfunc="mean") * 100)
    order = [c for c in ["deploy", "neutral_log", "eval_log", "eval_metadata", "eval_explicit"] if c in piv.columns]
    print(piv[order].reindex(LANGS).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.compare:
        compare()
    elif a.run:
        run(a.workers)
    else:
        ap.error("need --run or --compare")
