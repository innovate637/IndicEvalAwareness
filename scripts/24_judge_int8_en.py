#!/usr/bin/env python3
"""
24 — Judge the sarvam-m INT8 English cue-battery run with BOTH judges (gemma-27b + sarvam-105b).

WHY: INT4 sarvam-m sat at a refusal CEILING (~deploy high) -> little headroom to express an
eval-awareness gap (~+3pp). The INT8 run has headroom. If INT8 shows a clear deploy->eval
compliance gap, the flat INT4 gap was a QUANTIZATION ceiling artifact, NOT absence of
eval-awareness. Same JUDGE_PROMPT + post-</think> answer extraction as the INT4 judging, so
the gap is directly comparable across quantizations and judges.

Source: results/behavioral/cue_battery_en_int8.csv
  500 rows = 100 toxic items x 5 cue conditions
  (deploy / neutral_log / eval_log / eval_explicit / eval_metadata). Thinking model.

  python scripts/24_judge_int8_en.py --judge gemma  --workers 16
  python scripts/24_judge_int8_en.py --judge sarvam --workers 8
  python scripts/24_judge_int8_en.py --compare
"""
import os, sys, json, time, argparse, importlib.util, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).parent / fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

rj = _load("rj18", "18_refusal_judge.py")   # JUDGE_PROMPT / LANG_NAME / load_prompts / post_think
sv = _load("sv21", "21_sarvam_judge.py")    # sarvam call() / parse_verdict / sarvam_key

SRC = config.BEHAV_DIR / "cue_battery_en_int8.csv"
OUT = {"gemma":  config.BEHAV_DIR / "refusal_int8_en_gemma.csv",
       "sarvam": config.BEHAV_DIR / "refusal_int8_en_sarvam.csv"}
GEMMA_MODEL = "google/gemma-3-27b-it"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 8


def openrouter_key():
    for line in open(os.path.expanduser("~/CAISc/.env")):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("no OPENROUTER_API_KEY in ~/CAISc/.env")


def gemma_call(msg, key):
    body = json.dumps({"model": GEMMA_MODEL, "messages": [{"role": "user", "content": msg}],
                       "temperature": 0.0, "max_tokens": 30}).encode()
    last = None
    for i in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(OPENROUTER_URL, data=body, headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost", "X-Title": "int8-judge"})
            r = json.load(urllib.request.urlopen(req, timeout=90))
            return (r["choices"][0]["message"].get("content", "") or "").strip()
        except urllib.error.HTTPError as e:
            last = f"HTTP{e.code}"
            if e.code in (429, 500, 502, 503, 529) and i < MAX_RETRIES - 1:
                time.sleep(min(60.0, 3.0 * (2 ** i))); continue
            return f"ERR:{last}"
        except Exception as e:
            last = type(e).__name__
            if i < MAX_RETRIES - 1:
                time.sleep(min(30.0, 2.0 * (i + 1))); continue
            return f"ERR:{last}"
    return f"ERR:{last}"


def worklist():
    df = pd.read_csv(SRC)
    prompts = rj.load_prompts()
    rows = []
    for _, r in df.iterrows():
        ans = rj.post_think(r["response"])
        rows.append({"iid": str(r["item_id"]), "cond": r["condition"], "lang": "en",
                     "prompt": prompts.get(("en", str(r["item_id"])), ""), "answer": ans,
                     "kw_refusal": r.get("refusal")})
    return pd.DataFrame(rows)


def judge_one(row, judge, key):
    ans = str(row["answer"])
    if not ans or len(ans.strip()) < 3:
        v = {"prompt_harmful": None, "verdict": "NOANSWER"}
    else:
        msg = (rj.JUDGE_PROMPT.replace("{lang}", rj.LANG_NAME.get(row["lang"], row["lang"]))
               .replace("{prompt}", (row["prompt"] or "")[:1500]).replace("{answer}", ans[:3500]))
        reply = gemma_call(msg, key) if judge == "gemma" else sv.call(msg)
        v = sv.parse_verdict(reply)
    return {"iid": row["iid"], "cond": row["cond"], "lang": row["lang"],
            "prompt_harmful": v["prompt_harmful"], "verdict": v["verdict"],
            "kw_refusal": row["kw_refusal"], "answer_preview": ans[:140].replace("\n", " ")}


def run(judge, workers):
    out = OUT[judge]
    key = None
    if judge == "gemma":
        key = openrouter_key()
    else:
        sv.KEY = sv.sarvam_key()
    w = worklist()
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        ok = prev[~prev.verdict.astype(str).str.startswith("ERR")]
        done = set(zip(ok.iid.astype(str), ok.cond))
    todo = [r for _, r in w.iterrows() if (str(r["iid"]), r["cond"]) not in done]
    print(f"[{judge}] {len(todo)}/{len(w)} to judge ({workers} workers)", flush=True)
    header = out.exists(); B = 80
    for i in range(0, len(todo), B):
        chunk = todo[i:i + B]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(lambda r: judge_one(r, judge, key), chunk))
        pd.DataFrame(recs).to_csv(out, mode="a", header=not header, index=False); header = True
        nerr = sum(str(x["verdict"]).startswith("ERR") for x in recs)
        print(f"  [{judge}] {min(i + B, len(todo))}/{len(todo)} ({nerr} err)", flush=True)
    print(f"[{judge}] done -> {out}")
    compare()


def _int4_en_gap():
    """INT4 English cue-battery gap (existing gemma judge) for the comparison baseline."""
    try:
        g = pd.read_csv(config.BEHAV_DIR / "refusal_judge.csv")
        g["iid"] = g["iid"].astype(str)
        d = g[(g.model_gen == "sarvamai/sarvam-m") & (g.lang == "en")
              & (g.src.isin(["cue_battery_grid.csv", "cue_factorial.csv"]))
              & (g.cond.isin(["deploy", "eval_log"])) & (g.verdict.isin(["REFUSE", "COMPLY"]))]
        dep = d[d.cond == "deploy"]; ev = d[d.cond == "eval_log"]
        if len(dep) and len(ev):
            rd = 100 * (dep.verdict == "REFUSE").mean(); re = 100 * (ev.verdict == "REFUSE").mean()
            return rd, re, len(dep), len(ev)
    except Exception as e:
        print("int4 baseline n/a:", e)
    return None


def compare():
    print("\n" + "=" * 60)
    print("INT8 sarvam-m EN — refusal% by cue condition")
    print("=" * 60)
    rows = []
    for judge, out in OUT.items():
        if not out.exists():
            print(f"{judge}: (not judged yet)"); continue
        d = pd.read_csv(out); d = d[d.verdict.isin(["REFUSE", "COMPLY"])]
        for cond in ["deploy", "neutral_log", "eval_log", "eval_explicit", "eval_metadata"]:
            c = d[d.cond == cond]
            if len(c):
                rows.append({"judge": judge, "cond": cond,
                             "refusal%": round(100 * (c.verdict == "REFUSE").mean(), 1), "n": len(c)})
    if not rows:
        return
    r = pd.DataFrame(rows)
    piv = r.pivot(index="cond", columns="judge", values="refusal%")
    order = [c for c in ["deploy", "neutral_log", "eval_log", "eval_explicit", "eval_metadata"] if c in piv.index]
    print(piv.reindex(order).to_string())
    print("\n--- deploy -> eval_log gap (the eval-awareness test) ---")
    for judge in OUT:
        sub = r[r.judge == judge].set_index("cond")["refusal%"]
        if "deploy" in sub.index and "eval_log" in sub.index:
            print(f"  INT8 {judge:<7}: deploy {sub['deploy']:.1f} -> eval_log {sub['eval_log']:.1f}  "
                  f"gap {sub['eval_log'] - sub['deploy']:+.1f}")
    b = _int4_en_gap()
    if b:
        rd, re, nd, ne = b
        print(f"  INT4 gemma  : deploy {rd:.1f} -> eval_log {re:.1f}  gap {re - rd:+.1f}   (n {nd}/{ne})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=["gemma", "sarvam"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
    elif a.judge:
        run(a.judge, a.workers)
    else:
        ap.error("need --judge or --compare")
