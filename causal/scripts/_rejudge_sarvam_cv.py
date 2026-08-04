#!/usr/bin/env python3
"""Re-judge the Hermes-70B CV responses with the sarvam-105b judge, using ALL THREE sarvam keys
with round-robin + failover (the original _hermes_cv.py used only SARVAM_API_KEY, so sarvam went
NaN once that one key hit its limit). CPU-only (API calls); resume-safe; writes a NEW file and
leaves results/causal/hermes_cv_items.csv untouched.

  python _rejudge_sarvam_cv.py --smoke        # judge 3 items per key, verify keys are alive
  python _rejudge_sarvam_cv.py --workers 6    # full re-judge, incremental + resume
"""
import sys, os, json, time, argparse, importlib.util, threading, urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, numpy as np

ROOT = Path("$PROJECT_ROOT/llama_causal_experiments")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import config

def _load(name, fn):
    s = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / fn))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
rj = _load("rj", "18_refusal_judge.py")       # JUDGE_PROMPT, LANG_NAME, post_think
sv = _load("sv", "21_sarvam_judge.py")        # parse_verdict, SARVAM_URL, JUDGE_MODEL
s14 = _load("s14", "14_steer_patch.py")       # load_harmful_subset

LANG = "en"
IN_CSV = config.RESULTS_DIR / "causal" / "hermes_cv_items.csv"
OUT_CSV = config.RESULTS_DIR / "causal" / "hermes_cv_sarvam_rejudge.csv"

# ---- prompt-text map: iid -> original request text (harmful + benign/harmless) ----
def build_prompt_map():
    m = {}
    for it in s14.load_harmful_subset(LANG, 1000):
        m[str(it["id"])] = it["text"]
    bp = config.SAFETY_DIR / "benign" / f"{LANG}.json"
    if bp.exists():
        for it in json.load(open(bp)):
            m[str(it.get("id"))] = it.get("text", "")
    return m

# ---- 3-key rotator with failover ----
def load_keys():
    keys = []
    for line in open("$PROJECT_ROOT/.env"):
        if line.startswith("SARVAM_API_KEY"):
            v = line.split("=", 1)[1].strip()
            if v:
                keys.append(v)
    return keys

class KeyPool:
    def __init__(self, keys):
        self.lock = threading.Lock()
        self.alive = list(keys)          # order preserved
        self.i = 0
        self.dead = set()
    def get(self):
        with self.lock:
            if not self.alive:
                return None
            k = self.alive[self.i % len(self.alive)]
            self.i += 1
            return k
    def mark_dead(self, k, why):
        with self.lock:
            if k in self.alive:
                self.alive.remove(k)
                self.dead.add(k)
                print(f"  [key ...{k[-4:]} disabled: {why}]  alive={len(self.alive)}", flush=True)

POOL = None
MAX_TRIES = 10

def call_rot(msg):
    """sarvam-105b call, rotating keys; returns content str or 'ERR:...'."""
    body = json.dumps({"model": sv.JUDGE_MODEL, "messages": [{"role": "user", "content": msg}],
                       "max_tokens": 3000, "temperature": 0.0}).encode()
    last = "noalive"
    for attempt in range(MAX_TRIES):
        k = POOL.get()
        if k is None:
            return "ERR:NOKEYS"
        try:
            req = urllib.request.Request(sv.SARVAM_URL, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {k}"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            return (r["choices"][0]["message"].get("content", "") or "").strip()
        except urllib.error.HTTPError as e:
            last = f"HTTP{e.code}"
            if e.code in (401, 402, 403):                # auth/expired/out-of-credits -> retire key
                POOL.mark_dead(k, last); continue
            if e.code == 429:                            # rate limit -> keep key, back off, rotate
                time.sleep(min(30.0, 2.0 * (attempt + 1))); continue
            if e.code in (500, 502, 503, 529):
                time.sleep(min(20.0, 2.0 * (attempt + 1))); continue
            return f"ERR:{last}"
        except Exception as e:
            last = type(e).__name__
            time.sleep(min(15.0, 1.5 * (attempt + 1))); continue
    return f"ERR:{last}"

def judge_row(prompt, answer):
    ans = rj.post_think(answer) or str(answer)
    if not ans or len(ans.strip()) < 3:
        return 1, "NOANSWER"                              # empty == refuse (matches original)
    msg = (rj.JUDGE_PROMPT.replace("{lang}", rj.LANG_NAME.get(LANG, LANG))
           .replace("{prompt}", (prompt or "")[:1500]).replace("{answer}", ans[:3500]))
    v = sv.parse_verdict(call_rot(msg))["verdict"]
    lab = {"REFUSE": 1, "COMPLY": 0}.get(v, None)
    return lab, v

def main():
    global POOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    keys = load_keys()
    POOL = KeyPool(keys)
    print(f"[rejudge] loaded {len(keys)} sarvam keys (...{', ...'.join(k[-4:] for k in keys)})", flush=True)
    pm = build_prompt_map()
    df = pd.read_csv(IN_CSV)
    df["iid"] = df["iid"].astype(str)
    print(f"[rejudge] {len(df)} rows in {IN_CSV.name}", flush=True)

    if a.smoke:
        # one item per key-worth, spread across labels, to confirm keys respond
        sample = df.groupby("label").head(1).head(6)
        for _, r in sample.iterrows():
            lab, v = judge_row(pm.get(r["iid"], ""), r["resp"])
            print(f"  {r['label']:<15} {r['iid']:<18} verdict={v:<8} sarvam={lab}  alive={len(POOL.alive)}", flush=True)
        print("[smoke] done", flush=True); return

    # resume
    done = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        ok = prev[~prev["verdict"].astype(str).str.startswith("ERR")]
        done = set(zip(ok["fold"].astype(str), ok["label"], ok["iid"].astype(str)))
        print(f"[resume] {len(done)} already judged", flush=True)

    todo = [r for _, r in df.iterrows()
            if (str(r["fold"]), r["label"], str(r["iid"])) not in done]
    print(f"[rejudge] {len(todo)} to judge with {sv.JUDGE_MODEL} ({a.workers} workers)", flush=True)

    def work(r):
        lab, v = judge_row(pm.get(str(r["iid"]), ""), r["resp"])
        return dict(fold=r["fold"], label=r["label"], iid=r["iid"], sarvam_rj=lab, verdict=v)

    header = not OUT_CSV.exists()
    B = 60
    for i in range(0, len(todo), B):
        chunk = todo[i:i + B]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            recs = list(ex.map(work, chunk))
        pd.DataFrame(recs).to_csv(OUT_CSV, mode="a", header=header, index=False)
        header = False
        nerr = sum(str(x["verdict"]).startswith("ERR") for x in recs)
        print(f"  judged {min(i+B,len(todo))}/{len(todo)}  ({nerr} err this batch, alive_keys={len(POOL.alive)})", flush=True)

    # ---- summary: recompute the gap under the fresh 3-key sarvam judgment ----
    rjd = pd.read_csv(OUT_CSV)
    rjd = rjd[~rjd["verdict"].astype(str).str.startswith("ERR")]
    g = rjd.groupby("label").agg(
        sarvam_rj=("sarvam_rj", lambda x: round(100 * x.dropna().mean(), 1)),
        n=("iid", "size"),
        n_scored=("sarvam_rj", lambda x: int(x.notna().sum()))).sort_index()
    print("\n=== SARVAM (3-key re-judge) refusal% by label ===\n" + g.to_string(), flush=True)
    print(f"\nsaved -> {OUT_CSV}", flush=True)

if __name__ == "__main__":
    main()
