#!/usr/bin/env python3
"""
23 — Judge the INT8 quantization-sanity generations with the SAME gemma judge, then compare to the
INT4 (judged) baseline on the SAME items. Answers: is the INT4-vs-INT8 refusal / eval-cue-gap
difference real, or just keyword noise? (Keyword is unreliable for bn native-script.)

INT8 source : results/behavioral/quant_sanity_int8.csv  (re-derives prompt from item_id)
INT4 baseline: results/behavioral/refusal_judge.csv, src=cue_battery_grid.csv, model_gen=sarvamai/sarvam-m
Writes a SEPARATE file (no race with a concurrent 18 run). Cost: ~104 gemma calls (~$0.01).
"""
import sys, os, json, importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent)); import config
sys.path.insert(0, str(Path(__file__).parent))
_spec = importlib.util.spec_from_file_location("j18", str(Path(__file__).parent / "18_refusal_judge.py"))
j18 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(j18)

def okey():
    for l in open(os.path.expanduser("~/CAISc/.env")):
        if l.startswith("OPENROUTER_API_KEY="):
            return l.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found")
KEY, MODEL = okey(), j18.DEFAULT_MODEL
CONDS = ["deploy", "eval_log"]

df = pd.read_csv(config.BEHAV_DIR / "quant_sanity_int8.csv")
df["item_id"] = df["item_id"].astype(str)
prompts = {}
for lang in df.lang.unique():
    items = json.load(open(config.SAFETY_DIR / f"{lang}.json"))
    prompts[lang] = {str(it.get("id", "")): it["text"] for it in items}

def _j(row):
    p = prompts.get(row.lang, {}).get(row.item_id, "")
    try:
        v = j18.judge(KEY, MODEL, p, str(row.response), row.lang)
    except Exception as e:
        v = {"prompt_harmful": None, "verdict": f"ERR:{type(e).__name__}"}
    return {"lang": row.lang, "condition": row.condition, "item_id": row.item_id,
            "prompt_harmful": v.get("prompt_harmful"), "verdict": v.get("verdict")}

print(f"judging {len(df)} INT8 generations with {MODEL} ...")
with ThreadPoolExecutor(max_workers=6) as ex:
    jd = pd.DataFrame(list(ex.map(_j, df.itertuples(index=False))))
jd.to_csv(config.BEHAV_DIR / "quant_sanity_int8_judged.csv", index=False)
print(f"→ wrote {len(jd)} judged rows  (NOANSWER {100*(jd.verdict=='NOANSWER').mean():.0f}%, "
      f"ERR {100*jd.verdict.astype(str).str.startswith('ERR').mean():.0f}%)")

# INT4 baseline (judged), matched items
J = pd.read_csv(config.BEHAV_DIR / "refusal_judge.csv", on_bad_lines="skip")
b4 = J[(J.src == "cue_battery_grid.csv") & (J.model_gen == "sarvamai/sarvam-m")
       & J.lang.isin(df.lang.unique()) & J.cond.isin(CONDS)].copy()
b4["iid"] = b4.iid.astype(str)

def decided_refusal(d, vcol="verdict"):
    d = d[(d.prompt_harmful == 1) & d[vcol].isin(["REFUSE", "COMPLY"])]
    return ((d[vcol] == "REFUSE").mean() if len(d) else float("nan")), len(d)

print("\n══ JUDGED refusal on matched items — INT4 vs INT8 ══")
print(f"{'lang':4} {'cond':9} {'INT4%':>7} {'n4':>4} {'INT8%':>7} {'n8':>4} {'Δpp':>7}")
S = {}
for lang in sorted(df.lang.unique()):
    for cond in CONDS:
        d8 = jd[(jd.lang == lang) & (jd.condition == cond)]
        ids = set(d8.item_id)
        d4 = b4[(b4.lang == lang) & (b4.cond == cond) & (b4.iid.isin(ids))]
        r8, n8 = decided_refusal(d8); r4, n4 = decided_refusal(d4)
        S[(lang, cond)] = (r4, r8)
        d = (100*(r8-r4)) if (r4 == r4 and r8 == r8) else float("nan")
        print(f"{lang:4} {cond:9} {100*r4:7.1f} {n4:4} {100*r8:7.1f} {n8:4} {d:+7.1f}")

print("\n══ eval-cue gap (eval_log − deploy), judged — does the gap survive at each precision? ══")
for lang in sorted(df.lang.unique()):
    if (lang, "deploy") in S and (lang, "eval_log") in S:
        (r4d, r8d), (r4e, r8e) = S[(lang, "deploy")], S[(lang, "eval_log")]
        print(f"  {lang}: INT4 gap {100*(r4e-r4d):+.1f}pp   |   INT8 gap {100*(r8e-r8d):+.1f}pp")
print("\nNote: n is small (sanity sample) — reads gross distortion, not subtle effects.")
