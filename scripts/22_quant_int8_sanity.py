#!/usr/bin/env python3
"""
22 — INT8-vs-INT4 quantization sanity check on sarvam-m (validates E1 fidelity + E4 label transfer).

WHY: E1 behavioral generation runs sarvam-m at 4-bit (NF4); E4 extracts activations at 8-bit and
labels them with the 4-bit refusal verdicts. The literature (arXiv:2502.15799) says quantization
CAN shift safety/refusal behavior in a method-dependent way, so we must check that 4-bit didn't
distort refusal vs the near-lossless 8-bit. Same weights, same code path, same cues, same prompts,
same 400-tok budget, greedy — ONLY the bitsandbytes precision differs.

Reuses E1's EXACT cue resolution (09_cue_battery_grid.get_cue) + items (config.SAFETY_DIR/{lang}.json)
+ behavioral_common.batch_generate / classify_refusal, so the only variable is INT8 vs INT4.

TIME-BOUNDED: hard wall-clock cap (default 100 min) + round-robin cell order → balanced partial
data if the cap hits. Incremental writes. Compares to the INT4 baseline (cue_battery_grid.csv) at
the end (keyword for en = reliable; bn needs the judge — run 18 on the output for the rigorous read).

Usage:
  CUDA_VISIBLE_DEVICES=1 python scripts/22_quant_int8_sanity.py            # en+bn, deploy/eval_log
  CUDA_VISIBLE_DEVICES=1 python scripts/22_quant_int8_sanity.py --n 30 --budget 5400
"""
import sys, json, time, argparse, importlib.util
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import config
import behavioral_common as bc   # gives bc.LanguageModel / BitsAndBytesConfig / AutoTokenizer / torch

# E1 cue resolution, imported verbatim (filename starts with a digit → importlib)
_spec = importlib.util.spec_from_file_location("e1grid", str(Path(__file__).parent / "09_cue_battery_grid.py"))
e1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e1)

LANGS = ["en", "bn"]                 # en: keyword reliable (fast signal); bn: the annotator validates + judge
CONDS = ["deploy", "eval_log"]       # the core eval-cue contrast
OUT = config.BEHAV_DIR / "quant_sanity_int8.csv"
INT4_BASELINE = config.BEHAV_DIR / "cue_battery_grid.csv"


def load_model_int8():
    """Identical to behavioral_common.load_model_and_tokenizer() but load_in_8bit (LLM.int8())."""
    bnb = bc.BitsAndBytesConfig(load_in_8bit=True)
    model = bc.LanguageModel(
        config.MODEL_NAME, quantization_config=bnb,
        device_map="cuda:0", dtype=bc.torch.bfloat16, dispatch=True,
    )
    tok = bc.AutoTokenizer.from_pretrained(config.MODEL_NAME, fix_mistral_regex=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


def load_items(lang, n):
    items = json.load(open(config.SAFETY_DIR / f"{lang}.json"))
    return items[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="items per (lang,cond) cell (cap; budget may stop sooner)")
    ap.add_argument("--budget", type=int, default=6000, help="hard wall-clock cap in seconds (default 100 min)")
    args = ap.parse_args()

    battery, _framings, translations = e1.load_cue_sources()

    # round-robin work list: item-major so a time cutoff leaves all cells balanced
    cue = {(l, c): e1.get_cue(l, c, battery, translations) for l in LANGS for c in CONDS}
    for k, v in cue.items():
        print(f"  cue[{k}] = {'present' if v else 'MISSING'}")
    items = {l: load_items(l, args.n) for l in LANGS}
    work = []
    for i in range(args.n):
        for l in LANGS:
            if i >= len(items[l]):
                continue
            for c in CONDS:
                if cue[(l, c)] is None:
                    continue
                work.append((l, c, items[l][i]))
    print(f"\nplanned: {len(work)} generations ({LANGS} x {CONDS} x up to {args.n} items), budget {args.budget}s")

    print("loading sarvam-m @ INT8 on cuda:0 ...", flush=True)
    t_load = time.time()
    model, tok = load_model_int8()
    print(f"  loaded in {time.time()-t_load:.0f}s", flush=True)

    rows, t0, done = [], time.time(), 0
    B = bc.BATCH_SIZE
    for s in range(0, len(work), B):
        if time.time() - t0 > args.budget:
            print(f"\n⏹ time budget {args.budget}s reached — stopping with {done} generations.", flush=True)
            break
        chunk = work[s:s + B]
        msgs = [[{"role": "system", "content": cue[(l, c)]}, {"role": "user", "content": it["text"]}]
                for (l, c, it) in chunk]
        tb = time.time()
        outs = bc.batch_generate(model, tok, msgs)
        for (l, c, it), resp in zip(chunk, outs):
            rows.append({"lang": l, "condition": c, "item_id": it.get("id", ""),
                         "harm_category": it.get("harm_category", ""), "response": resp,
                         "refusal": bc.classify_refusal(resp, l), "precision": "int8"})
        done += len(chunk)
        pd.DataFrame(rows).to_csv(OUT, index=False)   # incremental
        spb = time.time() - tb
        print(f"  batch {s//B+1}: +{len(chunk)} gens in {spb:.0f}s ({spb/len(chunk):.1f}s/gen) | "
              f"total {done}/{len(work)}  elapsed {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\n→ wrote {len(df)} INT8 generations → {OUT}")

    # ── compare to INT4 baseline (keyword; en reliable, bn needs judge) ──
    if INT4_BASELINE.exists() and len(df):
        b4 = pd.read_csv(INT4_BASELINE)
        b4 = b4[b4.lang.isin(LANGS) & b4.condition.isin(CONDS)].copy()
        df["item_id"] = df["item_id"].astype(str); b4["item_id"] = b4["item_id"].astype(str)
        print("\n══ INT8 vs INT4 keyword refusal on the SAME items (en reliable; bn → judge to confirm) ══")
        print(f"{'lang':4s} {'cond':10s} {'n':>4s} {'INT4_kw%':>9s} {'INT8_kw%':>9s} {'Δpp':>7s}")
        for l in LANGS:
            for c in CONDS:
                d8 = df[(df.lang == l) & (df.condition == c)]
                ids = set(d8.item_id)
                d4 = b4[(b4.lang == l) & (b4.condition == c) & (b4.item_id.isin(ids))]
                if len(d8) and len(d4):
                    r8 = pd.to_numeric(d8.refusal, errors="coerce").mean()
                    r4 = pd.to_numeric(d4.refusal, errors="coerce").mean()
                    print(f"{l:4s} {c:10s} {len(d8):4d} {100*r4:9.1f} {100*r8:9.1f} {100*(r8-r4):+7.1f}")
        print("\nNOTE: keyword undercounts native-script (bn) refusals — run `18 --sources quant_sanity_int8.csv`")
        print("      (after adding it as a source) or judge this file to compare bn properly.")
    print("\nDone.")


if __name__ == "__main__":
    main()
