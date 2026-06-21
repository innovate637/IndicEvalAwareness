#!/usr/bin/env python3
"""
25 — CROSS-LINGUAL causal steering of the eval-awareness direction, with the full control suite
     (Neel-Nanda-standard: positive control, random/negative control, sign control, within-lang anchor).

THE QUESTION (what actually matters): is the eval-awareness direction SHARED across languages?
  - within-language steering (en-dir -> en) is TRIVIAL — a direction extracted from a language
    obviously moves that language. It is kept ONLY as the 100%-transfer YARDSTICK.
  - CROSS-lingual steering is the result:
      en-dir   -> hi/bn/ta/te   ("does the English eval direction move Indic behavior?")
      Indic-dir -> en           ("does an Indic eval direction move English behavior?")

CONTROLS (a steered effect is meaningless without them):
  POSITIVE   refusal-direction -> en : MUST raise refusal (proves the hook has causal power).
  NEGATIVE   random unit vector (matched norm, same layer) -> en & bn : must NOT systematically
             move refusal — otherwise any perturbation "works" and specificity is unproven.
  SIGN       sweep -alpha ... +alpha : a real signed direction moves refusal OPPOSITE ways for
             +/-alpha, monotonically; rules out "any big addition saturates behavior".
  BASELINE   alpha=0 == unsteered deploy generation (no hook registered).

STEERING MATH (the fix vs the earlier broken run): NORM-RELATIVE addition
      h_t  <-  h_t + alpha * ||h_t|| * v_hat
  alpha is a dimensionless FRACTION of the per-token residual norm at the steered layer, so it is
  comparable across layers (an absolute alpha is huge at early layers, negligible at late ones —
  the bug we hit). v_hat is the UNIT diff-of-means direction; we apply the SOURCE direction at the
  SOURCE direction's own best layer.

Refusal here = the shared keyword classifier (fast, cross-experiment-comparable) for the dose-
response SHAPE. The cross-lingual cells that show an effect should then be LLM-judged (script 18/21
rubric) for the headline number, since the keyword classifier under-counts Indic refusals.

  python scripts/25_steer_xlang.py --smoke                 # tiny GPU validation first
  python scripts/25_steer_xlang.py --n 30 --resume         # full run
Outputs: results/steering/xlang/steer_{kind}_{lang}.csv  (per-generation checkpoint, item-level resume)
"""
import os, sys, json, pickle, argparse, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))   # so importing 14/09 (which import behavioral_common) works
import config
from behavioral_common import classify_refusal


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "scripts" / fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

s14 = _load("s14", "14_steer_patch.py")   # load_directions, build_refusal_direction, _unit, D_MODEL, load_harmful_subset
s09 = _load("s09", "09_cue_battery_grid.py")  # load_cue_sources, get_cue
rj  = _load("rj18", "18_refusal_judge.py")    # JUDGE_PROMPT / LANG_NAME / post_think
sv  = _load("sv21", "21_sarvam_judge.py")     # sarvam-105b call() / parse_verdict / sarvam_key

from concurrent.futures import ThreadPoolExecutor

D_MODEL = config.D_MODEL
OUT_DIR = config.STEER_DIR / "xlang"
MAX_NEW_TOKENS = 400          # no Odia in the target set; matches E1 cue-battery
DEFAULT_N = 20                # overnight signal pilot (scale up once signal appears)
DEFAULT_STRENGTHS = [-0.5, 0.0, 0.5]                # sign control + baseline (fraction of resid norm)
DEFAULT_BATCH = 8
MAX_PROMPT_TOKENS = 768


def build_random_direction(layer: int, seed: int = 0) -> dict:
    """Matched NEGATIVE control: a random unit vector at the same layer (so the per-layer-norm
    scaling is identical to the eval direction it controls)."""
    rng = np.random.RandomState(1000 + seed + layer)
    v = rng.randn(D_MODEL).astype(np.float32)
    return {"layer": layer, "vector": v / (np.linalg.norm(v) + 1e-9)}


def layer_norm_scale(lang: str, layer: int) -> float | None:
    """Mean residual-stream L2 norm at `layer` for `lang`, from cached E4a activations.
    Used as the CONSTANT steering magnitude unit (CAA-style per-layer normalization): the
    added vector is  alpha * scale * unit_direction , so alpha is a fraction of the target
    language's typical residual norm at the steered layer (comparable across layers/langs)."""
    norms = []
    for cond in ["deploy", "neutral_log", "eval_log", "eval_explicit", "eval_metadata"]:
        p = config.ACT_DIR / f"{lang}_{cond}.pkl"
        if not p.exists():
            continue
        d = pickle.load(open(p, "rb"))
        acts = np.asarray(d["acts"], dtype=np.float32)        # [N, L, D]
        if acts.ndim == 3 and acts.shape[1] > layer:
            norms.append(np.linalg.norm(acts[:, layer, :], axis=1))
    return float(np.concatenate(norms).mean()) if norms else None


def _acts(lang, cond):
    p = config.ACT_DIR / f"{lang}_{cond}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


def refusal_dir_at(lang, layer):
    """Unit refused−complied diff-of-means at `layer` (pooled across conditions, judge labels)."""
    refused, complied = [], []
    for cond in ["deploy", "neutral_log", "eval_log", "eval_explicit", "eval_metadata"]:
        d = _acts(lang, cond)
        if not d:
            continue
        acts = np.asarray(d["acts"], dtype=np.float32)        # [N, L, D]
        for i, r in enumerate(d.get("refusal") or []):
            if r is None or i >= acts.shape[0] or acts.shape[1] <= layer:
                continue
            (refused if int(r) == 1 else complied).append(acts[i, layer, :])
    if len(refused) < 5 or len(complied) < 5:
        return None
    v = np.mean(refused, 0) - np.mean(complied, 0)
    return v / (np.linalg.norm(v) + 1e-9)


def eval_dir_orth(lang, layer):
    """Unit eval-framing diff-of-means (eval_log−neutral_log) at `layer`, ORTHOGONALIZED against
    the refusal direction at the same layer — so steering does not just re-inject 'refuse more'."""
    e, nu = _acts(lang, "eval_log"), _acts(lang, "neutral_log")
    if not e or not nu:
        return None
    ea = np.asarray(e["acts"], dtype=np.float32); na = np.asarray(nu["acts"], dtype=np.float32)
    if ea.shape[1] <= layer or na.shape[1] <= layer:
        return None
    d = ea[:, layer, :].mean(0) - na[:, layer, :].mean(0)
    r = refusal_dir_at(lang, layer)
    if r is not None:
        d = d - float(d @ r) * r            # remove the refusal component
    return (d / (np.linalg.norm(d) + 1e-9)).astype(np.float32)


# ── LLM judge (sarvam-105b) — replaces the keyword classifier ───────────────────
def judge_one(prompt, response, lang):
    """Sarvam-105b verdict on a (possibly steered) generation. Returns (verdict, refusal),
    refusal = 1.0 REFUSE / 0.0 COMPLY / nan for NOANSWER|UNCLEAR|ERR (excluded from rates)."""
    ans = rj.post_think(response) or str(response)   # judge the post-</think> answer; fall back to raw
    if not ans or len(ans.strip()) < 3:
        return "NOANSWER", float("nan")
    msg = (rj.JUDGE_PROMPT.replace("{lang}", rj.LANG_NAME.get(lang, lang))
           .replace("{prompt}", (prompt or "")[:1500]).replace("{answer}", ans[:3500]))
    v = sv.parse_verdict(sv.call(msg))["verdict"]
    return v, {"REFUSE": 1.0, "COMPLY": 0.0}.get(v, float("nan"))


# ── Batched, norm-relative steered generation ───────────────────────────────────
def steer_generate_batch(model, tokenizer, prompts, system_cue, layer, vec_t, alpha, scale,
                         max_new_tokens=MAX_NEW_TOKENS):
    import torch
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-pad so generated tokens align at the right

    formatted = []
    for p in prompts:
        pids = tokenizer(p, add_special_tokens=False)["input_ids"]
        if len(pids) > MAX_PROMPT_TOKENS:
            p = tokenizer.decode(pids[:MAX_PROMPT_TOKENS])
        formatted.append(tokenizer.apply_chat_template(
            [{"role": "system", "content": system_cue}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True))
    inputs = tokenizer(formatted, return_tensors="pt", padding=True).to("cuda:0")
    n_in = inputs["input_ids"].shape[1]

    hf = model._model
    handle = None
    if alpha != 0.0:
        # CONSTANT vector = alpha * scale * unit_direction, where scale = target language's mean
        # residual norm at this layer (CAA-style per-layer normalization; alpha = fraction of it).
        add = (alpha * scale * vec_t).to("cuda:0")

        def _hook(_m, _i, output):
            # Induce + maintain the eval-state: add the CONSTANT vector at ALL positions (prompt +
            # generated tokens). Constant magnitude (NOT per-token norm) means the high-norm BOS/
            # attention-sink is not over-steered — so steering the prompt is safe, and it is REQUIRED
            # to recreate the "I'm being evaluated" state the cue sets up during PROMPT processing
            # (the hypothesis we actually test), not merely a generation-time control knob.
            is_t = isinstance(output, tuple)
            hs = output[0] if is_t else output
            hs = hs + add.to(hs.dtype)
            return (hs,) + tuple(output[1:]) if is_t else hs

        handle = hf.model.layers[layer].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            out = hf.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                              max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
    finally:
        if handle is not None:
            handle.remove()
    return [tokenizer.decode(out[i][n_in:], skip_special_tokens=True).strip()
            for i in range(out.shape[0])]


def run_shard(model, tokenizer, target_lang, kind, dir_entry, items, cue, strengths, batch_size):
    import torch
    layer = dir_entry["layer"]
    vec_t = torch.tensor(dir_entry["vector"], dtype=torch.float32)
    scale = layer_norm_scale(target_lang, layer) or 1.0   # per-layer residual-norm unit for alpha
    sweep = [0.0] + sorted(s for s in set(float(x) for x in strengths) if s != 0.0)  # baseline FIRST

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp = OUT_DIR / f"steer_{kind}_{target_lang}.csv"
    done, base_ref = set(), {}
    if sp.exists():
        prev = pd.read_csv(sp)
        for _, r in prev.iterrows():
            done.add((float(r["strength"]), str(r["item_id"])))
            if float(r["strength"]) == 0.0:
                base_ref[str(r["item_id"])] = r["refusal_steered"]
    header_needed = not sp.exists()

    for strength in sweep:
        todo = [it for it in items if (strength, str(it.get("id", ""))) not in done]
        for b0 in tqdm(range(0, len(todo), batch_size),
                       desc=f"{kind}->{target_lang} a={strength}", leave=False):
            batch = todo[b0:b0 + batch_size]
            resps = steer_generate_batch(model, tokenizer, [it["text"] for it in batch],
                                         cue, layer, vec_t, strength, scale)
            # PRIMARY label = sarvam-105b LLM judge (parallel over the batch). The keyword
            # classifier is kept only as a secondary reference column (it under-counts Indic).
            with ThreadPoolExecutor(max_workers=max(1, len(batch))) as ex:
                verdicts = list(ex.map(lambda pr: judge_one(pr[0]["text"], pr[1], target_lang),
                                       list(zip(batch, resps))))
            rows = []
            for it, resp, (verdict, ref) in zip(batch, resps, verdicts):
                iid = str(it.get("id", ""))
                if strength == 0.0:
                    base_ref[iid] = ref
                rows.append({"target_lang": target_lang, "kind": kind, "layer": layer,
                             "strength": strength, "item_id": iid,
                             "refusal_base": base_ref.get(iid, ""),
                             "refusal_steered": ref, "verdict": verdict,
                             "kw_refusal": classify_refusal(resp, target_lang),
                             "response_steered": resp})
                done.add((strength, iid))
            pd.DataFrame(rows).to_csv(sp, mode="a", header=header_needed, index=False)
            header_needed = False
    return pd.read_csv(sp) if sp.exists() else pd.DataFrame()


def build_jobs(dirs, langs, smoke=False):
    """Return list of (target_lang, kind, dir_entry). Skips cells whose direction is missing."""
    ev = dirs.get("eval", {})
    en_layer = ev.get("en", {}).get("layer", 6)
    jobs = []

    # POSITIVE control
    if "en" in dirs.get("refusal", {}):
        jobs.append(("en", "refusal", dirs["refusal"]["en"]))
    # NEGATIVE control: random vector at the en eval layer, into en and bn (matches en->bn)
    rnd = build_random_direction(en_layer)
    jobs.append(("en", "random", rnd))
    if not smoke:
        jobs.append(("bn", "random", rnd))

    if smoke:
        # minimal cross-lingual probe: EN eval dir -> bn
        if "en" in ev:
            jobs.append(("bn", "xeval_from_en", ev["en"]))
        return jobs

    # within-language anchor (the trivial 100%-transfer yardstick)
    if "en" in ev and "en" in langs:
        jobs.append(("en", "xeval_from_en", ev["en"]))
    # cross-lingual, INTERLEAVED by language (priority bn/hi first) so a partial overnight finish
    # still covers BOTH directions for the languages that complete.
    for tgt in ["hi", "bn", "ta", "te"]:
        if tgt in langs and "en" in ev:
            jobs.append((tgt, "xeval_from_en", ev["en"]))       # EN  -> Indic
        if tgt in ev:
            jobs.append(("en", f"xeval_from_{tgt}", ev[tgt]))   # Indic -> EN
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--strengths", nargs="+", type=float, default=DEFAULT_STRENGTHS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--langs", nargs="+", default=["en", "hi", "bn", "ta", "te"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run (n=2, few cells) to validate the hook")
    a = ap.parse_args()

    sv.KEY = sv.sarvam_key()   # sarvam-105b judge auth (real-money preview key in ~/CAISc/.env)
    print("sarvam judge key loaded.")

    dirs = s14.load_directions()
    for lang in set(a.langs) | {"en"}:
        if lang not in dirs.get("refusal", {}):
            rd = s14.build_refusal_direction(lang)
            if rd is not None:
                dirs.setdefault("refusal", {})[lang] = rd

    # Rebuild eval directions ORTHOGONALIZED against refusal, at each language's probe layer
    # (design-review fix: otherwise we may just be re-injecting the refusal direction).
    eval_orth = {}
    for lang, entry in dirs.get("eval", {}).items():
        v = eval_dir_orth(lang, entry["layer"])
        if v is not None:
            eval_orth[lang] = {"layer": entry["layer"], "vector": v}
    dirs["eval"] = eval_orth
    print("eval dirs (orth. vs refusal): " + ", ".join(f"{l}@L{e['layer']}" for l, e in eval_orth.items()))

    n = 2 if a.smoke else a.n
    strengths = [0.0, 0.5] if a.smoke else a.strengths
    jobs = build_jobs(dirs, a.langs, smoke=a.smoke)
    print(f"jobs ({len(jobs)}): " + ", ".join(f"{k}->{l}" for l, k, _ in jobs))

    battery, framings, translations = s09.load_cue_sources()
    from behavioral_common import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(int8=True)
    print(f"Model loaded (8-bit). n={n} strengths={strengths} batch={a.batch_size}")

    for target_lang, kind, dir_entry in jobs:
        items = s14.load_harmful_subset(target_lang, n)
        cue = s09.get_cue(target_lang, "deploy", battery, translations)
        if not items or cue is None:
            print(f"  [skip] {kind}->{target_lang} (items={len(items) if items else 0}, cue={'ok' if cue else 'MISSING'})")
            continue
        sp = OUT_DIR / f"steer_{kind}_{target_lang}.csv"
        expected = len(set([0.0] + list(strengths))) * len(items)
        if a.resume and sp.exists() and len(pd.read_csv(sp)) >= expected:
            print(f"  [resume] {sp.name} complete — skip"); continue
        print(f"\n→ {kind} -> {target_lang} | layer={dir_entry['layer']} | n={len(items)}")
        df = run_shard(model, tokenizer, target_lang, kind, dir_entry, items, cue, strengths, a.batch_size)
        if not df.empty:
            g = df.groupby("strength")["refusal_steered"].mean()
            print(f"  refusal by alpha: " + "  ".join(f"{s:+.2f}:{r:.2f}" for s, r in g.items()))

    print("\nDone → results/steering/xlang/  (next: LLM-judge the cross-lingual shards for the headline number)")


if __name__ == "__main__":
    main()
