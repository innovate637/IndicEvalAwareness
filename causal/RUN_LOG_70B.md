# Hermes-70B Causal Replication — Run Log

> Living record of the Hermes-3-**70B** scale replication of the 8B causal eval-awareness positive
> (`llama_causal_experiments/`). All paths relative to
> `$PROJECT_ROOT/`. Results writeup → `RESULTS_70B.md`.
> The 8B originals are `RESULTS.md` / `HYPOTHESIS.md` / `README.md`; the sarvam-m E4 pipeline log is the
> top-level `../RUN_LOG.md`.

**Server:** university compute cloud — node `2xh100-nvl-cluster-vm2` (2× H100 NVL), Slurm. Driver
565.57.01 / CUDA 12.7. Partition `h100-full` (whole 94 GB GPU). venv `../.venv` (torch 2.12.1+cu126).
GPU code via `sbatch`; each job exports `HF_HOME=../.hf_cache` and the API keys from `../.env`, and runs
from `scripts/`.

---

## 1. Goal & design
Replicate the one clean positive (Hermes-3-Llama-3.1-**8B**, `RESULTS.md`) on **70B** to answer the "n=1 /
maybe it's just the 8B refusal tune" objection (`HYPOTHESIS.md §5`).

| Item | Detail |
|---|---|
| Model | `unsloth/Hermes-3-Llama-3.1-70B-bnb-4bit` — 4-bit NF4, **80 layers**, d_model 8192, non-reasoning |
| Method | identical engine `34_ablate_eval.py`; scripts `_hermes_cv.py`, `42_localize_steer.py`, `43_poscontrol_hermes.py` |
| Layer rescale 32→80 | steering window 6-22 → **15-55**; localize width 8 / starts 6·10·14·18 → **width 20 / starts 15·25·35·45**; poscontrol single mid-layer 16 → **40** (band 6-22 → 30-50) |
| Judges | gemma-3-27b (OpenRouter) + sarvam-105b (Sarvam API), T=0. Keyword scoring never used (undercounts Indic) |
| sbatch wrappers | `sbatch_smoke70b.sh`, `sbatch_poscontrol70b.sh`, `sbatch_localize70b.sh`, `sbatch_cv70b.sh` — all `--signal=B:USR1@300` self-requeue on walltime, `.done_*` sentinel to avoid double-runs |

---

## 2. Smoke test (`smoke_hermes70b.py`, job **266**, 2026-07-03) — ✅ PASS
`smoke_266.out`: model loads at 4-bit — **NL=80, D=8192**, weights bfloat16. Residual @L40 shape (1,32,8192),
norm(last)=14.8. Baseline "The capital of France is Paris."; **ZERO-ADD hook is a clean no-op** (identical
output) → the add-hook plumbing is correct on the 80-layer model. `SMOKE OK`.

## 3. Positive control (job **273**, 2026-07-03 13:03→14:15) — ✅ VALID
`43_poscontrol_hermes.py --n 40 --layers 30 40 50 --band 30-50`. Arditi refusal dir, ±α.
- **L40** is the clean validator: α=+1.0 55→**92** (+37.5pp, McNemar p=0.0001, 83% compliers flipped, deg 2%);
  α=+0.5 +20pp (p=0.008); α=−0.5 55→35 (−20pp, p=0.039). **Bidirectional, sign-consistent, coherent.**
- band 30-50 over-steers into garbage at ±α (deg 92–100%; judges then call it 100% "refuse") — single mid
  layers are the clean regime, same as the 8B 6-22 band.
⇒ steering apparatus genuinely moves refusal → eval-direction nulls/positives are interpretable. → `RESULTS_70B.md §3`.

## 4. Localization (job **279**, 2026-07-03 23:56→2026-07-04 01:01) — DISTRIBUTED
`42_localize_steer.py --width 20 --starts 15 25 35 45 --alpha 0.1 --n 50`.
Full 15-55: eval **100** vs null 84 vs random 62 (**+16 / +38pp**). No 20-layer sub-band reproduces the full
window (best L25-44 eval−null +12) → the effect is **spread across layers**, matches 8B. → `RESULTS_70B.md §2`.

## 5. Steering headline — n=100 5-fold CV (job **287**, 2026-07-04 14:07→15:17) — ✅ POSITIVE
`_hermes_cv.py`: window 15-55, coeffs {0.1, 0.25}, ADD-eval vs norm-matched null, dual-judge, **response
text saved** for the raw-output check. `results/causal/hermes_cv_items.csv`.
- **α=0.1 (clean regime):** ADD-eval **98% (gemma)** vs ADD-null **24%** → **+74pp**; harmless collateral 15%;
  deploy 54 → eval 89 natural gap.
- **α=0.25 excluded:** deg 94–100% — pure repetition loops (`"I will not refuse to provide a response…"`) that
  both judges mislabel REFUSE. The `deg` filter catches it; the headline rests on α=0.1.
- **Raw-text artifact check (as ../RUN_LOG §12/§14 discipline):** at α=0.1 ADD-eval = coherent on-topic
  refusals (87/100 open "I will not…"), ADD-null = coherent compliance; 0 script-contamination, 0 truncation.
  §12 derailment mode absent at α=0.1. → `RESULTS_70B.md §1, §4`.

**⇒ The 8B steering positive replicates at 70B, and larger (+74 vs +23pp).**

## 6. Sarvam dual-judge completion — 3-key re-judge (2026-07-04)
The original `cv_287` run set `sv.KEY = sarvam_key()`, which reads **only** `SARVAM_API_KEY` (key 1). Key 1
hit its quota — **HTTP 402 (out of credits), not 401** — after fold 0, so the sarvam column was **NaN for
folds 1–4** and all harmless items. (`.env` has 3 keys: key 1 dead ...WSKr; keys 2 ...A9qC / 3 ...dOVQ live.)

Re-judged all 800 stored responses with **`scripts/_rejudge_sarvam_cv.py`** — a standalone, CPU-only,
resume-safe judge that **rotates across all 3 keys with failover** (retire a key on HTTP 401/402/403, back
off on 429). Key 1 retired on its first call; keys 2/3 carried the run; **800/800, 0 errors**. Output
`results/causal/hermes_cv_sarvam_rejudge.csv` (original `hermes_cv_items.csv` untouched).

Dual-judge headline (α=0.1): ADD-eval **98% (gemma) / 99% (sarvam)**; ADD-null **24% / 50%**;
harmless **15% / 15%** → gap **+74pp (gemma) / +49pp (sarvam)**.
**Caveat:** re-judge ran on the stored **400-char** `resp`, which inflates the sarvam *null* rate upward
(long compliance payloads truncated → opening hedge scored REFUSE). gemma's full-text 24% is the reliable
null; the sarvam gap is a conservative floor. ADD-eval refusals are short (fully within 400 chars) → sarvam
99% is unaffected.

---

## 7. Patching — COMPLETE ✅ (jobs 310/311, 2026-07-06)
`39_patch_hermes.py` + `40_patch_multi_hermes.py` ported to 70B 4-bit (model id; `device_map="cuda:0"` no
dtype; bf16 donor inject; layer sets rescaled 32→80 — multi win15-55 / low15-34 / up35-55, single
15/25/35/45/55). Outputs → `results/causal/` (8B patch CSVs in `results/`, no collision); pristine 8B in
`code/`. Ran after a memory-limit fix (see §9) and a resume.
- **Multi-layer (job 310, deliverable) — POSITIVE.** Identity no-op valid at every set (IDENT ≈ deploy clean,
  deg 0 → 4-bit patching machinery valid). Clean isolate (deploy←EVAL mismatch − deploy←deploy mismatch):
  **win15-55 +18pp g / +12pp s**, low15-34 +13/+13, up35-55 +14/+8; harmless collateral 1–2%. Mirrors 8B
  (+16pp). → `RESULTS_70B.md §5`.
- **Single-layer (job 311) — valid, more localized than 8B.** IDENT clean everywhere; `EVAL patch − IDENT`
  L35 +14, L45 +11, L55 +13 (L15/L25 ≈ null). Unlike 8B (null at every single layer), 70B concentrates the
  effect in upper-mid layers, though full window still strongest. → `RESULTS_70B.md §5b`.

## 8. Cross-lingual EN→Indic steering — COMPLETE, positive-but-not-specific ⚠️ (job 312, 2026-07-06)
`41_xlingual_hermes.py` ported (window 15-55). `--langs hi bn --n 100 --win 15-55`. EN-built direction → Indic
harmful prompts, judged in-language. Transfer is strong and dual-confirmed — **hi +40pp g / +29pp s, bn +24pp g
/ +33pp s** (eval − null) — replicating the *direction* of the 8B result. **BUT not specific at α=0.1:** benign
collateral **hi 44% / bn 76%** (vs 8B's clean 6–8%) → the direction over-steers on Indic. Needs a coefficient
sweep (≈0.03–0.07) + raw-text check of the benign refusals (deg 4–8%) before it's a clean claim. α=0.25
degenerate. → `RESULTS_70B.md §6`. **This is the one open item.**

## 8b. Sarvam judge — 3-key rotation baked into the scripts
Added `scripts/_sarvam_pool.py` (`install(sv)` monkeypatches `sv.call` to round-robin all 3 keys, retire on
HTTP 401/402/403). Wired into **39, 40, 41** — they previously used `sv.KEY = sarvam_key()` (only key 1, dead /
HTTP 402). Confirmed working in all three job logs (`key ...WSKr disabled: HTTP402  alive=2`, no NaN). So
patching + cross-lingual are natively dual-judged, no post-hoc re-judge needed. See [[reference-sarvam-keys]].

## 9. Memory-limit incident + fix (2026-07-05)
The first patch run (job 300) reserved the **entire node RAM** (AllocMem=515755M = 100%) because the sbatch
scripts had no `#SBATCH --mem`, blocking the node's other GPU (instructor complaint). Actual usage was tiny
(MaxRSS ≈ 2.6 GB; weights live in GPU VRAM). Cancelled 300/301/303; added **`--mem=96G`** to all six
`sbatch_*70b.sh`; resubmitted as **310/311/312** (verified `mem=96G`, 1 GPU each). Codified as CLAUDE.md Hard
Rule 4 (never reserve the whole GPU/node; always set a bounded `--mem`). See [[project-env-setup]].

## 10. Overall status
**Successful two-method causal replication.** Steering ✅ (+74/+49pp) and patching ✅ (+18pp) both positive,
agree, text-verified, distributed, on a validated hook → the 8B positive replicates at 70B and is larger.
**One open item:** cross-lingual transfer is positive but needs a coefficient sweep to become *specific*.
Everything else (steering, patching, controls, artifact checks, dual-judge) is complete.
