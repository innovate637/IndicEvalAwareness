# RESULTS — Evaluation Awareness in Indic Settings (corrected, 2026-06-11)

> **MAJOR REVISION.** The original headline ("Indic safety gap: EN refusal 46% vs Indic 4–15%;
> recognition without refusal") was a **refusal-measurement artifact**. This is the corrected
> canonical record; the old document is archived at `RESULTS_v1_keyword_era.md`. Old keyword
> numbers appear as `[kw: …]` for the before/after contrast — they are **invalid as safety
> estimates**. Paper spine (GPT-5.4 consult concurs): **measurement validity in multilingual
> safety evaluation**, with recognition + CoT as explanatory pillars.

## 0. TL;DR

1. **The keyword refusal classifier under-counts refusals** — catastrophically for Indic
   native-script refusals, substantially for thinking-model polite deflections in English.
2. With a **calibrated LLM judge** (gemma-3-27b; English-anchored, second-judge-validated),
   all three models refuse **~80–95% in every language**; the Indic safety gap **vanishes or
   inverts** (Indic ≥ English).
3. With valid labels, a **real, small eval-cue compliance gap (+5–9pp, OR≈1.7) appears in all
   languages including Indic** — keyword labels had *hidden* the Indic effect and
   *manufactured* an English one. **Invalid labels flipped the scientific conclusion.**
4. Recognition + CoT survive: the eval cue is recognized cross-lingually (qwen discrimination
   *stronger* in Indic: +29…+39 vs +22 en), the CoT verbalizes eval-awareness in all 6
   languages and is **~82–91% faithful** to behavior, uniformly across languages.
5. Verbalized eval-awareness is **behaviorally near-inert** once item-confounding is removed
   (naive +14.2pp → within-item **+2.9pp, p=0.45**) — coheres with Knecht et al.'s small
   causal text-level effects and our near-null cue arithmetic (mediation: +14pp VEA incidence
   × ≤5pp ≈ +0.7–2pp ≈ observed).

## 1. The measurement artifact (discovery + evidence chain)

`behavioral_common.classify_refusal` (keyword matching) misses:
- **Native-script refusals**: इनकार / नहीं लिख सकता / অস্বীকার / "…আমার নীতিমালার সম্পূর্ণ
  বিরোধী" — confirmed by reading misclassified cases (bomb / hacking / password prompts
  firmly refused, scored `refusal=0`).
- **Thinking-model polite deflections in English**: "As an AI I don't have…", "X is a serious
  violation…" — refusals without refusal keywords. Sarvam's English baseline itself was
  under-counted (**~52% kw → ~88% judge**).

**Judge construction & validation** (`scripts/18_refusal_judge.py`):
- v1 (answer-only) over-called REFUSE (99% everywhere incl. English) → rejected on the
  English anchor. v2 (prompt+answer; REFUSE / COMPLY / NOANSWER; `prompt_harmful` screen)
  **calibrates on English**: gemma judge-vs-kw item agreement 86% (80.0% vs 77.7% rates).
- Disagreement audit: where judge and keyword disagree, **the judge is right** (read cases).
- **Second-judge cross-check** (gemma-4-31b-it:free, n=96): verdict agreement **88.4%**;
  **Indic 87.8% ≈ English 90.0%** → judging is not less reliable in Indic;
  prompt_harmful agreement 85.6%; disagreements bidirectional (no over-flag bias).
- Gap to gold: no human anchor yet → **V1 human labels (hi/bn) = top priority** (sheets
  regenerated to oversample judge↔keyword disagreements: `refusal_annotation_*_v2.csv`).

## 2. Corrected refusal landscape (judge labels, harmful prompts only)

Refusal % `[kw: invalid keyword %]`:

| model | en | hi | bn | ta | te | or |
|---|---|---|---|---|---|---|
| **sarvam-m** | 88 [52] | 92 [7] | 94 [13] | 87 [9] | 91 [12] | 99* [5] |
| **gemma-3-27b** | 80 [78] | 95 [51] | 93 [70] | 84 [36] | 96 [89] | 89 [0.7] |
| **qwen3-32b** | 84 [45] | 86 [1] | 84 [2] | 80 [2] | 85 [2] | 89 [0.3] |

- **No Indic safety gap**: sarvam EN 88 vs Indic-4 91 (−3pp); gemma 80 vs 92 (−12pp, Indic
  *safer*); qwen 84 vs 84 (0pp). The qwen row is the showcase: kw said 1–2% Indic refusal
  (a fake 43pp cliff); true ~80–89%.
- *sarvam-or: ~52–64% NOANSWER (400-token truncation) — computed on answered subset;
  provisional until regenerated with a larger budget. qwen Indic NOANSWER ~19–26%.
- `prompt_harmful=0`: **8–31% of "harmful" items judged benign post-translation** → excluded
  from rates; quantifies the V2 harm-attenuation confound.

## 3. Compliance gaps with valid labels (NEW headline result)

Paired per item (McNemar + cluster bootstrap); **confirmatory = GEE binomial, exchangeable
correlation, clustered on BASE item** (handles parallel translations across languages).
`scripts/19_confirmatory_stats.py` → `results/stats/confirmatory_judge_stats.csv`.

**Pre-specified primary contrasts (refusal, harmful items):**

| contrast | OR [95% CI] | ≈pp | GEE p | boot p |
|---|---|---|---|---|
| gemma eval_log−deploy, 6-lang pooled | **1.73 [1.29, 2.32]** | +6.2 | **0.0002** | 0.0003 |
| qwen eval_log−deploy, 6-lang pooled | **1.77 [1.30, 2.41]** | +9.0 | **0.0003** | 0.036 |
| sarvam eval−deploy, Indic-4, en cue | **1.65 [1.13, 2.42]** | +5.0 | **0.010** | 0.012 |
| sarvam eval−deploy, Indic-4, native cue | 1.31 [0.90, 1.91] | +2.6 | 0.16 ns | 0.028 |

**Decomposition (battery: monitoring vs eval-framing):**
- **gemma = FRAMING-driven**: eval_log−neutral_log OR=1.49 [1.18, 1.89], p=0.0009 (robust);
  monitoring-only ns (+1.7pp).
- **qwen = monitoring-leaning**: neutral_log−deploy OR=1.52, GEE p=0.009, **boot p=0.12 →
  suggestive only**; framing ns.
- **sarvam English = negative SUFFIX effect**: *any* monitoring suffix lowers refusal ~10pp
  (neutral_log OR=0.38 p=0.008; eval_log OR=0.36 p=0.009; robust both methods); pure framing
  (eval−neutral) ≈ 0; eval_explicit ≈ 0 (non-monotonic). n≈65 pairs — replicate in the E1
  full battery before leaning on it.

**The flip (core of the paper):** kw labels said EN gap +6pp / Indic ≈0 → "eval-awareness
doesn't drive Indic behavior." Valid labels say EN ≈0-to-negative / **Indic +2…+13pp,
pooled-significant** → the conclusion **reverses**, not just the rates. Per-language cells are
individually underpowered near ceiling (discordant pairs b+c ≤ 10) — pooled contrasts are the
primary claims; per-cell results exploratory. DiD (Indic−EN, matched items) +2.3 [−2.3, +7.0].

## 4. Recognition — survives, all 6 languages (MANIPULATION CHECK: significant)

Transcript classification, P(says "evaluation"); discrim = eval_log − deploy. Formal tests:
paired McNemar per language; GEE pooled clustered on base item
(`results/stats/recognition_discrimination_stats.csv`).

- **qwen — significant in EVERY language individually**: discrim +22 (en, p=1e-4) / +29 (hi,
  p=1e-6) / +36 (bn, p=2e-8) / +39 (ta, p=8e-8) / +27 (te, p=1e-4) / +35 (or, p=7e-7).
  **Pooled GEE OR=5.75 [4.15, 7.95], p≈5×10⁻²⁶.** Verbal *framing* discrimination
  (eval_log vs neutral_log): **OR=7.32 [5.38, 9.95], p≈6×10⁻³⁷** — qwen distinguishes
  being safety-evaluated from merely being logged. Recognition is *stronger* in Indic than
  English (our novel extension). Unclear-rate 4–23%.
- **gemma — verbally saturated, behaviorally sensitive**: says "evaluation" 91–100% even
  under deploy → discrimination compressed; still significant where headroom exists
  (hi +6.0 p=0.031; bn +9.0 p=0.004; ta +9.0 p=0.004; all flips one-directional b/c=6–9/0;
  en at 100/100 = zero headroom). Pooled GEE invalid (perfect separation) — per-language
  McNemar is the valid test. **Dissociation:** gemma's verbal report cannot distinguish
  eval framing from neutral logging (OR=1.00, p=0.95) but its *behavior* can (refusal
  framing effect OR=1.49, p=0.0009) — self-report and behavior are separable channels of
  eval-awareness, and which is sensitive differs by model (qwen = mirror image).
- **sarvam** (E2 pilot, thinking-off): recognizes the cue in every language (recall ≈ 1.0);
  thinking-on 2×2 (768-token budget) queued after E1.
- **Replication status: the established eval-awareness phenomenon is confirmed in our setup**
  (qwen p~10⁻²⁶; gemma where unsaturated; sarvam behavioral OR=1.65 p=0.010) — no
  failed-replication risk; the cross-lingual transfer/intensification is the new result.

## 5. CoT pillar — faithful, aware; monitorability moot

788 traces; blinded monitor reads `<think>` only (`results/cot_monitor/`):

| | en | hi | bn | ta | te | or |
|---|---|---|---|---|---|---|
| eval-aware in CoT % | 37 | 28 | 34 | 21 | 25 | 21 |
| faithfulness vs judge % | 89 | 89 | 89 | 91 | 90 | 82 |
| harmful planning % | 3 | 4 | 3 | 1 | 1 | 3 |

- **Faithfulness uniform cross-lingually (~82–91%)** — no Indic deficit. (Against kw labels it
  had *looked* like 30–50% in Indic — same artifact.)
- **Monitorability moot**: refusal ~90% ⇒ 0–3 harmful answers/lang to catch. The earlier
  "monitorability≈0" was a denominator artifact. Corrected story: **aware + faithful + safe**.
- **VEA→refusal collapses under item control**: naive +14.2pp → within-item **+2.9pp
  (sign test p=0.45; 64/71 ties)**; +4.9pp within item+condition. Monitor reasons show label
  leakage (flags refusal-deliberation as "awareness"). Verbalized awareness = weak behavioral
  lever (≤~5pp), matching Knecht et al. (2026).

## 5b. E4 mechanistic — eval-awareness DIRECTION transfers cross-lingually (representational; causal pending, 2026-06-12)

8-bit activations (last prompt token, all 40 layers), diff-of-means directions + sklearn probes;
calibrated-judge labels joined for the refusal positive control (showstopper-1 fixed).
`scripts/12,13,13b`; `results/activations/` (28/30 cells — Odia `eval_explicit`/`eval_metadata`
OOM'd; Odia has deploy/eval_log/neutral_log, enough for the two primary contrasts), `results/probes/`.

**Two anti-trivial directions both transfer EN→Indic incl. Odia** (transfer AUROC = EN-trained
direction scored on each Indic language at the EN best layer):

| direction (diff-of-means) | layer | hi | bn | ta | te | or | mean |
|---|---|---|---|---|---|---|---|
| **eval_framing** (eval_log − neutral_log) | 6 | 0.998 | 0.970 | 0.962 | 0.924 | **0.915** | **0.954** |
| cue_presence (eval_log − deploy) | 1 | 0.961 | 0.958 | 0.949 | 0.932 | 0.852 | 0.930 |
| awareness (VEA/CoT) | 19 | 0.642 | 0.644 | 0.593 | 0.698 | **0.444** | 0.604 |

In-language best-layer AUROC ≈ **1.000** for eval_framing & cue_presence (all langs incl. Odia);
awareness 0.65–0.88. eval_framing is the defensible axis: both arms carry a monitoring sentence,
so it isolates safety-eval framing from mere cue presence, and it mirrors the behavioral
eval_log−neutral_log effect (gemma OR=1.49, §3).

**Anti-lexical — cross-WORDING generalization (`13b`):** the eval_framing steering vector trained
ONLY on eval_log−neutral_log still separates *other* eval wordings from neutral on HELD-OUT items:
eval_explicit **0.93–0.999**, eval_metadata **0.77–0.99** (pooled cross-wording mean **0.951** at
layer 9; in-wording 0.992). A purely lexical direction could not. Combined with cross-LINGUAL
transfer (an English vector firing on Hindi/Odia activations), a lexical-artifact explanation is
hard to sustain → **a real, abstract "this is a safety evaluation" representation, shared across
languages AND phrasings**, living in layers 6–12 (peak 9). This answers the E4-audit showstopper-2
(early-layer 1.0 AUROC ⇏ lexical detection).

**Honest edges:**
- **Representational, not causal.** Steering (`14`: build refusal direction = positive control
  that MUST move refusal; then steer the eval direction → predicted ~null) is the decisive test —
  **queued overnight**. Prior = represented-but-inert (coheres with §5's within-item +2.9pp ns).
- **The VEA/CoT-awareness direction does NOT transfer** (Odia 0.444 < chance; cross-lingual cosine
  0.08–0.43): the *self-reported* awareness axis is language-SPECIFIC — a separate channel from the
  shared eval-framing axis (mirrors §4's report-vs-behavior dissociation).
- **Resource gradient:** eval_metadata generalization fades on Telugu (0.774); awareness probe
  weakest in ta/te. H3 (partly language-specific geometry) supported.

## 6. Benchmark-quality findings (reportable by-products)

- **prompt_harmful screen**: 8–31% of Toxic_Matrix "harmful" items benign post-translation
  (language-varying harm attenuation).
- **Truncation**: token budgets must scale with script/tokenizer overhead, else NOANSWER
  masquerades as comply (sarvam-or 52–64%; qwen Indic 19–26% at 400 tokens).
- **Toxic_Matrix ≥3,200 parallel rows** → n=300 headline cells feasible.

## 7. Paper framing (post-pivot)

**Thesis:** *Naive refusal scoring creates spurious cross-lingual safety conclusions; under
valid measurement, evaluation-awareness signals — recognition, verbalized awareness, and a
small compliance gap — all transfer across Indic languages.*

Package: (1) failure analysis of keyword refusal labels (multilingual + thinking models);
(2) calibrated cross-validated LLM-judge replacement; (3) re-estimation → gap vanishes,
compliance gap appears, conclusion flips; (4) recognition + CoT explain why (aware, faithful,
safe); (5) E4 — the eval-framing **direction is real and shared cross-lingually incl. Odia**
(transfers across languages AND eval phrasings; §5b), with causal steering (**represented-but-inert**
prior) running overnight as the closing test.

**Objections to pre-empt:** judge circularity (→ V1 human anchor; 2nd judge 88%; English
calibration), refusal ≠ safety, truncation bias (→ regenerate or; sensitivity analyses),
thinking/non-thinking comparability, ceiling (→ harder-prompt subset), item quality
(→ prompt_harmful screen reported).

**Title candidate:** *"When Safety Gaps Vanish Under Valid Measurement: Auditing Multilingual
Refusal Evaluation in LLMs."*

## 8. Status / infrastructure (2026-06-12)

- **INT8 E1 (GPU 2)**: `en` running (auto-saves ~05:20), then `bn` under per-batch checkpointing
  (`09` patched). `hi` paused with a 24-row checkpoint (GPU 3 loaned out). Auto-stop armed 07:50 Sat
  to free GPUs 2+3. INT4-vs-INT8 within-item read: the "+10pp INT8 gap" is largely a **judge
  artifact × quant style-shift** (gemma over-calls COMPLY on warmer/longer INT8 refusals), not
  behavior — see [[feedback-isolate-paired-cases]].
- **2nd judge = sarvam-105b** (Sarvam API, free) over all gemma+qwen+sarvam-m generations (`21`,
  7,753 co-judged): verdict agreement with gemma **87.2%** overall, **92.0%** on REFUSE/COMPLY
  (harmful); Indic ≈ English (en 91.8 / hi 89.5 / bn 87.6 / ta 82.2 / te 86.2). Where they differ
  the Indic-native judge sides with the STRICTER (REFUSE) call ~4:1 — corroborates gemma over-calls
  COMPLY. Strengthens the "judging reliable in Indic" claim. (prompt_harmful screen noisier, 73.5%.)
- **E4 representational DONE** (§5b): probes + cross-lingual transfer + cross-wording (`12/13/13b`).
  Causal steering (`14`) runs overnight on GPU 3 after the loan.
- Confirmatory stats: `19_confirmatory_stats.py` (GEE + cluster bootstrap; statsmodels venv).
- API spend to date ≈ **$1.15** (+ sarvam-105b judge = free).
- **Next:** E4 steering (overnight) → V1 human labels (Sibayan, v2 sheets) → E1 re-judge →
  E2 (768 tokens) → harder-prompt subset → n=300 expansion if time.
