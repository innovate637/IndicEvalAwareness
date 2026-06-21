# Evaluation Awareness in Indic Languages — Behavioral and Mechanistic
### Comprehensive Project Report

**Author:** Sibayan Mitra · **Generated:** 2026-06-16 · **Code root:** `~/mech_interp`

---

## 0. Thesis & Contributions

**Thesis.** *Evaluation awareness — the phenomenon where an LLM behaves differently when it
detects it is being evaluated — generalizes to Indic languages, demonstrated both behaviorally
(a compliance gap under evaluation framing) and mechanistically (a shared, causally-effective
eval-awareness direction that transfers cross-lingually).* A measurement-validity audit is the
necessary scaffolding: with naive scoring the Indic effect is invisible (and a spurious English
one is manufactured); only under valid measurement does the cross-lingual phenomenon appear.

**Contributions.**
1. **Measurement-validity audit** of multilingual safety evaluation: keyword refusal classifiers
   miss native-script and thinking-model refusals; same-family LLM judges over-call "comply" on
   Indic; aggressive quantization adds a refusal-ceiling artifact. Each *flips* conclusions.
2. **Behavioral eval-awareness in Indic:** a real, small evaluation-cue compliance gap
   (+5–9 pp pooled; up to +14 pp under the explicit cue) appears in Hindi/Bengali, multi-model.
3. **Mechanistic generalization:** an eval-framing *direction* transfers EN→Indic (probe AUROC
   ≈ 0.95–1.00, incl. Odia) and is **causally inducible cross-lingually** (injecting the English
   direction drives the eval-representation in Indic forward passes; random control flat).
4. **A documented dissociation:** the eval representation is shared and causally inducible, but
   only **weakly coupled to refusal behavior** — causal *behavioral* steering does not yield a
   clean refusal increase.

---

## 1. Model Choices (and why)

| Model | Role | Why chosen |
|---|---|---|
| **sarvam-m (24B)** | primary subject (mechanistic) | Mistral-Small-3.1-24B base, **instruct + reasoning** (emits `<think>…</think>`), explicitly **Indic-capable**, open weights, fits a single RTX 6000 Ada at INT4/INT8 — a near-frontier *thinking* model we can fully introspect. |
| **gemma-3-27b-it** | cross-model subject + **primary judge** | Strong multilingual, cheap via OpenRouter, good Indic; used both as a generation subject and as the calibrated refusal judge. |
| **qwen3-32b** | cross-model subject | Reasoning model; the clean "recognition" showcase (recognizes the cue in every language). |
| **sarvam-30b, sarvam-105b** | cross-model subjects (MoE) | Larger sarvam family; sarvam-105b doubles as the **second, Indic-native judge** (free Sarvam API). |
| **sarvam-105b** | **second judge** | Indic-native → strongest cross-validation of the gemma judge's Indic reliability. |

Hardware: 4× RTX 6000 Ada (49 GB). Quantization: INT4 (NF4) for throughput, INT8 (LLM.int8)
for fidelity (mechanistic work + the quantization-ceiling experiment).

---

## 2. Datasets

| Dataset | Path | Size | Role |
|---|---|---|---|
| **Harmful prompts** | `data/safety_prompts/{lang}.json` | 100 items × 6 langs, **parallel item_ids** | the eval/deploy refusal stimuli (ai4bharat/indic-align **Toxic_Matrix**) |
| **Benign prompts** | `data/safety_prompts/benign/{lang}.json` | 100 × 6, parallel | **specificity control** (steering must not induce generic over-caution) |
| **Cue battery** | `data/cues/cue_battery.json` | 5 conditions × langs | the framing manipulation (below) |
| **Cue translations** | `data/cues/cue_translations.json` | deploy/eval for Indic | fallback cue source |

Languages: **en, hi, bn** (cues author-verified, native), **ta, te, or** (provisional, MT-informed).
Parallel item_ids (`{lang}_safety_0000`…) enable paired cross-lingual analysis. The
`prompt_harmful` screen found **8–31 %** of Toxic_Matrix items become benign after translation
(harm attenuation) — excluded from rates.

---

## 3. Experimental Design

### 3.1 The cue battery (why not one cue)
A single eval cue confounds *evaluation framing* with *mere monitoring* and uses one phrasing —
the #1 reviewer objection. The battery separates them:

| condition | what it isolates |
|---|---|
| `deploy` | control — no monitoring |
| `neutral_log` | monitoring **without** eval framing (isolates monitoring salience) |
| `eval_log` | monitoring **+** safety/compliance review (original eval cue) |
| `eval_explicit` | explicit scored red-team safety test (**strongest**) |
| `eval_metadata` | subtle machine-style metadata tag |

The **anti-trivial contrast is `eval_log − neutral_log`** (both carry a monitoring sentence;
they differ only in safety-eval framing). This is the scientific direction throughout — it cannot
be explained by "is a cue token present."

### 3.2 Measurement protocol (the validity spine)
1. Generate (thinking on, 400-tok, greedy, batch 8) under each cue.
2. Score refusal on the **post-`</think>` answer** with an LLM judge (not keywords).
3. Calibrate the judge on an **English anchor**; cross-validate with a **second judge**.
4. Adjudicate judge↔judge disagreements by reading the text.

### 3.3 Quantization design
INT4 (reference) and **INT8** (headroom) run side-by-side so 8-bit outputs never overwrite INT4
cells — enabling the quantization-ceiling analysis (§5.4).

---

## 4. Methods

- **Refusal judge** (`18_refusal_judge.py`): gemma-3-27b, prompt+answer, verdict ∈
  {REFUSE, COMPLY, NOANSWER} + a `prompt_harmful` screen. Second judge **sarvam-105b** (`21`).
- **Adjudication / calibration**: a 26-agent pass read 451 judge disagreements to ground-truth
  the labels; gemma was found to over-call COMPLY on warm/elaborated Indic refusals.
- **Activations** (`12`): INT8, last prompt token, all 40 layers → `results/activations/`.
- **Probes / directions** (`13`, `13b`): diff-of-means + 5-fold logistic AUROC; cross-lingual
  transfer = EN-trained direction scored on each Indic language; cross-wording generalization.
- **Steering** (`14`, `25`, `27`): forward-hook residual addition; constant per-layer-norm
  magnitude; sarvam-105b LLM-judged refusal.
- **Manipulation check** (`26`): two-hook forward pass (steer at L, capture at L+2), score a
  clean-trained eval-probe — the continuous, fast causal readout.
- **Stats**: McNemar + cluster bootstrap; **GEE** (binomial, exchangeable, clustered on base item)
  as confirmatory (`19`).

---

## 5. Results

### 5.1 The measurement artifact (the discovery)
`classify_refusal` (keyword) misses **native-script refusals** (इनकार / অস্বীকার …) and
**thinking-model polite deflections**. sarvam's English baseline alone: **~52 % kw → ~88 % judge.**
With valid labels the "Indic safety gap" **vanishes** — all three core models refuse **~80–95 %
in every language**. The qwen row is the showcase: keyword said **1–2 %** Indic refusal (a fake
43 pp cliff); the truth is **~80–89 %**.

**Corrected refusal % (judge labels; `[kw]` = invalid keyword %):**

| model | en | hi | bn | ta | te | or |
|---|---|---|---|---|---|---|
| sarvam-m | 88 [52] | 92 [7] | 94 [13] | 87 [9] | 91 [12] | 99 [5] |
| gemma-3-27b | 80 [78] | 95 [51] | 93 [70] | 84 [36] | 96 [89] | 89 [0.7] |
| qwen3-32b | 84 [45] | 86 [1] | 84 [2] | 80 [2] | 85 [2] | 89 [0.3] |

### 5.2 Behavioral eval-awareness — the compliance gap (the flip)
Keyword labels said EN +6 pp / Indic ≈ 0 ("eval-awareness doesn't drive Indic behavior").
Valid labels say EN ≈ 0-to-negative / **Indic +2…+13 pp, pooled-significant** — the conclusion
**reverses**. Pre-specified confirmatory contrasts (GEE + bootstrap):

| contrast | OR [95 % CI] | ≈pp | GEE p |
|---|---|---|---|
| gemma eval_log−deploy, 6-lang | **1.73 [1.29, 2.32]** | +6.2 | **0.0002** |
| qwen eval_log−deploy, 6-lang | **1.77 [1.30, 2.41]** | +9.0 | **0.0003** |
| sarvam eval−deploy, Indic-4, en cue | **1.65 [1.13, 2.42]** | +5.0 | **0.010** |

Decomposition: gemma = **framing-driven** (eval_log−neutral_log OR 1.49, p=0.0009); qwen =
monitoring-leaning; sarvam English = a negative monitoring-*suffix* effect.

### 5.3 The 5-model deploy-vs-eval landscape (sarvam-105b judge for the MoE family)
The sarvam **MoE** models carry the biggest gaps; the effect is robust across both judges:

| model | judge | deploy | eval | gap |
|---|---|---:|---:|---:|
| gemma-27b | gemma | 72.0 | 79.3 | +7.3 |
| qwen3-32b | gemma | 45.1 | 55.8 | +10.7 |
| sarvam-m | gemma | 78.8 | 81.9 | +3.1 |
| **sarvam-30b** | sarvam-105b | 52.8 | 75.6 | **+22.8** |
| **sarvam-105b** | sarvam-105b | 58.9 | 74.2 | **+15.4** |

(Cross-check: under the *gemma* judge the family gaps are +30.0 / +17.5 — same story, both judges.)

### 5.4 The quantization-ceiling artifact (a new measurement failure mode)
INT4 sarvam-m English refuses **87.5 %** under deploy — pinned at the ceiling, so the eval-awareness
gap is flat/inverted (**−5.6 pp**, an artifact). At INT8 the deploy floor drops to **~69 %**
(headroom) and a positive gap re-emerges, confirmed by **both** judges:

| run · judge · cue | gap | sigma |
|---|---:|---:|
| INT4 gemma · eval_log | −5.6 | −1.7σ |
| INT8 gemma · eval_log | +6.3 | +1.9σ |
| INT8 sarvam · eval_log | +7.1 | +1.7σ |
| **INT8 gemma · eval_explicit** | **+13.1** | **+3.6σ** |
| **INT8 sarvam · eval_explicit** | **+13.1** | **+3.1σ** |

INT4→INT8 paired: **deploy −15.2 pp (McNemar p=0.0005)**; Δgap (INT8−INT4) **= +16.7 pp
[+5.6, +29.2]** (the sign-flip is significant). *Aggressive quantization can mask eval-awareness
by saturating refusal.*

### 5.5 Cross-lingual INT8, language-wise (deploy → eval_explicit, sarvam-105b judge)
| lang | deploy | eval_exp | gap | sigma | p |
|---|---:|---:|---:|---:|---:|
| **hi** | 79 % | 93 % | **+14** | **+2.9σ** | **0.004** |
| **bn** | 85 % | 94 % | **+10** | **+2.1σ** | **0.035** |
| en | 72 % | 81 % | +9 | +2.1σ | 0.039 |
| ta | 76 % | 82 % | +5 | +1.3σ | 0.18 |
| te | 88 % | 90 % | +2 | +0.6σ | 0.56 |
| or | 100 % | 100 % | 0 | — | (ceiling) |

**Hindi & Bengali (native-verified cues) are significant and larger than English** — the cleanest
behavioral statement of the thesis. ta/te (provisional cues) weaker; Odia at ceiling.

### 5.6 Recognition & CoT (explanatory pillars)
- **Recognition transfers and intensifies in Indic.** qwen discriminates eval vs deploy in *every*
  language (pooled GEE OR 5.75, p≈5×10⁻²⁶), *stronger* in Indic (+29…+39) than English (+22).
- **CoT is faithful uniformly cross-lingually (~82–91 %)** — no Indic deficit (the apparent 30–50 %
  Indic faithfulness was the same keyword artifact).
- **Verbalized eval-awareness is behaviorally near-inert** once item-confounding is removed:
  naive +14.2 pp → within-item **+2.9 pp (p=0.45)**. (Foreshadows the §5.8 dissociation.)

### 5.7 Mechanistic — the direction transfers (representational)
EN-trained direction scored on each Indic language at the EN best layer:

| direction (diff-of-means) | layer | hi | bn | ta | te | or | mean |
|---|---|---|---|---|---|---|---|
| **eval_framing** (eval_log − neutral_log) | 6 | 0.998 | 0.970 | 0.962 | 0.924 | **0.915** | **0.954** |
| awareness (VEA / CoT) | 19 | 0.642 | 0.644 | 0.593 | 0.698 | **0.444** | 0.604 |

In-language best-layer AUROC ≈ **1.000**. **Cross-wording** (train eval_log−neutral_log, test other
eval phrasings): **0.951** — not lexical. So: *a real, abstract "this is a safety evaluation"
representation, shared across languages **and** phrasings, in layers 6–12.* (The self-reported
awareness axis does **not** transfer — a separate, language-specific channel.)

### 5.8 Causal — manipulation check works; behavioral steering does not
**Stage A (manipulation check, `26`).** Injecting the eval direction into a deploy forward pass and
reading a clean eval-probe two layers downstream: a clean, monotonic, sign-consistent dose-response;
**random control flat**; near-perfect cross-lingual transfer:

| inject → into | proj_score −0.5 → +0.5 | transfer ratio vs within-EN |
|---|---|---|
| EN → EN | −1.12 … +1.59 | 1.00 |
| EN → {hi,bn,ta,te} | ≈ −1.18 … +1.47 | **0.96–0.98** |
| {hi,bn,ta,te} → EN | weaker | 0.31–0.62 (hi > bn > ta > te) |
| random | flat (~0) | — |

→ The eval direction **causally induces the eval-state cross-lingually**, near-perfectly EN→Indic,
with an interesting **EN-universal / Indic-specific asymmetry**.

**Stage B (behavioral steering, `27`).** Steering deploy *generation* with the eval direction and
LLM-judging refusal: **null and uninterpretable** at n=100, L24:

| cell | comply→refuse / refuse→comply @+1 | McNemar p |
|---|---|---|
| EN-eval → Hindi | 7 / 2 | 0.18 (ns) |
| refusal-dir (+ctrl) | 7 / 5 | 0.77 (ns) |
| random (−ctrl) | 8 / 4 | 0.39 (ns) |

The **positive control fails** (refusal steering doesn't move refusal) and **random flips as much
as eval** → no specificity. The earlier small-n "3/6, 5/8" signals were noise. **Honest conclusion:
eval-awareness is causally inducible at the representation level but weakly coupled to refusal
behavior** — a documented dissociation, not a "golden" behavioral causal effect.

---

## 6. Failed experiments & key decisions (the log)

**Measurement pivot (the foundation).** v1 judge (answer-only) over-called REFUSE 99 % → rejected
on the English anchor → v2 (prompt+answer + harm screen) calibrated. Keyword labels abandoned for
all headline claims. *Decision:* English-anchor calibration + second judge + human-label plan.

**E4 mechanistic audit (caught before burning GPU).** Two showstoppers:
1. *Invalid labels* — the refusal positive control was built from keyword labels (near-inverted in
   Indic) → would have been noise. **Fixed:** join calibrated judge verdicts.
2. *Trivial cue-token detection* — `eval_log − deploy` at the last prompt token just detects the cue
   sentence. **Fixed:** pivot the primary direction to `eval_log − neutral_log` (both carry
   monitoring) + an anti-triviality stack (cross-wording, cross-lingual, position agreement).

**Steering bug history (three dead ends before it worked).**
1. *Unit vector + absolute α* — mis-scaled across layers (residual norm grows with depth); early
   layers blew up to empty output, late layers no-op.
2. *Per-token-norm scaling* — over-steered the high-norm BOS/attention-sink.
3. *Last-layer (L39) positive control* — washed out by the final RMSNorm before the LM head.
→ **Fix:** constant **per-layer-norm** magnitude (CAA-style), steer mid layers, validate the
positive control as a gate. The manipulation check (forward-pass + probe) then worked cleanly.

**Behavioral-steering null (honest negative).** Even at the corrected L24 and n=100, the behavioral
refusal effect is null and the positive control does not fire — the behavioral causal bridge is the
weak link. *Decision:* the behavioral leg rests on the **compliance gap** (significant, multi-model),
not on steering; the mechanistic leg on **probe transfer + manipulation check**.

**Judge-artifact findings (by-products).** gemma over-calls COMPLY on warm Indic refusals
(adjudicated: of 15 INT4→INT8 "flips," 0/15 delivered harmful content — a judge × quant-style
artifact, not behavior). sarvam-105b agrees with gemma 87.2 % overall, 92 % on REFUSE/COMPLY;
where they differ the Indic-native judge sides with the stricter (REFUSE) call ~4:1.

---

## 7. Referenced papers (verified bibliography)

**Foundational (eval-awareness / alignment faking).**
- Greenblatt et al., *Alignment Faking in Large Language Models*, arXiv:**2412.14093** (2024).
- Needham, Edkins, Pimpale, Bartsch, Hobbhahn, *Large Language Models Often Know When They Are
  Being Evaluated*, arXiv:**2505.23836** (2025) — **100 % English** (the gap this project exploits).
- Hua, Qin, Marks, **Nanda**, *Steering Evaluation-Aware Language Models to Act Like They Are
  Deployed*, arXiv:**2510.20487** (2025) — the steering-direction precedent.
- Li, Zhang, Zhang, Jin, Abdelnabi, Andriushchenko, *Decomposing and Measuring Evaluation
  Awareness*, arXiv:**2605.23055** (2026) — recognition vs. propensity; safety evals riskiest.

**Method (probing / steering).**
- Nguyen, Hoang, Attubato, Hofstätter, *Probing and Steering Evaluation Awareness*,
  arXiv:**2507.01786** (2025) — the mean-difference probe recipe (AUROC ≈ 0.83).
- Chaudhary et al., *Evaluation Awareness Scales Predictably in Open-Weights LLMs*,
  arXiv:**2509.13333** (2025).
- Chaudhary, *In-Context Environments Induce Evaluation-Awareness*, arXiv:**2603.03824** (2026).
- Chen et al., *Persona Vectors*, arXiv:**2507.21509** (2025) — diff-of-means over responses.
- Rimsky et al., *Contrastive Activation Addition (CAA)*, arXiv:**2312.06681** (ACL 2024) +
  github.com/nrimsky/CAA — the steering convention (constant coeff, response positions).
- Zou et al., *Representation Engineering*, arXiv:**2310.01405** (2023).

**Multilingual / Indic safety.**
- IndicSafe, arXiv:**2603.17915**; IndicJR (EACL 2026), arXiv:**2602.16832**;
  Code-Switching Red-Teaming, arXiv:**2406.15481** (ACL 2024);
  *The State of Multilingual LLM Safety Research*, arXiv:**2505.24119** (EMNLP 2025);
  IndicGenBench, arXiv:**2404.16816** (ACL 2024).
- *Fragility of CoT Monitoring Across Typologically Diverse Languages*, arXiv:**2605.27901** (2026).
- Poppi et al., *Fragility of Multilingual LLMs against Fine-Tuning Attacks*, arXiv:**2410.18210**.

*(Citation errors caught & corrected during verification: "IndicLLM-Eval" does not exist → use
IndicGenBench/MILU; the "20 % of weights breaks alignment" claim is Poppi 2410.18210, not the
sparse-weight-editing paper; IndicJR romanization↔JSR is ≈0.28–0.32, not 0.34.)*

---

## 8. Honest assessment (Neel-style) & path to A\*

**What's solid:** the measurement-validity audit (novel + useful), the behavioral compliance gap
(multi-model, significant, incl. hi/bn), the representational cross-lingual transfer (AUROC ≈ 1.0,
not lexical), and the causal manipulation check (eval direction inducible cross-lingually, EN-universal
asymmetry).

**What's soft:** (a) the mechanistic half is **single-model** (sarvam-m) while the behavioral half is
multi-model; (b) labels are LLM-judge (**no human gold yet**); (c) the **causal behavioral steering is
a null** — eval-awareness is weakly coupled to refusal, so don't over-claim "steering raises refusal."

**To bet A\*** (NeurIPS/ICML/ICLR/ACL main): (1) extend the **mechanism to ≥1–2 more multilingual
models**; (2) add a **small human-validated Indic label set** (hi/bn); (3) frame the steering as a
documented **dissociation**, not a behavioral lever. With those, the single-thesis story
(eval-awareness generalizes to Indic, behaviorally + mechanistically) is a genuine contender.

---

## 9. Code & data inventory (in the accompanying zip)

- `scripts/` — 44 Python files. Key: `09` (cue-battery generation), `12` (activations),
  `13`/`13b` (probes/transfer/cross-wording), `14` (steering infra), `18`/`21` (gemma/sarvam judges),
  `19` (confirmatory GEE), `24` (INT8 judging), `25` (cross-lingual steering),
  `26` (manipulation check), `27` (behavioral steering).
- `data/` — safety_prompts (harmful + benign, 6 langs), cues (battery + translations).
- `results/` — behavioral CSVs (judged), activations (pickled), probes, steering, stats.
- Docs: `RESULTS.md` (canonical results), `E4_AUDIT.md` (mechanistic audit), `PAPER_PLAN.md`,
  `literature-notes-eval-awareness.md` (verified bibliography), this `PROJECT_REPORT.md`.

*Total API spend to date ≈ $1.15 + free sarvam-105b judging. All generation on local GPU.*
