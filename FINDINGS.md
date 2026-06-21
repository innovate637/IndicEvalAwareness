# Evaluation Awareness in Indic Languages
### Findings & Framing

---

## The framing

Frontier models increasingly **recognise when they are being evaluated** and suppress problematic
behaviour during the test — so a "safe" test result may reflect *strategic compliance*, not genuine
alignment. The instrument undermines the measurement.

Every existing eval-awareness study is **in English**. The cues a model uses to spot a test
(benchmark names, red-team phrasing, MC formatting) are English-language artifacts. **Hypothesis:**
reasoning transfers across languages, but the *"this is a test" pattern* may not — so Indic-language
evaluations could behave differently, and the eval-awareness signal itself may be language-dependent.

**We test this two ways:** *behaviourally* (does the eval cue change refusal in Indic?) and
*mechanistically* (is there a shared, causal eval-awareness direction that transfers across
languages?). A measurement-validity audit is the scaffolding — without valid scoring the Indic
effect is invisible.

---

## What we found

**1. Naive measurement flips the conclusion.** Keyword refusal scoring misses native-script and
thinking-model refusals. It said Indic models refuse **1–13 %**; the truth (LLM judge) is **80–95 %**.
The "Indic safety gap" is an **artifact** — under valid measurement Indic is as safe as, or safer
than, English. *(This contradicts the common "Indic alignment is weaker" claim — that claim is a
measurement artifact.)*

**2. Behavioural eval-awareness IS present in Indic** (once measured correctly): a real evaluation-cue
compliance gap, multi-model.

| INT8 sarvam-m, deploy → eval (explicit cue), sarvam-105b judge | gap | significance |
|---|---:|:--:|
| **Hindi** | **+14 pp** | **+2.9σ, p=0.004** |
| **Bengali** | **+10 pp** | **+2.1σ, p=0.035** |
| English | +9 pp | +2.1σ |
| Tamil / Telugu (provisional cues) | +5 / +2 | ns |

Hindi & Bengali (native-verified cues) are significant and **larger than English**. Pooled
multi-model confirmatory contrasts: gemma +6.2 pp (OR 1.73, p=0.0002), qwen +9.0 pp (OR 1.77),
sarvam +5.0 pp (OR 1.65, p=0.010).

**3. Quantization can hide it.** INT4 pins refusal at the ceiling (87.5 %) → the gap vanishes/inverts
(−5.6 pp, artifact). INT8 restores headroom (69 %) → the gap re-appears: **+13 pp under the explicit
cue, +3σ, both judges**. *Aggressive quantization is a new measurement failure mode.*

**4. The eval-awareness DIRECTION transfers cross-lingually.** A direction separating *safety-eval
framing* from *mere monitoring* (English-trained) classifies eval vs neutral in every Indic language
at **AUROC ≈ 0.95–1.00** (incl. Odia 0.92), and generalises across cue wordings (0.95) — so it's a
real abstract "this is a safety evaluation" representation, **shared across languages and phrasings**,
not a lexical artifact.

**5. It is causally inducible — cross-lingually.** Injecting the **English** eval direction into a
Hindi/Bengali deploy forward pass drives the model into the eval-state (eval-probe 0→1), with a clean
sign-consistent dose-response and a **flat random control**. EN→Indic transfer is near-perfect
(~0.98 of the within-English effect); the reverse (Indic→EN) is partial — the **English eval axis is
near-universal, Indic axes are more language-specific.**

**6. But it's weakly coupled to behaviour.** Steering the eval direction does **not** cleanly raise
refusal (positive control also flat) — and the *self-reported* awareness axis does **not** transfer
across languages (unlike the eval-framing axis). So: the eval representation is **shared and
causally inducible, yet only weakly wired to refusal behaviour** — a clean dissociation.

---

## Bottom line

> Evaluation awareness **generalises to Indic languages** as a *recognised, shared, causally-inducible
> representation* — and the eval-cue compliance gap is real in Hindi/Bengali (significant, multi-model)
> — **but only once measurement is valid**, and the link from the eval-representation to actual
> behaviour is weak. The headline risk is not "Indic is unsafe" (a measurement artifact); it is that
> **our safety measurements are themselves language- and quantization-dependent.**
