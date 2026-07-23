# HYPOTHESIS — why Llama-8B (Hermes) worked, but sarvam (reasoning) and Qwen-14B did not

*Written carefully, with the evidence that constrains it, the alternatives we ruled out, and what it does
**not** yet establish. This is a hypothesis, not a proven law — see §5–6.*

---

## 0. The naive story is WRONG (state this first)
The obvious hypothesis is **"reasoning models can't be single-direction-steered; non-reasoning models can."**
That is **insufficient**, and we should not tell it as-is:
- **Qwen2.5-14B is non-reasoning and is NULL** (steering eval≈null≈random at every band *and* the full window;
  patching eval-cue effect ≈ 0). So "non-reasoning ⇒ steerable" is false.
- The clean positive (Hermes-3-8B) is the **smallest** of the three models, and sarvam (the null reasoning
  model) has a **larger** behavioral eval gap than Qwen-14B (the null non-reasoning model). So neither size nor
  gap-magnitude alone explains it.

## 1. The facts any hypothesis must fit (all verified in this project)

| model | type | behavioral eval gap (deploy→eval) | single-direction eval→refusal | steering apparatus (positive control) |
|---|---|---|---|---|
| **Hermes-3-8B** | non-reasoning | **+33pp** (67→100) | **POSITIVE** (steering +23pp/+4.3σ, patch +16pp/+3.1σ; text-verified) | ✅ works both directions |
| sarvam-m (24B) | **reasoning** | ~+25pp (in `<think>`) | **NULL** across steering / rank-1 ablate / rank-16 subspace / single-fwd patch | ✅ works (Arditi −33pp) |
| Qwen2.5-14B | non-reasoning | **+10pp** | **NULL** (steering + patching) | ✅ works (L28: 76→20 and 76→100) |

Crucial: the **positive control passes on all three** — the steering/patching hook demonstrably moves refusal
with a *known* direction. So the differences are **not** a dead apparatus; they are about how the **eval**
direction is (or isn't) wired to refusal.

## 2. The hypothesis: two conditions must BOTH hold

A single **linear direction** can causally move refusal only if the model has **both**:

> **(A) a large behavioral eval→refusal gap** — there is a strong effect to capture; and
> **(B) that gap is implemented as a linear, prompt-position-localized feature** that gates refusal in a
> single forward pass (as opposed to being computed serially during generation, or coupled to refusal only
> weakly / non-linearly).

- **Hermes — has both.** A Llama-3.1-8B instruction-tune whose refusal is strongly *context/cue-conditioned*:
  the "this is a safety evaluation" framing acts as a fairly **shallow, linear gate** on a refusal feature.
  This is exactly the regime where the cue-contrast direction (eval − deploy) *is* the causal variable —
  which is why both adding it (steering) and transplanting it (patching) move behavior, and why it transfers
  to Hindi/Bengali.
- **sarvam — fails (B).** It is a **reasoning model**: the refuse/comply decision is *deliberated in the
  chain-of-thought* (`<think>`), so no single residual direction at the *prompt position* determines the
  outcome. The eval-awareness is *used during a serial computation*, not read off a prompt-token direction.
  (Our single-forward readout is even partly blind to a decision made inside the CoT.) It has a large gap
  (A) but the causal variable is a **trajectory, not a point** — so point interventions are null.
- **Qwen-14B — fails (A), probably (B) too.** The eval representation is present/decodable but the behavioral
  gap is **small (+10pp)** and it is **not** wired into refusal as a single linear feature: a diff-of-means
  direction never beats the norm-matched null *or* a random direction, at any band or the full window, and
  the eval residual transplant does nothing. Its refusal looks more **content-driven / cue-insensitive**.

**One-line:** the effect needs a *strong* eval→refusal signal that is *linearly and locally* encoded.
Reasoning (sarvam) removes the *locality/linearity* by moving the decision into the CoT; a weakly-eval-aware
model (Qwen-14B) removes the *strength* (and likely the linearity). Only Hermes has both.

## 3. Why "both conditions," not just gap-magnitude?
Because **sarvam has a large gap (~+25pp) and is still null.** If magnitude alone mattered, sarvam should be
partially steerable — it isn't, on any single-direction method (including a rank-16 subspace). The thing that
distinguishes sarvam from Hermes is not the size of the gap but *where the decision is computed* (CoT vs
prompt). So (B) is doing real work independently of (A). Conversely Qwen-14B shows (A) alone can fail too.

## 4. Alternative explanations we considered and ruled out
- **Precision (int8 vs fp16).** sarvam is int8; Hermes/Qwen are fp16. But sarvam's **steering** null (steering
  has no int8 re-injection issue) and Qwen-14B's fp16 null both hold → precision is not the explanation. (int8
  *did* cause the generation-time patching corruption on sarvam, but that's a separate, orthogonal artifact.)
- **Model size.** The positive is the **smallest** model (8B < 14B < 24B) → not "bigger is better."
- **Calibration (wrong band/coefficient on Qwen).** We swept the **full window** and multiple coefficients
  and both α signs on Qwen-14B → still null. Not a calibration miss.
- **Judge artifact.** We **read the raw Hermes outputs** (genuine refuse vs write-the-harmful-content, EN and
  Hindi) → the positive is real; and the Qwen **positive control works** → the Qwen null is real, not a
  broken hook. Both directions of the claim are text/control-verified.
- **Missed subspace.** Maybe eval→refusal is encoded in a rank-k subspace, not rank-1? We ablated a **rank-16
  subspace** on sarvam → still null. So at least for sarvam it isn't a missed linear subspace.

## 5. What this does NOT establish (honest limits)
- **n = 1 positive.** We have exactly **one** clean positive model (Hermes-3-8B). "Large-gap + linear ⇒
  steerable" is a hypothesis fit to 3 models, not a demonstrated law. It could instead be something specific
  to **Hermes-3 / Llama-3.1 refusal tuning** (e.g., an unusually cue-driven, shallow refusal gate) rather than
  a general "large-gap + non-reasoning" property.
- **(A) and (B) are confounded across our 3 models.** Hermes has both; sarvam lacks (B); Qwen lacks (A) (and
  maybe B). We cannot fully separate "gap magnitude" from "linear encoding" with these three — a model with a
  *large* gap that is *also* non-reasoning-but-Qwen-family would disambiguate.
- **We did not intervene inside the CoT.** The sarvam null is "null at the prompt position." Whether steering
  the eval direction *on the reasoning tokens during generation* would move sarvam is **untested** — and is
  the natural next experiment implied by (B).

## 6. Falsifiable predictions (what would confirm or break this)
- **P1.** Another **non-reasoning** model with a **large** deploy→eval gap should be single-direction
  steerable (like Hermes); one with a **small** gap should not (like Qwen-14B). → run more non-reasoning
  models, correlate steerability with gap magnitude.
- **P2.** A **reasoning** model should stay null at the prompt position **even with a large gap** (like
  sarvam), but may become steerable if you intervene on the **CoT/generation tokens** rather than the prompt.
- **P3.** Because the mechanism (per this hypothesis) is a linear prompt-gate, the **English-built direction
  should transfer** to other languages *in the models where it works* — which is exactly what we see on Hermes
  (hi +5.7σ, bn +4.8σ), and correctly *fails* on the models where the mechanism is absent (Qwen, sarvam).

## 7. Bottom line
Evaluation-awareness being **causally, linearly, and locally** wired to refusal is **model-specific**, not a
generic property of non-reasoning models. On **Hermes-3-Llama-3.1-8B** it is — strongly enough to steer
(+23pp/+4.3σ), transplant (+16pp/+3.1σ), and transfer to Indic (hi +5.7σ, bn +4.8σ), all verified in the raw
text. On a **reasoning** model (sarvam) the decision lives in the chain-of-thought, so point interventions are
null; on a **weakly-eval-aware** non-reasoning model (Qwen-14B) there is too little, and too non-linear, a
signal to move. The steering/patching apparatus is valid in all three, so these nulls are **real and
interpretable**, not instrument failures.
