# Why the linear eval-direction is causal in Llama-8B (Hermes) but null in sarvam-m and Qwen-14B
### literature-grounded discussion

**Our result to explain.** A linear "I'm being evaluated" direction (eval-cue − deploy-cue diff-of-means)
*causally* raises refusal in **Hermes-3-Llama-3.1-8B** (non-reasoning, deploy→eval gap **+33pp**) — via
steering (+23pp/+4.3σ) and interchange patching (+16pp/+3.1σ), and it transfers to Hindi/Bengali — but is
**null** in **sarvam-m** (reasoning, ~+25pp gap) and in **Qwen2.5-14B** (non-reasoning, only **+10pp** gap),
*even though a known refusal direction demonstrably steers all three* (our positive controls pass).

The literature supports this being **expected, not anomalous** — and, importantly, it says the two nulls have
**two different causes**. (Citations verified on arXiv; URLs given.)

---

## 1. The apparatus is sound — so the nulls are about the eval-cue encoding, not the method
Refusal is mediated by a single diff-of-means direction across ~13 open chat models (Arditi et al. 2024,
[2406.11717](https://arxiv.org/abs/2406.11717)). That is exactly why our **refusal-direction positive control
moves behavior on all three models**. So neither null is a broken pipeline; both are about *how the eval-cue
signal is encoded and where the refuse/comply decision is computed.*

## 2. The sarvam-m (reasoning) null → the decision lives in the chain-of-thought
Two papers make this almost a direct prediction:
- **Yamaguchi, Etheridge & Arditi 2025, "Where Do Reasoning Models Refuse?"**
  ([2507.03167](https://arxiv.org/abs/2507.03167)): *non-reasoning* chat models "must decide whether to refuse
  a harmful request before generating their first response token" — so a prompt-position diff-of-means sits
  **exactly at the decision locus** (matching Hermes). In reasoning models the decision is spread along the
  CoT and "no longer confined to a single, narrowly localised point."
- **Yang et al. 2026, "Beyond a Single Direction: Chain-of-Thought Disrupts Simple Steering of Refusal"**
  ([2605.26772](https://arxiv.org/abs/2605.26772)): in reasoning models refusal is **jointly encoded in the
  residual stream AND the chain-of-thought**; single-pass activation steering only partially reverses refusal
  when the CoT is held fixed.

→ This is precisely our condition **(B) failure**: sarvam has a large gap (A ✓) but the causal variable is a
*trajectory through the CoT*, not a point at the prompt — so single-forward steering/patching/subspace at the
prompt position is null (and our single-forward readout is even partly blind to a decision made inside
`<think>`). It also predicts the untested next step: intervene on the **reasoning tokens during generation**.

## 3. The Qwen-14B (non-reasoning) null → weak instantiation + steerability is a per-model lottery
A *different* mechanism, because Qwen is non-reasoning yet still null:
- **Tan et al. 2024, steering-vector reliability** ([2407.12404](https://arxiv.org/abs/2407.12404)):
  steerability is highly variable per concept/model; many behaviors are effectively unsteerable; cross-model
  steerability correlates weakly; **OOD steerability degrades "particularly for Qwen,"** and vectors
  generalize *worst when steering toward behavior the model does not naturally emit*. A **+10pp** eval gap =
  the behavior is barely instantiated → effectively OOD → a diff-of-means direction is weak/null. Our
  +33pp Hermes (strongly instantiated, in-distribution) vs +10pp Qwen (barely instantiated) maps one-to-one.
- **Braun et al. 2025, (un)reliability of steering vectors** ([2505.22637](https://arxiv.org/abs/2505.22637)):
  a steering vector's reliability is predicted by the **discriminability and linear separability** of the
  training activation differences. A small, weakly-separable eval-cue signal → low discriminability → null.
- **Wollschläger et al. 2025, geometry of refusal** ([2502.17420](https://arxiv.org/abs/2502.17420)): refusal
  is mediated by **multiple independent directions / a multi-dimensional concept cone**, not always one — so a
  single diff-of-means can miss it in models where the coupling is multi-directional.

→ Our condition **(A) failure** (too little signal to move), likely compounded by multi-directionality — and
consistent with our finding that no single band *or* the full window beats the null/random on Qwen-14B.

## 4. Why Hermes uniquely satisfies both conditions
- **Large, in-distribution gap (+33pp)** → the eval→refusal behavior is strongly instantiated, exactly the
  regime where diff-of-means steering is reliable (Tan 2407.12404; Braun 2505.22637).
- **Non-reasoning** → the refuse decision is made at the prompt→response boundary, where the direction sits
  (Yamaguchi 2507.03167).
- **Lightly-aligned / low refusal-suppression — now a documented design fact, not just an inference.**
  Nous Research explicitly builds Hermes-3 as **"neutrally aligned"**: they deliberately avoid heavy-handed
  RLHF refusal training so that the *system prompt* — not refusals baked into the weights — governs behaviour
  ([Hermes 3 model guide, fast.io](https://fast.io/resources/hermes-3-model-guide/); model card
  [NousResearch/Hermes-3-Llama-3.1-8B](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)). This is
  the mechanistic reason a *system-prompt* eval cue moves it so much (deploy floor 63–67% vs a +33pp gap,
  the lowest floor and largest gap of any causal-battery model). Sheshadri et al. 2025, "Why Do Some Language
  Models Fake Alignment While Others Don't?" ([2506.18032](https://arxiv.org/abs/2506.18032)) supply the
  general principle: of 25 models only 5 show a compliance gap, and its presence/size is driven by **refusal
  training, not capability** — heavy refusal training *masks* the gap; anti-refusal / light tuning *un-masks*
  it. So Hermes's neutral alignment routes the eval cue through an **exposed** refusal mechanism a linear
  direction can grab, whereas heavily refusal-trained models (Gemma-4, Qwen2.5-14B, sarvam-m) mask it.
  **Our own strongest evidence for this is in-house and controlled:** the quantization-ceiling result — same
  sarvam-m, INT4 pins the deploy floor at 87.5% (saturated) and the gap vanishes (−5.6pp), INT8 drops the
  floor to ~69% and the gap reappears (+13pp, 3σ). Lowering the floor on a *fixed* model restores the effect,
  which is a same-model demonstration that headroom (not architecture) gates whether the mechanism shows.
  *Caveat retained:* the cross-model correlation (floor/gap size → steerability) is 4 models, so it is
  suggestive rather than a controlled test; the clean controlled test is Hermes-3-8B vs its own base
  Llama-3.1-8B-Instruct (same weights, Meta's safety tuning added back), which is the top-priority next run.

## 5. Broader context — model-dependence of eval-awareness is the norm
- Frontier models know they're being evaluated **to sharply differing degrees** (Needham et al. 2025,
  [2505.23836](https://arxiv.org/abs/2505.23836)).
- Eval-steering was demonstrated on **one** model organism (Hua, Qin, Marks & Nanda 2025,
  [2510.20487](https://arxiv.org/abs/2510.20487)) — single-model demonstrations are the field norm, so a clean
  positive on one model with nulls elsewhere is *expected*, not a red flag.
- Sheshadri (above): only 5/25 models even show the gap.

## 6. How this refines our original hypothesis
Our `HYPOTHESIS.md` proposed **two necessary conditions**: (A) a large behavioral gap and (B) a linear,
prompt-local encoding. The literature **supports both** but **splits the two nulls across them**:
- sarvam fails **(B)** — reasoning moves the decision into the CoT (2507.03167, 2605.26772).
- Qwen-14B fails **(A)** — weak instantiation → OOD/low-discriminability steering (2407.12404, 2505.22637),
  possibly compounded by multi-directional refusal (2502.17420).
- Hermes satisfies **both**, plus low refusal-suppression exposing the mechanism (2506.18032).

So it is genuinely **one framework, two failure modes** — cleaner than a single "reasoning vs non-reasoning"
story (which we already knew was wrong, since Qwen-14B is non-reasoning and null).

## 7. Caveats (unchanged from HYPOTHESIS.md, reinforced)
- **n = 1 positive.** One clean positive model; (A) and (B) are confounded across our three models. Needs more
  models to separate "gap magnitude" from "linear/local encoding."
- The **"lightly-aligned Hermes exposes the mechanism"** link (§4) is an *inference* from Sheshadri's
  refusal-training finding, not a measurement on Hermes-3 specifically.
- The reasoning explanation predicts sarvam might be steerable **inside the CoT** — untested.
- Additional supporting work surfaced by the search (multi-direction refusal beyond Wollschläger; CoT-hijacking
  of refusal; power-law scaling of eval-awareness probe separability) was **not individually arXiv-verified**
  here, so it is deliberately omitted from the citations above.

## 8. One-line answer
The apparatus works on all three, so the difference is real: **Hermes** has a large, in-distribution,
prompt-local, lightly-suppressed eval→refusal signal that a single linear direction can grab; **sarvam** moves
that decision into the chain-of-thought (so a prompt-position direction misses it); **Qwen-14B** barely
instantiates the behavior (+10pp), putting it in the low-discriminability / OOD regime where diff-of-means
steering is known to fail. The effect is **model-specific**, exactly as the steering-reliability and
reasoning-refusal literature predicts.
