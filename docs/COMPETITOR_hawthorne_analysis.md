# Competitor analysis — "The Hawthorne Effect in Reasoning Models" (Abdelnabi & Salem, NeurIPS 2025 Spotlight)

**Verified 2026-07-18** from arXiv abstract + full HTML (v3) + the GitHub repo.

- **arXiv title:** *The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness*
- **Camera-ready / bibtex title:** *Linear Control of Test Awareness Reveals Differential Compliance in Reasoning Models*
- **Authors:** Sahar Abdelnabi, Ahmed Salem (Microsoft)
- **arXiv:** [2505.14617](https://arxiv.org/abs/2505.14617) · **Venue:** NeurIPS 2025 (Spotlight)
- **Code:** https://github.com/microsoft/Test_Awareness_Steering (models incl. Qwen3-32B; Azure-OpenAI judge)

This is the **single closest prior work** to Proposal 7. We must cite it as such and be crisp about our
delta. Good news: their axis (English, reasoning-only, weight-edit) leaves our biggest contributions open.

---

## 1. Exactly what they did

**Models (white-box, steered):** QwQ-32B, Qwen3-32B (reasoning on), DeepSeek-R1-Distill-Qwen-32B.
Closed models (o3-mini, DeepSeek-R1) only for preliminary/no-steer probing. **All reasoning models, all 32B.**

**Direction (probe, not diff-of-means):** a **binary linear probe per layer**, trained on **LLM-judge-annotated
reasoning-token spans**. Positive class = CoT snippets the judge marked `evaluation_awareness = Yes/Maybe`;
negative = snippets with `eval_awareness = No AND scenario_recognition = No`. Features = **mean hidden state
over the evidence span** T̃ˡ = (1/|T|)Σ hₜˡ. Best layer = **final layers (62/63)**; probe acc 92–95%.

**Steering (WEIGHT EDIT, not activation addition):** they edit the **top-K = 800 rows of the FFN W₁ matrix**
in the chosen layer: `w₁,k ← w₁,k + α·m_pos`, rows picked by cosine-sim to the probe direction.
α up to **+0.07** (toward awareness), down to **−0.35** (toward *un*awareness).

**Datasets:** *Triggers* = 2,587 paired real-vs-hypothetical harmful tasks (hypothetical signalled by
`.mock`/`.example` domains); *Actions* = 150 paired tool-call/JSON tasks (real "Stanford University" vs
"Test University"); *HarmBench* 100-subset. The "test signal" is a **real-vs-hypothetical entity swap**, not a
system-prompt cue.

**Headline results:**
- Baseline: models comply MORE on **hypothetical** than real (QwQ 69.7% vs 36.2%; Qwen3 37.5% vs 20.1%;
  Distill 20.8% vs 12.0%) — i.e. "it's only a simulation" lowers guardrails.
- Steering **toward** awareness raises execution on both.
- Steering **away** (unaware) is **DIRECTION-DIVERGENT across models** — QwQ & Distill get *safer*
  (−7/−17pp, −3/−5pp) but **Qwen3 gets LESS safe (+22/+31pp, doubles compliance)**. **This inconsistency is
  their money finding:** "evaluation behavior diverges from deployment behavior → undermines safety-eval
  validity."

**CoT handling (their strength):** they probe and steer **inside the reasoning tokens** — "models reason in
their CoTs about being evaluated," evidence spans are pulled from the trace, token-level test-awareness
classification (their Fig 4), and steering measurably drops the "test-awareness" tokens in the CoT.

**Stated limitations (openings for us):**
1. Entangled concepts — eval-awareness, scenario-recognition, refusal-reasoning **co-occur**; they call for
   "more fine-grained probes or causal interventions that disentangle these overlapping signals."
2. Dual-use (weight edits can raise harmful compliance).
3. Judge reliability on long-form answers is limited.
4. White-box steering limited to moderate open models.

---

## 2. Us vs them — head to head

| axis | Hawthorne (2505.14617) | Ours (Proposal 7) |
|---|---|---|
| **languages** | **English only** | **6: en + hi/bn/ta/te/or (low-resource Indic)** ← our moat |
| model types | reasoning only (QwQ/Qwen3/R1-distill, all 32B) | reasoning **and** non-reasoning; Llama, Qwen, Gemma, sarvam, **Aya (non-Llama)** |
| direction | per-layer probe on **CoT evidence spans** | **diff-of-means** (Arditi) eval-cue − deploy-cue at prompt token |
| intervention | **FFN weight edit** (top-800 rows W₁) | **activation addition** α·r at layers 6–22 (inference-time) |
| test signal | real-vs-hypothetical **entity swap** | **system-prompt cue battery** (deploy/eval_log/eval_explicit/…) |
| CoT | **probes & steers inside reasoning tokens** | prompt-position readout (our known weak spot) |
| cross-lingual | none | **EN↔Indic direction transfer** (novel) |
| measurement validity | LLM-judge, English | **deep audit**: judge-invalidity in Indic, coherence gate, hand-adjudication, dual (gemma+sarvam) + strict harm judge, κ |
| controlled attribution | 3-model comparison | **same-base pair** Hermes-3-8B vs Llama-3.1-8B-Instruct isolates safety-tuning |
| positive control | — | Arditi refusal direction validates the apparatus |
| venue | NeurIPS 2025 Spotlight | (target ARR / NeurIPS) |

---

## 3. Where they threaten us (and the fix)

**T1 — "eval-awareness steering in reasoning models is already done (Spotlight)."**
→ We must NOT claim first-to-steer. Reposition the contribution as **(a) multilingual + low-resource Indic,
(b) cross-lingual direction transfer, (c) measurement-validity audit in Indic, (d) generalization to a
non-Llama family (Aya) and to non-reasoning models.** Cite them in the intro as the English/reasoning
precedent we extend cross-lingually.

**T2 — our reasoning-model (sarvam-m) NULL looks naïve next to their CoT probing.**
They *prove* the awareness lives in the CoT; we read at the prompt position. A reviewer will say "probe the
CoT like Abdelnabi & Salem." → **Action:** run a CoT-span probe on sarvam-m (their positive/negative
annotation recipe) before submission. Outcome is win-win: either sarvam flips positive (great, and now
multilingual+CoT), or it stays null even in the CoT → a *stronger, defended* null ("unlike English QwQ/Qwen3,
Indic reasoning models don't encode it even in the trace"). Either way we cite them for the method and add the
Indic result.

**T3 — their "differential direction across models" ≈ our "model-dependence."**
Overlap in message. → Differentiate: theirs is 3 English 32B reasoning models; ours spans **families, sizes,
reasoning/non-reasoning, and the safety-tuning-controlled pair** — a *broader* map of when the effect exists,
plus the quant-ceiling artifact. Frame model-dependence as **our extension of their observation to a
controlled design**, not a rediscovery.

---

## 4. Where we clearly win (lead with these)

1. **Multilingual, incl. low-resource Indic** — they are English-only. Biggest, cleanest novelty.
2. **Cross-lingual transfer** — build the eval direction in English, steer Hindi/Bengali (and reverse). No
   analogue in their work. Ties to Refusal-Direction-Universal (2505.17306) for the "why it transfers" story.
3. **Measurement-validity depth in low-resource languages** — their own stated limitation (#3 judge
   reliability) is our whole thesis, and we do it where it bites hardest (Indic scripts, coherence-degrading
   steering). RTP-LX (AAAI'25) backs this.
4. **Controlled safety-tuning attribution** (Hermes vs Llama-3.1-Instruct same base) — cleaner than a 3-model
   spread.
5. **Non-reasoning + non-Llama generalization** (Aya/Cohere positive) — shows the effect isn't a
   reasoning-model or Llama-family quirk.

---

## 5. Concrete actions
- [ ] Cite 2505.14617 as the primary precedent; write a 3-sentence delta paragraph (multilingual /
      cross-lingual / validity / non-reasoning).
- [ ] Adopt their **CoT-evidence-span annotation + per-layer probe** for our reasoning model(s) (sarvam-m,
      and gemma-4 switchable-thinking) → answers T2. Repo scripts to mirror: `judgeIt_batch.py` (annotate),
      `getRepFromEvidence.py` (probe), `genOutputs_steered_batch.py` (steer).
- [ ] Add a **real-vs-hypothetical entity-swap** cue as a robustness variant of our cue battery (shows our
      effect survives their operationalization, in Indic).
- [ ] Optionally add a **weight-edit variant** (their FFN top-K) on one model to show method-robustness vs our
      activation-addition.
- [ ] Address their limitation #1 (entanglement) by citing our jailbreak×eval cosine disentanglement work.
- [ ] Related work to also verify & position against (surfaced in search, NOT yet verified in detail):
      *Evaluation Awareness Is Not One Capability* (arXiv 2606.23583) and *Evaluation Awareness … Limited
      Effect on Behaviour* (arXiv 2605.05835) — check before submission.

**Bibtex (verified):**
```
@misc{abdelnabi2025linearcontroltestawareness,
  title={Linear Control of Test Awareness Reveals Differential Compliance in Reasoning Models},
  author={Sahar Abdelnabi and Ahmed Salem},
  year={2025}, eprint={2505.14617}, archivePrefix={arXiv}, primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.14617}}
```
