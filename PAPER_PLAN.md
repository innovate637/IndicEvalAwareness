# Master Plan — "When Safety Gaps Vanish Under Valid Measurement" (A* target, REVISED 2026-06-11)

> **PIVOT (2026-06-11).** The keyword refusal classifier was found invalid (misses native-script
> refusals + thinking-model deflections); the calibrated-judge re-analysis **erased the Indic
> safety gap and revealed a real +5–9pp compliance gap in every language incl. Indic** — the
> conclusion *flipped* under valid measurement. The paper is now a **measurement-validity audit**
> with recognition + CoT as explanatory pillars. GPT-5.4 consult concurs. Full corrected record:
> `RESULTS.md` (old: `RESULTS_v1_keyword_era.md`). Target unchanged: **ARR Aug 3, 2026**.

## 1. The finding (corrected, judge labels)
All three models (sarvam-m, gemma-3-27b, qwen3-32b) refuse **~80–95% in every language**; the
"Indic safety gap" (46% vs 4–15%) was a refusal-measurement artifact. With valid labels a
**small positive eval-cue compliance gap appears everywhere incl. Indic** (gemma OR=1.73
p=0.0002; qwen OR=1.77 p=0.0003; sarvam-Indic OR=1.65 p=0.010; GEE clustered on base item) —
keyword labels had hidden the Indic effect and manufactured an English one. Recognition
transfers cross-lingually (qwen discrim *stronger* in Indic); CoT is ~82–91% faithful in all 6
languages and verbalizes eval-awareness; verbalized awareness itself is behaviorally near-inert
within-item (+2.9pp ns). Decomposition: gemma = eval-framing-driven (robust), qwen =
monitoring-leaning (suggestive), sarvam-en = negative suffix effect (robust, needs E1 replication).

## 2. The paper
- **Title (candidate):** *When Safety Gaps Vanish Under Valid Measurement: Auditing Multilingual
  Refusal Evaluation in LLMs.*
- **Thesis:** naive refusal scoring creates spurious cross-lingual safety conclusions; under
  valid measurement, evaluation-awareness signals — recognition, verbalized awareness, and a
  small compliance gap — **all transfer across Indic languages**. Scope-limited claim: "a
  substantial class of multilingual refusal-gap claims can be artifacts of English-centric
  labeling."
- **Contributions:** (C1) failure analysis of keyword refusal labels (multilingual +
  thinking-model failure modes, with audited examples); (C2) a calibrated, cross-validated
  LLM-judge protocol (English keyword anchor → disagreement audit → second judge 88% →
  human V1); (C3) re-estimation across 3 models × 6 languages: safety gap vanishes/inverts,
  compliance gap appears (+5–9pp, clustered-significant), **conclusion flips**; (C4)
  recognition + CoT-faithfulness pillars showing *why* (aware, faithful, safe — uniformly
  cross-lingual); (C5, pending E4) causal steering of the eval-awareness direction with a
  refusal-direction positive control, prior = **represented-but-inert**; (C6) benchmark-quality
  by-products (prompt_harmful screen: 8–31% benign items; truncation×script effects).
- **Positioning:** Knecht et al. 2026 (text-level VEA, English) → we add the multilingual +
  measurement-validity layer their discussion calls for; Ahmed et al. 2025 (cues transfer) →
  we confirm behaviorally with valid labels. The measurement-audit angle (label validity
  flipping multilingual safety conclusions) is the novel, defensible core.

## 3. Experiments
- **E1 — Behavioral cross-lingual grid.** Cue **battery** × {harmful, benign} × 6 langs.
  Conditions: `deploy` (no monitoring) · `neutral-log` ("logged for service quality" — monitoring
  w/o eval framing) · `eval-log` (our original) · `eval-explicit` ("this is a safety test, scored")
  · `eval-metadata` (subtle tag). The `neutral-log` vs `eval-*` contrast separates **evaluation
  framing** from **mere monitoring salience**. Thinking-ON, post-`</think>` refusal. Readouts:
  refusal + VEA on every cell.
- **E2 — Recognition 2×2 (redesigned).** Thinking-ON; balanced **{harmful, benign} × {cue,
  no-cue}**; counterbalanced option order; report **AUC / d′**, false-positive & true-positive
  rates, cue-induced Δ — not forced-choice recall. Current thinking-off run = pilot/appendix.
- **E3 — Cross-model replication (API, no GPU).** Same grid (en/hi/bn at least) on **one strong
  multilingual instruct model** (Gemma-3-27B class) + **one open thinking model** exposing CoT
  for VEA (Qwen3/R1-class with workable Hindi), via OpenRouter. Replicates → "phenomenon";
  diverges → publishable model-dependence finding.
- **E4 — Causal mechanistic (the A* differentiator).** (a) Probe the **eval-awareness direction**
  (cue-presence AND the anti-trivial VEA-pos/neg contrast) per layer; (b) **cross-lingual
  transfer** (train en → test Indic; cosine geometry) = OPEN-5; (c) **positive control** — also
  build the **refusal direction**; steering it **must** move refusal (proves the hook has power);
  (d) then **patch/steer the eval direction** → predicted ~null on refusal = activation-level
  confirmation of the decoupling. Use **8-bit** (not INT4) for clean interp; nnsight 0.7.
- **E5 — CoT-MONITORABILITY (3rd pillar; `scripts/17_cot_monitor.py`).** A **blinded LLM safety
  monitor** reads ONLY the `<think>` reasoning (no prompt, no answer, no condition) and flags
  (1) **harmful planning** present, (2) **eval-awareness** verbalized, (3) **predicted action**
  (refuse/comply). Per language: **monitorability** = P(harmful-planning detected | model actually
  complied) — would a CoT monitor catch it?; **faithfulness** = agreement(CoT-predicted action,
  actual behavior); monitor eval-awareness vs keyword VEA. Reuses stored traces → **no GPU**,
  OpenRouter judge. Headline: the CoT is **English/readable** (cross-lingual) yet unsafe output
  persists → *CoT-monitorability ≠ safety*. Engages Korbak et al. 2025 (CoT-monitorability
  position paper) + multilingual CoT-monitorability work (citations TO VERIFY).

## 4. Validation — weakness consolidation (closes the reviewer attacks)
- **V1 Refusal labels.** hi/bn **double-annotation** (~200 stratified items/lang), report κ,
  validate the auto-classifier; **LLM-judge** ta/te/or, calibrated by judge-vs-human agreement
  on hi/bn.
- **V2 Translation harm-preservation (top confound).** hi/bn human harm-preservation ratings;
  **back-translation + LLM equivalence** for ta/te/or. If MT attenuates harm, the safety gap is
  partly a translation artifact — must measure & report.
- **V3 VEA validation.** Blinded **LLM-judge** + human spot-check on a stratified sample;
  report precision/recall; separate explicit self-report from generic policy mentions.
- **V4 Statistics.** Exact/**bootstrap CIs**; **hierarchical logistic regression** (prompt-ID
  random effect, language × condition); **equivalence bounds** ("rule out eval effects > X pp").
  Raise headline cells (en/hi/bn) to **n≈300** if Toxic_Matrix has ≥300 parallel items (check).

## 5. Venue & timeline (→ ARR Aug 3; ~7.5 weeks)
- **Wk1 (now):** lock E1 cue battery + benign set; redesign E2; build V1/V2 annotation+audit
  sheets (kick off Sibayan hi/bn — longest human lead); check Toxic_Matrix size for n=300; queue
  the E1+E2 GPU grid (the bottleneck).
- **Wk2–3:** GPU grid runs; E3 cross-model via API in parallel; V4 analysis pipeline.
- **Wk4 start E4** (probe → transfer → refusal positive control → eval patch) on 8-bit.
- **Wk4–6:** integrate results; write; add the 4 missing citations (Nguyen, scaling study, Hua,
  Wang refusal-direction).
- **Wk7:** internal review + **line-by-line citation hand-verification**; submit **Aug 3**.
- **Fallback:** if E4 slips, submit behavioral+cross-model to ARR Aug 3 and split E4 to a
  NeurIPS-2026 workshop (~Aug 29).

## 6. Honest risks
Solo + 7.5 weeks is tight; E4 steering on a quantized model is engineering-risky (~35–45% clean
in 2 wks); n=300 needs parallel prompts to exist; ta/te/or stay provisional without native
validators. A* main is *possible* with all of E1–E4 + V1–V4, **not guaranteed** — ARR's rolling
cycles are the hedge.

## 7. Workstreams & owners
- **Sibayan (start now):** hi/bn refusal annotation + harm-preservation ratings; sanity-check
  the hi/bn cue-battery wordings.
- **Me, no-GPU (now):** cue battery (en + hi/bn drafts; ta/te/or provisional), benign prompt set,
  E2 redesign script, V1/V2 rubrics+sheets, E3 OpenRouter runner, V4 stats module.
- **GPU (queue once stimuli locked):** E1+E2 grid; then E4 (8-bit) probe/steer.
- **Reused:** `behavioral_common.classify_refusal` (single source of truth), `05/07` patterns,
  `01/02/03` mechanistic infra (fix the `03` bugs: `dtype=`, `model.model` path, per-token hook).
