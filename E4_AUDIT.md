# E4 Mechanistic Audit — targeting, bugs, redesign (2026-06-11)

Audited `12_extract_activations_v2.py`, `13_probe_transfer.py`, `14_steer_patch.py` against
(a) the corrected behavioral story (RESULTS.md) and (b) the persona-vectors method
(Chen et al. 2025, arXiv:2507.21509): *difference-of-means over RESPONSE activations, responses
labelled trait+/− by an LLM judge, best layer by steering effect, causally validated.*

## Verdict: the chain would run, but it targets the WRONG thing and uses INVALID labels.
Two showstoppers + a targeting redesign. None are subtle — they'd produce a confident-looking
but uninterpretable result. Good call to audit before burning GPU.

---

## SHOWSTOPPER 1 — labels come from the keyword classifier we just proved invalid
`12_*.labels_for()` → `classify_refusal()` (keyword) for the `refusal` label, and
`detect_vea()` (keyword) for `vea`. This session established those labels are wrong, *worst in
Indic*: keyword refusal ≈ 1–13% where the judge says ≈ 80–95%. Consequences:
- **The positive control is destroyed.** `14_*.build_refusal_direction()` pools
  `refused_acts` vs `complied_acts` using these labels. For Indic the two pools are
  **near-inverted** → the "refusal direction" is noise → steering it won't move refusal →
  the whole E4 becomes uninterpretable (the eval-direction null can't be read without a
  working positive control). This is the single most important fix.
- **The probe targets are mislabelled** the same way.
- **Fix:** join the calibrated **judge** verdicts (`results/behavioral/refusal_judge.csv`,
  REFUSE→1 / COMPLY→0 / NOANSWER→None) by (lang, condition, item_id). DONE in 12 (below);
  14 inherits the corrected labels automatically via the pkl.

## SHOWSTOPPER 2 — the "eval direction" is trivial cue-token detection, not awareness
The primary contrast is `cue_presence` = `eval_log` vs `deploy` at the **last prompt token**.
Those conditions differ by the literal eval sentence in the system prompt, so a last-prompt
probe just detects "is the eval sentence in my context" — **cue-token presence**, not an
internal "I'm being safety-evaluated" state. Persona vectors avoid exactly this by reading the
**response**, not the prompt cue. Our `awareness` (VEA) contrast was meant to fix it but is
itself leaky (the CoT-monitor audit showed `eval_aware` partly flags refusal-deliberation) and
keyword-derived.
- **Fix (targeting redesign):** make the **eval-framing contrast** the primary scientific
  direction: **`eval_log` vs `neutral_log`** — *both have a monitoring sentence*, differing only
  in safety/compliance-eval vs service-quality-logging framing. A probe separating them cannot
  be reading "is a cue present"; it reads "is this monitoring a safety evaluation." This also
  **directly mirrors the behavioral result that was significant** (gemma eval_log−neutral_log
  OR=1.49, p=0.0009). Added to 13 as contrast `eval_framing`; it feeds 14's eval direction.
- **Anti-triviality stack** (all now available): (i) framing contrast (cue present in both
  arms); (ii) cross-cue-WORDING generalization — train eval_log−neutral_log, test
  eval_explicit−neutral / eval_metadata−neutral; (iii) cross-lingual EN→Indic transfer;
  (iv) last-prompt-token vs think-mean position agreement.

## TARGETING REFINEMENT — position (persona-vectors-aligned), do at GPU time
Persona vectors average over **response** tokens; we use the **last prompt token** only. For a
*thinking* model the "I'm being evaluated" state forms inside `<think>`, not at the prompt
boundary. Last-prompt-token = good for *pre-generation monitoring* (project & predict, à la
persona vectors) but weak for the *direction itself*.
- **Recommended (add at GPU time — deliberately NOT yet in code):** also extract the **mean
  over `<think>` tokens**, teacher-forced from the stored greedy response, and probe both
  positions; report agreement. Implementation spec: tokenize
  `apply_chat_template(msgs, add_generation_prompt=False) + stored_response`, locate the
  `</think>` token id, mean-pool each layer's activations over the think span
  `[gen_start : </think>)`, and save alongside the prompt-last-token acts. Validate
  token-alignment on 2–3 examples first. Left out of 12 on purpose — unvalidated forward-pass
  code should not ship as a default; the prompt-last-token extraction (current) is still valid
  for cue-encoding + pre-generation monitoring.

## CONCRETE BUGS (beyond the two showstoppers)
1. **12 `resolve_cue` reads a now-empty source for ta/te/or.** It routes ta/te/or to
   `cue_translations.json["languages"]`, which is **empty** (we moved those cues into
   `cue_battery.json` this session). → ta/te/or would be SKIPPED entirely. **Fixed:** read
   `cue_battery.json` for all 6 languages (it now carries deploy/neutral_log/eval_log/
   eval_explicit/eval_metadata for every language).
2. **14 Format-A direction mapping is stale.** `load_directions()` maps `awareness`→eval
   (preferring the leaky VEA direction). **Fixed:** prefer `eval_framing`→eval, fallback
   `cue_presence`; `awareness` kept only as a secondary/exploratory direction.
3. **14 steering hook + sign convention remain NEEDS-GPU-VALIDATION** (already flagged in-file).
   Keep the alpha=0 byte-identity check and the positive-control gate. Now that the positive
   control uses *valid* labels, that gate is finally meaningful.
4. **Prediction updated for the pivot.** Old E4 predicted a *null* eval→refusal effect (the
   dead "dissociation"). Corrected prediction is **graded**: steering the eval-framing direction
   should produce a **small positive** refusal change (≈ the +5–9pp behavioral compliance gap),
   while the refusal direction (positive control) produces a **large** change. Effect-size
   ordering (refusal ≫ eval-framing > 0) is the result, not "null."

---

## The clear story E4 should tell (coherent with the corrected paper)
> **The eval-awareness representation is real, linearly decodable, and shared across Indic
> languages — but only weakly wired to refusal.** (1) A direction separating *safety-eval
> framing* from *mere monitoring* is decodable (probe AUROC), generalizes across cue wordings,
> and **transfers EN→Indic** (cosine geometry + transfer AUROC) — the mechanistic form of the
> cross-lingual recognition result. (2) Steering that direction causally shifts refusal **a
> little** (matching the small behavioral compliance gap), while the refusal-direction positive
> control shifts it **a lot** — quantifying *represented-but-weakly-coupled*. This is the
> mechanistic explanation of why a strongly-recognized eval cue moves behavior only modestly,
> in every language.

## Gates (unchanged thresholds, now on valid labels)
- Decodable: EN held-out AUROC > 0.80 on `eval_framing`.
- Transfers: EN→Indic transfer AUROC > 0.65 AND cross-wording generalization.
- Positive control works: refusal-direction steering raises refusal substantially
  (else STOP — hook broken, do not report the eval null).

## Run order when a GPU frees up (E1 has priority)
1. Ensure `refusal_judge.csv` covers the conditions you extract (re-judge E1 grid first).
2. `12 --position prompt_last` (safe) → validate think-mean on 2–3 ex → `12 --position both`.
3. `13` (CPU) — eval_framing primary; check gates.
4. `14 --selftest` → GPU `14 --langs en hi --strengths 0 2 4 8`; **verify positive control
   first**, then read the eval-framing effect.
