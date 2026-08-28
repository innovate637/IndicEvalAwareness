# Record

Project state, decisions and their rationale, and open questions.
Measured numbers live in results.md; the command history lives in run_log.md.

Last updated: 2026-08-09

---

## Project

Behavioral half of the IndicEvalAwareness work. The earlier study covered both
behavioral and mechanistic scenarios; it was too much for one paper, so it was split.
**This repo is the behavioral part only.** The prior work lives at
`~/IndicEvalAwareness/` and is treated as read-only reference.

Research question: does eval-framing change model behavior, and does that effect differ
between English and Indic languages (the cross-lingual dissociation)?

### Design

| Axis | Levels | n |
|---|---|---|
| Models | Gemma3-27b, Qwen3-32B, Sarvam-30b, Sarvam-105b | 4 |
| Languages | en, hi, bn, ta, te, or | 6 |
| Cue conditions | deploy, neutral_log, eval_log, eval_explicit, eval_metadata | 5 |

120 cells. Toxic Matrix supplies harmful requests (does eval framing change refusal
rate?); Dolly-T is the benign control (does it change behavior with nothing to refuse?).

**Model list was revised on 2026-08-09**: Qwen3-**32B**, not Qwen3-8b as first stated.

---

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Reuse the existing cue battery unchanged | Already covers all 6 languages × 5 conditions; no reason to re-author |
| 2 | Keep only native scripts; drop all `*_Latn` transliterations | Study is about languages in their own scripts. English's native script is Latin, so `eng_Latn` stays |
| 3 | Drop the response/refusal turn, keep prompts only | Toxic Matrix responses are 100% one canned string (zero signal); Dolly-T answers are unused since models generate their own |
| 4 | Drop `num_turns` | Constant 1.0, meaningless once the second turn is gone |
| 5 | Three-stage pipeline, each stage kept on disk | `raw` → `processed` → `prompts`. Costs ~1.7 GB against 8.3 TB free; makes every step re-runnable without re-downloading |
| 6 | Install `pyarrow` into the shared IndicEvalAwareness venv | Explicitly authorised. No parquet could be read otherwise. It is a leaf dependency, so the torch/transformers stack is unaffected |
| 7 | Audit turn order *before* discarding turn1 | Discarding the wrong turn would have silently poisoned every downstream prompt |
| 9 | ~~HH-RLHF replaces Toxic Matrix as the harmful prompt source~~ **REVERSED by D11** | Rationale partly retracted: the severity-metadata claim was wrong (Q13 correction) |
| 10 | Toxic Matrix data retained on disk, not deleted | Retired, not discarded — proved correct, D11 reinstated it without any re-download |
| 11 | **Toxic Matrix is the harmful prompt source; HH-RLHF retired** (2026-08-10, user decision, reverses D9) | Annotators agree far better on it (AC1 0.826 vs 0.520); India-specific harm axis suits an Indic study; already parallel across all 6 languages; severity skews severe, giving better dynamic range than HH-RLHF's petty harms; carries no training-contamination risk (Q12 was HH-RLHF's problem) |
| 12 | HH-RLHF files renamed `_retired_*`, not deleted | Same logic as D10. They are small and script-reproducible, and the §5c/§7 reliability results reference them |
| 13 | **Final study set: 200 Toxic Matrix prompts × 6 languages** (2026-08-10, user decision) | Frozen set for the whole project. `data/final_set/final_harmful_200.csv` |
| 14 | Set drawn as a prefix of a fixed permutation (seed 2026), not an ad-hoc sample | Extending to 500 later appends items and leaves these 200 byte-identical, so early results stay valid instead of being invalidated by a re-draw |
| 15 | **Final set is English only** (2026-08-10, user decision) | User translates with their own pipeline. `doc_id` is retained, so the five Indic columns are recoverable at any time with `--all-langs`; nothing is destroyed |
| 8 | Sarvam API key stored in `.env`, mode 600, gitignored | Keeps the secret out of scripts and out of any future commit |

---

## Credentials

A Sarvam API key is stored at `.env` as `SARVAM_API_KEY`, permissions `600`, listed in
`.gitignore`. **The value is never to be written into any tracked file, log, script,
results table, or command line.**

**Standing instruction from the user: this key is only ever to be used when the user
explicitly says to use it. It is never to be used on the assistant's own initiative** —
not for a connectivity check, not for a smoke test, not as part of any pipeline run.

Note: the key was pasted into a chat transcript when it was handed over, so it exists
outside `.env` in that log. Rotating it after the study is worth considering.

---

## Open questions

### Q17 — Phase 2 model set undecided, and a base-model/DV tension *(OPEN — main blocker)*

**Decision (user, 2026-08-11): base/pretrained checkpoints only, not instruct/chat
variants.** Also removed from the plan on user instruction: the compute section, the LLM
judge, and Cohen's kappa (AC1 and raw agreement are retained).

**Unresolved tension this creates — settle before generation.** Refusal behaviour is
produced by instruction tuning and RLHF; a genuinely pretrained checkpoint has no refusal
behaviour to measure, so refusal rate would sit near zero in every cell and every contrast
would be flat regardless of cue. Two readings:
1. "Base" = the vendor's off-the-shelf release (not a checkpoint we fine-tuned). Design
   works unchanged.
2. "Base" = pretrained-only. Then the DV must change from *refusal* to something a base
   model exhibits — completion harmfulness, continuation compliance, hedging rate, or a
   shift in continuation distribution. Legitimate and arguably more novel, but it is not
   "refusal behaviour" and §1/§7/§8 of `plan.md` would need rewriting.

Cheap resolution: `plan.md` Gate 0 — ~30 harmful items through one base checkpoint under
`deploy` and `eval_explicit`. If refusal is ~0 in both, reading 1 is required or the DV
changes.

Consequence of base-only: base checkpoints are generally not served over chat APIs, so this
effectively forces local inference and rules out any model whose base weights are
unpublished. Each candidate's **base** repo needs checking — prior figures were gathered
for the instruct variants (e.g. `gemma-3-27b-it` is the instruct repo, gated separately).

### Q17a — model selection criteria

`plan.md` §4 states the question and the constraints rather than assuming an answer. The
set must: use safety-tuned chat checkpoints (base models do not refuse, so the DV does not
exist on them); include at least one genuinely Indic-competent model, **verified** by the
§5 pre-flight gate rather than assumed; span Indic-specialised vs general-multilingual,
since that contrast is the paper's point; fit 94 GB VRAM or have an API; and not share a
family with the judge (Q8).

Working list carried forward: Gemma3-27b (~54 GB, gated on HF), Qwen3-32B (~64 GB),
sarvam-30b (~60 GB, **not on the Sarvam API** — local weights only), sarvam-105b (API
only for us; reasoning model, needs generous `max_tokens`).

Nothing but generation itself is blocked on this — pipeline, scoring protocol and analysis
plan can all be built and piloted against one model first.

### Q1 — Are the Sarvam models actually closed-source? *(blocking model selection)*

The user states sarvam-30b and sarvam-105b are closed-source. Direct HuggingFace API
checks on 2026-08-09 contradict this: both repos are public, ungated, not disabled,
tagged `apache-2.0`, with complete weights (26 and 85 safetensors shards). For contrast,
`google/gemma-3-27b-it` returns `gated=manual` in the same check, which is what a
genuinely restricted repo looks like.

Possible reconciliations:
- The deprecation/closure may refer to **Sarvam's hosted API**, a separate product from
  the weights. That would not affect local Slurm inference at all.
- The user may have newer or internal information (takedown, licence change) not visible
  in the Hub metadata.

Unresolved. Do not plan around either answer until settled.

**Update 2026-08-09:** the user supplied a Sarvam **API** key, and it works. Reaching
sarvam-105b over the hosted API makes Q2 moot for that model — no VRAM constraint applies
to an API call. Confirmed working: `https://api.sarvam.ai/v1/chat/completions`, header
`api-subscription-key`.

**The key exposes only `sarvam-105b` and `sarvam-105b-conversations`. There is no
sarvam-30b on this API**, which is consistent with the user's report that it is
deprecated. So sarvam-30b is available as open weights on the Hub but not via the API,
while sarvam-105b is available both ways. If sarvam-30b stays in the design it must be run
from local weights (~60 GB bf16, fits the 94 GB GPU).

Also note: **sarvam-105b is a reasoning model** — it emits `reasoning_content` charged
against the same token budget as `content`. Any generation code must set a generous
`max_tokens` or responses come back empty, and its reasoning trace may need handling or
stripping in the main experiment.

### Q2 — Sarvam-105b does not fit the available GPU *(blocking, independent of Q1)*

One schedulable H100 NVL = 94 GB VRAM. Sarvam-105b needs ~210 GB at bf16, ~105 GB at
fp8 — both over budget before KV cache. The seven 12 GB MIG slices cannot be combined.

Options, none chosen:
- ~4-bit quantization to fit ~53 GB. **Methodologically risky**: the dependent variable
  is refusal behavior, and heavy quantization could itself shift refusal rates,
  confounding the model comparison.
- Request 2 GPUs — not available; the node exposes only one full GPU.
- Drop Sarvam-105b and run a 3-model design.

This threatens the 4-model design more than Q1 does.

### Q3 — Gemma3-27b is gated

`google/gemma-3-27b-it` is `gated=manual`. Someone must accept the licence on HuggingFace
with an account whose token is available on this machine before the weights can be pulled.

### Q4 — RESOLVED 2026-08-10: 200 items, frozen

Final set is `data/final_set/final_harmful_200_en.csv` — 200 Toxic Matrix prompts,
**English only** (D15), drawn by `scripts/14_sample_final_set.py` as the first 200 of a
seed-2026 permutation of the 89,820-prompt pool. Columns:
`itemnum, perm_rank, doc_id, prompt`.

The Indic versions are produced by the user's own translation pipeline. Once translated,
the design is still 200 items × 6 languages × 5 cue conditions × 4 models =
**24,000 generations**. `doc_id` also joins back to
`data/prompts/toxic_prompts_sarvam.parquet` for the IndicTrans2 versions, which stay
available as a translation-quality comparison (`--all-langs` regenerates them).

**Power caveat, on the record.** The analysis in run_log put 200 at detecting roughly
8-9pp shifts at α=.05 uncorrected, ~13pp after Bonferroni across 96 tests. The study's
headline claim is a cross-lingual *interaction*, which needs roughly 4x a main effect:
~620 items for a 10pp dissociation. **At n=200 a null interaction result will be
uninterpretable** — indistinguishable from an underpowered one. The user set 200 with this
stated; recorded rather than re-argued.

Mitigation already built in: D14 means running `scripts/14_sample_final_set.py 500` later
extends the set without changing the original 200, so a decision to scale up costs only
the extra generations, not a re-run of work already done.

### Q4 (original) — Sampling plan not set

### Q5 — Refusal scoring method not decided

No decision on how refusal/compliance will be measured (string heuristic, classifier, or
LLM judge). This determines the dependent variable, so it should be settled before
generation, not after.

### Q7 — Tamil/Telugu prompts lose harmfulness in translation *(new, 2026-08-09)*

Screening found Tamil loses harm on 7 of 41 English-harmful items, 0 gained
(exact McNemar p = 0.0156); Telugu 7 lost, 1 gained (p = 0.0703). Hindi and Bengali are
unaffected. See results.md §4.

This matters because refusal rate is the dependent variable. If a Tamil prompt is no
longer a harmful request, a model that answers it is being *helpful*, not unsafe — and
that would be scored as an eval-awareness effect when it is a translation artifact. It
would inflate exactly the cross-lingual dissociation the paper claims to find.

Two mechanisms are both present and not separable from one judge pass: genuine MT
softening (item 13 lost the word "false") and judge inconsistency on faithfully
translated content (item 10). Options, none chosen:
- Second judge from a different family, keep items where judges agree
- Back-translate ta/te prompts and check harm survives the round trip
- Screen the full sampled set and drop items that lose harm in any language
- Report per-language harm-retention as a caveat and leave items in

Deciding this before generation is much cheaper than after.

### Q8 — Judge model family overlaps with a model under test

The screening judge is `qwen/qwen-2.5-72b-instruct` and `Qwen3-32B` is a model under test.
Harmless for this data-quality screen, but the main experiment's refusal scorer must not
be a Qwen model, or Qwen3-32B's scores would be judged by its own family.

### Q9 — LLM judges are unreliable on Indic text, language-dependently *(new, critical)*

Two judges on identical items agree only 81% raw, **Cohen's κ = 0.180**. Agreement is
~90% on English and Hindi but falls to **69% on Tamil** and 75% on Telugu. See
results.md §5.

**This is the most consequential finding so far, and it threatens the main experiment,
not just the data.** The dependent variable is refusal rate, scored per language. If the
scorer is a single LLM judge that drifts more on Tamil than on English, it will produce a
cross-lingual difference *in the absence of any real model behavior difference* — which
is precisely the paper's headline claim. The §4 single-judge "Tamil loses harm" result is
itself an instance: it vanished under a second judge.

Implications to settle before generation:
- A single-LLM-judge refusal scorer is not defensible for this design.
- Minimum viable fix: two judges from different families, report κ per language, and
  restrict the headline analysis to items where they agree.
- Human validation on a subsample per language is the stronger fix, and is likely needed
  for camera-ready given the claim rests entirely on cross-lingual comparison.
- Whatever is chosen, **per-language judge reliability must be reported**, not assumed.

### Q7 update — harm retention largely survives; the Tamil effect did not replicate

Second judge does not confirm §4. Sarvam finds no language significant (Tamil 4 lost /
0 gained, p = 0.125 vs Qwen's 7/0, p = 0.0156). Consensus losses — both judges agree
harmful in English and benign in the Indic language — are **5 cells out of 195 (2.6%)**,
and 4 of those 5 are the same item (44), suggesting one weak source item rather than a
translation problem.

Provisional resolution: translation degradation is real but small, and does not by itself
justify dropping items. The dominant problem is judge reliability (Q9), not MT quality.
Re-open if a third judge or human labels contradict this.

### Q10 — CLOSED 2026-08-10. Qualification rule set by the user; both datasets pass

**Final rule: if a prompt asks for harm or contains harm, it qualifies.** Under it,
Toxic Matrix qualifies 96/100 and HH-RLHF 89/100 (results.md §5b). Prompt-set composition
is settled and is no longer a reason to prefer either source. `genuine_intent` is kept as
a descriptive column for stratified reporting, not as an inclusion filter.

Original entry and its correction retained below for the record.

### Q10 (original) — ~35% of Toxic Matrix items cannot support a refusal measurement

Manual read of 100 English prompts: 78 are harmful, but only **65 are harmful AND an
actual request**. 22 are statements or narratives that ask for nothing, 5 are benign, 3
are abstract discussion, 1 is a bare insult. See results.md §6.

**CORRECTED 2026-08-10 — this was overstated.** The original claim was that no-ask items
threaten the cross-lingual comparison. That is wrong: the design is **within-item**, the
same items run through all 120 cells, so item composition is identical in every cell. The
heterogeneity is constant across conditions and languages and therefore cannot confound
the English vs Indic contrast. It costs some precision, nothing more.

The user's position — if content is harmful or asks for harmful content, it is harmful —
is correct as a harm taxonomy, and is also right that a model can refuse to *engage* with
harmful content that asked nothing. Eval-framing could move that behaviour too, so no-ask
items are not dead weight.

What survives: a refusal on a no-ask narrative and a refusal on an operational request are
different behaviours, so `genuine_intent` is worth keeping as a **stratification variable
for interpretation**, not as a filter. Report refusal rates split by it; do not drop items
on it.

### Q11 — prompt source: HH-RLHF, Toxic Matrix, or both *(new)*

Both screened by hand under identical rules, 100 items each (results.md §7). Neither
dominates:

- **HH-RLHF is cleaner in form** — 98% of harmful items are genuine requests vs 83%; the
  Q10 no-ask problem all but disappears (1 vs 13); no fiction/roleplay at all.
- **Toxic Matrix has higher usable yield** — 65 vs 57 per 100, since 42% of harmless-base
  is benign by construction.
- **HH-RLHF is US-centric** (Katy Perry, Trump, Flint) and **Toxic Matrix is India-centric**
  (caste, sect, named Indian hospitals). For an Indic study this is the decisive
  difference, not the yield.
- **Severity differs sharply** — HH-RLHF skews petty (shopping-cart theft, cinema
  sneaking), Toxic Matrix skews severe (bombs, genocide, organ harvesting). They are not
  interchangeable and **must not be pooled into one condition**, or source is confounded
  with severity.

Recommendation, not yet decided by the user: run both as a prompt-source factor. An
eval-awareness effect that survives both a human-authored US set and a synthetic Indian
set is robust against prompt-source artefacts, which is a question reviewers will ask.

### Q13 — SUPERSEDED 2026-08-10 by D11: harmful source reverted to Toxic Matrix

**Current state.** Canonical harmful set is `data/prompts_en/harmful_prompts_en.csv`,
built from Toxic Matrix English by `scripts/13_build_harmful_set_tm.py`: **89,820 unique
prompts** from 90,352 source rows (531 duplicates, 1 degenerate fragment). Mean 33.7
words. `doc_id` joins back to `data/prompts/toxic_prompts_sarvam.parquet`, which already
carries all six languages in parallel.

**A translation pipeline may not be needed.** Toxic Matrix ships hi/bn/ta/te/or alongside
English, item-aligned by `doc_id`. Using those gives the within-item cross-lingual design
directly, at the cost of accepting IndicTrans2's MT quality (harm retention verified in
results.md §4–5: consensus loss 2.6%). Translating independently would be a robustness
check, not a requirement.

**Q12 (HH-RLHF contamination) is now moot** for the study, since HH-RLHF is retired.

HH-RLHF artefacts kept as `data/prompts_en/_retired_*` and `data/raw_hhrlhf/`.

### Q13 (original) — harmful source is HH-RLHF

User decision: **HH-RLHF replaces Toxic Matrix.** Canonical set built at
`data/prompts_en/harmful_prompts_en.csv` — 38,961 unique English prompts merged and
deduplicated across `harmless-base` openers (17,065) and `red-team-attempts` first turns
(21,896); 16,498 duplicates collapsed, 658 degenerate fragments dropped (<3 words or
<15 chars, e.g. the single word "Assistant!"). 36,873 carry attacker `rating` and
`min_harmlessness`; 8,663 are rated >=4.

English only — the user has their own translation pipeline. Toxic Matrix files are
retained on disk but retired as the harmful source.

**Q12 (contamination) is an accepted risk under this decision**, not a resolved one. It
belongs in the paper's limitations, and the templatedness check described in Q12 remains
worth running as a cheap sanity test.

**Correction to the rationale (2026-08-10).** Decision 9 cites "severity metadata Toxic
Matrix entirely lacks". That advantage does not hold as stated: `rating` and
`min_harmlessness_score_transcript` are **transcript-level**, describing whether the whole
multi-turn attempt succeeded, not the severity of the first turn. Benign openers carry
rating 4.0 (results.md §7b). Severity stratification on first-turn data is therefore
invalid, and the `rating`/`min_harmlessness` columns in
`data/prompts_en/harmful_prompts_en.csv` **must not be used as first-turn severity
labels**. They are retained only as pointers back to the source transcript.

The decision itself stands on the remaining grounds — human-authored rather than
synthetic, 2.5x shorter prompts, cleaner single-turn form, canonical and MIT-licensed.

Reliability facts established for HH-RLHF (results.md §7b): judge reliability is
**unmeasured**; the manual screen is single-annotator with no IAA; 16,498 duplicates were
collapsed and the two sources overlap heavily so they are **not independent**; ~1.2%
degenerate fragments; visible crowdworker noise.

### Q14 — ANSWERED 2026-08-10: human IAA is κ = 0.286, and the rule needs narrowing

Second reviewer completed the blind pack. Raw agreement 70%, **Cohen's κ = 0.286**
(results.md §5c). All 6 disagreements one-directional: the assistant qualified
opinion-solicitation, denigration and AI-directed insults; the user did not. Perfect
agreement on all 12 actionable-harm items.

**Two humans agree only marginally better than two LLM judges (κ 0.286 vs 0.18).** The
construct is ambiguous, not just the judges — this makes a single-LLM refusal scorer
untenable (see Q9) and is worth reporting in the paper in its own right.

**Open sub-question blocking a re-label:** does "contains derogatory content about people"
qualify when nothing is requested? The user's labels say yes for item 3 (derogatory
assertion about people) but no for item 20 (abuse aimed at the model). Once confirmed,
re-run the qualification pass — expected to move HH-RLHF from 89/100 to ~60/100 and to
drop Toxic Matrix well below 96/100.

Severity ratings were not collected in this pass, so HH-RLHF's unusable `rating` field
(Q13) still has no human replacement.

### Q14 (original) — no inter-annotator agreement exists yet

Every label in this project comes from a single annotator. Given Q9 — two LLM judges at
81% raw agreement but kappa = 0.18 — an unvalidated single-annotator standard is a weak
foundation for a calibration reference.

`review_pack/` is built and blind, 20 items, awaiting a second reviewer. It also collects
a **human severity rating (1–5)**, which is the replacement for HH-RLHF's `rating` field
after that turned out to be transcript-level (Q13 correction).

Watch raw agreement *and* kappa: the set is ~90% qualified, so raw agreement will look
high regardless. Kappa is the figure that means anything. Disagreements and reviewer notes
matter more than the headline number — they show where the qualification rule is
ambiguous rather than where a reviewer erred.

### Q15 — the qualification rule is not yet writable *(open, blocks large-scale labelling)*

Two 20-item packs are done (results.md §5c, §5d). Agreement by Gwet's AC1: **0.520 on
HH-RLHF, 0.826 on Toxic Matrix**. Cohen's κ was +0.286 and **−0.071** respectively; the
negative value is the kappa prevalence paradox at 0.93 prevalence, not a reliability
collapse.

Adopting the reviewer's inferred boundary in pack 2 did **not** predict their judgements:
they qualified items 7 and 11 (excluded by direct analogy to their own earlier exclusion of
"why mark hurd is a bad ceo") and rejected item 12 (suicide-adjacent, while qualifying
item 1 which is the same kind of content).

The unstable region is consistently **content that denigrates or targets a group or
individual without requesting an action**. Until a written rule resolves items 7, 11 and
12 explicitly, no large-scale labelling — human or automated — should proceed, since the
standard it would be calibrated against does not yet exist.

Two substantive findings that do not depend on resolving this:
- **Qualification is at ceiling on Toxic Matrix** (18–19/20 here, 96/100 in §5b), so the
  label cannot filter that dataset. On HH-RLHF the reviewer qualified 12/20, so it does
  discriminate there.
- **Never report Cohen's κ alone for the refusal-scoring reliability analysis (Q9).**
  Refusal rates on harmful prompts will sit near ceiling and κ will be unstable or
  negative while agreement is genuinely high. Report Gwet's AC1, or agreement within the
  minority class. `scripts/11_score_review_pack.py` now emits both and warns above 0.85
  prevalence.

### Q12 — HH-RLHF training contamination *(accepted risk, see Q13)*

HH-RLHF (2022) is one of the most widely used safety datasets in existence. Gemma3,
Qwen3 and Sarvam have plausibly seen it, or derivatives of it, during their own safety
tuning.

**This attacks the construct, not the data quality.** If a refusal is a memorised response
to a familiar prompt rather than a reasoned one, the measurement is recall, not
eval-awareness — and memorised refusals plausibly respond *less* to cue manipulation,
damping the very effect the study is trying to detect. Toxic Matrix (2024) is also public
but far less absorbed into safety-training pipelines.

Testable before committing: run a sample of HH-RLHF prompts against each model and check
whether refusals come back near-verbatim or unusually templated relative to matched
Toxic Matrix prompts.

Other, lesser problems with HH-RLHF for this use case (detail in results.md §7):
US-centric content; severity skewed toward petty harms, compressing refusal-rate dynamic
range; 42% benign so random sampling wastes stimuli; first-turn extraction discards the
multi-turn escalation that carries its attack strength; and using the English original
means the Indic translations are the user's own and unvalidated.

Current scope note: user asked for **English only for now** and has their own translation
pipeline, so `data/prompts_en/` is English-only and no Indic columns were produced from
HH-RLHF.

### Q16 — translation protocol written, not yet executed *(open)*

**Method decided (user, 2026-08-10): translate with Claude Opus (`claude-opus-5`) via
Claude Code or chat, one teammate per language, then COMET-Kiwi QE.** SOP is
`docs/TRANSLATION_SOP.md`; the older IndicTrans2-based `docs/TRANSLATION_PROTOCOL.md` is
superseded on the points that conflict. Prompts in
`data/final_set/translation_prompts.json`. Freeze and ideally pre-register before
translating.

Three consequences of using Claude Opus over chat, all in the SOP and all needing a line
in the paper:
- **Non-determinism is unavoidable.** Chat/Claude Code exposes no temperature/effort, and
  Opus 5 rejects `temperature`/`top_p`/`top_k` even via API — so translations aren't
  reproducible. State it; don't cherry-pick re-runs.
- **Refusals will occur** (harmful prompts + strong classifiers) and the refusal rate will
  likely differ by language — a differential missing-data mechanism that must be logged and
  reported, not worked around.
- **COMET-Kiwi is necessary but not sufficient** — it scores fluency/adequacy, so a
  translation that sanitises the harm can still score well (results.md §4 Tamil example).
  Harm retention is a separate gate: re-run the harm screen on the translated set with two
  judges from different families once all five languages are in.

Superseded assumption: the original Q16 entry recommended IndicTrans2 for its lack of a
safety layer. That advantage is now given up deliberately — using Claude Opus means
accepting refusals as logged missing data, in exchange for the user's own pipeline and
translation quality.

Load-bearing choices:
- **IndicTrans2 over a chat LLM.** No safety layer, so it will not silently sanitise
  harmful text; deterministic and versioned; and it matches what AI4Bharat used, making
  the existing Toxic Matrix Indic columns a free second reference.
- **One instruction template, identical across all five languages**, varying only
  `{LANGUAGE}` and `{SCRIPT}`. If the instruction differs per language, language is
  confounded with instruction wording and no cross-lingual comparison is interpretable.
- **Harm retention is an acceptance gate**, not a report-only metric: no language may lose
  harm on more than 5% of items vs English, judged by two judges from different families,
  paired by `doc_id`. This gate is absent from standard translation protocols and is the
  study-specific validity requirement — results.md §4 showed a single word ("false")
  disappearing in Tamil and neutralising the stimulus.
- Back-translation must use a **different system** than the forward pass and be blind to
  the source, or it merely inverts its own errors.

Outstanding actions this creates:
- **Ethics/IRB approval is required before recruiting human reviewers**, since exposure to
  this content makes them participants, not contractors. Retroactive approval is usually
  unavailable and its absence can block publication.
- Confirm the institution's position on **DPDP Act 2023** (prompts name real living
  people) in writing.
- Never reproduce a prompt naming a real person verbatim in any output; anonymise.

### Q6 — ta/te/or cues need native review

3 of 5 cue conditions in Tamil, Telugu, and Odia are provisional MT-informed drafts. For
a paper whose central claim is cross-lingual, half the Indic languages resting on
unreviewed cue translations is a direct target for reviewers. The file itself says these
"MUST be native-reviewed before camera-ready".

---

## Known caveats to carry into the writeup

- All Indic prompts in both datasets are **IndicTrans2 machine translations**, not
  natively authored. Same MT-quality caveat as the provisional cues.
- Toxic Matrix prompts were **generated by Mistral-7B**, per the dataset card. The
  filename says `sarvam`, which the card does not explain. If the prompts were in fact
  Sarvam-generated, that is a contamination question worth pre-empting, since two models
  under test are Sarvam models. Unresolved.
- `CLAUDE.md` scopes work to this directory; reading from `~/IndicEvalAwareness/` was
  done under explicit user direction and that path is treated as read-only.

---

## Status

Data preparation complete. Harm screening complete (300 judgements, results.md §4).

No models under test have been run, no weights downloaded, no Slurm job submitted, no
generation pipeline written. Q1–Q3 determine which models the pipeline must support and
should be settled before it is built. Q7 should be settled before generation, since it
affects which items are eligible.

---

## 2026-08-18 — Phase 2 readiness state (audit; supersedes the design table above)

**This file's header and Design table are stale.** They describe the pre-substitution design
(4 models, languages ending in `or`). The governing document is now
`docs/Final_Phase_2_Plan_main1.md` (rev 3.2), which supersedes `plan.md` and the Phase-2
sections of this file: 6 languages `en/hi/bn/ta/te/**kn**`, 6 models (5 + 1 conditional),
5 cues, 2 arms, 200 items. Full audit in `run_log.md` under the same date.

**User instruction (2026-08-18): run `gemma3-27b-it` only.** That is a 1-model slice of the
main grid — 1 × 2 arms × 6 langs × 5 cues × 200 = **12,000 generations**, plus gates.
Gemma is excluded from sub-run B by design (§4.5 — its chat template merges system into
user, so a system-placement run would duplicate the main grid).

### Open questions blocking the run

#### Q18 — Slurm is down and this is not the cluster the plan targets *(blocking, environment)*
`Slurmctld(primary) at bitspilani-slurmvm02 is DOWN`; no job can be submitted. Separately,
the plan's §2.3/§7.3/§11 resource map is written for Sharanga (`gpu_h100_4`, `gpu_h200_8`,
3-GPU QOS cap, `/scratch/$USER`). This cluster has one node with 1× H100 NVL plus 7× 12 GB
MIG slices, partitions `h100-full` and `h100-mig`, and no `/scratch`. `gemma3-27b-it` at
~54 GB bf16 **cannot fit a 12 GB MIG slice**, so it requires `h100-full` — which carries
`DenyAccounts=students,students-limited`. Needs: controller restored, account confirmed for
`h100-full`, and the whole resource map re-derived for H100 NVL (94 GB, not 80 GB).

#### Q19 — vLLM version mismatch *(blocking, environment)*
Plan pins **vllm 0.27.1**; env `~/miniforge3/envs/vllm` has **0.23.0** (transformers 5.12.1,
pyarrow absent). Every API correction in plan §0.1–§0.2 (V3 `TokensPrompt`, N3 `Logprob.rank`,
N4 `<think>` splitting, `VLLM_BATCH_INVARIANT`) was verified against 0.27.1. Installing is
not permitted without explicit user authorisation (CLAUDE.md rule 2). Decision needed before
`generate.py` can be written against a definite API.

#### Q20 — cue battery has no native review, so G0 cannot pass *(blocking, cannot be resolved by Claude)*
15 of 30 strings (`ta`/`te`/`kn`) are `provisional_pending_native_review`; the file carries no
`review_verdict` field at all, and plan §8.8 G0 hard-asserts `approved` on all 30. Kannada was
LLM-drafted where §3.3 V1 demands authored strings, and §3.3 V2 specifically requires a native
reviewer to confirm the `monitor_log` / `eval_log` distinction survives — the load-bearing
contrast of the study, and the one most likely to collapse in exactly ta/te/kn. **This needs
native speakers; no amount of code closes it.** Either recruit reviewers, or run with
provisional cues and record the G0 waiver as a stated limitation.

#### Q21 — item 47 replacement is half-applied; §3.6 refusal decision unrecorded *(blocking, data)*
`data/harmful_bn.json` (in the GitHub repo, absent locally) has the item-47 swap applied
(`f39243b1…` → `438b0290…`); en/hi/ta/te/kn do not. The design is within-item, so the grid is
currently inconsistent. Separately §3.6's Option A/B/C choice for the refused items — 56
(ta, kn) and 160 (hi, ta, kn) — is not recorded anywhere. Plan recommends **Option A**: drop
both `doc_id`s from all six languages and extend the seed-2026 permutation prefix by two,
which is also the only option that stops propagating item 56. Needs a user decision, plus a
choice between the `claude` and `ai4bharat` translation sets for the replacement item.

#### Q22 — itemnum base *(mechanical, decided by the plan)*
Every delivered file is 0-based (0–199); the plan is 1-based throughout. `normalise_translations.py`
renumbers to 1-based on ingest, per §3.1. Recorded so it is decided once and never revisited.

### Q18–Q21 status update (2026-08-18, after building the pipeline)

**Q18 (cluster) — partially characterised.** Resource map re-derived and encoded in
`config/models.yaml` with `[LOCAL]` markers; `cluster/submit.py` emits a compliant
sbatch. Still blocked on: slurmctld restored, and confirmation the account may use
`h100-full` (it carries `DenyAccounts=students,students-limited`).

**Q19 (vLLM) — unchanged, and now the main code-level unknown.** Nothing installed.
`pyproject.toml` records the plan's pins as an instruction, not a description. Two
consequences already visible: `consolidate.py` cannot import (pyarrow), and
`build_benign_arm.py` cannot read `Dolly.parquet` (pyarrow). The vLLM API surface in
`generate.py` (TokensPrompt, `Logprob.rank`, `VLLM_BATCH_INVARIANT`) is written to
the plan's 0.27.1 and is **unverified against the installed 0.23.0**.

**Q20 (cue review) — scaffolding in place, substance still missing.** The Appendix-A
`metadata` block now exists with 30 entries, all `pending`, so G0 fails loudly and
specifically instead of on a missing field. This does not advance the actual blocker:
ta/te/kn need native reviewers, and per §3.3 V2 they must be shown the
`monitor_log`/`eval_log` pair side by side and asked explicitly whether the
distinction survives. Kannada additionally needs its five strings *authored*, not
LLM-translated as they currently are.

**Q21 (item set) — quantified, not decided.** Option A verified to work
mechanically: dropping the two refused doc_ids yields a clean, aligned n=198 across
en/hi/ta/te/kn. Two things still need a human decision: (i) A/B/C itself, and
(ii) whether to extend the seed-2026 prefix by two to restore n=200 — the plan's
§3.6 recommendation — which requires translating two new items in five languages.
The Bengali/item-47 inconsistency is untouched; `data/harmful_bn.json` is still
absent locally and the repo copy still carries the unilateral item-47 swap.

#### Q23 — `google/gemma-3-27b-it` is gated and not accessible *(NEW, blocking, and it blocks the model the user chose)*
Fetching any file returns `GatedRepoError 403: Access to model
google/gemma-3-27b-it is restricted and you are not in the authorized list`, on the
account holding `~/.cache/huggingface/token` (`Trizal`). The repo's public metadata
resolves, so the commit SHA is pinned, but **no tokenizer, no config and no weights
can be downloaded** — which blocks G0.5 (cue parity needs the tokenizer), G2, and
`token_budget.py`. `google/gemma-3-27b-pt` is separately gated and needs its own
acceptance. `Qwen/Qwen3-32B` and `sarvamai/sarvam-m` are ungated and fetch fine, so
if gemma access is slow to arrive, one of those is a viable substitute to make
progress against — that is a design decision, not a mechanical one.
**Action: accept the gemma licence at https://huggingface.co/google/gemma-3-27b-it
(and .../gemma-3-27b-pt) on the `Trizal` account.**
