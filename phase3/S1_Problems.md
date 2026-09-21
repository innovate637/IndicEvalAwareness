# S1 — Why the Truncation Sensitivity Analysis Could Not Be Run As Written

**Status:** S1 removed from `analyse.py` in commit `8a89fe8`.
**Audience:** PI review (Trizal), and the project record. Companion to `S2_Problems.md`.
**Date:** 2026-09-20
**Sources:** `phase3/j0_triage_results.jsonl`, the four `GEN_ROOTS` generation trees.

Short version: S1's exclusion rule removes **82.5% of all items** and **93% of
harmful items**, leaving 14. The survivors are not a random 14 either — they are
systematically the shortest, simplest items in the set. S1 as written cannot
answer the question it was designed to ask.

---

## 1. What S1 Is

From `analysis_plan_frozen.md` §3.4, verbatim:

> | S1 | primary contrast with all `trunc_clean` and `trunc_degenerate` rows excluded **listwise by `doc_id` across the whole grid** | result does not depend on truncated rows |

**Purpose.** A robustness check on the headline C1 result. Truncation rates in
this corpus correlate with language — Phase 2 §1.2 recorded up to 21.6% in
Kannada for `qwen25-72b-instruct` against 0.0% in English for `gemma3-27b-it` —
and the headline claim runs along the language axis. If truncated responses were
driving C1, the result would be an artifact of generation length rather than a
finding about eval awareness. That is risk **P3**, rated Critical. S1 exists to
close it by re-running C1 with every truncated item removed and checking the
answer does not change.

S1 is an appendix analysis. Per §3.4 it is "never used to pick the headline."

---

## 2. How S1 Works

### 2.1 Which rows are targeted

Truncation flags come from J0 triage (`phase3/j0_triage_results.jsonl`), set
during Phase 2 generation where `finish_reason == "length"`:

| flag | definition | rows |
|---|---|---:|
| `trunc_clean` | `truncated == true` and tail repetition score ≤ 0.5 | 719 |
| `trunc_degenerate` | `truncated == true` and tail repetition score > 0.5 | 463 |
| **either — S1's target set** | | **1,182** |

**1,182 of 47,880 rows = 2.47%.**

By itself that is a small number. The problem is not how many rows are flagged;
it is how the exclusion rule converts flagged *rows* into excluded *items*.

### 2.2 The exclusion rule

S1 does not drop the 1,182 flagged rows. It drops **`doc_id`s**, listwise, across
the whole grid: if *any* row belonging to a `doc_id` is truncated, *every* row
for that `doc_id` is removed — every model, every language, every cue.

The plan's reason for this is sound and was argued in §3.2: "every drop is a lost
**pair**, not a lost observation." C1 is a paired contrast computed on matched
items, so an item present under one cue but absent under another would break the
pairing. Listwise exclusion keeps the grid fully crossed. The rule is not a
mistake; its interaction with this grid's shape is what fails.

---

## 3. The Cascade Problem

| | count | share |
|---|---:|---:|
| `doc_id`s in the grid | 399 | |
| `doc_id`s excluded by S1 | **329** | **82.5%** |
| `doc_id`s surviving | 70 | 17.5% |

Split by arm:

| arm | before | surviving | lost | % lost |
|---|---:|---:|---:|---:|
| **harmful** | 199 | **14** | 185 | **93.0%** |
| benign | 200 | 56 | 144 | 72.0% |

**2.47% of rows flagged removes 93% of harmful items.**

The harmful arm is hit harder because truncation is itself concentrated there
(3.33% of harmful rows vs 1.61% of benign), and more items means more chances to
be hit.

### 3.1 Why 14 items cannot support the analysis

C1 is a proportion difference computed on matched items, with a 10,000-resample
item-level cluster bootstrap. At n = 14:

- **Granularity.** One item moves `delta` by 1/14 = **0.071**. The primary C1
  estimates range from +0.010 to +0.106, with a four-model mean of **+0.046**.
  The measurement step is larger than the effect being measured. S1 cannot
  resolve the quantity it exists to check.
- **Bootstrap validity.** Resampling 14 items with replacement gives a discrete,
  lumpy distribution. The percentile interval is an artifact of which of 14 items
  were drawn, not a characterisation of sampling error.
- **Cell coverage.** C1 is reported across 4 models × 6 languages = 24 cells, all
  reading the same 14 items. Any apparent variation between cells is 14 items
  being re-sliced, not independent evidence.

When S1 was run before removal it produced exactly one sign flip against the
primary — `gemma3-27b-it / te`, primary +0.0151 vs S1 −0.0714. That flip is a
single item changing its label. It is noise presented as a disagreement.

### 3.2 The survivors are a biased subsample, not just a small one

This is the part that matters most, and it is not a sample-size problem.

An item survives S1 only if **none** of its 120 responses hit the token cap. That
is not a neutral filter — it selects for items that are short and simple enough
that no model, in any language, under any cue, ever produced a long answer.

Comparing the 14 harmful survivors against the 185 dropped:

| | median item length | median response length |
|---|---:|---:|
| S1 survivors (14) | 112 chars | 475 chars |
| dropped (185) | 188 chars | 655 chars |
| ratio | **0.60×** | **0.73×** |

The survivors' prompts are 40% shorter and their responses 27% shorter than the
items S1 discards. S1 does not re-run C1 on a random subsample of the corpus. It
re-runs C1 on the short, simple tail of it.

So even a hypothetical version of S1 with a workable n would be answering a
different question: not "does the result depend on truncated rows?" but "what is
the result among items that are short enough never to truncate anywhere?" Those
are not the same question, and only the first is the one §3.4 asked.

---

## 4. Why the Listwise Rule Is So Aggressive

### 4.1 The correct grid arithmetic

Each `doc_id` appears in **120 rows**:

```
4 models  ×  6 languages  ×  5 cues  =  120
```

**Arm is not a factor here.** Each `doc_id` belongs to exactly one arm — the
harmful and benign sets are 199 and 200 *different* items — so a `doc_id` is
never present in both. (An earlier framing of this document described 40 rows per
`doc_id` as "4 models × 2 arms × 5 cues". That is wrong on both counts: it
includes arm, which is constant, and omits language, which is the largest factor.
Verified directly: every `doc_id` in the corpus appears in exactly 120 rows, with
4 model levels, 6 language levels, 5 cue levels and 1 arm level.)

The correction matters. At 120 exposures rather than 40, the cascade is three
times more severe than the wrong figure implies, and the observed 82.5% loss is
only explicable at 120.

### 4.2 The probability mechanism

With a per-row truncation rate of p = 0.0247 and 120 independent exposures:

```
P(at least one truncated row per doc_id) = 1 − (1 − 0.0247)^120 = 0.950
```

Observed loss is 82.5%, somewhat below the independent-draw prediction because
truncation clusters within items — a long item tends to truncate in several
cells at once rather than in one. But the order of magnitude is right, and the
conclusion holds: at 120 exposures, *almost every item will be hit by something*.

The distribution of truncated rows per `doc_id` shows how thin the damage is
spread:

| truncated rows (of 120) | doc_ids | effect |
|---|---:|---|
| 0 | 70 | survive |
| **1** | **89** | **all 120 rows dropped for a single truncation** |
| 2 | 73 | dropped |
| 3 | 49 | dropped |
| 4 | 39 | dropped |
| 5–22 | 78 | dropped |

**89 `doc_id`s — 22% of the corpus — are removed entirely because exactly one of
their 120 responses hit the token cap.** One model, in one language, under one
cue, running long is enough to delete that item from all four models, all six
languages and all five cues.

### 4.3 The loss is non-random along the axis that matters

Truncation is not spread evenly, which is why the survivors are biased:

| by model | truncated | rate |
|---|---:|---:|
| qwen25-72b-instruct | 726 | 6.07% |
| qwen3-32b | 373 | 3.12% |
| sarvam-m | 56 | 0.47% |
| gemma3-27b-it | 27 | 0.23% |

| by language | truncated | rate |
|---|---:|---:|
| kn | 411 | 5.15% |
| te | 402 | 5.04% |
| ta | 262 | 3.28% |
| hi | 49 | 0.61% |
| bn | 35 | 0.44% |
| en | 23 | 0.29% |

Worst cells, harmful arm: `qwen25-72b-instruct / kn` at 21.6%,
`qwen25-72b-instruct / te` at 20.0%, `qwen25-72b-instruct / ta` at 12.2%.

Because exclusion is listwise, a Kannada truncation under `qwen25-72b-instruct`
deletes that item from the **English** `gemma3-27b-it` cells too. The single
worst-behaved model–language combination in the corpus therefore drives item
loss across the entire grid, and it does so preferentially for Dravidian-heavy,
longer items. This is the same language-correlated truncation that P3 and Phase 2
R5 warned about — and S1, the check designed to neutralise it, propagates it
instead.

---

## 5. Could S1 Be Fixed?

**Row-level exclusion instead of listwise.** Dropping only the 1,182 flagged rows
would preserve nearly all items. But it breaks the pairing that C1 depends on: an
item present under `eval_log` and missing under `monitor_log` cannot contribute a
discordant pair, so the denominators would differ per cue and the paired contrast
would no longer be computed on matched items. §3.2 rejected this explicitly —
"never drop a single cell" — and it was right to.

**Listwise within (model, language) rather than across the whole grid.** This
keeps pairing intact within each cell while stopping one model's Kannada
truncation from deleting another model's English data. It would retain far more
items. It is the most defensible repair.

**But any of these is a post-hoc deviation from a frozen plan.** The entire value
of a pre-registered sensitivity analysis is that its rule was fixed before the
labels existed. Rewriting the rule after seeing that the pre-registered version
is inconvenient converts S1 from evidence into decoration, and a referee is
entitled to say so. If S1 is ever restored under a modified rule, the
modification must be recorded as explicitly as this removal is.

**And the underlying question is weaker than it looks.** The primary analysis
already includes every truncated row. Those rows were judged on whatever text the
450-token window contained, and per §3.3 the window rule was designed precisely
so that truncation would be **near non-differential by construction** — a
response cut at token 5,824 and one that ended naturally at token 800 present the
judge with the same shape of input. S1 was always a belt-and-braces check on a
confound the window rule was built to neutralise, not the primary defence
against it.

---

## 6. Team Decision

The team reviewed the surviving-item count and decided S1 is uninformative as
written.

**Decision: drop S1 entirely rather than report a meaningless sensitivity
analysis.** Publishing a contrast computed on 14 non-randomly selected items,
with intervals that cannot be interpreted, would be worse than publishing
nothing — it would invite a reader to weigh it against the primary as though the
two were comparable.

The deviation is recorded in `run_log.md` under CC Prompt 9. The frozen plan is
**not** modified; the record of departure lives in the log, as it should.

**The plan's own tie-break rule is moot.** §3.4 ends:

> If S1 and the primary disagree in direction, report both and take the more
> conservative as the headline. Decide that now.

That rule presumes S1 can produce a directional estimate worth comparing. At
n = 14, with one item worth 0.071 against effects of 0.046, it cannot. The single
observed sign flip is one item moving. Applying the rule literally would have let
a single Telugu item in one model override a result computed on 199 items across
24 cells. The rule was written in good faith for a version of S1 that had a
usable sample; it does not survive contact with a 93% item loss.

---

## 7. Mitigation

Dropping S1 does not leave truncation unexamined.

**1. The primary analysis includes all rows.** No exclusions. Every truncated
response is in the headline numbers, judged on the text that exists in its
window. The result is not conditional on a filtering choice.

**2. The window rule already addresses the confound by construction (§3.3).**
Judging a fixed 300-head + 150-tail token window means a truncated response and a
naturally-ended one present the judge with inputs of the same shape. Truncation
was designed out of the DV rather than filtered out of the sample. §3.3 lists
this as the first of its four properties: "the 21.6 percent Kannada truncation
rate stops being a length confound."

**3. `unusable_rate` is reported per cell in §8.7.** If truncation were producing
unjudgeable responses at different rates across cells, it would surface there.
Across all 47,880 rows the judge returned only 11 `UNUSABLE` labels (0.02%), and
10 of those are a single sarvam-m generation artifact rather than truncation
damage — so truncated responses are, empirically, still being judged.

**4. S3 remains planned and addresses this more directly.** §3.4's third
sensitivity analysis re-judges a 4,000-row stratified subsample on the **full
response** rather than the window. That tests the window rule itself, which is
the real question underneath S1: not "what if we delete truncated items?" but
"does judging a window instead of the whole response change the answer?" S3 needs
a judging run that has not happened. It is deferred, not dropped.

**What is genuinely lost.** S1 would have given a pre-registered, one-line answer
to "does the result depend on truncated rows?" There is now no such answer, and
the mitigations above are arguments rather than a measurement. If a referee asks
the question directly, the response has to be §3.3's construction argument plus
S3 when it exists — which is defensible, but it is reasoning rather than a
number. That cost should be stated in the limitations section rather than left
for someone else to notice.

---

## Appendix — verification

Every figure in this document was recomputed read-only from
`phase3/j0_triage_results.jsonl` and the four `GEN_ROOTS` generation trees on
2026-09-20.

| quantity | value |
|---|---:|
| total rows | 47,880 |
| `trunc_clean` | 719 |
| `trunc_degenerate` | 463 |
| either | 1,182 (2.47%) |
| rows per `doc_id` | 120 (4 models × 6 langs × 5 cues; arm constant) |
| `doc_id`s total | 399 (199 harmful, 200 benign) |
| `doc_id`s excluded by S1 | 329 (82.5%) |
| harmful items surviving | 14 of 199 (93.0% lost) |
| benign items surviving | 56 of 200 (72.0% lost) |
| `doc_id`s killed by exactly one truncated row | 89 |
| P(≥1 of 120 truncated), p = 0.0247, independent | 0.950 |
| observed `doc_id` loss rate | 0.825 |
| median item length, survivors vs dropped (harmful) | 112 vs 188 chars |
| median response length, survivors vs dropped (harmful) | 475 vs 655 chars |

Related: `phase3/S2_Problems.md` documents the parallel investigation into S2.
