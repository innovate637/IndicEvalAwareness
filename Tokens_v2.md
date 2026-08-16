# Token budget — Phase 2 (revised)

Supersedes the previous `Tokens.md`. Two things changed: **Odia is replaced by Kannada**, and
every number below is now **measured** — against the real tokenizers and the completed Phase-1
generations — rather than derived from published fertility figures.

The headline is uncomfortable: **both Phase-1 runs were themselves truncated**, so the old plan
was calibrated against censored data and its budgets were far too small.

---

## 0. What was wrong with the previous version

| Claim in old doc | Status | Reality |
|---|---|---|
| Gemma Odia ratio `2.62` | **wrong** | measured **3.57** — under-budgeted by 38% |
| "computed values land at or below the table" | **wrong for the language that mattered** | true for hi/bn/ta, false for or |
| Only Qwen3-Odia hits the ceiling | **wrong** | on Qwen3, **all four** Indic languages hit 2048 |
| "worst case ≈ 600 tokens of prompt" | **wrong** | item 47 is 10,810 tokens on Qwen3-Odia — 2.6× the whole context |
| `r` measured over 400 prompts (200 harmful + 200 benign) | **input missing** | no benign set exists in the repo |
| Per-char vs per-word bias (caveat b) | **not a real problem** | per-char and per-item ratios agree within ~5% |

The *rule* was arithmetically sound — the formula and all six table values reproduce exactly. What
failed was every input fed into it.

---

## 1. Why the old rule's shape was wrong

The old rule was `max_tokens = clip(ceil(BASE_EN × r / 32) × 32, 512, 2048)` with `BASE_EN = 512`.

The defect is that **512 is arbitrary**. Nothing measured it; it was assumed, then everything else
scaled off it. If the anchor is wrong, per-language calibration just distributes the error
proportionally — every cell is wrong together, and `CEIL` then silently flattens the four cells
that need the most room.

The corrected rule anchors on **what the models actually emitted** in Phase 1:

```
budget(model, lang) = ceil32( cot_allowance(model) + answer_p(en, model) × r(model, lang) × BUFFER )
```

- `answer_p(en, model)` — the English answer-length percentile, measured, **uncensored**
- `r(model, lang)` — measured per-item token ratio vs English (§2)
- `cot_allowance` — reasoning models only; **does not scale with language** (§3)
- `BUFFER` — 1.25, the anti-rerun margin (§5)

---

## 2. Measured fertility, including Kannada

Per-item token ratio vs English, on the aligned 200-item set (`hi`/`bn`/`ta` from the real
translations; `kn` from a 33-item length-stratified sample, since `harmful_kn.json` does not exist
yet; `te` from the Phase-1 sets as the anchor that validates `kn`).

| Language | Gemma-3-27B | Qwen3-32B |
|---|---|---|
| `en` | 1.00 | 1.00 |
| `bn` | 1.21 | 5.31 |
| `hi` | 1.38 | 5.01 |
| `ta` | 1.50 | 6.41 |
| `te` *(anchor)* | 1.80 | 7.59 |
| **`kn`** | **2.00** | **7.61** |
| ~~`or`~~ *(dropped)* | ~~3.63~~ | ~~10.26~~ |

**Kannada is a real improvement over Odia.** On Gemma it costs 2.00× English instead of 3.63×; on
Qwen3, 7.61× instead of 10.26×. The single most expensive cell in the campaign gets ~30% cheaper.

Kannada lands almost exactly on Telugu (1.80/7.59), which is the expected result — sister Dravidian
languages with closely related scripts — and is the cross-check that makes the 33-item sample
trustworthy. **Re-measure once `harmful_kn.json` exists**; treat 2.00/7.61 as provisional.

---

## 3. Chain-of-thought does not scale with language

Splitting Qwen3's Phase-1 output into reasoning vs answer:

| lang | CoT median | CoT p99 | answer median |
|---|---|---|---|
| en | 319 | 1010 | 200 |
| hi | 292 | 932 | 1333 |
| bn | 299 | 946 | 1407 |
| ta | 331 | 1515 | 1665 |
| te | 300 | 940 | 1695 |

CoT length is **flat across all six languages** (median 264–331) while the answer swings 8×. Qwen3
reasons in English regardless of the prompt language. So the budget is not one quantity scaled by
`r` — it is a **language-invariant CoT allowance plus a language-scaled answer**. The old rule
scaled both, which inflates English and starves Indic simultaneously.

`cot_allowance = 1024` (covers p99 in five of six languages with margin).

---

## 4. What Phase 1 actually cost us

Observed truncation, answer span, at each run's own cap:

| lang | Gemma-3 @ cap 400 | Qwen3 @ cap 2048 |
|---|---|---|
| en | **93.0%** | 0.0% |
| hi | 17.7% | 12.0% |
| bn | 0.7% | 15.0% |
| ta | 18.3% | 24.2% |
| te | 7.0% | 27.3% |
| or | 51.0% | 25.2% |

Every Indic cell in Qwen3 blows Gate G6's 5% limit by 2.4–5.5×, and Gemma English truncated
**93%** of responses. This is the concrete reason not to reuse the old budgets: they are the
budgets that already failed.

It also means the true output distribution is **right-censored in both runs** — the real p95/p99
are above what we can observe. Every number in §5 is therefore a *lower bound*, which is exactly
why the buffer exists and why §7 matters.

---

## 5. The budgets

`BUFFER = 1.25`. Applied on top of a percentile that already covers the bulk of the distribution,
this is the margin that keeps a cell from needing a re-run when the real distribution turns out
slightly heavier than the censored estimate predicts.

### Gemma-3-27B (non-reasoning → no CoT allowance)

Anchored on the uncensored Qwen3 English **answer** p99 = 1534, which is the best available
estimate of answer content length (Gemma's own run was censored at 400 and yields nothing).

| lang | r | budget |
|---|---|---|
| en | 1.00 | **1920** |
| bn | 1.21 | **2336** |
| hi | 1.38 | **2656** |
| ta | 1.50 | **2880** |
| te | 1.80 | **3456** |
| **kn** | **2.00** | **3840** |

### Qwen3-32B (reasoning → +1024 CoT)

Two options. Pick one deliberately.

| lang | r | **G6-safe** (p95 base 933) | **no-rerun** (p99 base 1534) |
|---|---|---|---|
| en | 1.00 | 2208 | 2944 |
| hi | 5.01 | 6880 | 10656 |
| bn | 5.31 | 7232 | 11232 |
| ta | 6.41 | 8512 | 13312 |
| te | 7.59 | 9888 | 15584 |
| **kn** | **7.61** | **9920** | **15616** |

**Qwen3 on Indic is genuinely expensive** and this is the finding, not a rounding error. A Kannada
answer that would take 933 English tokens costs ~7,100 Qwen3 tokens before any margin. If that
cost is unacceptable, the levers are: run Qwen3 with `enable_thinking=False` (saves the flat 1024),
accept p90 instead of p95, or drop Qwen3 for Indic — **not** to shrink the budget and eat the
truncation, which is what Phase 1 did.

---

## 6. `max_model_len` must rise

The old value of 4096 cannot hold these. Longest **prompt** in the set (item 47, the degenerate
414-term wordlist):

| | en | hi | bn | ta | kn *(est.)* | or *(dropped)* |
|---|---|---|---|---|---|---|
| Gemma-3 | 1741 | 2227 | 2337 | 616 | ~3100 | 4457 |
| Qwen3 | 1860 | 5413 | 5909 | 1731 | ~11800 | 10810 |

Required context = longest prompt + budget:

| model | required `max_model_len` |
|---|---|
| Gemma-3-27B | **8192** (3100 + 3840 = 6940) |
| Qwen3-32B, G6-safe | **24576** (11800 + 9920 = 21720) |
| Qwen3-32B, no-rerun | **32768** |

Note `CEIL` is no longer an independent constant — it is `max_model_len` minus the longest prompt
for that model × language, and must be computed, not assumed.

**Item 47 deserves a separate decision.** It is a degenerate artifact of the source corpus, it is
1/200 items, and on Qwen3-Kannada it alone would force the context window from 8k to 24k. Dropping
it (documented, applied identically to all six languages) is defensible and buys back most of the
memory budget. If it stays, it must be budgeted for.

---

## 7. The one thing that removes the remaining uncertainty

**Every number in §5 is extrapolated from censored data.** You cannot set a no-rerun budget from
runs that were themselves truncated — that is the trap the previous version fell into.

Before committing 25+ GPU-hours, run a **calibration probe**:

- 40 items × 6 languages × 2 models, `max_tokens = 16384`, no cap that can bind
- ~1 GPU-hour
- gives the genuinely uncensored length distribution per cell

Then set `budget = ceil32(p99_observed × 1.25)` from real numbers and write it to
`config/max_tokens.json`. Everything above becomes the fallback if the probe is skipped.

The probe is cheap insurance: it costs ~1 GPU-hour against a ~25–40 GPU-hour main grid that would
otherwise have to be partly re-run — which is precisely what happened in Phase 1.

---

## 8. Gates

- **G6 (output truncation) ≤ 5% per cell** — unchanged, but now satisfiable by construction rather
  than discovered after the fact.
- **G6-input (new).** G6 measures *output* truncation and is blind to *input* overflow: a prompt
  longer than `max_model_len` is dropped or errors, never truncated, so it never appears in G6.
  Add a preflight assertion that `max(prompt_tokens) + max_tokens ≤ max_model_len` for every cell.
  Item 47 fails this today on four of six languages.
- **Override rule.** If a cell still fails G6, raise *that cell only* by 50%, delete its shard,
  re-run it, and record it in `preflight/token_budget_overrides.json`. If the raise would exceed
  the context window, raise `max_model_len` — never silently clamp, which reintroduces the
  censoring this document exists to remove.

---

## 9. Cost

Per model per language: 2 arms × 5 cues × 200 items = 2,000 generations.

Worst case (every cell hits its budget), Gemma-3 across six languages: **34.2 M** output tokens.
Qwen3 G6-safe: **89.3 M**. These are ceilings, not expectations — mean output runs well below cap.

The old doc's realistic estimate (~25–35 M for the whole six-model grid) applied a **flat**
350–500 mean across all languages, which contradicts the document's own premise that Indic costs
several times more per unit of content. Weighting the mean by measured `r` raises it by ~1.7× for
Gemma and far more for Qwen3. **The ~25–40 GPU-hour figure in §11.4 should be treated as light by
at least 2×**, and the calibration probe in §7 is what would replace that guess with a number.

Kannada now carries ~19% of the six-language token load on Gemma (Odia carried 26%), which is the
main practical benefit of the swap.

---

## 10. Provenance

| Number | How obtained |
|---|---|
| `r` for hi, bn, ta | 198 aligned items, `harmful_{en,hi,bn,ta}.json`, both tokenizers |
| `r` for kn | 33-item length-stratified sample, translated for measurement |
| `r` for te | Phase-1 `data/safety_prompts/te.json`, used to validate `kn` |
| output distributions | `results/behavioral/crossmodel_*_{en,hi,bn,ta,te,or}.csv`, n≈300/cell |
| truncation rates | at-cap fraction (within 2 tokens of that run's max) |
| prompt lengths | `harmful_*.json` tokenized directly |

Not yet measured, and flagged rather than guessed: Kannada on the full 200 items; the four models
beyond Gemma-3 and Qwen3; the uncensored tail of every distribution (§7).

Also outstanding, unrelated to tokens but blocking the pipeline: `harmful_ta.json` does not parse
(missing closing brace, item 56), and item 56 is `ok` in Tamil but `refused` in Bengali and Odia.
