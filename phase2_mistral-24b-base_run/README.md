# mistral-24b-base — Phase 2 behavioural run (BASE model)

`mistralai/Mistral-Small-3.1-24B-Base-2503` @ `ba6496e3dce1d0bdc93848804b1d4b9d5f3c57bc`
Run id `20260907T170523Z-mistralbase` · Slurm job 609 · completed 2026-09-08 (09:01:37)
vLLM 0.19.1 · torch 2.10.0+cu128 · transformers 4.57.6 · H100 NVL · bf16

This is the **true pretrained ancestor of `sarvam-m`**, so the two form the campaign's
one genuinely matched base/instruct pair.

## Contents
| path | rows | what |
|---|---|---|
| `generations/` | **11,970** | main grid: 2 arms x 6 langs x 5 cues x 199/200 items |
| `g1_determinism/` | 398 | two identical passes, harmful/en/deploy |
| `probe_calibration/` | 960 | budget probe, ceiling 3300 |
| `preflight/` | — | G2 verdict, G0, manifest |
| `config/` | — | models.yaml, run.yaml, max_tokens.json, cue battery, sbatch |
| `results.md` `run_log.md` `record.md` | — | full campaign record, all five models |

Grid verified: 60 cells, 11,970 unique `record_id`, 11,970 unique `prompt_sha`, one run id.

## Gates
- **G2 PASS.** Loads bf16, 1 BOS/prompt, token ids are ints.
- **G1 PASS = 1.0000** on `response_text`, `response_token_ids`, `first_token_logprobs`,
  `n_completion_tokens`, `finish_reason`. Bitwise reproducible.
- **G6 (truncation) 87.43 %** — advisory for a base model, but see (2): the rate is
  strongly language-dependent here, unlike on gemma3-27b-pt.

## READ THIS BEFORE SCORING

**1. THE TOKENIZER NEEDED A FIX, AND THIS RUN HAS IT.** This repo ships a **broken
pre-tokenizer regex**. Without `fix_mistral_regex=True`, 100 % of hi/bn/ta/te/kn prompts
tokenize differently (and ~35 % longer) while English is **byte-identical** — a
language-correlated corruption that would have handicapped exactly the five languages
under test and left the control clean. This run used the FIX (`config/models.yaml`,
`tokenizer_kwargs`). Any re-tokenisation you do downstream must pass the same flag or it
will not match `prompt_token_ids`. `sarvam-m`, though post-trained from this checkpoint,
is NOT affected.

**2. TRUNCATION IS STRONGLY LANGUAGE-CORRELATED — treat it as a covariate.**

| lang | en | hi | ta | bn | te | kn |
|---|---|---|---|---|---|---|
| natural stop % | **36.49** | 18.10 | 7.57 | 4.61 | 4.36 | **4.31** |
| mean tokens | 2089.7 | 2618.9 | 2892.3 | 2969.4 | 2963.9 | 2968.9 |

English finishes naturally more than a third of the time; Telugu and Kannada almost never
do — a **32 pp** spread (vs 2.6 pp on gemma3-27b-pt). Cross-language comparison of
continuation-harmfulness is confounded by English having far more complete continuations.
Raising the budget does not fix this: the median already sits at the ceiling in every
language, and the ordering is a property of the model.

**3. Budgets are a design choice, not a measurement.** The probe saturated in all six
languages, so `max_tokens` clipped to a flat **3072** rather than fertility-scaled values.

**4. `response_text` WAS REPAIRED; `detok_repaired: true` marks which rows.**
vLLM detokenises incrementally, so multi-byte UTF-8 characters split across two tokens
rendered as U+FFFD — **42.42 % of rows**, language-correlated (en 0.1 %, te 67.0 %),
because Tekken splits Indic scripts into byte pieces. Re-decoding the stored
`response_token_ids` in one batch restores them. **4,990 rows repaired; 88 (0.74 %)
remain damaged** where truncation severed a character at the very end — those are
unrecoverable. `response_char_len` was updated with the text; token ids were never
touched. Filter on `detok_repaired` if you need to isolate them.

**5. Cue echo is LOW — 11.0 % overall** (en 5.6, hi 7.9, bn 19.8, ta 12.7, te 5.6,
kn 14.3), without the cue-ordering seen on gemma3-27b-pt (41.5 %). Mistral continues the
*item*, not the cue. Use this model, not gemma3-27b-pt, if you need a base arm whose DV
is not contaminated by cue reproduction.

**6. Refusal-shaped text may be transcript continuation, not refusal** — same caveat as
every base model here. Per plan §2.1 the base DV is **continuation-harmfulness**.

**7. Language match:** en 99.3, te 97.2, hi 95.5, bn 94.1, kn 93.5, ta 90.7 %.

**8. G1 run-id labelling defect (cosmetic).** The G1 rows carry
`run_id = g1_gemmapt_A/B` because the sbatch was derived from gemma-pt's and the
identifiers were not rewritten. `model_slug` is correctly `mistral-24b-base` and the
directory is model-scoped, so the data are unambiguous — only the labels are wrong.

Activations (60 shards, 19 GB, shape (n, 41, 5120), 4 tensors/prompt, bf16) are NOT in
this zip.
