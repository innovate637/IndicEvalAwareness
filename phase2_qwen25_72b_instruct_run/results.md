# Results — Qwen2.5-72B-Instruct Phase-2 run

**Status: probe complete, token budgets measured.** Jobs A+B not yet submitted.

---

## Run identity

| | |
|---|---|
| Model | `Qwen/Qwen2.5-72B-Instruct` |
| Revision | `495f39366efef23836d0cfae4fbe635880d2be31` |
| Slug | `qwen25-72b-instruct` |
| Weights | `/scratch/jagatsesh/hf` (145 GB, 37/37 shards verified 2026-08-30) |
| Cluster | Sharanga (`hpc01.sharanga.local`) — the real one, unlike both precedents |
| Partition / QOS | `gpu_h200_8` / `qos_gpu_h200`, account `bits` |
| Precision | bf16 (no quantisation anywhere — plan §5.4) |
| Run id | assigned at Job A launch, persisted to `preflight/RUN_ID` |

## Design grid

2 arms × 6 languages × 5 cues = **60 shards**.
Harmful n=199, benign n=200 → **11,970 generations** expected
(199 × 6 × 5 = 5,970 harmful + 200 × 6 × 5 = 6,000 benign). Matches the
qwen3-32b run's verified total exactly.

## Measured before launch

Longest rendered prompt per language (all arms × all cues, tokenizer-measured
2026-08-30), against `max_model_len = 8192`:

| lang | max prompt tokens | headroom |
|---|---:|---:|
| en | 307 | 7885 |
| hi | 1521 | 6671 |
| bn | 1594 | 6598 |
| ta | 2026 | 6166 |
| te | **2343** | 5849 |
| kn | 2320 | 5872 |

Longest prompt anywhere: **2343 tokens**, leaving **5849 tokens** of completion
headroom. The te/kn figures are ~7.6× English, consistent with the qwen3-32b
precedent's finding that Qwen tokenizers are far less efficient on Indic scripts.

## Probe results (job 336608, 2026-09-09)

**Job:** 336608, COMPLETED, exit 0:0, elapsed 05:51:31 (2026-09-08 22:24 → 2026-09-09 04:16).
2,400 generations (40 items × 6 langs × 5 cues × 2 arms), `max_tokens_override=5824`.
Model at 67.8 GiB/GPU on 2×H200, TP=2, TRITON_ATTN, KV cache 371,536 tokens.

### Measured token budgets (`config/max_tokens.json`)

| lang | heuristic | measured | Δ | note |
|---|---:|---:|---:|---|
| en | 512 | **1280** | +768 | FLOOR artifact confirmed (2.5× under-estimate) |
| hi | 2272 | **2880** | +608 | |
| bn | 2592 | **3648** | +1056 | 1/400 truncated (0.25%) — negligible |
| ta | 2688 | **5824** | +3136 | 45/400 truncated (11.25%) — hit ceiling |
| te | 3616 | **5824** | +2208 | 52/400 truncated (13%) — hit ceiling |
| kn | 3360 | **5824** | +2464 | 62/400 truncated (15.5%) — hit ceiling |

### Saturation verdict

ta/te/kn p99 saturated at the 5824 ceiling. Spot-check of the longest kn response
(5824 tokens, `finish_reason=length`): coherent numbered points in Kannada in the
first portion, then the same phrase repeated 3× with the third cut off mid-word.
**Confirmed degenerate repetition**, not genuine content truncation.

The p50 values (866–976 for ta/te/kn) confirm the vast majority of responses finish
naturally well within budget. Raising `max_model_len` beyond 8192 would only extend
the repetition loop. Cross-model comparability preserved — Gemma3 and Qwen3-32B
exhibited the same pattern.

**Budgets accepted. `config/max_tokens.json` is authoritative.**
Heuristic backup at `config/max_tokens.json.bak_heuristic`.

---

## Job A — Generation (job 337646, 2026-09-10)

| | |
|---|---|
| JobId | 337646 (`iea_q72b_gen`) |
| State | COMPLETED, exit 0:0 |
| Elapsed | **10:04:04** |
| Period | 2026-09-09 15:52:00 → 2026-09-10 01:56:04 |
| Node | gpunode7, 2×H200 NVL 141GB, TP=2 |
| Run ID | `20260909T102224Z-qwen25-72b-instruct-invariant` |

**60/60 JSONL shards** produced in `generations/qwen25-72b-instruct/`.
**11,970 total rows** (5,970 harmful + 6,000 benign).

Actual runtime (10h) was well under the 24h walltime — no resume cycle required.
Both FlashInfer fixes held for the full run with zero errors.

### Truncation deep audit (probe data, pre-launch)

Manual review of all 80 truncated probe responses:
- ~80–85% show clear phrase/sentence-level repetition in final 150 chars
- ~10–15% ambiguous (degrading/circular)
- **0 cases** of clearly novel content being cut off

Team decision: budgets accepted as-is. Truncation stops junk, not real content.

---

## Pending — to be filled after the run

- [ ] G0–G8 gate outcomes
- [ ] G1 determinism (target ≥0.99 byte-identical; qwen3-32b reached 1.0000)
- [ ] G6 truncation rate per language (qwen3-32b **failed** here: te 6.93%, kn 7.44%)
- [ ] Refusal rates by arm × lang × cue
- [ ] `eval_log − monitor_log` contrast, the load-bearing comparison
- [x] Wall-clock: **10:04:04** — comfortably within 24h, no resume needed
- [ ] Activation capture completeness (60 shards × 4 variants)

## Standing caveats to carry into any writeup

1. **Qwen2.5 injects its own system prompt** into every row; neither precedent run
   has a system block. Accepted deliberately — see `docs/PLAN_DEVIATIONS.md`
   D-72B-1. Any three-model comparison must state this.
2. **`ta`/`te`/`kn` cue strings were never natively reviewed** (inherited C1/C2).
   The `eval_log − monitor_log` contrast is unconfirmed to survive translation in
   any Dravidian language.
3. **No native Indic refusal lexicon exists**, so the G7/G8 heuristic refusal
   scoring is reliable only for en/hi.
4. **Harmful n=199, not 200** — `hi` item 160 (`b6d11d84…`) was refused by the
   translation model on NCII grounds and dropped from all six languages. The
   paper must state n=199.
