# qwen3-32b — Phase 2 behavioural run

Results only. Pipeline code, configs and activations are not included here.

| | |
|---|---|
| model | `Qwen/Qwen3-32B` @ `9216db5781bf21249d130ec9da846c4624c16137` |
| run_id | `20260822T041507Z-qwen3-invariant` |
| manifest_sha | `e059b2088842edd2` |
| generations | **11,970** = 6 languages × 5 cues × (199 harmful + 200 benign) |
| engine | vLLM 0.19.1 · torch 2.10.0+cu128 · transformers 4.57.6 |
| decoding | bf16 · greedy (`temperature=0`) · seed 2026 · thinking **disabled** |
| determinism | `batch_invariant: true`, `attention_backend: TRITON_ATTN` |
| context | `max_model_len: 8192` (raised from 4096 — Indic prompts ~4× gemma's) |
| wall clock | 8h31m |

## Layout

```
generations/<arm>/<lang>/<cue>.jsonl      60 shards, one row per generation
preflight/manifest.json                    input hashes, versions, doc_id sets
preflight/g0.json g0parity_qwen.json       input + cue-parity gates
preflight/g0fit_qwen.json g3dry_qwen.json  context-fit + cue-integrity gates
preflight/g2_qwen3-32b.json                model-load gate
g1_determinism/                            duplicate shard proving G1 = 1.0000
results.md run_log.md record.md            findings, history, decisions
```

## Gate results

| gate | result |
|---|---|
| G0 inputs | PASS |
| G0.5 cue parity | **FAIL as coded** (max 8.25 vs 2.5) — assessed a false positive; see results.md |
| G0 context-fit | PASS (near-zero headroom by construction) |
| G2 model load | PASS — **no `<think>` block**, thinking suppression verified |
| G3 cue integrity | PASS — cue present 11,970/11,970 |
| **G1 determinism** | **1.0000** byte-identical, token ids too |
| G4 / G5 language match | PASS |
| **G6 truncation** | **FAIL** — `te/harmful` 6.93%, `kn/harmful` 7.44% (limit 5%) |

## Read before using

- **n = 199 on the harmful arm, not 200.** Hindi item 160 was refused by the
  translation model; under the within-item rule that doc_id is dropped from all six
  languages. Benign is an independent draw at 200. **Item 160 is deliberately not in
  this bundle.**
- **G6 fails for `te` and `kn`, and no budget fixes it.** 88 of Kannada's 120
  truncations are repetition loops that never terminate; `kn` is provisioned at 5,888
  of a maximum possible 5,889 tokens. **Phase 3 must exclude degenerate rows and report
  the exclusion rate per language**, or Dravidian cells rest on fewer usable
  observations than the others.
- **Degeneration is ~60× gemma's**, concentrated in Dravidian languages
  (kn 4.46%, ta 4.16%, te 3.86% vs en 0.05%).
- **`response_text` on the harmful arm is raw model output to harmful prompts.**
- **No native-authored Indic refusal lexicon exists**, so refusal is not scored for
  hi/bn/ta/te/kn. Any cross-lingual refusal figures elsewhere are provisional.
- Qwen generates ~7× more tokens than gemma in Indic but ~0.35× in English — the
  length relationship reverses between models.
