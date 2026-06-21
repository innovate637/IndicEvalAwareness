# Scripts — guide to the pipeline

All scripts are run from the repo root with the project venv: `.venv/bin/python scripts/<name>.py`.
Shared config in `config.py` (paths, `MODEL_NAME=sarvamai/sarvam-m`, `NUM_LAYERS=40`, `D_MODEL=5120`).
Shared helpers in `scripts/behavioral_common.py` (`load_model_and_tokenizer(int8=)`,
`classify_refusal`, `generate`/`batch_generate`). GPU: set `CUDA_VISIBLE_DEVICES=N`.

The numbering is roughly chronological; the **canonical pipeline** is marked ★.

---

## A. Data preparation
| script | purpose |
|---|---|
| `00_smoke_test.py` | sanity-load the model, confirm 40 layers / d_model 5120 |
| `00b_build_prompt_pairs.py` | build eval/deploy prompt pairs |
| `08_build_benign.py` / `08b_build_benign_public.py` | build the matched **benign** control set (public data, no MT) → `data/safety_prompts/benign/` |
| `make_cue_review.py` | emit human-review sheets (one Markdown/lang) for the cue translations |
| `rerun_corrected_cells.py` | regenerate specific (lang, condition) cue-battery cells after a cue text was corrected |

## B. Behavioral generation (the eval/deploy refusal grid)
| script | purpose |
|---|---|
| `04_behavioral.py` | first behavioral pass (eval vs deploy refusal) |
| `05_cue_factorial.py` | the original 2×2 (eval/deploy × English/translated) cue |
| ★ `09_cue_battery_grid.py` | **main E1 generator** — the 5-condition cue battery (deploy / neutral_log / eval_log / eval_explicit / eval_metadata) × harmful prompts × langs. `--int8` for the 8-bit run, `--lang a,b`, `--items s,e` (shard across GPUs), `--resume` (per-batch checkpoint). |
| `11_crossmodel_openrouter.py` | run gemma-3-27b / qwen3-32b via OpenRouter → `crossmodel.csv` |
| `20_sarvam_grid.py` | sarvam-30b / sarvam-105b (MoE family) via Sarvam API |
| `22_quant_int8_sanity.py` | INT8-vs-INT4 sarvam-m generation (validates 8-bit fidelity; feeds the quant-ceiling result) |

## C. Refusal judging (the measurement-validity core)
| script | purpose |
|---|---|
| ★ `18_refusal_judge.py` | **primary judge** — gemma-3-27b, prompt+answer, verdict ∈ {REFUSE,COMPLY,NOANSWER} + `prompt_harmful` screen. Holds the shared `JUDGE_PROMPT`. → `refusal_judge.csv` |
| `18b_judge_crosscheck.py` | re-judge a stratified sample with a 2nd free model (judge-reliability check) |
| ★ `21_sarvam_judge.py` | **second judge** — sarvam-105b (Indic-native, free API), same rubric → `refusal_judge_sarvam105.csv`; `--compare` prints agreement |
| `23_quant_sanity_judge.py` | judge the INT8 quant-sanity generations (gemma) |
| ★ `24_judge_int8_en.py` | judge the INT8 **English** cue battery with **both** judges; `--compare` → the quant-ceiling table |
| ★ `28_judge_int8_all.py` | judge the INT8 cue battery for **all languages** (sarvam-105b) → the language-wise gap table |

## D. Statistics
| script | purpose |
|---|---|
| `16_stats_and_judge.py` | V3/V4 stats + VEA-judge module (no GPU) |
| ★ `19_confirmatory_stats.py` | **confirmatory** GEE (binomial, exchangeable, clustered on base item) + cluster bootstrap for the compliance-gap contrasts |
| `21_residual_analyses.py` | residual analyses (note: number collides with `21_sarvam_judge`) |

## E. Recognition & CoT pillars
| script | purpose |
|---|---|
| `07_recognition.py`, `10_recognition_v2.py` | does the model verbally recognize the eval cue (P(says "evaluation")) |
| `17_cot_monitor.py` | CoT-monitorability: blinded monitor reads `<think>` only; faithfulness vs behavior |

## F. Mechanistic (E4) — the core of the cross-lingual claim
| script | purpose |
|---|---|
| `01_extract_activations.py`, `02_train_probes.py`, `03_build_steering.py` | v1 mechanistic (superseded by 12–14) |
| ★ `12_extract_activations_v2.py` | **activations** — INT8, last prompt token, all 40 layers; joins calibrated **judge** refusal labels → `results/activations/{lang}_{cond}.pkl` (acts[100,40,5120], item_id, refusal, vea) |
| ★ `13_probe_transfer.py` | diff-of-means directions + 5-fold logistic AUROC; **cross-lingual transfer** (EN→Indic); contrasts: `eval_framing` (eval_log−neutral_log), `cue_presence`, `awareness` → `directions.pkl`, `auroc_all.csv`, `transfer.csv` |
| ★ `13b_cross_wording.py` | **anti-lexical** test — train the eval-framing vector on eval_log−neutral_log, test it on other eval phrasings (held-out) |
| `14_steer_patch.py` | steering infrastructure — `load_directions`, `build_refusal_direction` (positive control), the forward-hook, `load_harmful_subset` |
| ★ `25_steer_xlang.py` | **cross-lingual causal steering** — forward-hook residual addition (constant per-layer-norm magnitude), sarvam-105b LLM-judged refusal, full control suite; reusable funcs: `steer_generate_batch`, `run_shard`, `judge_one`, `eval_dir_orth`, `build_random_direction`, `layer_norm_scale` |
| ★ `26_steer_probe_readout.py` | **manipulation check (readout A)** — two-hook forward pass: steer at L, capture at L+2, score a clean-trained eval-probe. The fast, continuous causal readout (does injecting the eval direction induce the eval-state, cross-lingually?) |
| ★ `27_steer_behavioral.py` | **behavioral readout (readout B)** — steer deploy generation at a mid layer, LLM-judge refusal on harmful + **benign** (specificity); positive (refusal-dir) + random controls; output namespaced by layer |

## G. Consults (advisory, paid GPT via OpenRouter — run on demand)
`consult_gpt55_*.py` — logged GPT-5.x consultations on design / mechanistic plausibility / novelty /
publishability. Outputs to `results/consult/`. (Also `25→27` design review: `results/consult/gpt_xlang_steer_review.md`.)

---

### Typical end-to-end run
```
# 1. generate behavioral grid (per language, INT8)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/09_cue_battery_grid.py --lang hi --int8 --resume
# 2. judge it
.venv/bin/python scripts/28_judge_int8_all.py --run
# 3. mechanistic: activations → probes → transfer
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/12_extract_activations_v2.py
.venv/bin/python scripts/13_probe_transfer.py && .venv/bin/python scripts/13b_cross_wording.py
# 4. causal: manipulation check then behavioral
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/26_steer_probe_readout.py --L 9 --resume
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/27_steer_behavioral.py --L 24 --resume
# 5. confirmatory stats
.venv/bin/python scripts/19_confirmatory_stats.py
```
Full results narrative: `RESULTS.md`. Mechanistic design/audit: `E4_AUDIT.md`.
