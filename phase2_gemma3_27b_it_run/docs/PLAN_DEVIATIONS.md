# Deviations from `Final_Phase_2_Plan_main1.md` (rev 3.2)

Audited 2026-08-18. **D4 corrected after re-verification — see that row.** Every row is either **reality ≠ plan** or **a deliberate choice
I made**. Marked `[BLOCKING]` where it stops a gate, `[STALE]` where the plan's own
text is now out of date, `[MINE]` where it is my deviation.

---

## 1. Environment — the plan targets a different machine

| # | Plan | Reality | Impact |
|---|---|---|---|
| E1 **[resolved 2026-08-19]** | Sharanga: `gpu_h100_4`, `gpu_h200_8`, per-user cap 3 GPUs | one node `2xh100-nvl-bitspilani-vm2`; partitions `h100-full` (DenyAccounts=students*) and `h100-mig`; 1 full GPU + 7× 12 GB MIG | §2.3 resource map rewritten; array throttle `%3` → `%1` |
| E2 | 80 GB H100 | **94 GB H100 NVL** | KV re-derived: ~27 GB KV ≈ 54k tokens, ~2× the plan's budget. `4096`/`48` kept, now with *more* headroom |
| E3 | `--mem` 120–160 G per model | **32 G everywhere** | CLAUDE.md rule 4 is a hard rule and overrides the plan: an unbounded job reserves ~503 GB and blocks the node |
| E4 | `/scratch/$USER/phase2/...` | **no `/scratch`**; 8.3 TB on `/localstorage` | all §7.4 paths → `./phase2_scratch` |
| E5 | vLLM **0.27.1** pinned | **0.23.0** installed | **Verified: every pinned API fact still holds** — `TokensPrompt.prompt_token_ids` ✓, `generate()` has no `prompt_token_ids` kwarg (V3) ✓, `Logprob.rank` ✓ (moved to `vllm.logprobs`), `VLLM_BATCH_INVARIANT` + `batch_invariant` module ✓, all `SamplingParams`/`EngineArgs` kwargs ✓. So this is a **reproducibility-claim mismatch, not a code break** |
| E6 | `transformers` pinned (V1 / Gemma-3 system-role / #40849 are version-dependent) | **5.12.1** — a major-version jump | **The riskiest version gap.** V1's whole finding is about Gemma 3 template behaviour in a specific transformers line. Must be re-verified at G2, not assumed |
| E7 | `pyarrow` in the env | **absent** | `consolidate.py` cannot import; `build_benign_arm.py` cannot read `Dolly.parquet` |
| E8 | `pytest` (implied by C5) | **absent** | tests ship with a ~30-line shim; `pytest tests/` works unchanged once installed |
| E9 `[BLOCKING]` **CONFIRMED** | Driver 580.126.20 / CUDA 13.0, "do not port the CSIS version pin" | **Compute node: driver 565.57.01, CUDA 12.7.** Installed torch is `2.11.0+cu130` (needs driver ≥580) → `torch.cuda.is_available() == False` **on the GPU node**. Verified by jobs 434/435 | **NO GPU WORK CAN RUN.** Needs either a CUDA 12.x torch/vLLM install (rule 2, user) or a driver upgrade to ≥580 (rule 1, admin) |
| E10 | conda env `p2` under `~/miniconda3` | env `vllm` under `~/miniforge3` | sbatch template rewritten |
| E11 | repo root `slay-eval-phase2/` | `Behavioral-IndicEvalAwareness/` | cosmetic, but §7.1 and Appendix C paths do not match |

## 2. Model access `[BLOCKING]`

| # | Finding |
|---|---|
| M1 | **`google/gemma-3-27b-it` — `gated=manual`, files return 403.** *"Access to model google/gemma-3-27b-it is restricted and you are not in the authorized list"* on account `Trizal`. Blocks G0's "Gemma licences accepted", **G0.5** (needs the tokenizer), **G2**, and `token_budget.py`. This is the model selected for this run |
| M2 | **`google/gemma-3-27b-pt` — separately gated, also 403.** Needs its own acceptance |
| M3 | `Qwen/Qwen3-32B` and `sarvamai/sarvam-m` are ungated and fetch fine |
| M4 | Plan §3.5/G0 require all repos "staged and SHA-pinned". **Nothing is staged**; only `gemma3-27b-it` has a pinned SHA (`005ad3404e59…`, readable from public metadata without a licence) |

## 3. Inputs and data

| # | Plan | Reality |
|---|---|---|
| D1 `[BLOCKING]` | `itemnum` 1..200 contiguous (§3.1, G0's `range(1,n+1)`, `verify.py`, §6) | **0-based (0–199) in every harmful file**, including `final_harmful_200_en.json`. Benign is already 1-based — so the two arms disagree with each other |
| D2 `[BLOCKING]` | six canonical `final_harmful_200_<lang>.json` | **only `en` exists**, and it is 0-based → G0 reports `harmful/en: MALFORMED`, the other five `MISSING FILE` |
| D3 `[BLOCKING]` | `data/` holds en,hi,bn,ta,or (§0.5) | **The local working copy is STALE, not inconsistent.** `harmful_bn.json` is absent locally; the other five local files are byte-different from their GitHub counterparts. The GitHub repo has all six |
| D4 **CORRECTED** | within-item design: identical `doc_id` set in all six (§1.1) | **In the GitHub repo the design is INTACT** — item-47 swap (`f39243b1…` → `438b0290…`) applied consistently in **all six** languages, and all six `doc_id` sequences are identical to `en`. My earlier claim that only `bn` had the swap was an artefact of comparing repo-`bn` against **local** copies of the other five. Only the local checkout lacks the swap |
| D5 `[STALE]` | §3.6: refusals in `hi` 160, `bn` 56, `or` 56 | **Superseded upstream.** In the GitHub repo only **`hi` item 160** is still `refused`; `bn`, `ta`, `te`, `kn` are all 200/200 `ok`. Items 56 and 160 were evidently re-translated after the plan was written |
| D6 | §3.6 refusals to resolve | On **repo** data the §3.2 all-or-nothing drop removes **1** doc_id → **n = 199** (not the n=198 I computed on stale local data). Resolving `hi` 160 alone would restore n = 200 with no prefix extension |
| D6b `[NEW]` | §3.1 rows must be complete | **Two doc_ids carry no COMET-Kiwi score in some languages**: `b6d11d84…` (item 160) unscored in hi/ta/kn, `e6efb528…` (item 56) unscored in bn/ta/kn. They pass a `status == ok` check but have **no QE evidence**, so the SOP's 0.70 threshold was never applied to them |
| D7 | §0.4: *"`data/harmful_or.json` is retired, **not deleted**. Keep it committed and reference it in the manifest as retired-input provenance"* | **It is deleted** — gone locally *and* from the GitHub repo. The plan's retired-input provenance claim is currently unbackable |
| D8 | `translation_refusals.json` — *"required even if empty"* (§3.1) | missing |
| D9 | `data/final_set/_incoming/` as the delivery drop point (§3.1) | does not exist |
| D10 | benign arm built by a deterministic, reportable script (C2) | **files exist but no script produced them.** No stratification report, determinism unverifiable, length-matching unaudited — exactly what C2 calls unacceptable for a manifest-hashed input that is half the grid |
| D11 | §3.6 A/B/C decision + item-47 swap | **unrecorded.** Both `harmful_replacement_candidate.json` and `item47_replacement_and_56_160_scores.json` still say *"CANDIDATE, not yet applied"* — while `bn` has it applied |
| D12 `[STALE]` | §0.5 B1/B2/B3 (Tamil malformed; te and kn missing) | **all three fixed** — ta parses, te 200/200 `ok`, kn 198 `ok` |
| D13 | `config/max_tokens.json`, `docs/analysis_plan_frozen.md` | both missing (§3.2 says the language-substitution paragraph must be copied into the latter) |

## 4. Cue battery

| # | Finding |
|---|---|
| C1 `[BLOCKING]` **CORRECTED** | The "15 of 30 provisional" figure came from a **stale local copy**. The GitHub repo now claims `author_verified: [en,hi,bn,ta,te,kn]` — all six — and has **deleted** `provisional_pending_native_review`, `native_review_todo` and the `_meta.status` note recording kn as LLM-drafted. Still blocking: **no `reviewed_by`/`reviewed_on`/`review_verdict` anywhere**, and "author_verified" is not the named-native-speaker sign-off §3.3 V1 requires. **Needs confirming whether a real review happened** — see run_log 2026-08-18 Correction 3 |
| C2 `[BLOCKING]` | **Kannada was LLM-drafted** ("no prior kn cue text existed anywhere"), where §3.3 V1 requires the five kn strings to be **authored, not translated** |
| C3 | No `reviewed_by`/`reviewed_on`/`review_verdict` existed at all. `[MINE]` I added the Appendix-A `metadata` block — 30 entries, **all `pending`**. I marked en/hi/bn `pending` too, not `approved`: `_meta` says "author_verified" but no named reviewer is on record and G0 wants a signed verdict |
| C4 **CORRECTED** | The rename **had** been applied — in the GitHub repo, on schedule. My local copy was stale. `[MINE]` applied 2026-08-18 (backup kept) — `run.yaml` already listed `monitor_log`, so nothing could load the file before this |
| C5 | **G0.5 cue parity is uncomputable** — it needs 6 tokenizers and gemma is gated. Char-length proxy is 0.81–1.29 vs `en`, comfortably under 2.5, but **that is not the gate**: Indic token fertility is far worse than character ratio suggests |
| C6 | `deploy` is a non-null neutral framing, identical in choice across all six — **consistent** with Appendix A. Side effect: the I5 empty-system-message guard is now unreachable in the main grid |

## 5. Scope — running `gemma3-27b-it` alone

| # | Plan | With this model set |
|---|---|---|
| S1 | M = 6, main grid **72,000** | M = 1, main grid **12,000** |
| S2 | Sub-run B, 2,400 generations | **0.** §4.5 excludes Gemma by design (its template merges system into user), so the cue-placement robustness annex **cannot exist at all** here |
| S3 | §2.1 base↔instruct planned comparison, 2 matched pairs | **impossible** — gemma's pair is `gemma3-27b-pt`, also gated |
| S4 | §2.2 Indic-specialist ↔ general-multilingual axis | **gone** — that axis needs sarvam/qwen |
| S5 | §10: G8 discordance measured on **`sarvam-m`** across 6 languages | would have to run on gemma instead, changing the empirical basis of the 200-vs-500 decision |
| S6 | H3 cross-lingual dissociation | **survives** — 6 languages × 5 cues × 2 arms is intact on one model. This is the headline claim, and it is the part that still works |

## 6. TRANSLATION_SOP.md (blocker B8)

Plan §0.5 B8: *"numbers the same retry section §7a in two places and §8a in two others"*.
**Reality has mutated, not resolved:** exactly one heading `## 8a` now exists, but there
are **10 prose references to `§7a` and no `§7a` heading anywhere**. The duplicate was
fixed by renaming; the cross-references were never updated. For a document written to be
"executed literally with no clarifying questions", dangling references are arguably worse
than duplicated ones. Still blocks a clean `te`/`kn` re-run.

## 7. My deliberate deviations `[MINE]`

| # | Deviation | Why |
|---|---|---|
| X1 | `ModelCfg` gains a `gres` field; `models.yaml` rows carry `gres: gpu:nvidia_h100_nvl:1` | the plan's bare `gpu:N` does not resolve against this node's Gres declaration |
| X2 | `submit.py` **hard-refuses** to emit a job with no `--mem` or no `gres`; log dir absolutised | CLAUDE.md rule 4 enforced in code rather than trusted; N10 says Slurm won't create the log dir |
| X3 | `normalise_translations.py` adds a 4th provenance value **`source_en`** | §3.1 allows only opus/indictrans2/google_translate_manual. English is the *source*, not a translation; stamping it "opus" would be a false manifest claim |
| X4 | same script accepts already-1-based input and normalises fields **without** renumbering | the plan's idempotence wording implies a hard reject, but the benign files are 1-based *and* un-normalised (`text`, no provenance). A hard reject would leave them permanently unprocessable |
| X5 | `tests/_runner.py` shim; `test_assemble.py` uses a **stub tokenizer** | pytest absent and installing unauthorised; and real templates are G2/G3's job — C5 says tests test the code, gates test the data |
| X6 | C8 hash list names `scripts/02_screen_harm.py` as the harm-retention script | the plan describes it but never names a file |
| X7 | Nothing installed, nothing staged, nothing submitted, no data file rewritten | CLAUDE.md rules 2/3; and the §3.6 drop is a research decision |

## 8. Plan claims I could not verify

- §8.14: *"Item 47 fails `g0_context_fit` today on four of six languages."* Uncheckable without a tokenizer (gemma gated). Item 47 is slated for replacement anyway (COMET 0.54–0.58 in every language).
- C8: *"verify `14_sample_final_set.py` prefix stability empirically — generate 200 and 203, diff the first 200."* Not done. §3.2, §3.6 and §10's extend-to-500 branch all depend on it, and the plan flags that it is asserted throughout and never tested.
