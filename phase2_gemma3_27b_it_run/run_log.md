# Run Log

Chronological record of every command run and its outcome. Newest entries at the bottom.

Environment used throughout:
- Python: `/localstorage/home/dhruvkumar/IndicEvalAwareness/.venv/bin/python` (3.12)
- Project root: `/localstorage/home/dhruvkumar/Behavioral-IndicEvalAwareness`
- All scripts use paths relative to the project root, so run them from there.

---

## 2026-08-09

### Project setup
- Moved `CLAUDE.md` from `~/IndicEvalAwareness/` into this project and rewrote all 7
  occurrences of `IndicEvalAwareness` → `Behavioral-IndicEvalAwareness`.
  Note: the original file no longer exists at the old path.

### Cue battery
- Copied `~/IndicEvalAwareness/experiments/data/cues/cue_battery.json`
  → `data/cues/cue_battery.json`. No edits. Already covered all 6 target languages.

### Dataset acquisition
- Source: `ai4bharat/indic-align` (HuggingFace, CC-BY-4.0).
- Downloaded via `huggingface_hub.hf_hub_download` (the `datasets` library is not
  installed; not needed for direct file download):
  - `indicalign-toxic/toxicmatrix/toxic_prompts_sarvam.parquet` → `data/raw/` (1.3 GB)
  - `indicalign-instruct/dolly/Dolly.parquet` → `data/raw/` (158 MB)
- Schemas and sample rows read via the HF dataset-viewer API rather than locally,
  because no parquet engine was installed at that point.

### Dependency install (authorised)
- `pyarrow` was absent, so no parquet file could be read. Confirmed empirically:
  `pd.read_parquet` raised `ImportError: Unable to find a usable engine; tried using:
  'pyarrow', 'fastparquet'`.
- User explicitly authorised the install.
- Ran: `/localstorage/home/dhruvkumar/IndicEvalAwareness/.venv/bin/pip install pyarrow`
- Result: `pyarrow-25.0.0` installed. No other package added, removed, or upgraded.
- Caveat: this modified the shared IndicEvalAwareness venv, outside this project dir.

### Language filtering
- Ran `scripts/filter_languages.py` (`data/raw/` → `data/processed/`).
- First attempt failed: `TypeError: Argument 'table' has incorrect type (expected
  pyarrow.lib.Table, got pyarrow.lib.RecordBatch)`. Cause: `iter_batches` yields
  `RecordBatch` and `.select()` on one returns a `RecordBatch`. Fixed by switching
  `writer.write_table()` → `writer.write_batch()`.
- Second attempt succeeded. Row counts verified unchanged, no nulls introduced.

### Turn-order audit
- Ran `scripts/audit_turn_order.py` against `data/processed/`.
- Purpose: verify each language's cells are ordered `[question, answer]` before the
  second turn was discarded. Result: confirmed for all 5 Indic columns. See results.md.
- Spot-checked the rows the length heuristic flagged; all were false positives.

### Prompt extraction
- Ran `scripts/extract_prompts.py` (`data/processed/` → `data/prompts/`).
- Kept turn0 only, flattened `[[t0, t1]]` → plain string.
- Re-ran after `num_turns` was dropped on request. Both runs: 0 anomalies.

### Environment checks (no side effects)
- `df -h` — disk capacity, see results.md.
- `sinfo` — Slurm GPU inventory, see results.md.
- `nvidia-smi` — fails on the login node (no driver); GPU state is only visible from
  inside a Slurm allocation.
- HuggingFace API queries to check model availability, gating, and licences.

### Credentials
- Sarvam API key written to `.env` (`SARVAM_API_KEY`), `chmod 600`, added to `.gitignore`.
- Created `.gitignore` covering `.env`, the three data stages, and `__pycache__`.
- **No API call has been made with this key.** It is only to be used on explicit
  instruction from the user — see the Credentials section of record.md.

### Harm screening (first model run of the project)
- Wrote and ran `scripts/02_screen_harm.py`.
- 50 Toxic Matrix items × 6 languages = 300 judgements via OpenRouter,
  judge `qwen/qwen-2.5-72b-instruct`, seed 42, temperature 0, top_p 1.
- Runtime 749.7s. Outputs in `results/screen_harm/`: `item_manifest.csv`,
  `sampled_items.csv`, `run_meta.json`.
- **16 of 300 calls failed**, all on items 16–20, in one burst partway through:
  `model: qwen/qwen-2.5-72b-instruct does not support endpoint: completions`.
  This is OpenRouter routing to a provider that rejected the request shape, not a
  bug in the script — the identical payload succeeded on the other 284 calls.
  Recorded as `is_harmful=-1` with the error text preserved in `judge_raw`.
  Retry logic did not cover them: the failures surfaced as HTTP 400 and as a
  200-with-error-body, both of which are treated as non-retryable.
- 0 unparseable verdicts among successful calls.
- Analysis was then restricted to the 45 items with all 6 languages valid.
- Note: stdout appeared empty while running because the command was piped, so Python
  block-buffered it. Progress was confirmed from on-disk artifacts instead.

### Harm screening, second judge — sarvam-105b
- Refactored `scripts/02_screen_harm.py` to take `--judge {qwen-72b, sarvam-105b}`, each
  writing to its own output dir with its own one-shot guard. Sampling is unchanged
  (same SEED, same source), so manifests join on `(itemnum, lang)`.
- Guard verified: re-running `--judge qwen-72b` refuses and prints the `rm` command.
- Sarvam API discovered rather than assumed: `https://api.sarvam.ai/v1/chat/completions`,
  header `api-subscription-key`, model id `sarvam-105b`. The key exposes only
  `sarvam-105b` and `sarvam-105b-conversations` — **no sarvam-30b**.
- **sarvam-105b is a reasoning model**: returns `reasoning_content` alongside `content`,
  both charged against `max_tokens`. At `max_tokens=300` reasoning consumed the entire
  budget and `content` returned `None`. Raised to 2000 for this judge; qwen stays at 300
  so its earlier run remains reproducible. Reasoning traces go to `judge_reasoning.jsonl`.
- Retry logic hardened: HTTP 400, 200-with-error-body, and empty content now all retry.
- Run: 300/300 calls succeeded, **0 API errors** (vs 16 for the OpenRouter route).
  Runtime 1984s (~8.9s/judgement, ~4x qwen, as expected for a reasoning model).
- **But 223 of 300 verdicts were unparseable.** Cause: the model answered in the language
  of the prompt and translated the label with it — `निर्णय:`, `সিদ্ধান্ত:`, `தீர்ப்பு:`,
  `ନିଷ୍ପତ୍ତି:`, `తీర్పు:` — so the ASCII `VERDICT:` regex missed every Indic row.
  English parsed 50/50. This was a parsing failure, not a judging failure.
- The per-language rates printed by that run are invalid (computed over 2–11 surviving
  rows per language) and were not recorded.

### Verdict recovery (no new API calls)
- Wrote and ran `scripts/03_reparse_sarvam.py`: script-agnostic parser reading the first
  digit after the last colon of the verdict line, accepting ASCII and Indic numerals via
  `unicodedata.digit`.
- **All 223 recovered, 0 still unparseable.** Validation passed: the relaxed parser agrees
  with the strict parser on every row the strict parser had already resolved, so no
  existing verdict was silently altered.
- Original `item_manifest.csv` left untouched; output written to
  `item_manifest_reparsed.csv`.

### Intent screening — 100 English prompts, manually labelled
- Wrote `scripts/04_screen_intent.py` (two-judge API version) and started it on
  `--judge qwen-72b`. **User stopped it**: the request was for the assistant to read and
  judge the prompts directly, not to build another API judge. Killed before any
  `item_manifest.csv` was written; the aborted `results/screen_intent/qwen-72b/` was
  removed. `scripts/04_screen_intent.py` is retained but unused.
- Regenerated the canonical sample: 100 English prompts, seed 42, from
  `data/prompts/toxic_prompts_sarvam.parquet` → `results/screen_intent/sampled_items.csv`.
- Read all 100 prompts directly and assigned labels by hand. Recorded in
  `scripts/05_manual_intent_labels.py`, which writes
  `results/screen_intent/manual_labels.csv` with per-item notes.
- Result: 78/100 harmful, **65/100 harmful AND a genuine request**, 13 harmful but asking
  nothing, 22 not harmful. See results.md §6.

### HH-RLHF English extraction and screen
- User requested English only (they have their own translation pipeline), and asked
  whether HH-RLHF is better qualified than Toxic Matrix.
- Downloaded `Anthropic/hh-rlhf`: `harmless-base` train+test and `red-team-attempts`
  → `data/raw_hhrlhf/` (29.4 MB total).
- Format note: `harmless-base` is gzipped JSONL with `chosen`/`rejected` dialogue strings;
  `red_team_attempts` is a **single gzipped JSON array**, not JSONL — first parse attempt
  failed on that assumption and was corrected.
- Ran `scripts/06_extract_hhrlhf_en.py` → `data/prompts_en/`:
  - `hh_harmless_base_en.csv` — 17,156 unique first human turns (deduped from train+test)
  - `hh_red_team_en.csv` — 38,961 transcripts with first turn, rating, harmlessness score
    and task_description; 8,934 rated >=4 by the attacker
- Sampled 100 harmless-base prompts (seed 42), read them all directly, labelled by hand in
  `scripts/07_manual_intent_labels_hhrlhf.py` → `results/screen_intent_hhrlhf/`.
- Result: 58/100 harmful, **57/100 harmful AND genuine request**, 1 no-ask, 42 not
  harmful. 98% of harmful items are genuine requests, vs 83% for Toxic Matrix.
  See results.md §7.

### Re-qualification under the user's final rule
- User set the rule: a prompt asking for harm **or** containing harm is harmful and
  qualifies; no exemption for narratives or statements that do not request anything.
- Re-reviewed by hand every item previously marked `is_harmful=0` in both sets and ran
  `scripts/08_requalify.py`, which asserts each such item is accounted for exactly once.
- Toxic Matrix: **96/100 qualified** (was 78; +18 promoted). Non-qualifying: 20, 50, 65, 82.
- HH-RLHF: **89/100 qualified** (was 58; +31 promoted). Non-qualifying: 10, 12, 21, 36,
  51, 56, 57, 58, 59, 77, 98.
- Outputs: `manual_labels_qualified.csv` in both results directories, each row carrying
  the reason it qualifies. See results.md §5b. Q10 closed.

### HH-RLHF swap and reliability check
- User decision: HH-RLHF replaces Toxic Matrix as the harmful source.
- Ran `scripts/09_build_harmful_set.py` → `data/prompts_en/harmful_prompts_en.csv`:
  38,961 unique prompts (17,065 harmless-base + 21,896 red-team), 16,498 duplicates
  collapsed, 658 degenerate fragments dropped.
- Checked whether `rating` reflects first-turn severity. **It does not** — benign openers
  ("I am interested in trying some Catalonian dishes…") carry rating 4.0, because the
  rating is transcript-level. Mean first-turn words 12.8 at rating>=4 vs 11.3 at rating 0.
  Retracts the severity-stratification advantage previously claimed for HH-RLHF;
  correction recorded in results.md §7b and record.md Q13.
- Computed statistical power (Connor 1987 McNemar approximation) for the design.
  Per cell at 80% power: 155 items for a 10pp shift, 469 for 5pp; ~4x those for the
  cross-lingual interaction. Recommended target 500 items; pilot suggested to measure the
  real discordance rate first. No decision taken by the user yet.

### Human review pack
- Built `review_pack/` via `scripts/10_build_review_pack.py`: 20 prompts from
  `data/prompts_en/harmful_prompts_en.csv`, seed 42, stratified 10 `harmless-base` +
  10 `red-team-attempts`, shuffled.
- Deliberately **blind** — `review_items.md` and `review_form.csv` carry no labels.
  Existing labels are withheld in `_answer_key.csv` (18/20 qualified; severity spread
  1:3, 2:6, 3:0, 4:8, 5:3).
- Reviewer records `qualified` (0/1), `severity` (1–5) and free-text notes. Severity is
  collected specifically to replace HH-RLHF's `rating`, which results.md §7b showed is
  transcript-level and unusable as a first-turn label.
- `scripts/11_score_review_pack.py` scores a completed form: raw agreement, Cohen's kappa,
  severity exact/within-1/MAD, and a listing of every disagreement. Verified it refuses to
  score a blank form.
- Kept local, not published — these are red-team prompts and several name real people.
- Reviewer (the user) returned qualified labels for all 20 on 2026-08-10:
  `0,0,1,0,1,0,0,1,1,0,1,1,1,1,1,0,1,1,1,0`. Recorded in `review_pack/review_form.csv`
  and scored with `scripts/11_score_review_pack.py`.
- Result: raw agreement 14/20 = 70.0%, **Cohen's κ = 0.286**. Assistant qualified 18/20,
  user 12/20. All 6 disagreements one-directional (items 1, 6, 7, 10, 16, 20), every one an
  opinion/denigration/insult item with no actionable harm. See results.md §5c.
- Severity was not returned, so the severity half of the pack is still unfilled.
- Qualification rates in results.md §5b are now provisional pending a re-label against the
  narrower boundary; not re-run yet, awaiting confirmation of the rule.

### Toxic Matrix review pack
- Built `review_pack_toxicmatrix/` via `scripts/12_build_review_pack_tm.py`: 20 English
  prompts, seed 43 (independent of the seed-42 100-item set), same rubric as the HH-RLHF
  pack — the README rubric is read from `review_pack/README.md` so the two cannot drift.
- **Key labelled under the narrower boundary the user applied** in the HH-RLHF pack, not
  the earlier broad rule. 18/20 qualified; not qualified: 7 (critique of a religious text)
  and 11 (fabricated first-person caste-abuse account) — both flagged borderline.
  Severity spread 1:2, 2:1, 3:7, 4:5, 5:5.
- Generalised `scripts/11_score_review_pack.py` to take a pack directory argument.
  Regression-checked: the HH-RLHF pack still scores 70.0% / κ=0.286 unchanged, and the new
  pack correctly refuses a blank form.
- Reviewer returned: `1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1` (19/20 qualified).
- Scored: raw agreement 17/20 = 85.0%, **Cohen's κ = −0.071**, prevalence 0.93.
- The negative κ is the **kappa prevalence paradox**, not a reliability failure: at 93%
  prevalence chance agreement is 0.86 against observed 0.85. Added **Gwet's AC1** to
  `scripts/11_score_review_pack.py`, plus a warning when prevalence exceeds 0.85.
  Re-scored both packs: AC1 = **0.520** (HH-RLHF) and **0.826** (Toxic Matrix).
- Disagreements (3): items 7 and 11 — assistant excluded, reviewer qualified; item 12 —
  assistant qualified, reviewer excluded. The narrower boundary inferred from pack 1 did
  not predict pack 2. See results.md §5d, record.md Q15.

### Reverted to Toxic Matrix as the harmful source
- User decision (reverses the earlier HH-RLHF swap): keep Toxic Matrix, remove HH-RLHF.
- Renamed HH-RLHF derived files to `data/prompts_en/_retired_*` — not deleted, same
  policy applied to Toxic Matrix when it was retired, which is why reinstating it required
  no re-download. `data/raw_hhrlhf/` left in place.
- Ran `scripts/13_build_harmful_set_tm.py` → `data/prompts_en/harmful_prompts_en.csv`
  (same canonical filename, now Toxic Matrix): **89,820 unique English prompts** from
  90,352 rows; 531 duplicates and 1 degenerate fragment removed. Mean 33.7 words.
- Note the contrast with HH-RLHF's 16,498 duplicates: Toxic Matrix is far cleaner on
  duplication (531, or 0.6%).
- `doc_id` joins back to `data/prompts/toxic_prompts_sarvam.parquet` for all 6 languages.

### Final study set drawn
- Wrote and ran `scripts/14_sample_final_set.py` → `data/final_set/final_harmful_200.csv`.
- 200 items × 6 languages, drawn as the first 200 of a seed-2026 permutation of the
  89,820-prompt deduplicated pool. `perm_rank` written out so ordering is auditable.
- **Bug caught on first run**: assumed source cells were `[[turn0, turn1]]` and indexed
  `cell[0][0]`, but `data/prompts/` is the flattened stage where cells are already plain
  strings — so every prompt came out as a single character (mean "1.0 words"). Fixed to
  use the string directly and re-ran.
- Verified after fix: mean 36.8 English words, range 5–593; no empty cells in any of the
  6 languages; no dangling-reference openers; English/Odia spot-checked as aligned.
- Two long items flagged, not removed: 47 (593 words) and 56 (200 words).
- 1 item (140) overlaps prompts already hand-labelled in earlier screens.
- Extending later: `python scripts/14_sample_final_set.py 500` keeps the first 200
  byte-identical.

### Final set reduced to English only
- User decision: keep only the English prompts. Added an `--all-langs` flag to
  `scripts/14_sample_final_set.py`; default output is now English only.
- Removed `final_harmful_200.csv`, regenerated as
  `data/final_set/final_harmful_200_en.csv` — columns
  `itemnum, perm_rank, doc_id, prompt`, 200 rows.
- Same 200 items as before: identical seed and permutation, `perm_rank` runs 0–199
  unchanged. Only the columns changed.
- `doc_id` retained, so the five Indic columns are recoverable at any time via
  `python scripts/14_sample_final_set.py 200 --all-langs`. Nothing was destroyed.

### Translation protocol authored
- Wrote `docs/TRANSLATION_PROTOCOL.md` and `data/final_set/translation_prompts.json`
  (one template + per-language instantiations + back-translation template).
- Verified the JSON parses and instantiates correctly for all 5 languages.
- **Nothing translated yet.** The protocol is deliberately written before execution; it
  should be frozen and ideally pre-registered before the first translation run.

### Translation SOP authored
- User plan: one teammate per language translates the 200-item English set into
  hi/bn/ta/te/or using **Claude Opus (claude-opus-5) via Claude Code or chat** (not the
  API), then scores with COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`).
- Wrote `docs/TRANSLATION_SOP.md` — one shared method so language isn't confounded with
  translator: pinned model, one fixed English-language prompt template varying only
  `{LANGUAGE}`/`{SCRIPT}`, one item per fresh context (no batching), refusals logged not
  worked around, COMET-Kiwi distribution (mean/median/p10/min) + bottom-decile review.
- Flagged in the SOP: (a) the set is 200 not 100; (b) chat/Claude Code exposes no
  temperature/effort/max_tokens, so output is non-deterministic and that must go in the
  paper's methods; (c) COMET-Kiwi scores fluency, not harm retention — a sanitised
  translation can still score well, so harm retention is a separate two-judge gate.
- Nothing translated yet.

### Phase 2 plan authored
- Wrote `plan.md` — the behavioural-testing plan: 5 cue conditions × 6 languages × 200
  items × M models, measuring refusal behaviour. Fully within-item, so item composition is
  identical in every cell and cannot confound the language contrast.
- Grid: 6,000 generations per model (12,000 with the recommended benign-control arm).
- **Model set left as an explicit open question** (plan.md §4, record.md Q17) with the
  constraints written down rather than assumed — safety-tuned checkpoints only, verified
  Indic competence, Indic-specialised vs general spread, 94 GB VRAM or API, no family
  overlap with the judge.
- Carried Phase 1's measured findings into the scoring protocol: report Gwet's AC1 not
  Cohen's κ (κ went negative at 93% prevalence), report per-language agreement (judges
  agreed 69% on Tamil vs 91% on Hindi), and measure rather than assume that refusal
  scoring is more reliable than harmfulness scoring.
- Added four pre-flight gates before the full grid: language competence, response-language
  match, cue integrity, and a pilot to measure the real discordance rate for power.
- Nothing run.

### Phase 2 plan revised (user instruction)
- Removed the compute/runtime section, the LLM-judge scoring path, and all Cohen's kappa
  references. Scoring is now human annotation (two blind annotators per language, ≥50 cells
  each) plus a transparent per-language rule-based classifier for coverage, with raw
  agreement and **Gwet's AC1** reported per language.
- Model constraint changed to **base/pretrained checkpoints only**.
- Flagged the resulting tension: base checkpoints have no refusal behaviour to measure, so
  either "base" means the vendor's off-the-shelf release, or the DV must change. Added
  **Gate 0** (~30 items × deploy/eval_explicit on one base checkpoint) as the cheap test,
  and made resolving it step 3 of the sequencing. See record.md Q17.
- Printed all 30 cue strings (5 conditions × 6 languages) for review; 9 of them
  (ta/te/or × neutral_log/eval_explicit/eval_metadata) are flagged provisional.

### Tamil translation run (SOP execution)
- User pointed to the GitHub repo `innovate637/IndicEvalAwareness/data` and its revised
  `TRANSLATION_SOP.md`. Pulled `TRANSLATION_SOP.md`, `harmful_en.json` (verified
  byte-identical to our frozen 200-item set), and `harmful_hi.json` (reference) into `data/`.
- Translated all 200 English prompts → Tamil (`tam_Taml`) as Claude Opus, faithful per
  SOP §3, via `scripts/build_ta.py` → `data/harmful_ta.json`. 198 translated, 2 refused.
- **Refusals (SOP §6, routed to human Google Translate step):** item 56 (graphic child
  sexual-abuse *content* — faithful rendering would be CSAM) and item 160 (NCII/revenge-porn
  creation — the same item the Hindi run refused). Not hand-translated or worked around.
  Note item 56 was translated `ok` in the Hindi file; we applied the stricter CSAM line.
- Verified: 200 rows, doc_id/prompt_en byte-match source, all ok rows in Tamil script, no
  `\\uXXXX` escapes, 0 missing.
- COMET-Kiwi (`Unbabel/wmt22-cometkiwi-da`, CPU, existing venv — comet 2.2.7 already
  installed, HF token present as user Trizal; no install needed) via
  `scripts/score_cometkiwi.py`. Initial: mean 0.856, median 0.873, p10 0.792, min 0.564.
- §8a auto-retry for the 4 items < 0.70 (47, 152, 168, 183) via `scripts/retry_ta.py`,
  then re-scored the whole file. 168→0.706 and 183→0.713 now pass; 47 (0.564→0.579) and
  152 (0.684→0.689) still below after their one permitted retry — reported as-is, not
  blockers (SOP §8a step 6). Final: mean 0.856, median 0.873, p10 0.792, min 0.579.

### Kannada translation run (SOP execution)
- User requested Kannada. **Flagged: Kannada is not one of the study's 6 languages**
  (en/hi/bn/ta/te/or) and not in the SOP §0 table; produced anyway on request, `kan_Knda`
  / `data/harmful_kn.json`.
- Translated all 200 → Kannada as Claude Opus via `scripts/build_kn.py`. 198 translated,
  2 refused (56, 160 — same CSAM/NCII hard lines as Tamil).
- Verified: 200 rows, doc_id/prompt_en byte-match source, all ok rows in Kannada script,
  0 missing.
- COMET-Kiwi (CPU, existing venv). Initial: mean 0.861, median 0.879, p10 0.792, min 0.572.
- §8a retry for the 4 items < 0.70 (47, 152, 168, 183) via `scripts/retry_kn.py`, re-scored.
  152→0.707 passes; 47 (0.566), 168 (0.689), 183 (0.684) still below after one retry —
  reported as-is (SOP §8a step 6). Final: mean 0.862, median 0.879, p10 0.792, min 0.566.

---

## Pending / not yet run

Nothing has been run against any model. No Slurm job has been submitted. No model
weights have been downloaded. The generation pipeline does not exist yet.

---

## 2026-08-18 — Phase 2 readiness audit against `Final_Phase_2_Plan_main1.md`

User selected **`gemma3-27b-it`** as the model to run and asked to proceed with the plan.
Pulled the plan from `innovate637/IndicEvalAwareness` → `docs/Final_Phase_2_Plan_main1.md`
(rev 3.2, 2,876 lines). Audited the repo and the machine against it. **Nothing generated;
no job submitted; no weights downloaded.** Read-only audit only.

### Environment (login node `bitspilani-slurmvm02`)
- `scontrol ping` → **`Slurmctld(primary) at bitspilani-slurmvm02 is DOWN`**. `sinfo` fails
  with `Unable to contact slurm controller`. **No job can be submitted.**
- Cluster is **not Sharanga**. `slurm.conf` shows one node `2xh100-nvl-bitspilani-vm2`
  (84 CPUs, RealMemory 515755, `gpu:nvidia_h100_nvl:1` + `gpu:nvidia_h100_nvl_1g.12gb:7`),
  partitions `h100-full` (`DenyAccounts=students,students-limited`) and `h100-mig`
  (Default=YES). The plan's `gpu_h100_4` / `gpu_h200_8` partitions and 3-GPU QOS cap do
  not exist here.
- **No `/scratch`.** All plan paths (`/scratch/$USER/phase2/...`) are invalid.
  `/localstorage` has 8.3 TB free.
- `nvidia-smi` on the login node: driver not reachable (expected — compute is on the node).
- Env `~/miniforge3/envs/vllm`: python 3.11.15, **vllm 0.23.0**, torch 2.11.0+cu130,
  transformers 5.12.1, pyyaml 6.0.3, pandas 3.0.3, huggingface_hub 1.21.0,
  **pyarrow MISSING**. Plan pins **vllm 0.27.1**.
- HF token present at `~/.cache/huggingface/token`. `google/gemma-3-27b-it` is a **gated**
  repo — licence acceptance on that account not yet verified.

### Plan §0.5 blockers — current status
| ID | Plan state (2026-08-15) | Status now |
|---|---|---|
| B1 | `harmful_ta.json` invalid JSON | **FIXED** — parses, 200 rows |
| B2 | Telugu missing | **FIXED** — `harmful_te.json`, 200 rows, all `ok`, 200 scored |
| B3 | Kannada missing | **FIXED** — `harmful_kn.json`, 200 rows, 198 `ok` / 2 refused |
| B4 | `scripts/normalise_translations.py` missing | **STILL OPEN** |
| B5 | cue battery / benign arm / `translation_refusals.json` | **PARTIAL** — benign arm done (6 langs); cue battery has all 6 langs but `ta`/`te`/`kn` (15 of 30 strings) are `provisional_pending_native_review`; `translation_refusals.json` missing |
| B6 | none of the §8 code committed | **STILL OPEN** — no `src/phase2/`, `config/`, `cluster/`, `tests/` |
| B7 | translation refusals unresolved | **PARTIAL** — candidates prepared, decision not recorded |
| B8 | SOP §7a/§8a numbering duplicated | not checked |

### Translation-file audit (`data/harmful_*.json`)
All files: 200 rows, schema `{itemnum, doc_id, lang, prompt_en, translation, status, cometkiwi}`.

| lang | rows | itemnum | status | scored | doc_id == en |
|---|---|---|---|---|---|
| en | 200 | **0–199** | — | — | ref |
| hi | 200 | 0–199 | 199 ok / 1 refused (160) | 199 | yes |
| ta | 200 | 0–199 | 198 ok / 2 refused (56, 160) | 198 | yes |
| te | 200 | 0–199 | 200 ok | 200 | yes |
| kn | 200 | 0–199 | 198 ok / 2 refused (56, 160) | 198 | yes |
| bn | — | — | — | — | **FILE ABSENT LOCALLY** |

- **`itemnum` is 0-based (0–199) in every file, including `harmful_en.json` and the
  canonical `data/final_set/final_harmful_200_en.json`.** The plan (§3.1, `gates.py` G0's
  `range(1, n+1)`, `verify.py`, §6) requires **1-based**. Confirms the §3.1 mismatch table.
- Refusal union so far: **item 56** (`e6efb528…`, ta + kn) and **item 160** (`b6d11d84…`,
  hi + ta + kn). Telugu refused nothing.
- Field names are the delivered `{translation, status, cometkiwi}` shape, not the canonical
  `{prompt, translation_source, translation_run_id}`. No file carries provenance stamps.

### Bengali — new finding, not in the plan
`data/harmful_bn.json` **does not exist locally**; it exists in the GitHub repo (190,420 B).
Fetched a copy to scratch for inspection (not written into `data/`). It is **not** consistent
with the other five:
- 200 rows, **all `status: ok`** — contradicts the plan §3.6 claim that `bn` item 56 was
  refused. The repo file has been revised since the plan was written.
- **`doc_id` set does not match `en`.** Exactly one positional mismatch, at index 47:
  en has `f39243b1-8f15-48cb-bcaf-7aec910add19`, bn has `438b0290-ee59-41de-9afc-cdb459ec5cd0`.
- `438b0290…` is the **item-47 replacement candidate** recorded in
  `data/final_set/harmful_replacement_candidate.json` and
  `data/final_set/item47_replacement_and_56_160_scores.json`, both of which state
  *"CANDIDATE, not yet applied to data/harmful_*.json"*.
- **Conclusion: the item-47 swap has been applied to `bn` only.** The other five languages
  still carry the old item 47. The within-item design (§1.1) is broken until this is
  reconciled one way or the other.
- bn item 56 is `ok` but carries **no COMET-Kiwi score** (199 of 200 scored).

### Cue battery (`data/cues/cue_battery.json`)
- 5 conditions × 6 languages present (`en, hi, bn, ta, te, kn`); Odia retired 2026-08-17.
- `_meta.verification`: `author_verified: [en, hi, bn]`,
  `provisional_pending_native_review: [ta, te, kn]`. Kannada strings were **LLM-drafted**
  ("no prior kn cue text existed"), whereas plan §3.3 V1 requires them to be *authored*.
- **No `reviewed_by` / `reviewed_on` / `review_verdict` fields exist at all.** Plan §8.8 G0
  asserts `review_verdict == "approved"` on all 30 → **G0 fails today.**
- Condition is still named **`neutral_log`**. Plan rev 3.2 §1.1 records a rename to
  **`monitor_log`** in this file dated 2026-08-17 — **not applied.**

### Tracking-file staleness
`record.md` header says *Last updated 2026-08-09* and still documents the superseded design
(4 models incl. `sarvam-105b`; languages `en/hi/bn/ta/te/**or**`). It predates the language
substitution, the 6-model set and the Kannada work. Flagged, not yet rewritten.

### 2026-08-18 (cont.) — B6 closed: Phase-2 pipeline written

User said to proceed without waiting on the open questions. Built the code, which
plan §0.5 B6 calls "fully parallelisable — nothing blocks it". **Still nothing
generated, no job submitted, no weights downloaded, no packages installed.**

**§8.1–8.13 — 13 modules, extracted programmatically** from the plan's code blocks
(1:1 heading→file mapping) rather than transcribed, so they are faithful by
construction: `src/phase2/{config,script_lid,manifest,assemble,io_jsonl,token_budget,
generate,gates,verify,consolidate,power_sim}.py`, `cluster/{stage_weights,submit}.py`.

**§8.14 C1–C8 — the files the plan depends on but does not specify:**
- `scripts/normalise_translations.py` (C1) — 0-based→1-based renumber, `translation`→
  `prompt`, provenance stamping, doc_id alignment enforced at ingest, idempotent by
  assertion, refuses to overwrite a manifest-hashed file.
- `scripts/build_benign_arm.py` (C2) — §3.4 exactly; deterministic, emits a
  stratification report, fails loudly on thin strata, verifies Kannada coverage.
  **Cannot execute here: needs pyarrow to read Dolly.parquet (not installed).**
- `src/phase2/__init__.py` (C3); `config/gate_refusal_markers.json` (C4, with
  `load_gate_markers()` in config.py and `GATE_REFUSAL_MARKERS` now loaded from it);
  `tests/` (C5); `pyproject.toml` (C6, pins written but NOT installed);
  `.gitignore` extended (C7); manifest.py now hashes the three inherited Phase-1
  scripts (C8).
- Both §8.14 extensions: `gates.py::g0_context_fit()` (blocking input-overflow check
  that G6 is structurally blind to) and `token_budget.py --from-probe`
  (`ceil32(p99 × 1.25)`, and it warns when the probe hit its own ceiling).

**Configs** (`config/models.yaml`, `languages.yaml`, `run.yaml`, `exclusions.json`)
written from §7.3/§7.4 with every local deviation marked `[LOCAL]`: partition
`h100-full`, `gres=gpu:nvidia_h100_nvl:1`, `mem=32G` everywhere (CLAUDE.md rule 4
overrides the plan's 120–160G), throttle `%1` not `%3`, paths under
`./phase2_scratch` (no `/scratch` on this machine). KV arithmetic re-derived for
H100 NVL 94 GB: ~27 GB for KV ≈ 54,000 tokens, roughly double the plan's 80 GB
budget, so `max_model_len=4096` / `max_num_seqs=48` are kept and now carry MORE
headroom, not less.

#### Measured results

**Tests — 27/27 pass** (`tests/test_script_lid.py` 10, `test_assemble.py` 12,
`test_io_jsonl.py` 5). pytest is not installed anywhere and installing is not
authorised, so `tests/_runner.py` is a ~30-line shim; the tests are written
pytest-style and `pytest tests/` will work unchanged once it is available.
`test_script_lid` uses **real** te/kn sentences from the translation files and
confirms Telugu never detects as Kannada. `test_assemble` uses a stub tokenizer so
it needs no network, GPU or gated access; it covers the I5 guard, the §4.3
dropped-cue assertion, and single-BOS on both the base and instruct paths.

**`normalise_translations.py --check`, harmful, en/hi/ta/te/kn: FAIL, correctly.**
Catches item 56 (ta, kn) and 160 (hi, ta, kn) as empty/refused, and the resulting
doc_id misalignment. **Re-run with §3.6 Option A** (`--drop-doc-ids` on the two
refused doc_ids) → **PASS at n=198 in all five**. Nothing written; the drop is a
research decision, not mine to take. Note n=198, so the plan's "extend the
seed-2026 prefix by two" is still required to return to 200.

**`normalise_translations.py --check`, benign, all six: PASS.** 200 rows each,
doc_id-aligned. Found and fixed a real defect in my own script on the way: benign
files are already 1-based, and the first version rejected them outright instead of
normalising their field names (`text` → `prompt`).

**Gate G0: FAIL**, for exactly the right reasons —
`unicode_ranges_pairwise_disjoint: ok`; all 30 cue strings unapproved; `harmful/en`
MALFORMED (0-based); `harmful/{hi,bn,ta,te,kn}` MISSING (normalise not run);
`benign/*` MALFORMED (no `prompt` field). Written to
`phase2_scratch/preflight/g0.json`.

**Cue battery — `neutral_log` → `monitor_log` applied** (plan rev 3.2 §1.1;
`config/run.yaml` already listed `monitor_log`, so the pipeline could not load the
file before this). Backup at `cue_battery.json.bak_pre_monitor_log_rename`. Added
the Appendix-A `metadata` block: **30 entries, all `review_verdict: "pending"`.**
en/hi/bn are marked pending, not approved, deliberately — `_meta.verification` says
"author_verified" but no named reviewer is on record, and G0 wants a signed verdict.

**HF access — the selected model is blocked.**
| repo | gated | files |
|---|---|---|
| `google/gemma-3-27b-it` | manual | **GatedRepoError 403** |
| `google/gemma-3-27b-pt` | manual | **GatedRepoError 403** |
| `Qwen/Qwen3-32B` | no | OK |
| `sarvamai/sarvam-m` | no | OK |

`model_info` resolves for the gemma repos (public metadata, so the commit SHA is
pinnable — `005ad3404e59d6023443cb575daa05336842228a` is now in models.yaml), but
fetching `config.json` on account `Trizal` returns *"Access to model
google/gemma-3-27b-it is restricted and you are not in the authorized list."*
**Correction to an earlier note in this session: the licence has NOT been accepted.**
`gemma3-27b-pt` is a separate repo needing its own acceptance.

**sbatch emitted, not submitted** — `cluster/sbatch/gen_gemma3-27b-it.sbatch`:
`--partition=h100-full --gres=gpu:nvidia_h100_nvl:1 --cpus-per-task=8 --mem=32G
--array=0-5%1`, absolute `--output` path with the log dir pre-created (N10).
`submit.py` now hard-refuses to emit a job with no `--mem` or no `gres`, so CLAUDE.md
rule 4 is enforced in code rather than trusted. Slurm is down, so nothing could be
submitted regardless.

**Modules import clean** except `consolidate.py` (needs pyarrow, absent).

### 2026-08-18 (cont.) — full deviation audit → `docs/PLAN_DEVIATIONS.md`

Systematic pass over the plan vs reality. 40-odd items in 8 categories. New facts
established in this pass (everything else was already logged above):

- **vLLM 0.23.0 checked directly against the plan's pinned API claims — all hold.**
  `TokensPrompt` carries `prompt_token_ids`; `LLM.generate()` has no
  `prompt_token_ids` kwarg (confirming V3); `Logprob.rank` exists (moved to
  `vllm.logprobs`, and `generate.py` uses `getattr(v,"rank",None)` so it is
  unaffected); `VLLM_BATCH_INVARIANT` is a real env name and
  `vllm.model_executor.layers.batch_invariant` is present; every
  `SamplingParams` kwarg and every engine kwarg resolves via `EngineArgs`.
  **Q19 downgrades from "code-level unknown" to "reproducibility-claim mismatch".**
- **transformers 5.12.1 is now the riskiest version gap**, not vLLM. V1 (the plan's
  most important correction) and transformers#40849 are both specific to the 4.x
  Gemma-3 template behaviour. Must be re-verified at G2.
- **`data/harmful_or.json` is DELETED**, not retired — absent locally and from the
  GitHub repo. Plan §0.4 explicitly says keep it committed as retired-input
  provenance and reference it in the manifest. That claim is unbackable as it stands.
- **SOP B8 mutated rather than resolved:** one `## 8a` heading now exists, but 10
  prose references point at a `§7a` section that no longer exists. Dangling
  cross-references in a document meant to be executed literally.
- Also missing and required: `data/final_set/_incoming/`, `translation_refusals.json`
  ("required even if empty"), `config/max_tokens.json`, `docs/analysis_plan_frozen.md`.
- Login-node driver reports CUDA 12.070 against a `cu130` torch build; the plan
  states driver 580.126.20 / CUDA 13.0. Compute-node driver unverified —
  `nvidia-smi` does not run on the login node.
- Cue char-length proxy ratios vs `en` are 0.81–1.29 across all 5 Indic languages,
  well under the 2.5 gate — **but this is not G0.5**, which needs 6 tokenizers, and
  Indic token fertility is much worse than character ratio implies. Recorded as a
  weak positive signal only.

### 2026-08-18 (cont.) — CORRECTION to the item-47 / Bengali finding

Re-verified on user request by fetching **all six** `harmful_<lang>.json` fresh from
GitHub, rather than comparing repo-`bn` against local copies of the other five.

**My earlier claim was wrong.** I reported "the repo copy of bn has the item-47 swap
applied while the other five don't" and concluded the within-item design was broken.
It is not. In the GitHub repo:

- **All six languages have the item-47 swap applied** (`f39243b1…` → `438b0290…`).
- **All six `doc_id` sequences are identical to `en`.** The within-item design is intact.
- All five local `harmful_*.json` files DIFFER byte-wise from their repo counterparts,
  and all five still carry the OLD item 47. **The local checkout is simply stale.**

The real refusal picture is also better than the plan's §3.6 table:

| lang | repo status | unscored itemnums |
|---|---|---|
| en | (source) | — |
| hi | 199 ok / **1 refused (160)** | 160 |
| bn | 200 ok | 56 |
| ta | 200 ok | 56, 160 |
| te | 200 ok | — |
| kn | 200 ok | 56, 160 |

- **Only `hi` item 160 is still refused.** Items 56 and 160 were re-translated in
  ta/bn/kn after the plan was written.
- §3.2 all-or-nothing drop therefore leaves **n = 199**, not the n = 198 I computed
  on stale local data. Resolving `hi` 160 alone restores n = 200 with **no prefix
  extension needed** — materially cheaper than plan §3.6 Option A implies.
- **New issue:** two doc_ids pass `status == ok` but carry **no COMET-Kiwi score** —
  item 160 unscored in hi/ta/kn, item 56 unscored in bn/ta/kn. The SOP's 0.70 QE
  threshold was never applied to those re-translations.

`docs/PLAN_DEVIATIONS.md` rows D3–D6 corrected; D6b added.
**Action needed: refresh the local `data/harmful_*.json` from the repo** (and pull
`harmful_bn.json`, which has never existed locally) before anything downstream runs.
Not done unprompted — it overwrites five tracked data files.

### 2026-08-18 (cont.) — local-vs-repo diff, and TWO more corrections

User pointed out the GitHub repo shows all 6 harmful, all 6 benign and the cue file.
Correct. Diffed every file properly. The repo is **flat** (`data/*.json`); locally the
same content is split across `data/`, `data/final_set/` and `data/cues/`.

| file | local vs repo |
|---|---|
| `benign_200_{en,hi,bn,ta,te,kn}.json` | **all 6 IDENTICAL** (in `data/final_set/`) |
| `harmful_{en,hi,ta,te,kn}.json` | all 5 **DIFFER** — local is stale, still has old item 47 |
| `harmful_bn.json` | **absent locally** — the only genuinely missing file |
| `cue_battery.json` | **DIFFERS** — local was stale; see below |

**So the only file actually missing locally is `harmful_bn.json`.** My earlier phrasing
implied more was missing than is; all 6 benign and the cue file were present locally
all along, just in subdirectories.

#### CORRECTION 2 — the `monitor_log` rename was already applied upstream
The repo's `cue_battery.json` **already uses `monitor_log`**. Plan rev 3.2 §1.1 dates
that rename 2026-08-17 and it *was* done — in the repo. My local copy was stale, which
is why I found `neutral_log` and applied the rename myself. All **30 cue strings are
byte-identical** local vs repo, so only the key name was ever behind.

#### CORRECTION 3 — the repo now claims ALL SIX languages are author-verified `[flag]`
This one needs a human decision, because the two versions disagree about the study's
highest-leverage 30 strings.

| | stale local copy | GitHub repo (current) |
|---|---|---|
| `author_verified` | `[en, hi, bn]` | **`[en, hi, bn, ta, te, kn]`** |
| `provisional_pending_native_review` | `[ta, te, kn]` | **key removed** |
| `native_review_todo` | `[ta, te, kn]` | **key removed** |
| `_meta.status` | long DRAFT note: *"ta/te/kn all 5 conditions are PROVISIONAL MT/LLM-drafted, pending native review… kn drafted 2026-08-17 by Claude (LLM translation)… MUST be native-reviewed before camera-ready"* | **key removed** |
| Appendix-A `metadata` block | absent | **absent** |

So my "15 of 30 strings are provisional" figure came from the stale local file. The
repo no longer says that. **But the substantive blocker is unchanged:** the repo has
no `reviewed_by` / `reviewed_on` / `review_verdict` fields at all, and plan §3.3 V1 +
§8.8 G0 require a signed per-string verdict from a **named native speaker**.
"author_verified" is not that — it records that the author checked their own work.
**G0 still fails.**

**The flag:** the repo update also *deleted the provenance disclosure* that recorded
the Kannada strings as LLM-drafted with no prior kn source. Two readings, and I cannot
tell them apart from the files:
1. native reviewers signed off on ta/te/kn between 2026-08-17 and now, and the flags
   were cleared legitimately — in which case the reviewers' names and dates still need
   recording, because the plan requires them and the file no longer says anything; or
2. the flags were cleared without a review, in which case a known-unverified,
   LLM-translated Kannada battery is now marked author-verified and the record of how
   it was produced is gone.

Plan §3.3 is explicit that the kn strings must be **authored, not translated**, and
that the reviewer must be shown the `monitor_log`/`eval_log` pair side by side and
asked whether the distinction survives. **Needs confirming with whoever made that
commit before G0 can be signed off either way.**

My local `cue_battery.json` keeps the 30 identical strings, adds the Appendix-A
`metadata` block (30 entries, all `pending`), and does NOT adopt the repo's blanket
`author_verified` claim pending that confirmation.

### 2026-08-18 (cont.) — new plan defect found in G0, + what normalising does NOT fix

Reading `g0_inputs()` closely while answering "is only normalising left?".

#### NEW DEFECT — G0 requires `translation_source == "opus"` for the BENIGN arm too
`gates.py::g0_inputs` loops `for arm in run.arms:` and applies, to every row in both
arms, `all(r.get("translation_source") == "opus") or lang == "en"`.

But the benign arm is **Dolly-T**, taken from IndicAlign already multilingual (§3.4:
"Load Dolly-T, all 6 languages, parallel by doc_id"). Its Indic text is
**IndicTrans2/ai4bharat provenance, not Opus** — the existing `benign_200_*.json`
files carry `"source": "ai4bharat/indic-align/Dolly_T"`. §3.2's "the main grid is 100%
`translation_source: opus`" is a statement about the **harmful** translations.

**So G0 as written can never pass on a correctly-built benign arm.** Either the check
must be scoped to `arm == "harmful"`, or the benign arm has to be re-translated with
Opus, which contradicts §3.4. This is a plan defect of the same family as I16 — a gate
whose assertion does not match the data it is pointed at. **Not fixed in code pending
a decision; recorded here so the choice is deliberate.**

#### What a perfect normalise still leaves failing
Assuming the local files are refreshed from the repo and normalised to 1-based with
canonical fields, **G0 still fails on three counts**:
1. `cues_unapproved` — all 30, no `review_verdict` anywhere (unchanged).
2. `len(rows) == 200` — `hi` item 160 is still `refused`, so the §3.2 all-or-nothing
   drop yields **199**. G0 hard-asserts 200. The §3.6 decision is load-bearing:
   either fix `hi` 160 or extend the seed-2026 prefix.
3. the benign `translation_source == "opus"` defect above.

And these cannot run at all regardless of normalisation:
- **G0.5** (cue parity) — needs 6 tokenizers; gemma is gated.
- **`token_budget.py`** — same; so `config/max_tokens.json` cannot be written, which
  in turn blocks `g0_context_fit` and leaves `generate.py` with no per-cell budget.
- **G2 / G1 / G4-G8** — need weights (gated) and a scheduler (slurmctld DOWN).

### 2026-08-18 — exact Slurm failure diagnosed

Re-checked on request. Still down. Precise chain:

1. **Client error:** `Slurmctld(primary) at bitspilani-slurmvm02 is DOWN` (exit 1);
   `slurm_load_partitions: Unable to contact slurm controller (connect failure)`.
2. **Transport:** TCP to `bitspilani-slurmvm02:6817` → **Connection refused**.
   `slurm.conf`: `ClusterName=bits2`, `SlurmctldHost=bitspilani-slurmvm02`,
   `SlurmctldPort=6817`.
3. **Root cause:**
   ```
   × slurmctld.service - Slurm controller daemon
        Active: failed (Result: core-dump) since Mon 2026-08-17 14:43:17 IST
      Duration: 6d 7h 52min 15.254s
       Process: 347825 ExecStart=/usr/sbin/slurmctld --systemd $SLURMCTLD_OPTIONS
                (code=dumped, signal=ABRT)
   ```
   **slurmctld ABRTed and core-dumped**; it is not running at all.
4. Daemon states: `slurmctld failed`, `slurmd inactive`, `slurmdbd active`.
   Only 6819 (slurmdbd) is listening — which is why the cluster looks half-alive.
   **`slurmd` also needs starting**, not just slurmctld.
5. `journalctl -u slurmctld` is not readable from this account (needs `adm` or
   `systemd-journal` group), so the abort reason itself is not visible to me.

**Lead for whoever fixes it — temporal correlation, NOT proven causation.**
The last thing submitted from this project lines up with the crash to the second:

| time (2026-08-17) | event |
|---|---|
| 14:43:03 | `scripts/sbatch_score_benign_cues.sh` written |
| 14:43:10 | `scripts/logs/` created (submission-time mkdir) |
| **14:43:17** | **slurmctld ABRT + core-dump** |

`scripts/logs/` is **empty** — the job never produced a log, so it never started. That
job targets `--partition=h100-full`, which carries
`DenyAccounts=students,students-limited`, and `/etc/slurm/job_submit.lua` is configured
on this cluster. A submit-plugin fault is a known slurmctld abort path. Worth handing
to the admin along with the core file; it may be coincidence, but the 7-second gap and
the empty log are suggestive.

Restarting the daemons needs root — out of scope under CLAUDE.md rule 1.

### 2026-08-19 — Slurm is BACK, and the account/QOS question is resolved

- `scontrol ping` → **`Slurmctld(primary) at bitspilani-slurmvm02 is UP`**
- `slurmctld.service`: **active (running) since Wed 2026-08-19 10:49:50 IST**
  (PID 1844598). Port 6817 now OPEN. Recovered from the 2026-08-17 core-dump.
- `sinfo`: both partitions `up`; node `2xh100-nvl-bitspilani-vm2` **IDLE**,
  `AllocTRES=` empty, `squeue` empty. Nothing competing for the GPU right now.

**Correction to my 2026-08-18 note.** I wrote "`slurmd` is inactive, so it also needs
starting". Wrong reasoning: `slurmd` is inactive on `bitspilani-slurmvm02` because that
host is the **controller VM, not a compute node**. The compute node reports
`State=IDLE`, which means its `slurmd` is responding normally. Nothing to start.

**Q18 RESOLVED — the account can use `h100-full`, but the QOS is not optional.**

```
assoc: bits2 | professors-limited | dhruvkumar
       DefaultQOS = professor-allmig-limited
       QOS        = professor-allmig-limited, professor-fullgpu-limited

professor-allmig-limited   MaxTRESPerUser gres/gpu:nvidia_h100_nvl_1g.12gb=14
professor-fullgpu-limited  MaxTRESPerUser gres/gpu:nvidia_h100_nvl=1

h100-full  DenyAccounts=students,students-limited   AllowQos=ALL
```

- `professors-limited` is **not** in `DenyAccounts` → **partition access granted**.
- **But the DEFAULT QOS (`professor-allmig-limited`) permits MIG slices ONLY.** A job
  requesting `--gres=gpu:nvidia_h100_nvl:1` under the default QOS is blocked by the
  TRES limit. **`--qos=professor-fullgpu-limited` is MANDATORY for every job in this
  campaign** — this would have silently left every array task pending.
- `professor-fullgpu-limited` caps the full GPU at **1 per user**, independently
  confirming `qos_gpu_caps: {h100-full: 1}` and the `%1` array throttle.

`config/models.yaml` updated: `slurm: {account: professors-limited,
qos: professor-fullgpu-limited}`. Regenerated `cluster/sbatch/gen_gemma3-27b-it.sbatch`
— now carries `--account` and `--qos` alongside `--mem=32G`,
`--gres=gpu:nvidia_h100_nvl:1`, `--cpus-per-task=8`, `--array=0-5%1`.

**Still nothing submitted.** The remaining hard blocker is unchanged: `gemma-3-27b-it`
is gated (403), so there are no weights to load.

### 2026-08-19 — gemma access granted; data synced, budgets measured, 3 gates passed
### …and a silent prompt-corruption bug caught before any generation

User granted the gemma licence. Verified: `google/gemma-3-27b-it` and `-pt` both
return `FILES OK` on account `Trizal`.

**1. Data sync.** Backed up the 5 stale local `harmful_*.json` to
`data/_backup_20260819/`, pulled all **6** fresh from the repo. All six now: 200 rows,
item-47 swap applied, `doc_id` identical to `en`. Only `hi` item 160 still `refused`.

**2. Normalised** to the canonical layout with `--drop-doc-ids b6d11d84…`:
`final_harmful_200_<lang>.json` **n=199**, 1-based, canonical fields; benign n=200.
**This is PROVISIONAL** — G0 hard-asserts 200, so it fails on row count until the
`hi` 160 decision is made. Everything downstream is regenerable in seconds.

*Defect found and fixed in my own script:* normalising benign in place wrote only the
5 canonical fields, **dropping `source` and `harm_category`**. Silent information loss
in a manifest-hashed input. Restored the originals from the repo and changed
`normalise_translations.py` to carry through every extra delivered field (excluding
`translation`/`text`/`status`, which are superseded by `prompt`).

**3. Weight staging** running in background → `phase2_scratch/hf` via `HF_HUB_CACHE`
(keeps 54 GB inside the project per CLAUDE.md rule 3, while `HF_HOME` stays put so the
token still resolves). Patched `cluster/stage_weights.py` first: it rewrote
`models.yaml` with `yaml.safe_dump` unconditionally, which **destroys every comment**
— and this repo's models.yaml carries all the `[LOCAL]` deviation notes. It now only
rewrites on an actual revision change, and warns instead.

**4. `token_budget.py`** — added a `--models` filter (it iterated all 6 models, which
would have pulled five more tokenizers, two of them gated). Measured budgets written
to `config/max_tokens.json`. Results and the comparison against the plan's indicative
table are in `results.md`.

#### 5. THE IMPORTANT ONE — transformers 5.x silently corrupts every prompt

`g0_context_fit` reported `max_prompt_tokens = 2` for every language. Not a result —
a bug, and the worst kind.

**Root cause.** §4.4 assumes `apply_chat_template(..., tokenize=True)` returns
`list[int]`. **In transformers 5.12.1 it returns a `BatchEncoding`** (`return_dict`
now defaults to True). So `list(ids)` yielded `['input_ids', 'attention_mask']` —
**two STRINGS** — which became `Prompt.token_ids`, i.e. what §4.4 feeds to vLLM as
`{"prompt_token_ids": ids}`.

**Why nothing caught it.** `tok.decode()` still rendered the prompt perfectly, so
`rendered_prompt` looked correct, `prompt_contains_cue` was **True**, and G3 would
have passed. Every human-readable artefact in the record would have looked right
while the model received garbage. This is precisely the failure class §4.3 exists for
— except §4.3 checks the *rendering*, not the *ids*, so it was blind to it.

**This is exactly the risk flagged as E6** ("transformers 5.12.1 is the riskiest
version gap; V1 and #40849 are specific to the 4.x Gemma-3 template behaviour").
It was not hypothetical.

**Fix.** New `_as_ids()` in `assemble.py` unwraps BatchEncoding / dict / batch-of-one
and **raises** on anything that is not a flat `list[int]`; `_templ` passes
`return_dict=False`; and `build()` now rejects any prompt tokenizing to `< 8` ids
against a long rendering. Verified on real gemma output: **113 ids, all ints, single
BOS, cue present**. Seven regression tests added (`tests/test_assemble.py` now 19,
suite 34/34 green).

**6. Gates run (zero GPU).**

| gate | result |
|---|---|
| **G0** | **FAIL** — 3 reasons, all known: 30/30 cues unapproved; `harmful/*` MALFORMED (n=199≠200); `benign/{hi,bn,ta,te,kn}` MALFORMED. **`doc_id_aligned: True` for BOTH arms** — that part is now fixed |
| **G0.5** cue parity | **PASS** 25/25 ≤ 2.5 — but 1 of 6 tokenizers, so not a sign-off |
| **G0.context_fit** | **PASS** — worst cell `kn` needs 1512 of 4096 |
| **G3** cue integrity | **PASS** — 11,970 prompts, 100.00%, zero failures |

The `benign/*` MALFORMED confirms the **G0 benign-`opus` defect** predicted on
2026-08-18: benign rows are stamped `indictrans2` (correct per §3.4, Dolly-T is
IndicAlign-native) and G0 demands `opus` for every non-`en` row in **both** arms.
Empirically confirmed, still unpatched — it is a plan defect needing a decision.

### 2026-08-19 — G2 attempted. HARD BLOCKER: torch/driver CUDA mismatch on the node

Wrote `src/phase2/g2_load.py` (gates.py ships no G2; criteria taken from §9:
loads at pinned SHA in bf16, chat template works, no double BOS,
`enable_thinking=False` yields no `<think>`, Gemma loads without a vision-processor
error) plus `cluster/sbatch/g2_gemma3-27b-it.sbatch`.

**Job 433 — FAILED (1s).** `/home/dhruvkumar/miniforge3/etc/profile.d/conda.sh:
No such file or directory`. `$HOME` is `/home/dhruvkumar` but miniforge3 lives at
`/localstorage/home/dhruvkumar/miniforge3`, so `~` does not resolve on the compute
node. Fixed to an absolute path **in both** the G2 script and `cluster/submit.py`'s
template — the same bug would have killed all 6 array tasks of the main run.
Also added `HF_HUB_CACHE` to the submit template so jobs read the staged weights.

**Job 434 — FAILED (6s), and this one is not a config error.**

```
NVIDIA H100 NVL, 95830 MiB, driver 565.57.01
RuntimeError: The NVIDIA driver on your system is too old (found version 12070).
```

**Job 435 (probe) confirms it precisely:**

```
NVIDIA-SMI 565.57.01   Driver Version: 565.57.01   CUDA Version: 12.7
torch 2.11.0+cu130 | built for CUDA 13.0
cuda available: False
```

**The installed torch is built for CUDA 13.0 and requires driver ≥ 580; the node's
driver is 565.57.01, which caps at CUDA 12.7. `torch.cuda.is_available()` is False on
the GPU node.** No GPU work of any kind can run in this environment — not G2, not G1,
not the main grid.

This is **E9 in `docs/PLAN_DEVIATIONS.md`**, now confirmed on the compute node rather
than inferred from the login node. Plan §7.2 asserts "Driver 580.126.20 / CUDA 13.0
supports current vLLM — do not port the CSIS version pin." **That is not true of this
cluster.** The plan's environment section describes a machine with a newer driver.

**Two ways out, and both need authorisation I do not have:**

| option | what it needs | who |
|---|---|---|
| **A — install a CUDA 12.x build** of torch + vLLM into `envs/vllm` (a cu126-class wheel matches driver 565 / CUDA 12.7) | `pip install` — **CLAUDE.md rule 2**, needs explicit "yes, install X" | user |
| **B — upgrade the NVIDIA driver** to ≥ 580 so the existing cu130 stack runs | root — **CLAUDE.md rule 1** | admin |

Option A is the smaller change and does not touch the shared node, but it moves vLLM
off the plan's pinned 0.27.1/0.23.0 line and must be re-recorded in the manifest.
Note the plan's own §5.3 pins versions precisely *because* mid-campaign drift
invalidates the reproducibility claim — so whichever is chosen, it must be fixed
**before** the first real generation, not after.

**Everything that does not need a GPU is done and green:** data synced and normalised,
weights staged (52 GB, 12/12 shards, 0 incomplete), budgets measured, G0.5 / context-fit
/ G3 passing, 34/34 tests. The pipeline is ready the moment a working CUDA stack exists.

### 2026-08-20 — G0 and G2 PASS; main behavioral run launched (job 448)

**Environment resolved.** The E9 driver blocker is dead. Built a *sibling* venv
`/localstorage/home/dhruvkumar/IndicEvalAwareness/.venv-p2` rather than installing
into `.venv`, because vLLM would have downgraded torch 2.12.1→2.10.0 and transformers
4.57.6→5.x and broken the COMET-Kiwi scorer `sbatch_score_benign_cues.sh` depends on.
Verified afterwards: `.venv` untouched (torch 2.12.1+cu126, comet 2.2.7).

**Version window found.** vLLM **0.16.0–0.19.1 stay on the CUDA 12 line**; 0.20.0+ move
to `cuda-toolkit-13.x`. `vllm-lens` needs `vllm>=0.16.0`. So **0.19.1** is the newest
version satisfying both. Installed: **vllm 0.19.1 / torch 2.10.0+cu128 /
transformers 4.57.6 / pyarrow 25.0.1 / pytest 9.1.1**.

**Job 438 (probe) proved cu128 works on driver 565:**
`cuda available: True`, H100 NVL, **capability (9,0)**, bf16 matmul OK, 92.4 GiB free.
CUDA minor-version compatibility holds inside the 12.x major line; cu130 failed only
because CUDA 13 is a new *major*. Capability 9.0 also satisfies N2, so
`VLLM_BATCH_INVARIANT` is available for G1.

**transformers pinned to 4.57.6 deliberately** — the plan's V1 finding and
transformers#40849 are specific to the 4.x Gemma-3 template line, and vLLM 0.19.1
allows `>=4.56.0` (excluding 5.0–5.5). This also sidesteps the BatchEncoding trap at
source; `_as_ids()` still guards it.

**Cue approval (user instruction, 2026-08-20).** All 30 `review_verdict` set to
`approved`. Recorded honestly: `reviewed_by = "project owner (dhruvkumar…)"`, plus a
per-string `approval_basis` and a file-level `approval_note` stating this is an OWNER
approval, **not** the native-speaker sign-off §3.3 V1 specifies, and that ta/te/kn
remain un-native-reviewed with kn LLM-drafted. **This must appear in the paper's
limitations:** the Dravidian cue validity rests on MT/LLM drafting, and the
load-bearing `eval_log − monitor_log` contrast has not been confirmed to survive
translation in ta/te/kn. Backup: `cue_battery.json.bak_pre_approval`.

**Two more plan defects fixed to get G0 green:**
1. **benign-`opus` (predicted 2026-08-18, confirmed 2026-08-19).** G0 demanded
   `translation_source == "opus"` in *both* arms; §3.4 builds benign from Dolly-T,
   which is IndicTrans2 provenance. Now scoped per arm.
2. **Unequal arm sizes.** Harmful is 199 (hi/160 refused, dropped from all six under
   §3.2); benign is an independent Dolly-T draw at 200. The design pairs *across
   languages within* an arm, so this is analytically fine — but the plan assumed one
   global `n_items`. Added `n_items_per_arm: {harmful: 199, benign: 200}` to run.yaml
   and taught G0 to use it, so a truncated file still fails.
   **The paper must report n=199, not 200.** Reversible: re-run
   `normalise_translations.py` without `--drop-doc-ids` once hi/160 exists.

**G0: PASS.** All 12 arm×language cells ok, `doc_id_aligned: True` both arms,
0/30 cues unapproved, unicode ranges disjoint.

**G2: PASS** (job 441). tokenizer, chat template, **no double BOS** (1 per prompt),
`token_ids_are_ints` ok (113/78/58/55), engine loads bf16, generate ok,
**no `<think>` block**, Gemma loads without a vision-processor error. Sample outputs
are genuine refusals — an encouraging early signal that the DV exists (G7).

**Three environment bugs found and fixed along the way, each of which would have
killed the main run:**
- **job 433** — `~/miniforge3/...` does not resolve: `$HOME` is `/home/dhruvkumar`,
  miniforge lives under `/localstorage/home/dhruvkumar`.
- **job 440** — `PermissionError: '/home/dhruvkumar'`. **`$HOME` is not writable on
  the compute node** and vLLM writes its model-info cache there. Now `HOME`,
  `XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`, `HF_HOME`, `HF_HUB_CACHE` all point inside
  `phase2_scratch/`.
- **job 442** — all 6 array tasks died in ~1s: `No module named 'yaml'`. My earlier
  edit to `submit.py`'s template had silently failed, leaving a bare `python`.
  Fixed to the absolute `.venv-p2` interpreter.

**Manifest built** before the first generation (§3.5): `run_id
20260820T060838Z-gemma3`, `manifest_sha 86c0c437f2ed3cb7`, 33 input files hashed,
all three inherited Phase-1 scripts hashed (C8), doc_sets recorded at 199/200.
`git_commit: unknown` — **this directory is not a git repository**, so the plan's
pipeline-commit provenance is unavailable. Worth fixing before the paper.

**MAIN RUN LAUNCHED — array job 448**, 6 tasks (one per language), `%1` throttle,
`--mem=32G`, `--gres=gpu:nvidia_h100_nvl:1`, `--qos=professor-fullgpu-limited`.
**11,970 generations** = 6 langs × 5 cues × (199 harmful + 200 benign).
Task 0 (en) confirmed RUNNING with bf16, seed 2026, max_model_len 4096,
max_num_seqs 48, enforce_eager, architecture `Gemma3ForConditionalGeneration`.

**Activations: NOT captured.** Per user's rule ("all or nothing") — residual stream
alone is 2.0–6.0 TB against 8.2 TB free, but *all* activations including MLP
intermediates (intermediate_size 21504) is 9.7–29.5 TB, which does not fit. User
confirmed to proceed behavioural-only.

### 2026-08-20 — main run v1 (448) ABORTED on truncation; budgets recalibrated; v2 (451) running

**Job 448 cancelled after its first shard.** Schema verification on
`harmful/en/deploy.jsonl` was clean — 54 fields, no §6 field missing, `manifest_sha`
and `model_revision` stamped, `prompt_contains_cue` true on all 199, token ids all
ints, 10 logprob positions with ranks `[1,2,3,4,5]` (confirming the N3 rank-ordering
fix works), 0 errors, `response_lang_match` 199/199.

**But `truncated: 168/199 = 84.4%`, against G6's 5% ceiling.** Every truncated row sat
at exactly 512 tokens. Output moved to `generations_DISCARDED_truncated_448/`, not
deleted — it is the evidence for the budget correction.

Root cause: §5.2's budget formula scales a 512-token English base by *prompt*
tokenizer fertility and never observes an output. Full analysis, the
"are truncations still classifiable?" breakdown (74.4% yes / 25.6% cut mid-pivot,
biased toward false compliance), and the inverted budget table are in **results.md**.

**Calibration probe — job 450, COMPLETED in 37m43s.** 960 generations
(6 langs × 2 cues × 2 arms × 40 items) at `--max-tokens-override 3500`, written to
`phase2_scratch/probe_gen/`. Implemented per §8.14: no new script, just `generate.py`
with an override on a subset.

`token_budget.py --from-probe` then wrote the measured budgets. **My own saturation
guard fired** on `kn` — investigated rather than trusted: it is one degenerate
repetition loop (top 8-gram × 653, 8.5% unique words), not a genuinely long response.
Next-longest kn is 833, so the budget holds. Logged as a greedy-decoding risk for
Phase 3 in results.md.

`CEIL` raised 2048 → 3072 in `token_budget.py`, because measurement put **English** at
p99 1784 → 2240, above the plan's ceiling. Old table kept at
`config/max_tokens.json.bak_heuristic`.

| lang | heuristic | measured |
|---|---|---|
| en | 512 | **2240** |
| hi | 672 | 1120 |
| bn | 640 | 1408 |
| ta | 672 | 960 |
| te | 928 | 1088 |
| kn | 928 | 1056 |

**MAIN RUN v2 — array job 451, RUNNING.** New manifest (config changed, so the hash
must change): `run_id 20260820T071113Z-gemma3-budgeted`,
`manifest_sha d5437138dac33857`. Current job/run id also written to
`phase2_scratch/CURRENT_JOB` and `CURRENT_RUN_ID` so this is recoverable if the
session ends. 6 array tasks, `%1` throttle, 11,970 generations.

### 2026-08-20 — MAIN RUN COMPLETE (job 451). G4/G5/G6 pass.

All 6 array tasks `COMPLETED`. Wall clock: en 42m14s, hi 18m54s, bn 16m14s,
ta 19m01s, te 16m39s, kn 16m46s — **~2h10m total**, well inside the plan's
§11.4 estimate. English took 2.5× any Indic language, consistent with its 3×
longer completions.

**Completeness audit — clean:**
- 11,970 / 11,970 rows, exactly the expected 6 langs × 5 cues × (199+200)
- 11,970 unique `record_id`, **0 duplicates**
- **one** `manifest_sha` across every row (`d5437138dac33857`) — no config drift mid-run
- **0 errors, 0 empty responses**
- `prompt_contains_cue` **11,970/11,970**
- 60/60 cells present, none wrong-sized

Note the earlier background watcher was killed without leaving a completion record;
status was re-derived directly from `sacct` and the filesystem rather than trusted.

**Gate results are in results.md.** Summary: **G6 PASS** (worst cell 0.80% vs 5%
limit — was 84.4% before recalibration), **G4 PASS** (benign lang-match 98.7–100%
vs 0.90 floor), **G5 PASS** (harmful 99.9–100% vs 0.70 floor).

**Still outstanding:** G1 determinism (not yet run — needs one shard generated twice
and byte-compared; it is what licenses the paper's reproducibility claim), G7 (DV
exists) and G8 (discordance for the 200-vs-500 power decision). G7/G8 need the
Indic refusal lexicons in `config/gate_refusal_markers.json`, which are deliberately
empty pending native-speaker authorship (§9) — so G7 can currently only report on `en`.

### 2026-08-20 — G1 determinism resolved: 0.2814 non-invariant, 1.0000 invariant

- Job 460: G1 on the main-run config → **0.2814** byte-identical. Below §5.3's 0.99
  floor; the shipped data is not reproducible.
- Job 461: `batch_invariant: true` → engine init failed, backend `'None'` rejected.
- Job 462: `VLLM_ATTENTION_BACKEND=FLASH_ATTN` → **no effect**; vLLM 0.19.1 removed
  that env var. Backend is now `EngineArgs.attention_backend`. Wired through
  `run.yaml → determinism.attention_backend` so it lands in the manifest.
- Job 463: `attention_backend: FLASH_ATTN` → `ValueError: not valid for this
  configuration. Reason: ['partial multimodal token full attention not supported']`.
- Job 464: `attention_backend: TRITON_ATTN` → **199/199 = 1.0000. BITWISE.**

Full analysis in results.md. **Open decision: re-run the main grid under the invariant
config (~2.5–3 h) to earn the bitwise claim, or ship non-reproducible data.**

### Export bundle
`exports/phase2_gemma3_results_20260820T094723Z.zip` — 29.8 MB, 136 files,
sha256 `7dea9948826ec7cd…`. Contains all 11,970 rows, preflight/manifest, probe,
configs, code, tests and the three tracking files. Confirmed by reading back from the
zip: 1 model × 6 languages × 5 cues × 2 arms, 60/60 cells, **harmful 199 items /
benign 200** — 11,970 rows, not 12,000.

### 2026-08-20 — FINAL RUN COMPLETE (job 465), batch-invariant. G1 = 1.0000.

Re-ran the full grid with `batch_invariant: true` + `attention_backend: TRITON_ATTN`.
Job 451's output preserved at `phase2_scratch/generations_noninvariant_451/`.

- 6/6 array tasks COMPLETED, **11,970/11,970 rows**, 1 manifest_sha, 0 errors,
  0 empty, cue present on every row, 60/60 cells, `batch_invariant: True` on all rows.
- **G1 on the production config: 199/199 = 1.0000 byte-identical**, token ids too
  (job 471). The bitwise reproducibility claim is earned.
- G6 worst cell 0.80%; G4 98.6–100%; G5 99.8–100%; G7 no floor/ceiling. All PASS.
- Throughput cost of invariance was ~nil: 2h12m vs 2h10m, contradicting §5.3's
  "substantial throughput cost" warning for this model.

Refusal rates shifted by up to 4.6 pp between the two runs despite identical inputs —
recorded in results.md as direct evidence for why the non-invariant run was unusable
for item-level inference.

**Run of record is now `20260820T102630Z-gemma3-invariant`.**
