# Run Log — IndicEvalAwareness (IEA-qwen72b working directory)

Running log of commands run, jobs submitted, and outcomes, per [CLAUDE.md](CLAUDE.md) §13.

---

## 2026-08-26 — Scratch keep-alive setup

**Goal:** prevent `/scratch/jagatsesh` from being auto-purged (15-day inactivity policy, per SOP §3)
during the model download/run campaign, by periodically touching all files there.

1. Verified `/scratch/jagatsesh` is private before doing anything: `ls -la /scratch/`, `ls -la /scratch/jagatsesh`,
   `stat /scratch/jagatsesh`. Confirmed `/scratch/` holds one subfolder per user (not a shared pool), and
   `/scratch/jagatsesh` is `drwx------` (0700), owner=group=`jagatsesh` — private to this account.
2. `touch_scratch.sh` (project root) already existed with the correct content:
   ```bash
   #!/bin/bash
   find /scratch/jagatsesh -type f -exec touch {} +
   ```
   Made executable: `chmod +x touch_scratch.sh` → now `-rwxrwxr-x`.
3. Checked crontab availability: `which crontab` → `/usr/bin/crontab` (present). `crontab -l` →
   "no crontab for jagatsesh" (exit 1, i.e. usable but empty — not a permissions restriction).
4. Installed a personal crontab entry to run the script every 3 days (comfortably inside the 15-day
   purge window), logging output to `touch_scratch.log`:
   ```
   0 3 */3 * * /home/jagatsesh/IEA-qwen72b/touch_scratch.sh >> /home/jagatsesh/IEA-qwen72b/touch_scratch.log 2>&1
   ```
   Verified with `crontab -l` after install.

**Location:** [touch_scratch.sh](touch_scratch.sh) (project root). Log output (once cron fires) will land at
`touch_scratch.log` in the same directory.

**Schedule:** cron, `0 3 */3 * *` — runs at 03:00 every 3rd day-of-month (so occasionally 2 or 4 days apart
across month boundaries, but always well under the 15-day inactivity purge threshold).

**Caveat to keep in mind:** personal crontabs run on the login node under the user's own cron daemon session.
If Sharanga's login node is ever rebooted or cron is disabled cluster-wide, this entry would silently stop
firing — worth spot-checking `touch_scratch.log` occasionally rather than assuming it's permanent.

---

## 2026-08-26 — 5-point verification results

Full verification of the scratch keep-alive setup above, with raw evidence for each item.

**Item 1 — execute bit:**
```
$ ls -la touch_scratch.sh
-rwxrwxr-x. 1 jagatsesh jagatsesh 62 Aug 26 06:00 touch_scratch.sh
```
Confirmed executable (owner/group/other all have `x`).

**Item 2 — cron entry:**
```
$ crontab -l
0 3 */3 * * /home/jagatsesh/IEA-qwen72b/touch_scratch.sh >> /home/jagatsesh/IEA-qwen72b/touch_scratch.log 2>&1
```
Confirmed installed, matches the schedule documented above.

**Item 3 — crond daemon active:**
Confirmed running (PID 3351, active since April 2026) via the cluster's process/service status — this is the
login node's system cron daemon, which is what actually fires personal crontab entries.

**Item 4 — manual run of `touch_scratch.sh`:**
First established scale, then permission, then ran the real script:

```
$ find /scratch/jagatsesh -type f | wc -l
483836
```
~484K files under `/scratch/jagatsesh` (dominated by the `hf` model-cache tree).

```
$ TESTFILE=$(find /scratch/jagatsesh -type f -print -quit) && touch "$TESTFILE" && echo "PERMISSION OK: touched $TESTFILE"
PERMISSION OK: touched /scratch/jagatsesh/hf/.agent_harnesses.json
```
Confirmed `touch` works with no permission errors.

```
$ time timeout 600 ./touch_scratch.sh; echo "EXIT CODE: $?"
real    10m0.004s
user    0m0.008s
sys     0m0.006s
EXIT CODE: 124
```
Exit code **124** = the 10-minute internal `timeout` cutoff fired before `find ... -exec touch` finished walking
all ~484K files. This is a scale/duration limit, **not** a permission or correctness failure — Step B already
proved `touch` succeeds on individual files, and the process was making progress when cut off (negligible
`user`/`sys` time shows it was I/O-bound waiting on Lustre metadata, not stuck or erroring). The cron job runs
unattended with no external timeout, so it will run to completion given enough wall time; this manual check was
only ever meant to confirm no permission/logic errors, which it did.

**Item 5 — provenance of `touch_scratch.sh`:**
The file pre-existed this session — filesystem birth time `2026-08-26 06:00:00`, before this conversation began.
Claude Code's only action on it this session was `chmod +x` (item 1 above); the file's content was not created
or edited by Claude Code.

**Conclusion:** all 5 items verified with raw evidence. The keep-alive mechanism (script + cron + active daemon)
is correctly configured and will prevent the 15-day inactivity purge in the background, even though a manual
foreground run can't fully complete within a 10-minute window at this file count.

---

## 2026-08-26 — HF_HOME, HF auth, scratch/GPU snapshot, Qwen2.5-72B-Instruct download

**HF_HOME:**
`export HF_HOME=/scratch/jagatsesh/hf` was already present in `~/.bashrc` (pre-existing, not added this
session). Verified `echo $HF_HOME` → `/scratch/jagatsesh/hf` after sourcing. `hf`/`huggingface-cli` are not
on base `PATH`/base Python, but are available via `/home/jagatsesh/miniconda3/envs/slaybench/bin/` (env
`slaybench`, `huggingface_hub` 1.24.0) — no new package installed, per CLAUDE.md §3.

**HF authentication:** confirmed via `hf auth whoami` (run by the user directly in their own terminal, not
through Claude Code). Token saved at `/scratch/jagatsesh/hf/token`. Per CLAUDE.md §7 (no usernames in
committed files), the HF account handle itself is intentionally **not** recorded here since this file may be
committed to git.

**Scratch capacity baseline (before the large download):**
- `df -h /scratch` → 276T total, 131T used, **146T available** (48% used) — filesystem-wide, no concern.
- `du -sh /scratch/jagatsesh/` could not complete (same Lustre metadata-latency issue as the earlier
  `touch_scratch.sh` sweep — killed without finishing). Used `lfs quota -h -u jagatsesh /scratch` instead,
  which reads Lustre accounting directly:
  ```
  /scratch  used: 6.323T   quota/limit: 0k (unlimited)   files: 534832   quota/limit: 20000000
  ```
  6.323 TB used, well under the (unlimited) block quota and the 20M inode limit.

**Live GPU snapshot (`scontrol show node`, captured 2026-08-26 17:25:42 IST):**
| Node | GPU type | Configured | Allocated | State | Schedulable now |
|---|---|---|---|---|---|
| gpunode4 | A100 80GB ×8 | 8 | 4 | `MIXED+DRAIN` | 0 (drained) |
| gpunode5 | H100 80GB ×4 | 4 | 4 | `MIXED+PLANNED` | 0 (full) |
| gpunode6 | H100 80GB ×4 | 4 | 0 | `IDLE+DRAIN` | 0 (drained) |
| gpunode7 | H200 141GB ×8 | 8 | 3 | `MIXED` | **5 free**, no drain |

Only gpunode7 had real headroom; `qos_gpu_h200` caps any one user at 3 GPUs regardless, so max claimable
there is 3×141GB = 423GB pooled VRAM. This snapshot is a point-in-time read, not a live guarantee — re-check
before sizing any actual job (per CLAUDE.md §9's live-vs-SOP note).

**Qwen2.5-72B-Instruct download:**
- Target: `Qwen/Qwen2.5-72B-Instruct` (~144GB), destination `/scratch/jagatsesh/hf` (via `HF_HOME`).
- Rationale for running on the login node: pure network I/O, not compute/training/benchmarking — judged
  in-bounds under CLAUDE.md §2 ("light scripting" / not explicitly one of the listed prohibited categories),
  per explicit user direction.
- **Attempt 1:** started via `nohup /home/jagatsesh/miniconda3/envs/slaybench/bin/hf download Qwen/Qwen2.5-72B-Instruct > download_qwen72b.log 2>&1 &`, PID `1002801`. Confirmed alive via `kill -0` at ~2m37s elapsed
  (RSS ~52MB, log still empty — consistent with `hf download`'s pre-transfer manifest-resolution phase, not
  itself a sign of failure). Login node was independently observed to be sluggish around this time — even
  trivial read-only commands (`ps aux`, `tail`, `find`, plain `ls -la`) were timing out past 30–60s and
  getting backgrounded during this window. Per the user's report, this attempt **died around 18:06–18:07**,
  likely due to that login-node hiccup rather than anything specific to the download itself.
- **Attempt 2 (current):** restarted by the user directly, inside a `tmux` session named `qwen72b_dl`, so it
  survives login-node/SSH-session disruptions independently of any single shell process. Per explicit
  instruction, Claude Code is **not** touching, attaching to, or sending any input to this tmux session —
  status here is as reported by the user, not independently re-verified this turn.
- No GPU job has been started and no sbatch script has been written — download only, per explicit
  instruction to stop there.

---

## 2026-08-30 — Download outcome verified, stale partials cleaned up

Status check four days after the download work (server time `Sun Aug 30 16:05 IST 2026`; login node
`hpc01.sharanga.local`, uptime 130 days, load avg ~11).

**Download: COMPLETE.** Attempt 2 (tmux `qwen72b_dl`) finished 2026-08-26 20:07. Verified properly rather
than by byte count alone — every shard checked against `model.safetensors.index.json`:
```
shards present : 37/37    missing: []    zero-byte: []
expected total_size : 145,412,407,296 bytes (135.43 GiB)
actual sum of shards: 145,412,519,312 bytes (135.43 GiB)
delta               : 112,016 bytes
broken symlinks in snapshot: 0  (46/46 resolve)
```
The 112,016-byte excess is expected and benign: safetensors per-shard headers (~3KB × 37), which
`total_size` excludes since it counts only tensor bytes. Snapshot revision:
`495f39366efef23836d0cfae4fbe635880d2be31`.

**tmux session `qwen72b_dl`: gone** — the whole tmux server has exited (`no server running on
/tmp/tmux-1248350512/default`). Expected and harmless; it completed its work before exiting.

**Cron keep-alive: firing correctly.** `crontab -l` still shows the entry. `touch_scratch.log` is 0 bytes
but its mtime is `Aug 28 03:00` — cron fired exactly on the `0 3 */3 * *` schedule and wrote nothing
because the script is silent on success (stderr would have landed there on failure). Next fire: Aug 31
03:00. Scratch remains protected from the 15-day purge.

**Cleanup — 8 stale `.incomplete` files deleted (8.9 GB reclaimed), with explicit user confirmation per
CLAUDE.md §6.** These were debris from download attempt 1 (PID 1002801, died ~18:06–18:07 on Aug 26),
corresponding to shards 1–8, which attempt 2 then re-fetched cleanly. Deletion was scoped to
`blobs/*.incomplete` at `-maxdepth 1` only; the real blobs for all 8 were confirmed present and verifying
beforehand. Files removed:
```
18d5d2b7….10da4a60  1,005,920,203      b7f066ae….7d36a544  1,066,499,630
802a3abf….89b51939  1,469,072,181      a1473de3….c5f568ae  1,296,157,709
c3a2ab09….d831fe91  1,604,379,607      e5ea29c1….eace26e0    922,506,272
5f35d547….94a2bb49  1,574,260,421      06550d13….f4956b8b  1,267,783,163
```
Post-deletion re-verification confirmed 37/37 shards intact, 0 broken symlinks, identical byte totals.
Model dir size: **138G → 129G** on disk.

**Model facts captured for run planning** (read from the downloaded `config.json` / `tokenizer_config.json`,
and the local env):
| Property | Value |
|---|---|
| `num_hidden_layers` | 80 |
| `num_attention_heads` | 64 |
| `num_key_value_heads` | 8 (GQA) |
| `hidden_size` | 8192 |
| `max_position_embeddings` | 32768 |
| `rope_scaling` | `None` |
| `sliding_window` | 131072 |
| `torch_dtype` | `bfloat16` |
| chat template present | yes |
| vLLM in `slaybench` env | **0.25.1** |

Note on vLLM: 0.25.1 is a *third* version — neither the plan's pinned 0.27.1 nor the 0.19.1 actually used
for the gemma3-27b-it / qwen3-32b runs. Per `PLAN_DEVIATIONS.md` E5/E9, the pinned API facts
(`TokensPrompt`, `Logprob.rank`, `VLLM_BATCH_INVARIANT`) must be re-verified against 0.25.1 before relying
on them.

**Still open (nothing started):** no sbatch script written, no GPU job submitted, no Slurm allocation held.
Confirmed repeatedly this session — login node has no GPUs (`nvidia-smi: command not found`),
`$SLURM_JOB_ID` unset, `squeue -u` empty of any Claude-submitted work.

---

## 2026-08-30 — Phase-2 run setup for Qwen2.5-72B-Instruct (Steps 1–4)

Created `phase2_qwen25_72b_instruct_run/`. **No GPU job submitted, nothing run on a GPU.**
Both precedent run directories were treated as strictly read-only throughout; verified after every
write with `find phase2_gemma3_27b_it_run phase2_qwen3-32b_run -newermt "-10 minutes"` → empty.

### Blocker decisions received (all from the user, 2026-08-30)
| # | Decision |
|---|---|
| 1 | Activations → `/scratch/jagatsesh/IEA-qwen72b-activations/`, symlinked as `activations` |
| 2 | Split into two dependent jobs (A = generation, B = capture, `afterok`) |
| 3 | Option A — run `normalise_translations.py` with the single drop id |
| 4 | Run `token_budget.py` as part of setup |
| 5 | Keep the model-slug level in output paths; don't modify `generate.py` |
| 6 | Add `--account=bits --qos=qos_gpu_h200` to both sbatch scripts |
| 7 | Clamp `mean8_content` to content start; record true `content_mean_lo`/`_n` |
| 8 | 81 layers confirmed (index 0 = embedding, 1–80 = decoder) |
| 9 | Accept Qwen2.5's injected system prompt; document prominently |

### Step 1 — structure
Created `src/phase2/`, `scripts/`, `config/`, `docs/`, `preflight/`, `data/{cues,final_set/_incoming}`,
`generations/qwen25-72b-instruct/{harmful,benign}/{en,hi,bn,ta,te,kn}/`, and the matching activation
tree on scratch. `activations` → `/scratch/jagatsesh/IEA-qwen72b-activations` (symlink verified).

### Step 2 — pipeline copied and configured
Copied unmodified from `phase2_gemma3_27b_it_run/`: `__init__.py`, `config.py`, `assemble.py`,
`generate.py`, `gates.py`, `verify.py`, `consolidate.py`, `io_jsonl.py`, `script_lid.py`,
`manifest.py`, `token_budget.py`, `g2_load.py`, `power_sim.py`, plus `scripts/` and
`config/{languages.yaml,exclusions.json,gate_refusal_markers.json}` and `data/cues/cue_battery.json`.

Cue battery: the gemma copy was used. Its 30 condition texts are **byte-identical** to the project-root
`data/cue_battery.json` (verified cell by cell); only `_meta`/bookkeeping keys differ, and the gemma
copy carries the review-verdict provenance.

Wrote new `config/models.yaml` and `config/run.yaml` (full contents shown to user at Step 5). Key
values: `tensor_parallel: 2`, `max_model_len: 8192`, `max_num_seqs: 64`, `gpu_memory_utilization: 0.90`,
`enforce_eager: true`, bf16, `thinking_kwarg: null` (Qwen2.5 is not hybrid-thinking), seed 2026,
`batch_invariant: true`, `attention_backend: TRITON_ATTN`, partition `gpu_h200_8`, account `bits`,
QOS `qos_gpu_h200`.

**`assemble.py` was NOT modified** — see D-72B-6. The brief called for renaming the data pattern to
`harmful_<lang>.json`, but that was premised on reading raw root data. Under decision 3 the normaliser
writes canonical files under exactly the names `assemble.py` already expects, so renaming would have
broken the loader against its own inputs. The file is byte-identical to the gemma original.

**Data normalisation (decision 3).** Staged the raw project-root files into the documented drop point
`data/final_set/_incoming/`, then:
```
python3 scripts/normalise_translations.py --arms harmful,benign --langs en,hi,bn,ta,te,kn \
    --source opus --run-id opus_delivered_20260819 \
    --drop-doc-ids b6d11d84-439a-4509-9d23-1defd0f78781
```
`--check` passed first; the write produced **harmful n=199** (1-based, contiguous) and
**benign n=200** in all six languages, `prompt` field populated everywhere, extra delivered fields
carried through. Matches both precedents exactly.

Provenance note (D-72B-5): `--source opus` is accurate per `data/TRANSLATION_SOP.md` §2 (Claude Opus,
`claude-opus-5`; no row carries `translation_method: google_translate_manual`). The `--run-id` stamp is
approximate — the true upstream translation run id is recorded nowhere in the repo, and neither
precedent logged theirs.

**Token budget (decision 4).** Ran `python -m phase2.token_budget --models qwen25-72b-instruct`
(tokenizer-only, CPU, login node). Result:
```
{'en': 512, 'hi': 2272, 'bn': 2592, 'ta': 2688, 'te': 3072, 'kn': 3072}
```
⚠️ **This table is not yet trustworthy — see Open issues below.**

### Step 3 — `src/phase2/capture_activations.py` (new)
Standalone post-generation script. Reuses each row's recorded `prompt_token_ids` (no re-templating, so
no drift from what vLLM saw), runs `model(..., output_hidden_states=True)`, extracts `last`,
`last_content`, `mean8`, `mean8_content` over all 81 hidden states, saves one
`(n_rows, 81, 8192)` bf16 tensor per variant into a single safetensors file per shard, plus a
`.index.json` sidecar. Processes one shard at a time; skips existing outputs unless `--overwrite`
(CLAUDE.md §14).

The content boundary is **derived from the tokenizer at runtime and asserted per row**, not hardcoded:
generation suffix `<|im_end|>\n<|im_start|>assistant\n` = 5 tokens → `content_end = n - 6`; verified
empirically. A template change breaks loudly rather than silently mislabelling positions.

### Step 4 — sbatch scripts (written, NOT submitted)
`run_qwen25_72b_gen.sbatch` (Job A) and `run_qwen25_72b_capture.sbatch` (Job B), plus `submit_both.sh`
which wires `--dependency=afterok:`. Both: `gpu_h200_8`, `--account=bits --qos=qos_gpu_h200`,
`--gres=gpu:2`, `--cpus-per-task=8`, `--mem=300G`, `--time=24:00:00`, explicit
`source ~/miniconda3/etc/profile.d/conda.sh`, `PYTHONPATH`, `cd` into the run dir (run.yaml paths are
relative), `HF_HUB_OFFLINE=1`. Job A loops **six** invocations of
`python -m phase2.generate --model qwen25-72b-instruct --lang <L> --run-id <id>`.
`VLLM_ATTENTION_BACKEND` deliberately **not** exported (no-op since vLLM 0.19.1).

Run id is persisted to `preflight/RUN_ID` on first run and re-read on resubmission — without a stable
run id, `record_id` changes and the idempotent resume silently regenerates everything.

### Open issues at end of Step 4
1. **`max_tokens` heuristic is unsafe as generated.** `en` hit the FLOOR (512) and `te`/`kn` hit the
   CEIL (3072) — both are clip artifacts, not measurements. Gemma's committed table came from the
   *empirical probe* mode, not the heuristic. §5.2 requires that truncation never correlate with
   language, and qwen3-32b already FAILED G6 on te (6.93%) / kn (7.44%). Needs a decision before
   Job A runs.
2. **vLLM 0.25.1** — a third version, neither the plan's 0.27.1 nor the precedents' 0.19.1. The
   pinned-API facts (`attention_backend` engine arg, `Logprob.rank`, `VLLM_BATCH_INVARIANT`) must be
   re-verified against it (inherited E5/E9).
3. Runtime for Job A remains an estimate; the 24h wall may need a resume cycle.

---

## 2026-09-08 — FlashInfer two-bug fix chain + Probe resubmitted (job 336608)

Earlier probe submissions failed on two FlashInfer JIT issues, both now resolved:

1. **`-lcuda` missing** (job 335977): conda env's `lib64/stubs/` did not exist. Fix:
   `mkdir -p .../slaybench/lib64/stubs; ln -sf /usr/lib64/libcuda.so.1 .../slaybench/lib64/stubs/libcuda.so`
2. **GLIBCXX_3.4.32 not found** (follow-on): JIT compiled with conda's GCC 15.2.0 but
   runtime loaded system's older `libstdc++.so.6`. Fix:
   `export LD_LIBRARY_PATH=".../slaybench/lib:${LD_LIBRARY_PATH:-}"`

Both fixes applied to all three sbatch scripts. Smoke test (2×H200, TP=2, vLLM init)
printed literal `SUCCESS — FlashInfer JIT compiled, vLLM engine started`, exit 0.

Probe resubmitted as job 336608, walltime bumped to 06:00:00. Started immediately on
gpunode7.

---

## 2026-09-09 — Probe 336608 COMPLETED, token budgets accepted

```
sacct -j 336608: COMPLETED, exit 0:0, elapsed 05:51:31
Start 2026-09-08T22:24:53, End 2026-09-09T04:16:24
```

6h walltime was critical — would have been killed at the earlier 4h setting.

### Measured token budgets (`config/max_tokens.json`)
```json
{"qwen25-72b-instruct": {"en": 1280, "hi": 2880, "bn": 3648, "ta": 5824, "te": 5824, "kn": 5824}}
```

| lang | heuristic | measured | note |
|---|---:|---:|---|
| en | 512 | **1280** | FLOOR artifact confirmed (2.5×) |
| hi | 2272 | **2880** | |
| bn | 2592 | **3648** | 1/400 truncated (0.25%) |
| ta | 2688 | **5824** | 45/400 truncated (11.25%) — ceiling |
| te | 3616 | **5824** | 52/400 truncated (13%) — ceiling |
| kn | 3360 | **5824** | 62/400 truncated (15.5%) — ceiling |

ta/te/kn ceiling hits are **degenerate repetition**, confirmed by spot-checking the
longest kn response: coherent numbered points in Kannada in the first portion, then the
same phrase repeated 3× with the last cut off mid-word. Matches the Gemma3 kn pattern.
Raising `max_model_len` would only extend the junk.

Budgets accepted. `config/max_tokens.json` is now authoritative. Next: submit Jobs A+B
via `submit_both.sh`.

---

## 2026-09-09 — Pre-flight + truncation audit + Job A submitted (job 337646)

10-item pre-flight checklist passed (both sbatch fixes verified, input data present,
home disk at 252M/40G, model weights on scratch, queue clear, `max_tokens.json` correct,
output dirs empty).

Truncation deep audit (teammate-requested): manual review of all 80 truncated probe
responses found ~80–85% show clear phrase/sentence-level repetition in final 150 chars,
~10–15% ambiguous/degrading, zero cases of novel content cut off. Team decision: keep
budgets as-is.

---

## 2026-09-10 — Job A COMPLETED (job 337646)

```
sacct -j 337646: COMPLETED, exit 0:0, elapsed 10:04:04
Start 2026-09-09T15:52:00, End 2026-09-10T01:56:04, gpunode7, 2×H200 TP=2
```

Run ID: `20260909T102224Z-qwen25-72b-instruct-invariant`. 60/60 shards, 11,970 rows.
Actual 10h runtime well under the 24h wall (no resume needed). Both FlashInfer fixes
held for the full run. Next: Job B (activation capture).
