# GPU Cluster SOP — running LLM inference/generation jobs on a shared SLURM cluster

Generic playbook distilled from running Qwen3-32B generation across 6 languages x 3
conditions on a shared SLURM cluster (H100/H200/A100 pools, multi-tenant account).
Written to be reused on a different cluster or project — nothing here is specific
to this study's dataset or model. Every rule below traces back to a real incident;
the "why" is kept alongside the rule so it's clear when it still applies and when
it doesn't.

---

## 1. Before you submit anything

**Don't trust `sacctmgr show assoc` (or equivalent) for permission checks.** It can
report only `compute`/`cpulimit`-type associations even when GPU jobs succeed fine —
QOS can be assigned automatically at submit time rather than shown in the static
association table. The only reliable test is actually submitting a small job.

**Run a 10-minute interactive sanity check before any real job:**
```bash
srun -p <gpu_partition> --gres=gpu:1 -c 8 --mem=64G --time=00:10:00 --pty nvidia-smi
```
Confirms real GPU access, not just partition existence. Cheap insurance against
burning a multi-hour queue wait only to discover a permissions problem.

**Map partitions to per-user caps before sizing jobs**, and treat a documented cap
as provisional until cross-checked — clusters' own docs can be stale. Note both the
**per-partition per-user GPU cap** and the **per-account aggregate memory/CPU cap**
(see §4 — these are different limits and both bite).

**If a job gets cancelled fast with `Reason=Resources`:** check the partition's
current queue first. If it's genuinely empty, it was a transient scheduling hiccup —
just resubmit. Don't conclude "no access" from one failure.

---

## 2. Launching servers / long-running processes

**Give every concurrent process of yours a distinct, fixed port** if you're running
more than one server-style process (e.g. multiple inference servers) that might land
on the same physical node. A port is a property of the *host*, not the job — two of
your own SLURM jobs can land on the same node, and if both default to the same port,
one silently fails to bind. Assign ports by role/condition at submit time
(`--port-base 8000`, `8001`, `8002`, ...), don't rely on a shared default.

**Tag long-running-process log files with the launching process's own PID**
(`~/server_{role}_{gpu}_pid{PID}.log`), not just role+GPU-index. Two concurrent jobs
using the same role+GPU-index label will otherwise truncate each other's log on
startup — harmless to the actual computation, but it destroys your ability to debug
after the fact (you'll be reading one process's history and think it's another's).

**Fail fast on subprocess death, don't just poll a timeout.** A readiness-wait loop
that only checks "is the port answering yet" will burn its *entire* timeout on a
server that crashed instantly (e.g. an unrecognized CLI flag) instead of reporting
the real error in seconds. Check `proc.poll()` (or equivalent) each iteration and
bail immediately if the process has already exited, pointing at its log.

**Set `PYTHONUNBUFFERED=1`** (or run `python -u`) for anything whose output goes to
a job log file. Python's stdout is block-buffered when redirected to a file, not
line-buffered — if the process dies abruptly (OOM-killed, `scancel`, node failure),
everything sitting in the buffer is lost, including the one error message that would
have explained what happened. This one setting turns "job log shows nothing useful"
into "job log shows exactly what went wrong."

---

## 3. Coordinating multiple concurrent jobs safely

**Never use one shared mutable file (e.g. a single `servers.json`) as a registry
that multiple concurrent jobs read-modify-write.** A naive whole-file
read-then-write race means job B's write can silently clobber job A's registration
moments after A wrote it — and if A's own registration then vanishes mid-run, A
doesn't necessarily crash; it may just start failing every subsequent request. If
your resilience design writes `"failed": true` stubs instead of stopping on error
(reasonable in isolation), this combination is how you silently destroy an entire
run's worth of data without a single crash to alert you.

**Fix: per-owner files, not one shared file.** Give each independent entity (each
model/role/target) its own file (`registry/{target}.json`), written atomically
(temp file + `os.replace`/equivalent). Two jobs registering *different* targets can
never collide because they never touch the same file — no locking needed, and
deliberately so:

**Don't rely on advisory locks (`flock`) on a shared network filesystem for
cross-node coordination.** Lustre/NFS-style shared filesystems don't reliably
enforce advisory locks across different compute nodes. Structural avoidance (own a
file no one else touches) beats a lock you can't fully trust.

**Where possible, skip the shared registry entirely** for the common case: if a job
launches its own server and is the only consumer of it, have the launcher print the
assigned port directly (`LAUNCHED_PORTS=...`) and pass it straight to the consumer
via a CLI arg, instead of writing to *and reading from* any shared state. Reserve the
registry for the genuine cross-job-discovery case (a separate later job needs to find
a server some earlier job started).

---

## 4. Memory sizing — a bigger trap than it looks

**A per-job `--mem` fix must be checked against the *account's total* cap, not
considered alone**, whenever other jobs from the same account are running
concurrently. If the account cap is 300G and two of your jobs already claim 96G
each (192G), a third job asking for 160G will never fit — it queues forever with a
reason like `QOSMaxMemoryPerUser`, which can look identical to ordinary GPU
contention (`Priority`/`Resources`) unless you inspect the specific reason string
after enough time has passed. Don't diagnose a long queue wait as "the cluster is
just busy" without checking the reason field first.

**Workloads that hold extra per-request state in host memory (e.g. activation/
hidden-state capture, any buffering that isn't released until a save/flush) need
meaningfully more memory *and* proportionally more caution around concurrency
(worker count) than a plain generate-and-discard workload** — even at the identical
`--mem` limit, on the identical model, the two can behave completely differently.
If a framework's own defaults are lower for the heavier-state workload (e.g. a
capture-mode client defaulting to fewer concurrent workers than the plain-generate
client), that's a signal, not an arbitrary choice — don't override it back up to
match the lighter workload's concurrency without separately re-validating memory
headroom.

**An OOM kill is not always immediate** — sustained load over hours can cross a
memory ceiling that looked fine in the first 20 minutes. Don't declare a
configuration "safe" from a short observation window if the job is going to run
for many hours; watch actual `MaxRSS`/utilization trends, and if the failure is a
kernel OOM (`sacct` state `OUT_OF_MEMORY`, `oom_kill` event), take the process's own
log at face value (`ok` up to the last recorded line) rather than assuming the data
before the kill is corrupted — a clean OS kill and a mid-write corruption are
different failure modes with different data-integrity consequences.

---

## 5. Sync workflow between local machine and the cluster

**If the cluster copy of your code is a snapshot pushed by tarball (not a git
clone, e.g. because credentials can't live on a shared account) — never `rm -rf`
the destination directory as part of that push**, even if that's the simplest way
to guarantee a clean sync. The destination directory is also where the job *writes
its own output*. A "refresh code" step that wipes the whole directory will silently
destroy generated results sitting in a subdirectory the sync script doesn't
otherwise touch, with zero warning, the moment it runs — and it *will* run again,
routinely, as a matter of habit, long after everyone's forgotten the original
directory was ever empty of real data.

**Fix: extract the new code into a temp location, then sync it in with an explicit
exclude list for anything that holds generated output** (`rsync -a --delete
--exclude data --exclude outputs --exclude <registry-dir> tmp/ dest/`). This keeps
code fully refreshed/pruned to match source control while making it *structurally
impossible* for a code sync to delete run output, regardless of when it's next
triggered.

**Don't batch results-pulling to "end of session."** Pull generated results back
(and commit/push if using git) incrementally, as each meaningful unit of work
finishes — a full day (or week) of unpulled compute sitting only on cluster scratch
is both a data-loss risk (scratch purges after N days of inactivity; a code sync bug
like the one above is also just waiting there) and a real financial-risk conversation
if anyone's paying per GPU-hour, since a later failure discards work nobody's seen
yet. Automate this if you can (see below), but the underlying principle — pull often,
don't accumulate — matters more than the automation.

**A background "auto-pull on a timer" script is a reasonable idea but has one hard
limit worth stating explicitly: it can't run while the local machine is powered
off.** Don't oversell it as a fix for "I'm shutting down my laptop" — it only helps
while the machine is on. What actually protects you across a shutdown is making sure
the *cluster side* doesn't depend on the local machine being reachable: have each
job write its own consolidated/git-storable output as part of its own script (§6),
so results are already safe on cluster disk the moment they're produced, regardless
of whether anything ever pulls them promptly.

**On Windows/PowerShell specifically:** don't wrap native command calls (`ssh`,
`scp`, etc.) in `try/catch` combined with `2>&1` redirection under
`$ErrorActionPreference = "Stop"`. PowerShell 5.1 turns *any* line a native exe
writes to stderr — including completely normal banner/status text — into a
terminating `NativeCommandError`, so a command that actually exited 0 gets reported
and treated as a failure. Check `$LASTEXITCODE` explicitly instead of relying on
try/catch around native calls.

---

## 6. Consolidation — many small files vs. one big file

Per-request/per-run output as many small files (one JSON per request) is the right
format *during* generation — it's what makes resume-after-crash trivial (check "does
this specific file already exist and look successful" before redoing work). But it's
the wrong format for **git storage** at scale: git handles a very large file count
poorly regardless of total byte size (slow status/add/clone), independent of whether
the content itself would easily fit.

**Fix: consolidate into fewer, larger files (e.g. one JSONL per logical grouping),
generated automatically as part of the job, not as a separate manual step run
whenever someone remembers.** Two extra rules to keep this safe under concurrency,
mirroring §3:

- If multiple concurrent jobs each own a distinct slice of the output (e.g. one
  job per condition, all producing the same logical language/target), have each
  own its own intermediate consolidated file, not a shared one they all write to.
  A cheap final concatenation step (reading only, safe to rerun any number of
  times) merges the owned slices into the final combined file.
- Consolidation should be additive/non-destructive: never modify or delete the
  original per-run files while a job might still be resuming against them.

**Add a failure circuit breaker to any long batch loop that has resilient
per-item error handling** (i.e. doesn't crash on a single failed request). Track a
running failure rate; once it's clearly not transient (e.g. >50% failed after 20+
samples), stop submitting new work and exit non-zero, rather than burning through
the entire remaining batch writing failure stubs. Combined with `set -e` in the
calling script, this prevents a dead dependency (crashed server, network partition)
from letting the job silently "complete" on garbage — it aborts loudly instead,
leaving the good prefix intact for a clean resume.

---

## 7. Debugging checklist

- `squeue -u <user> -o "%i %j %t %M %R"` — quick state; the `%R` (reason) column
  matters more than it looks, since `Resources`, `Priority`, and
  `QOSMax*PerUser` all *look* like "just wait" but have different actual causes
  and different fixes.
- `squeue --start -j <jobid>` — SLURM's own scheduling estimate. Treat it as a
  worst-case, not a promise — jobs finishing early routinely beat it. If it shows
  `N/A`, that's itself informative (often a sign the request genuinely can't fit
  under a current constraint, not just "hasn't been estimated yet").
- `sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed` — the source of truth for
  what actually happened to a finished/failed job; don't infer failure mode from a
  vanished `squeue` entry alone.
- `sacct -j <jobid> --format=JobID,MaxRSS,ReqMem,AllocTRES%40` — check actual memory
  usage against the request before assuming a repeat OOM will recur at a bumped
  limit; sometimes the real fix is elsewhere (see §4).
- `scontrol show job <jobid>` — works briefly even after completion; shows exact
  node/GPU allocation, useful for confirming two jobs really are (or aren't) sharing
  a physical resource.
- **Cross-check `squeue -p <partition> -t RUNNING` against your own job's queue
  reason** before concluding "the cluster is busy" or "someone is hogging it" — it's
  worth distinguishing one long-running job monopolizing a shared pool (a real,
  nameable cause) from ordinary multi-user contention (nobody's fault, just wait) —
  the two look identical from your own job's perspective but call for different
  reactions.
- **Read the actual server/worker log, not just the job's own stdout log**, when
  debugging a failure — the job log may only show a launch banner if the process
  died before flushing buffered output (see §2's `PYTHONUNBUFFERED` note); the
  underlying server process's own log usually has the real story.
