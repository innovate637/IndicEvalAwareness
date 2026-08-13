# Fact-Check: Technical Experimental Plan Verification (as of August 2026)

## TL;DR
- **Most claims are CONFIRMED, but three need correction:** (a) the Gemma 3 *instruction-tuned* chat template no longer raises on a `system` role — it accepts and merges the system content into the first user turn, so your "raises an exception" premise is **WRONG for Gemma 3** (it was true only for Gemma 1/2); (b) `HF_HUB_ENABLE_HF_TRANSFER=1` is now a **deprecated no-op** — the Hub is fully on the Xet backend, so use `HF_XET_HIGH_PERFORMANCE=1`; (c) passing `prompt_token_ids=` as a keyword to `llm.generate()` has been **removed** — use a `TokensPrompt`/dict.
- **The core Sarvam claim is CORRECT:** `sarvam-m` was post-trained from `mistralai/Mistral-Small-3.1-24B-Base-2503` (not the 2501 or 3.2 checkpoint); `sarvam-30b` and `sarvam-105b` both exist as open-weight `sarvam_moe` MoE repos requiring `trust_remote_code`.
- **Both McNemar formulas are essentially correct,** with one caveat each: `(b+c)/n²` is the *null-approximation* of the paired-difference variance (full estimator subtracts `(b−c)²/n³`), and `sqrt(v1+v2)` for a difference-of-differences is valid **only if the two contrasts use disjoint items**.

## Key Findings

| # | Claim | Verdict |
|---|-------|---------|
| A1 | sarvam-m built on `Mistral-Small-3.1-24B-Base-2503` | **CONFIRMED** |
| A2 | `sarvamai/sarvam-30b` exists, MoE, `enable_thinking`, `trust_remote_code` | **CONFIRMED** |
| A3 | `sarvamai/sarvam-105b` exists, open weights | **CONFIRMED** |
| A4 | Both gemma-3-27b `-it` and `-pt` exist, gated, "gemma" license | **CONFIRMED** |
| A5 | Qwen3-32B: 64L / 64Q / 8KV / head_dim 128 | **CONFIRMED** |
| A6 | Gemma 3 chat template *rejects* system role | **WRONG** (merges it) |
| B7 | vLLM stable version + `LLM()` kwargs | **CONFIRMED** (v0.27.1) |
| B8 | `SamplingParams`, `logprobs=5` structure | **CONFIRMED** |
| B9 | `{"prompt_token_ids": [...]}` vs `TokensPrompt` | kwarg **REMOVED**; dict OK |
| B10 | `reasoning_content` server-only | **PARTLY** (offline exists but fragile) |
| B11 | `VLLM_BATCH_INVARIANT=1` | **CONFIRMED** |
| B12 | `enforce_eager=True` | **CONFIRMED** |
| D15 | `python-ulid` vs `ulid-py` | use **python-ulid** |
| D16 | `HF_HUB_ENABLE_HF_TRANSFER=1` | **DEPRECATED no-op** |
| D17 | `*.py` needed in allow_patterns | **CONFIRMED** |
| E18 | `%u` expands; Slurm doesn't mkdir | **CONFIRMED** |
| E20 | `--array=0-5%3` | **CONFIRMED** |
| F21 | Exact McNemar p-value formula | **CONFIRMED** |
| F22 | Paired-diff variance / DoD SE | **CONFIRMED w/ caveats** |

## Details

### A1. `sarvamai/sarvam-m`
**CONFIRMED.** The repo exists. Its `README.md` metadata reads verbatim: `license: apache-2.0 ... base_model: - mistralai/Mistral-Small-3.1-24B-Base-2503`, `base_model_relation: finetune`, with languages en + 10 Indic (bn, hi, kn, gu, mr, ml, or, pa, ta, te), and the description "sarvam-m is a multilingual, hybrid-reasoning, text-only language model built on Mistral-Small." Model size is **24B params**, architecture **`MistralForCausalLM`** (dense — `config.json` uploaded as "MistralForCausalLM"; the vision encoder from Mistral Small 3.1 was removed for text-only use). License **Apache-2.0**, and the repo is **NOT gated**.

Your specific claim is **CORRECT on every point**: it was post-trained from `mistralai/Mistral-Small-3.1-24B-Base-2503`; that is the right repo id; the base repo **exists** and is Apache-2.0; and it is **not** the `Mistral-Small-24B-Base-2501` (Mistral Small 3) checkpoint nor `Mistral-Small-3.2`. The base repo `Mistral-Small-3.1-24B-Base-2503` carries an `extra_gated_description` pointing to Mistral's privacy policy — i.e. a **light gate** (a privacy acknowledgement), not a full license-acceptance wall. sarvam-m supports think/non-think hybrid mode via `enable_thinking` (default `True`) in `apply_chat_template`; via the Sarvam/OpenAI API the equivalent control is `reasoning_effort` (low/medium/high).

### A2. `sarvamai/sarvam-30b`
**CONFIRMED under exactly that name.** From the model card: `model_type: sarvam_moe`, a Mixture-of-Experts, **~32B total params** (HF reports "32B params"), **2.4B non-embedding active params**. Architecture (verbatim from the card): "uses 19 layers, a dense FFN `intermediate_size` of 8192, `moe_intermediate_size` of 1024, top-6 routing, grouped KV heads (`num_key_value_heads=4`), and an extremely high rope_theta (`8e6`)... 128 experts with a shared expert, a routed scaling factor of 2.5, and auxiliary-loss-free router balancing." Attention: **64 query heads, 4 KV heads, head_dim=64** (GQA), 19 layers (1 dense + 18 MoE), sigmoid routing. Released **March 2026** under **Apache-2.0**.

Answers to your specific sub-questions: **total ≈ 32B / active 2.4B**; **MoE** (not dense); **19 layers**; **num_attention_heads = 64**; **num_key_value_heads = 4**; **Apache-2.0**; benchmarks run at **65,536 max context**; **supports `enable_thinking`** in the chat template; **requires `trust_remote_code=True`** (tagged `custom_code`). It is **NOT gated**. Caveat: native vLLM support was pending (PR #33942) at release; the card ships a `hotpatch_vllm.py` that pins vLLM 0.15.0 and registers the `sarvam_moe`/`sarvam-105b` executors — verify compatibility against your target vLLM (0.27.x) before relying on it, or use the `sarvamai/sarvam-30b-fp8` / `QuantTrio/sarvam-30b-AWQ` variants which document vLLM serving.

### A3. `sarvamai/sarvam-105b`
**CONFIRMED, open weights.** Downloadable from HuggingFace (`sarvamai/sarvam-105b`) and AI Kosh, and also served via the Sarvam API. It is an MoE with **10.3B active params**, 128 experts, top-8 routing, one shared expert, routed scaling factor 2.5, `intermediate_size=16384`, `moe_intermediate_size=2048`. So: open weights, not API-only.

### A4. Gemma 3 27B repos
Both **`google/gemma-3-27b-it`** (instruction-tuned) and **`google/gemma-3-27b-pt`** (pretrained/base) exist as **separate repos**. `-pt` genuinely is the pretrained base checkpoint — the `-it` README lists `base_model: google/gemma-3-27b-pt`. **Both are gated:** you must be logged in to HuggingFace and accept Google's Gemma usage license ("Acknowledge license"; "To access Gemma on Hugging Face, you're required to review and agree to Google's usage license"). The license name is exactly **`gemma`** (the Gemma Terms of Use), **not Apache**. Note "pt" here means *pretrained* and "it" means *instruction-tuned* — do not confuse with language codes. The `-it` repo class is `Gemma3ForConditionalGeneration` (multimodal, text+image).

### A5. `Qwen/Qwen3-32B`
Confirmed by the Qwen model card, config, and the Qwen3 technical report ("Qwen3-32B is a decoder-only Transformer model... with 64 layers, 64 query heads and 8 key/value heads"):
- **num_hidden_layers = 64**
- **num_attention_heads = 64** (query), **num_key_value_heads = 8** (GQA)
- **head_dim = 128**
- **hidden size ≈ 5120** *(inferred; see caveat below)*
- **context length: 32,768 native** (`max_position_embeddings` = 40,960), extendable to **131,072 via YaRN**
- **32.8B total params (31.2B non-embedding)**; **License: Apache-2.0**
- RoPE base = 1,000,000

**Legal tensor-parallel degrees:** TP must evenly divide both 64 attention heads and 8 KV heads, so **TP ∈ {1, 2, 4, 8}** are safe; TP=16 is illegal (only 8 KV heads). **`enable_thinking` IS supported** as an `apply_chat_template` kwarg (thinking is on by default; recommended thinking-mode sampling: temp 0.6, top_p 0.95, top_k 20, min_p 0 — do not use greedy decoding). A separate **`Qwen/Qwen3-32B-Base` repo** — **could not be confirmed** from primary sources in this pass (the base pretrained checkpoints for several Qwen3 dense sizes exist, but verify the exact `-Base` repo id before scripting against it). There is a confirmed `Qwen/Qwen3-32B-FP8` repo.

### A6. Gemma 3 chat template and the `system` role — **CORRECTION**
Your premise is **outdated for Gemma 3**. The classic Gemma 1/Gemma 2 instruction templates contain `{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}` and genuinely raise a `jinja2.exceptions.TemplateError: System role not supported`. **But the Gemma 3 instruction-tuned template accepts a `system` role and merges the system content into the first user turn rather than raising.** The current HuggingFace transformers Gemma 3 documentation shows an example that passes `{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]}` to `google/gemma-3-27b-it` via `apply_chat_template` and runs successfully. So for the current `google/gemma-3-*-it` template, a system role does **not** raise — it is silently merged/prepended.

**Flag:** an open bug (transformers #40849) reports that the smallest checkpoint, `gemma-3-270m-it`, may *silently omit* the system content from the rendered prompt. So while no exception is thrown, you should assert on the rendered string for your exact checkpoint (27B) to confirm the system text actually appears.

### B7. vLLM version and `LLM(...)` constructor kwargs
**Current stable vLLM is 0.27.1**, released Aug 11, 2026 (a patch on top of v0.27.0; also on PyPI as `vllm 0.27.1`). All of your listed constructor kwargs are **still accepted and unchanged** in the 0.27.x line: `model`, `revision`, `tokenizer_revision`, `dtype`, `tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `enforce_eager`, `seed`, `trust_remote_code`. None have been renamed, deprecated, or removed. (Note: vLLM has fully moved to the V1 engine; some *sampling/engine* legacy paths have changed, but these constructor args are intact.)

### B8. `SamplingParams` and the `logprobs` structure
`SamplingParams` accepts **all** of `temperature`, `top_p`, `top_k`, `seed`, `logprobs`, `n`, `stop`, `max_tokens`, `skip_special_tokens`. With `logprobs=5`:
- `CompletionOutput.logprobs` is a **list, one entry per generated token position**.
- Each entry is a **dict mapping `token_id (int) -> Logprob` object**. (Confirmed against vLLM's `_make_logprob_dict(...) -> dict[int, Logprob]`.)
- The `Logprob` object has **`logprob`, `rank`, and `decoded_token`** attributes — so **yes, `decoded_token` exists**.
- The **sampled token is always included even if it is not in the top-5**, so a position's dict may contain **up to 6 keys**.
- **Ordering:** do NOT rely on dict insertion order for rank — use the `Logprob.rank` field. (Recent vLLM also added a flattened `FlatLogprobs` container for GC efficiency that still supports the list-like `Sequence` API, so `list[dict[int, Logprob]]` access patterns keep working.)

### B9. Pre-tokenized prompts — **kwarg removed**
Passing **`prompt_token_ids=` as a keyword argument to `llm.generate()` has been removed** — it now raises `TypeError: LLM.generate() got an unexpected keyword argument 'prompt_token_ids'` (confirmed by users Dec 2025). The **correct current way** is to pass a `TokensPrompt`. Because `TokensPrompt` is a `TypedDict`, a **plain dict `{"prompt_token_ids": [...]}`** in the prompts list works and is equivalent to `TokensPrompt(prompt_token_ids=[...])`:
```python
from vllm.inputs import TokensPrompt
outputs = llm.generate([{"prompt_token_ids": ids}], sampling_params)   # OK
# or: llm.generate([TokensPrompt(prompt_token_ids=ids)], sampling_params)
```
So your `{"prompt_token_ids": [...]}` form is fine; the old kwarg form is not.

### B10. `reasoning_content` in offline mode
Historically `reasoning_content` / `--reasoning-parser` was **OpenAI-server-only**. In current vLLM, **offline support exists**: you can pass `reasoning_parser=` to the `LLM(...)` constructor (users confirm it works on v0.10.1.1/v0.11.0, and the `vllm.reasoning` module ships offline parsers, including `qwen3` and a Gemma reasoning parser). **However**, the offline `LLM.generate()` path does not use the streaming parser, and reasoning detection via `is_reasoning_end()` on prompt tokens is unreliable (documented failure with structured output). **Recommended for a robust offline batch job:** generate normally and **string-split the output yourself on `<think>...</think>`** (both sarvam-m and Qwen3 emit `<think>` blocks). This avoids dependence on parser edge cases.

### B11. Batch-invariant / deterministic mode
**CONFIRMED — the flag is exactly `VLLM_BATCH_INVARIANT=1`** (your usage is correct). It gives bitwise-identical results regardless of batch size (including prefill). Key constraints and costs:
- Requires an **NVIDIA GPU with compute capability ≥ 9.0** (Hopper/H100 and newer).
- **NOT supported for GDN_ATTN / hybrid Mamba models** (Qwen3-Next / Qwen3.6-style) — it aborts at startup with `RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN`. Your Gemma 3 / Qwen3-32B / sarvam targets are standard softmax-attention and are fine.
- **Throughput cost:** it is a deliberate performance trade-off; the docs state enabling it "may impact performance compared to the default non-deterministic mode" and it disables optimizations such as custom all-reduce. Expect a meaningful (roughly tens-of-percent) latency/throughput hit; treat the exact number as workload-dependent.
- Works in offline batch mode: set `os.environ["VLLM_BATCH_INVARIANT"] = "1"` **before** constructing `LLM(...)`.

### B12. `enforce_eager`
**CONFIRMED still exists.** `enforce_eager=True` disables CUDA-graph capture and runs the model in pure eager PyTorch — **lower memory footprint and faster startup, at some steady-state throughput/latency cost.** Default is `False` (CUDA graphs / piecewise compilation enabled). Note: batch-invariant mode may itself require a specific compilation config (`cudagraph_mode="PIECEWISE"` in examples), so test the two flags together.

### C13. KV cache per token (bf16)
General formula: **bytes/token = 2 (K+V) × num_layers × num_kv_heads × head_dim × 2 bytes.**

**Gemma 3 27B** — confirmed config (from `google/gemma-3-27b-it` config.json): `num_hidden_layers = 62`, `num_key_value_heads = 16`, `head_dim = 128`, `sliding_window = 1024`, interleaved **5 local sliding : 1 global**. Naïve (all-global) KV: 2 × 62 × 16 × 128 × 2 = **507,904 bytes ≈ 0.50 MB/token (~496 KiB)**.

**Qwen3-32B** — `num_hidden_layers = 64`, `num_key_value_heads = 8`, `head_dim = 128`: 2 × 64 × 8 × 128 × 2 = **262,144 bytes = 0.25 MB/token**.

**Sliding-window savings:** Gemma 3's 5:1 local:global pattern with a 1024-token window means that for **long sequences**, ~5/6 of layers cap their KV at 1024 tokens instead of growing with sequence length. **vLLM does implement a hybrid KV-cache allocator** that exploits this, so effective Gemma 3 KV memory is substantially lower than the naïve figure for long contexts. **For short sequences (< 1024 tokens) there is no saving** — and since your generations top out at ~1900 tokens total, the benefit is modest in your regime (only the ~900 tokens beyond the window, on 5/6 of layers). **Qwen3-32B is dense global attention with no sliding-window savings.**

### C14. Sizing a 27B bf16 model on an 80GB H100
Weights ≈ 54 GB. At `gpu_memory_utilization=0.90` → ~72 GB usable; subtract ~54 GB weights and ~3–5 GB activation/CUDA-graph overhead → **roughly 13–15 GB left for KV**. At ~0.5 MB/token that is on the order of **~26,000–30,000 KV tokens** in aggregate (more for Gemma 3 once sliding-window savings kick in). For prompts of 100–500 tokens and generations up to ~1400 (max sequence ~1900), sensible settings for a throughput-oriented offline batch job:
- **`max_model_len = 2048`** (tight fit) or **4096** for safety headroom.
- **`max_num_seqs`**: start around **128** and tune down if you see KV-cache preemption/recompute or OOM; with `max_model_len=2048` and ~26k KV tokens the hard ceiling on simultaneously *fully-packed* sequences is ~13, but continuous batching means many more short/in-progress sequences coexist, so 64–128 is a reasonable throughput target.
- Consider `max_num_batched_tokens` ≈ 8192–16384 to balance prefill/decode.

### D15. ULID package
The two packages **conflict on the top-level `ulid` import name**. `ulid-py` (`import ulid; ulid.new()`) has had **no release in over 12 months** and is effectively discontinued/low-attention. `python-ulid` (`from ulid import ULID; ULID()`) is **actively maintained with recent releases** and has Pydantic integration and a CLI. **Recommendation: use `python-ulid`** and never install both in the same environment (they clash). There is **no ULID type in the Python standard library**; if you don't strictly need lexicographic sortability, `uuid` (stdlib) is the safe alternative — or use `python-ulid` and treat its objects as strings for storage.

### D16. huggingface_hub
- `snapshot_download` still accepts `repo_id`, `revision`, `allow_patterns`, `max_workers` — all valid.
- `HfApi().model_info(repo).sha` **returns the commit SHA** of the requested revision (main by default) — correct.
- **CORRECTION: `HF_HUB_ENABLE_HF_TRANSFER=1` is now deprecated and a no-op.** In huggingface_hub v1.0+, "hf_xet is now the default package for uploading and downloading files to and from the Hub, replacing the previously optional hf_transfer, which has now been fully removed." The env-vars docs state: "Now that the Hugging Face Hub is fully powered by the Xet storage backend... hf_transfer can't be used anymore." Setting the legacy flag is silently ignored (a warning only fires if you *also* set `HF_XET_HIGH_PERFORMANCE`). **Use `HF_XET_HIGH_PERFORMANCE=1`** for fast transfer (and `HF_HUB_DISABLE_XET=1` to opt out). Related: the `huggingface-cli` was removed in v1.0 and replaced by the `hf` CLI (`hf download ...`).

### D17. `allow_patterns` and `trust_remote_code`
**CONFIRMED — you must include `*.py`.** If you restrict `snapshot_download(allow_patterns=[...])` to only `*.safetensors, *.json, *.model, *.txt, *.jinja, tokenizer*`, the custom `modeling_*.py` / `configuration_*.py` files that `trust_remote_code` models load via `auto_map` will **not be downloaded, and loading will fail**. For sarvam-30b / sarvam-105b (both `custom_code`), **add `*.py` to `allow_patterns`.**

### E18. Slurm `--output` path
**`%u` IS expanded to the username** in `--output=` (alongside `%j` job-id, `%A`/`%a` array ids). **However, Slurm does NOT create missing parent directories** — if the target directory does not exist, the job cannot write its output and effectively **fails/dies with no output file** (a common silent failure: no jobid visible, no file created). **Pre-create the directory before `sbatch`.** Also avoid `$HOME`, `~`, or shell variables in the `--output` path — use explicit absolute paths plus Slurm `%` tokens only.

### E19. Wrapping single-GPU vLLM in `srun`
For a single-node, single-GPU vLLM process, wrapping it in `srun` inside an sbatch script is **generally acceptable but unnecessary**, and can occasionally interfere with vLLM's multiprocessing worker spawning when CPU/task bindings are over-constrained (e.g., `--cpus-per-task`/affinity that starves worker processes). If you use it, prefer `srun --ntasks=1` and avoid restrictive CPU-affinity. **The simplest safe option is to launch the Python process directly** (no `srun`) since there is nothing to distribute on one GPU. *Sourcing on this exact interaction is thin — treat as best-practice guidance rather than a documented guarantee.*

### E20. Slurm job-array throttling
**CONFIRMED — `--array=0-5%3` is correct.** It creates **6 tasks (indices 0–5)** with **at most 3 running concurrently** (the `%N` suffix caps simultaneous tasks). You can also adjust a live job with `scontrol update ArrayTaskThrottle=<N> JobId=<id>`.

### F21. Exact two-sided McNemar (binomial) p-value
**CONFIRMED.** With discordant counts `b` and `c`, `n = b + c`, the two-sided exact p-value is:

**p = min(1, 2 · Σ_{i=0}^{min(b,c)} C(n, i) / 2ⁿ)**

Under H₀, conditional on `n = b+c`, one discordant cell ~ Binomial(n, 0.5), and `Σ_{i=0}^{min(b,c)} C(n,i)/2ⁿ` is exactly the binomial CDF at `min(b,c)`; doubling gives the two-sided value. This matches statsmodels' `mcnemar(..., exact=True)`, which computes `2 * binom.cdf(min(b,c), b+c, 0.5)` then applies `np.minimum(pvalue, 1)`.

**Edge cases you must get right:**
1. **The doubling can exceed 1**, which is why the **cap to 1.0 is mandatory** — when `b = c`, `min(b,c)` is at/above the median, the one-sided CDF ≥ 0.5, and doubling yields ≥ 1 → capped to exactly **1.0** (correct: no evidence against H₀).
2. **Use the SMALLER cell** as the CDF argument (`min(b,c)`), or equivalently the upper tail from `max(b,c)`. Using the larger cell in the lower-tail CDF gives the wrong tail.
3. **Here `n` means the discordant total `b+c`**, NOT the number of pairs — a very common confusion.
4. **The exact conditional test is conservative** (Fagerland, Lydersen & Laake 2013): for very small `b+c` it can never reach p < 0.05; mid-P or asymptotic variants are less conservative if power matters.

### F22. Paired-difference variance and difference-of-differences
The marginal difference estimator is **d̂ = (b − c)/n_pairs** (correct).

**Your `Var ≈ (b+c)/n²` is the null-approximation, valid when b ≈ c.** The full estimated variance (Agresti / PSU STAT 504) is:

**V̂(d̂) = [(b + c) − (b − c)²/n] / n²**, i.e. **SE(d̂) = √(b + c − (b − c)²/n) / n**, where `n` = number of pairs.

Under H₀ (b ≈ c) the `(b−c)²/n³` term vanishes and this reduces to your `√(b+c)/n`. So the claim is correct **as a test-statistic / null approximation** but **not** as the general confidence-interval SE — for a CI, use the full formula. **Do NOT use the independent two-proportion variance** `p₁(1−p₁)/n₁ + p₂(1−p₂)/n₂` for paired data; it ignores the within-pair correlation.

**Difference-of-differences (English vs. Tamil):**
Var(d̂₁ − d̂₂) = Var(d̂₁) + Var(d̂₂) − 2·Cov(d̂₁, d̂₂).
- **`√(v1 + v2)` is valid ONLY if Cov = 0**, i.e. the two contrasts come from **disjoint items/subjects**.
- **If the English and Tamil contrasts are computed on the SAME items** (shared pool, same models), they are **correlated**, Cov ≠ 0, and `√(v1+v2)` is **wrong** (it typically *understates* the SE when the contrasts are positively correlated). Estimate the covariance via a **cluster-robust / bootstrap SE resampling at the item level** (resample items, recompute d̂₁−d̂₂), or a GEE/delta-method treatment on the joint table. Determine your item overlap before choosing.

## Recommendations
1. **Fix the three confirmed errors in your plan now:** (a) drop any logic that expects Gemma 3 `-it` to *raise* on a system role — it merges it; instead assert the system text appears in the rendered prompt (guard against the 270m-style silent-omission bug); (b) replace `HF_HUB_ENABLE_HF_TRANSFER=1` with `HF_XET_HIGH_PERFORMANCE=1` (and drop `hf_transfer` from deps); (c) replace any `llm.generate(prompt_token_ids=...)` with `llm.generate([{"prompt_token_ids": ids}], ...)`.
2. **For sarvam-30b / sarvam-105b:** set `trust_remote_code=True`, **include `*.py` in any `allow_patterns`**, and pin/validate your vLLM version against the model's custom executor (native support arrived via a post-release PR; consider the FP8/AWQ variants that document vLLM serving, or run the card's hotpatch in an isolated env).
3. **Pin down two open items before finalizing:** confirm the exact `Qwen/Qwen3-32B-Base` repo id (and Qwen3-32B hidden size) directly from its `config.json`; and confirm whether your English/Tamil McNemar contrasts share items (this determines the DoD SE method).
4. **Determinism vs. throughput:** if you enable `VLLM_BATCH_INVARIANT=1` (H100 = OK), budget for a throughput hit and test it together with your compilation/`enforce_eager` settings; only enable it if bitwise reproducibility across batch sizes is a hard requirement.
5. **Slurm:** pre-create output directories (Slurm won't mkdir); `--array=0-5%3` and `%u` are fine as written; run single-GPU vLLM without `srun` unless you have a specific reason.
6. **Stats:** use `statsmodels.stats.contingency_tables.mcnemar(table, exact=True)` for p-values (it matches your formula and handles the cap), and use the **full** `√(b+c−(b−c)²/n)/n` SE (not the `√(b+c)/n` approximation) for reported confidence intervals; use a bootstrap for the difference-of-differences if items overlap.

**Thresholds that would change these recommendations:** if your sequences ever exceed ~1024 tokens materially, Gemma 3's sliding-window KV savings become significant and you can raise `max_num_seqs`; if discordant counts `b+c` are large (>~25), switch from `exact=True` to the asymptotic McNemar χ² (statsmodels default) for speed with negligible accuracy loss; if `b+c` is very small (<6), the exact test cannot reach significance, so consider mid-P.

## Caveats
- **Qwen3-32B hidden size (~5120)** is inferred, and the existence/exact id of a separate **`Qwen/Qwen3-32B-Base`** repo was not confirmed from a primary source this pass — verify against `config.json` before scripting.
- **Sourcing on the `srun`/vLLM worker-spawn interaction is thin** — treat E19 as best-practice, not a documented guarantee.
- Sarvam model **context-length and gating** details come from model cards and blog posts and can change with repo updates; re-check the live `config.json` at run time.
- vLLM internals (logprobs container, offline reasoning parsing) are evolving quickly across 0.2x releases; the specifics above reflect the 0.27.x line — pin your vLLM version in the experiment for reproducibility.
- The McNemar variance identities trace to McNemar (1947) via Agresti / PSU STAT 504; the exact-test code behavior is verified against the statsmodels source.