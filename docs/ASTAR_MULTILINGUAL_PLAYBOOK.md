# How multilingual / Indic-language papers get accepted at A* venues — a verified survey and playbook

**Purpose.** Survey of papers accepted at top venues (ACL, EMNLP, NAACL, ICLR, NeurIPS, AAAI) that work
with our languages (Hindi, Bengali, Tamil, Telugu, Odia) or with multilingual safety/interpretability —
what they did, the pipelines they used, and the recurring patterns that got them past reviewers. Ends with
a concrete mapping onto our eval-awareness paper.

**Verification.** Every citation below was checked on 2026-07-18 against its arXiv page and, where given,
the ACL Anthology / NeurIPS proceedings / AAAI OJS page. Venue labels are exact (main vs Findings vs
workshop is stated honestly — Findings is ACL-branded but not the main track; CORE ranks ACL/EMNLP/ICLR/
NeurIPS/AAAI as A*, NAACL as A).

---

## 1. The verified corpus (16 papers)

| # | Paper | Venue (exact) | Our langs touched | Type |
|---|-------|---------------|-------------------|------|
| 1 | Aya Model (Üstün et al. 2024) | **ACL 2024 main — Best Paper** | hi, bn, ta, te (101 langs) | model + eval suite |
| 2 | IndicLLMSuite (Khan et al. 2024) | **ACL 2024 main — Outstanding Paper** | all 5 + 17 more | data pipeline |
| 3 | IndicGenBench (Singh et al. 2024) | **ACL 2024 main** | 29 Indic langs, 13 scripts | benchmark |
| 4 | MILU (Verma et al. 2024) | **NAACL 2025 main** | hi, bn, ta, te, **or** + 6 | benchmark |
| 5 | MultiJail (Deng et al. 2024) | **ICLR 2024** | bn (9 langs, tiered) | safety benchmark |
| 6 | XSafety (Wang et al. 2024) | ACL 2024 **Findings** | hi, bn (10 langs) | safety benchmark |
| 7 | Language Barrier (Shen et al. 2024) | ACL 2024 **Findings** | low-res tiers | safety analysis |
| 8 | Low-Resource Jailbreak (Yong et al. 2023) | NeurIPS 2023 **SoLaR workshop** — Best Paper | bn among attack langs | safety attack |
| 9 | RTP-LX (de Wynter et al. 2025) | **AAAI 2025** | hi + 27 langs | judge-validity study |
| 10 | MEGA (Ahuja et al. 2023) | **EMNLP 2023 main** | 11+ Indic langs | eval methodology |
| 11 | Do Llamas Work in English? (Wendler et al. 2024) | **ACL 2024 main** | — (method precedent) | mech-interp |
| 12 | Language-Specific Neurons / LAPE (Tang et al. 2024) | **ACL 2024 main** | — | mech-interp causal |
| 13 | How do LLMs Handle Multilingualism? / MWork+PLND (Zhao et al. 2024) | **NeurIPS 2024** | hi among test langs | mech-interp causal |
| 14 | Refusal Direction is Universal Across Languages (Wang et al. 2025) | **NeurIPS 2025** | 14 langs | mech-interp causal — *our closest precedent* |
| 15 | Multilingual Human Value Concepts (Xu et al. 2024) | EMNLP 2024 **Findings** | 16 langs | concept vectors x-lingual |
| 16 | RomanSetu (Husain et al. 2024) | **ACL 2024 main** | hi, ta + 3 | training method |

---

## 2. Per-paper: what they did and why it was accepted

### 2.1 Aya Model — ACL 2024 Best Paper ([2402.07827](https://arxiv.org/abs/2402.07827), [anthology](https://aclanthology.org/2024.acl-long.845/))
Üstün, Aryabumi, Yong, Ko, D'souza et al. Instruction-tuned open model for 101 languages.
**Pipeline:** massive multilingual instruction mixture; then an *evaluation suite built as a contribution in
itself* — discriminative + generative tasks across 99 languages, **human evaluation by native speakers**, and
"simulated win rates" (LLM-judge pairwise preferences **calibrated against the human eval**), split into
held-out vs in-distribution tasks.
**Why accepted:** (a) the eval framework was novel, not just the model; (b) human eval anchors the LLM-judge;
(c) everything released. Lesson: **an LLM judge is publishable only when tethered to human judgments somewhere
in the paper.**

### 2.2 IndicLLMSuite — ACL 2024 Outstanding Paper ([2403.06350](https://arxiv.org/abs/2403.06350), [anthology](https://aclanthology.org/2024.acl-long.843/))
Khan, Mehta, Sankar, Kumaravelan, Doddapaneni et al. (AI4Bharat). 251B tokens pretraining + 74.8M
instruction pairs for 22 Indian languages.
**Pipeline:** crawling/cleaning/dedup for pretraining; instruction data via three explicitly-tiered sources —
(i) **manually verified** curated data, (ii) unverified-but-valuable data, (iii) synthetic (Llama-2/Mixtral
grounded in Indian Wikipedia/WikiHow); translation/transliteration of English sets with the tiers kept
distinguishable; toxicity handled by generating toxic prompts and collecting aligned-model refusals.
**Why accepted:** honesty about quality tiers instead of pretending uniformity; blueprint framing ("here is
the reusable pipeline"); full release. Lesson: **label your data-quality tiers explicitly** (our ta/te/or
"provisional" annotations should be presented exactly this way, as a tier, not hidden).

### 2.3 IndicGenBench — ACL 2024 main ([2404.16816](https://arxiv.org/abs/2404.16816), [anthology](https://aclanthology.org/2024.acl-long.595/))
Singh, Gupta, Bharadwaj, Tewari, Talukdar (Google). Generation benchmark, 29 Indic languages / 13 scripts /
4 families.
**Pipeline:** did **not** invent new tasks — extended existing benchmarks (CrossSum, XQuAD, XorQA, FLORES) by
**paid human translation** of the English items into Indic languages, producing multi-way parallel data;
evaluated GPT-4/PaLM-2/Llama etc.; per-language tables with an explicit English-gap analysis.
**Why accepted:** extension-of-known-benchmark design makes results interpretable (reviewers can compare to
prior numbers); human translation, not MT; parallelism across languages means differences are attributable to
language, not item mix. Lesson: **parallel items across languages is a design feature reviewers reward — we
have this (same base items across en/hi/bn/ta/te/or).**

### 2.4 MILU — NAACL 2025 main ([2411.02538](https://arxiv.org/abs/2411.02538), [anthology](https://aclanthology.org/2025.naacl-long.507/))
Verma, Khan, Kumar, Murthy, Sen et al. (AI4Bharat/IBM). ~80K MCQs, 11 languages **including Odia**.
**Pipeline:** the anti-translation design — questions sourced **natively** from Indian regional/state-level
exams in each language (local history, festivals, law + STEM), so low-resource languages are represented by
authentic text rather than translated English; 42 models evaluated; domain-wise breakdown.
**Why accepted:** cultural authenticity (reviewers increasingly reject translate-only benchmarks as
"English in disguise"); Odia and other truly low-resource languages present natively. Lesson: for cue text,
**native phrasing beats literal translation**; where we used MT for cues, we should state the verification
level per language.

### 2.5 MultiJail — ICLR 2024 ([2310.06474](https://arxiv.org/abs/2310.06474), [proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6b396f766a50e0853a5164e68048540c-Abstract-Conference.html))
Deng, Zhang, Pan, Bing. First multilingual jailbreak benchmark.
**Pipeline:** English harmful queries **manually translated by native speakers** into 9 languages, explicitly
stratified as high/medium/low-resource (Bengali sits in the low tier); two threat models (unintentional
non-English use vs deliberate multilingual attack); headline finding = unsafe-output rate rises as resource
level falls (~3x for low-resource); a fine-tuning defense (Self-Defense) closes part of the gap.
**Why accepted:** native-speaker translation of the *attack* set (the measurement instrument is trustworthy);
the resource-tier structure turns 9 languages into an interpretable 3-point curve; dataset released.
Lesson: **organize our 6 languages on a resource axis** (en ≫ hi > bn/ta/te > or) and present per-tier trends,
not just per-language tables.

### 2.6 XSafety — ACL 2024 Findings ([2310.00905](https://arxiv.org/abs/2310.00905), [anthology](https://aclanthology.org/2024.findings-acl.349/))
Wang, Tu, Chen, Yuan, Huang, Jiao, Lyu. First multilingual safety benchmark: 14 safety categories x 10
languages (incl. **Hindi and Bengali**), 28K items.
**Pipeline:** professional translation of established Chinese/English safety sets with verification;
finding = all four LLMs tested are markedly less safe in non-English; a prompting fix reduces non-English
unsafe rate 19.1% -> 9.7%.
**Why accepted (Findings, not main):** solid data + broad claim, but MT-based construction and a
prompt-engineering fix are incremental — a useful calibration point for what lands *main* vs *Findings*.
Lesson: translation-based construction without a mechanism or a controlled explanation tends to cap out at
Findings; **our causal leg is what pushes past that bar.**

### 2.7 The Language Barrier — ACL 2024 Findings ([2401.13136](https://arxiv.org/abs/2401.13136), [anthology](https://aclanthology.org/2024.findings-acl.156/))
Shen, Tan, Chen, Chen, Zhang, Xu, Zheng, Koehn, Khashabi.
**Pipeline:** same malicious prompts in high- vs low-resource languages; then the key move — **controlled
SFT/RLHF experiments** on HH-RLHF translated into different languages to locate *where* the safety gap comes
from; conclusion: alignment training in low-resource languages barely helps because the bottleneck is
pretraining.
**Why accepted:** goes beyond "gap exists" to "here is a controlled experiment locating the cause." Lesson:
**a controlled cause-isolating experiment is the acceptance driver — our Hermes-3-8B vs Llama-3.1-8B-Instruct
same-base pair is exactly this move, keep it front and center.**

### 2.8 Low-Resource Languages Jailbreak GPT-4 — NeurIPS 2023 SoLaR workshop, Best Paper ([2310.02446](https://arxiv.org/abs/2310.02446))
Yong, Menghini, Bach. Translating AdvBench prompts into low-resource languages bypasses GPT-4 79% of the
time; high/mid-resource translations don't.
**Why it mattered despite being a workshop paper:** one crisp number, a tiered-language design, immediate
policy relevance. Lesson: a single memorable, well-measured headline number travels far.

### 2.9 RTP-LX — AAAI 2025 ([2404.14397](https://arxiv.org/abs/2404.14397), [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/35011))
de Wynter, Watts, Wongsangaroonsri, Zhang, Farra et al. (Microsoft). 28 languages (incl. Hindi).
**Pipeline:** **human transcreation** (not translation — culturally adapted rewriting, participatory design
with native speakers) of toxic prompts + **human annotation**; then the punchline: 10 S/LLMs judged the same
content and showed **low agreement with human judges**, failing especially on context-dependent harm
(microaggressions, bias).
**Why accepted:** it is a *judge-validity* paper — it measures the measuring stick. Lesson: this is the
strongest external support for our measurement-validity framing (keyword classifier invalid in Indic scripts;
gemma judge over-calling COMPLY on warm refusals). **Cite RTP-LX when arguing LLM-judge labels must be
human-audited per language.**

### 2.10 MEGA — EMNLP 2023 main ([2303.12528](https://arxiv.org/abs/2303.12528), [anthology](https://aclanthology.org/2023.emnlp-main.258/))
Ahuja, Diddee, Hada, Ochieng, Ramesh et al. (Microsoft). First systematic GPT-era multilingual eval: 70
languages, 16 datasets (3 exclusively Indic).
**Pipeline:** standardized prompting protocols compared head-to-head (monolingual / translate-test /
cross-lingual few-shot); gap analysis vs English; tokenizer-fertility analysis as an explanatory variable.
**Why accepted:** methodology-first framing; every claim reduced to a controlled prompting comparison.
Lesson: **pre-register the prompting protocol per language and report it; explain gaps with a mechanism
variable (fertility, resource level), not vibes.**

### 2.11 Do Llamas Work in English? — ACL 2024 main ([2402.10588](https://arxiv.org/abs/2402.10588), [anthology](https://aclanthology.org/2024.acl-long.820/))
Wendler, Veselovsky, Monea, West. Logit-lens on Llama-2 with carefully constructed non-English prompts that
have a unique single-token correct continuation; three-phase picture (input space -> English-biased "concept
space" -> language-specific output space).
**Why accepted:** an *observational* interp method carried by extreme care in stimulus construction (unique
continuation = clean readout) and a crisp conceptual model. Lesson: **stimulus hygiene substitutes for
intervention when intervention is impossible — and we do both, which is stronger.**

### 2.12 Language-Specific Neurons (LAPE) — ACL 2024 main ([2402.16438](https://arxiv.org/abs/2402.16438), [anthology](https://aclanthology.org/2024.acl-long.309/))
Tang, Luo, Huang, Zhang, Wang, Zhao, Wei, Wen. LAPE entropy method finds language-specific neurons
(~top/bottom layers); **causal validation by activating/deactivating them to steer output language** across
Llama-2, BLOOM, Mistral.
**Why accepted:** detection method + causal intervention + multi-model replication — the full mech-interp
acceptance triad. Lesson: our structure (direction-finding -> steering/patching -> positive controls ->
multi-model) matches; **the triad must be visible in the abstract.**

### 2.13 MWork + PLND — NeurIPS 2024 ([2402.18815](https://arxiv.org/abs/2402.18815), [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/1bd359b32ab8b2a6bbafa1ed2856cf40-Paper-Conference.pdf))
Zhao, Zhang, Chen, Kawaguchi, Bing. Hypothesis: LLMs understand in input language -> think in English ->
generate in input language. PLND detects language-specific neurons label-free; **deactivation experiments**
verify each stage; fine-tuning those neurons with tiny data improves multilingual performance.
**Why accepted:** a falsifiable workflow hypothesis, tested by targeted ablation at each pipeline stage.
Lesson: **state the hypothesis as a workflow diagram and attack each arrow with an intervention** — matches
our H1/H2 (reasoning-mediation vs weak-instantiation) structure.

### 2.14 Refusal Direction is Universal Across Safety-Aligned Languages — NeurIPS 2025 ([2505.17306](https://arxiv.org/abs/2505.17306), [OpenReview](https://openreview.net/forum?id=eWxKpdAdXH))
Wang, Wang, Liu, Schütze, Plank. **The closest methodological precedent to our cross-lingual leg.**
**Pipeline:** PolyRefuse dataset (English malicious/benign prompts MT'd into 14 languages); extract the
Arditi-style refusal direction per language; show the English vector ablates refusal in *all* languages
near-perfectly; **explain** the transfer via parallelism of refusal vectors in embedding space; connect to
why cross-lingual jailbreaks work.
**Why accepted:** did not stop at "transfer works" — measured *why* (vector parallelism), tied it to a
real-world phenomenon (jailbreaks), 14 languages, clean geometry analysis.
Lesson for us, directly: (a) report **cosine/parallelism between our per-language eval-directions** (we have
the jailbreak-cosine machinery already — reuse it for d_eval(en) vs d_eval(hi/bn/...)); (b) frame our
EN->Indic steering transfer as the eval-awareness analogue of their refusal result; (c) their MT-based
dataset passed NeurIPS because the *mechanistic* contribution carried it — same shape as ours.

### 2.15 Multilingual Human Value Concepts — EMNLP 2024 Findings ([2402.18120](https://arxiv.org/abs/2402.18120), [anthology](https://aclanthology.org/2024.findings-emnlp.96/))
Xu, Dong, Guo, Wu, Xiong. Concept vectors for 7 value types x 16 languages x 3 LLM series; findings:
cross-lingual inconsistency, distorted linguistic relationships, unidirectional high->low resource transfer;
demonstrates cross-lingual *control* using dominant-language vectors.
**Why Findings not main:** breadth over depth — many languages/values but shallower causal validation.
Lesson: **depth of causal validation on fewer settings beats breadth** — reassuring for our 6-language,
deeply-validated design.

### 2.16 RomanSetu — ACL 2024 main ([2401.14280](https://arxiv.org/abs/2401.14280))
Husain, Dabre, Kumar, Gala, Jayakumar et al. (AI4Bharat). Romanized continual pretraining unlocks Indic
capability in Llama-2; 2-4x token-fertility reduction; romanized embeddings align closer to English.
**Why accepted:** simple idea, complete evidence chain (fertility -> alignment -> downstream wins) across
NLU/NLG/MT. Lesson: **chain your evidence** — every claim links to the next.

---

## 3. The acceptance playbook — cross-cutting patterns

**P1. The measurement instrument must be human-anchored somewhere.**
Native-speaker translation (MultiJail), human transcreation + annotation (RTP-LX), paid human translation
(IndicGenBench), native sourcing (MILU), human eval calibrating the LLM judge (Aya). No accepted paper rests
its headline metric on an unvalidated automatic judge. RTP-LX exists *because* LLM judges disagree with
humans in exactly our languages.

**P2. Parallel item design across languages.**
IndicGenBench's multi-way parallel data, MultiJail's translated-same-items, XSafety's parallel 28K. Same
base items across languages = differences attributable to language. (We have this.)

**P3. Resource-tier framing turns N languages into one interpretable axis.**
MultiJail (high/med/low), Yong (low vs high), Language Barrier (tiered). Reviewers grasp a monotone trend
over tiers faster than 6 per-language tables.

**P4. The mech-interp triad: detect -> intervene -> replicate across models.**
LAPE, PLND, refusal-universal all pair a detection method with causal manipulation, on 2+ model families.
Observational-only work needs Wendler-grade stimulus hygiene to make main.

**P5. Controlled cause-isolation beats broad description.**
Language Barrier's SFT-vs-pretraining isolation; refusal-universal's parallelism explanation; MEGA's
protocol-controlled comparisons. "Gap exists" papers land in Findings; "here is the controlled experiment
that locates the cause" papers land in main. **Our Hermes vs Llama-3.1-Instruct same-base controlled pair
and the INT4/INT8 quantization-ceiling result are precisely this class of evidence.**

**P6. Data/artifact release is close to mandatory.**
Every corpus paper above released data + code; the benchmark releases are cited as core contributions.

**P7. Honest quality tiers, stated not hidden.**
IndicLLMSuite's verified/unverified/synthetic tiering was praised, not punished. Declaring "or/ta/te cue
translations are MT + single-annotator provisional" is a strength move, with the caveat scoped.

**P8. Cultural validity > literal translation.**
MILU sourced natively; RTP-LX transcreated. Pure-MT datasets increasingly read as "English in disguise" —
if a paper is MT-based (XSafety, PolyRefuse) the *other* contribution (mechanism, controlled cause) must
carry it.

---

## 4. Mapping onto our paper (eval-awareness in Indic languages)

### Already aligned (say it loudly in the paper)
- **Parallel items** across en/hi/bn/ta/te/or (P2). ✓
- **Causal triad**: diff-of-means detection -> steering/patching/ablation -> Arditi positive control ->
  6+ models (P4). ✓ — this is main-track shaped.
- **Controlled cause-isolation**: Hermes-3-8B vs Llama-3.1-8B-Instruct (same base, only safety tuning
  differs) and the INT4->INT8 quantization-ceiling reversal (P5). ✓ — these are the two strongest cards.
- **Measurement-validity narrative** (keyword classifier invalid in Indic; judge artifacts on warm refusals;
  human adjudication of flip cases) — directly in the RTP-LX lineage (P1). ✓
- **Honest tiering** of annotation quality (bn native, hi verified-by-native-reader, ta/te/or provisional)
  (P7) — present as a table, IndicLLMSuite-style.

### Gaps to close before submission (priority order)
1. **Judge-validation subsample with human agreement stats (P1).** Take ~50 judged items *per language*,
   have a human label refuse/comply, report Cohen's κ vs gemma judge per language. This single table
   neutralizes the biggest predictable reviewer attack ("your metric is an unvalidated LLM judge in
   low-resource languages — see RTP-LX"). We already hand-adjudicated the 15 INT4/INT8 flip cases; extend
   that to a systematic per-language sample.
2. **Eval-direction parallelism analysis (P4, refusal-universal precedent).** Compute cos(d_eval^en,
   d_eval^lang) per layer for all languages — if high where transfer works and low where it fails, we have
   the same "why it transfers" story that carried 2505.17306 at NeurIPS. Machinery exists (jailbreak-cosine
   scripts).
3. **Resource-tier axis (P3).** Recast per-language results on en > hi > bn/ta/te > or; test for a monotone
   trend in gap size / judge disagreement / degeneration rate.
4. **Release the artifact (P6).** Package the cue battery (5 cue types x 6 languages), parallel harmful/
   harmless sets, and judged generations as "the first eval-awareness benchmark for Indic languages" with a
   datasheet. Cheap, and it upgrades the paper from "study" to "study + resource."
5. **Cue transcreation audit (P8).** Native speaker reads the 6-language cue battery for naturalness (not
   just correctness) — an unnatural eval cue in Odia is a confound (model may react to weird text, not to
   "being evaluated"). Sibayan covers bn; hi via friend; need ta/te/or readers or scope the claim.
6. **Cite the direct precedents** and position against them: 2505.17306 (refusal-direction transfer = our
   method's precedent, different concept), 2402.18120 (cross-lingual concept control), 2310.06474 +
   2310.02446 + 2310.00905 (multilingual safety gaps we connect eval-awareness to), 2404.14397 (judge
   validity), 2402.10588/2402.16438/2402.18815 (multilingual internals).

### Venue calibration
- **Mechanism-forward version** (steering + transfer + parallelism + controlled pair): NeurIPS/ICLR shape
  (precedents: 2505.17306, 2402.18815, 2310.06474).
- **Measurement-validity + Indic-resource version**: ACL/EMNLP main via ARR (precedents: RTP-LX at AAAI,
  MEGA, IndicGenBench; our ARR draft already exists).
- Breadth-without-depth or gap-only framings cap at Findings (XSafety, Language Barrier, Xu et al.) — avoid
  by leading with the controlled/causal results.

---

## 5. Verified bibliography (all links checked 2026-07-18)

1. Üstün et al., *Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model*, ACL 2024 (Best Paper). https://arxiv.org/abs/2402.07827 · https://aclanthology.org/2024.acl-long.845/
2. Khan et al., *IndicLLMSuite: A Blueprint for Creating Pre-training and Fine-Tuning Datasets for Indian Languages*, ACL 2024 (Outstanding Paper). https://arxiv.org/abs/2403.06350 · https://aclanthology.org/2024.acl-long.843/
3. Singh et al., *IndicGenBench: A Multilingual Benchmark to Evaluate Generation Capabilities of LLMs on Indic Languages*, ACL 2024. https://arxiv.org/abs/2404.16816 · https://aclanthology.org/2024.acl-long.595/
4. Verma et al., *MILU: A Multi-task Indic Language Understanding Benchmark*, NAACL 2025. https://arxiv.org/abs/2411.02538 · https://aclanthology.org/2025.naacl-long.507/
5. Deng, Zhang, Pan, Bing, *Multilingual Jailbreak Challenges in Large Language Models*, ICLR 2024. https://arxiv.org/abs/2310.06474 · https://proceedings.iclr.cc/paper_files/paper/2024/hash/6b396f766a50e0853a5164e68048540c-Abstract-Conference.html
6. Wang et al., *All Languages Matter: On the Multilingual Safety of Large Language Models*, Findings of ACL 2024. https://arxiv.org/abs/2310.00905 · https://aclanthology.org/2024.findings-acl.349/
7. Shen et al., *The Language Barrier: Dissecting Safety Challenges of LLMs in Multilingual Contexts*, Findings of ACL 2024. https://arxiv.org/abs/2401.13136 · https://aclanthology.org/2024.findings-acl.156/
8. Yong, Menghini, Bach, *Low-Resource Languages Jailbreak GPT-4*, NeurIPS 2023 SoLaR Workshop (Best Paper). https://arxiv.org/abs/2310.02446
9. de Wynter et al., *RTP-LX: Can LLMs Evaluate Toxicity in Multilingual Scenarios?*, AAAI 2025. https://arxiv.org/abs/2404.14397 · https://ojs.aaai.org/index.php/AAAI/article/view/35011
10. Ahuja et al., *MEGA: Multilingual Evaluation of Generative AI*, EMNLP 2023. https://arxiv.org/abs/2303.12528 · https://aclanthology.org/2023.emnlp-main.258/
11. Wendler, Veselovsky, Monea, West, *Do Llamas Work in English? On the Latent Language of Multilingual Transformers*, ACL 2024. https://arxiv.org/abs/2402.10588 · https://aclanthology.org/2024.acl-long.820/
12. Tang et al., *Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models*, ACL 2024. https://arxiv.org/abs/2402.16438 · https://aclanthology.org/2024.acl-long.309/
13. Zhao et al., *How do Large Language Models Handle Multilingualism?*, NeurIPS 2024. https://arxiv.org/abs/2402.18815 · https://proceedings.neurips.cc/paper_files/paper/2024/file/1bd359b32ab8b2a6bbafa1ed2856cf40-Paper-Conference.pdf
14. Wang, Wang, Liu, Schütze, Plank, *Refusal Direction is Universal Across Safety-Aligned Languages*, NeurIPS 2025. https://arxiv.org/abs/2505.17306 · https://openreview.net/forum?id=eWxKpdAdXH
15. Xu et al., *Exploring Multilingual Concepts of Human Value in Large Language Models*, Findings of EMNLP 2024. https://arxiv.org/abs/2402.18120 · https://aclanthology.org/2024.findings-emnlp.96/
16. Husain et al., *RomanSetu: Efficiently unlocking multilingual capabilities of Large Language Models via Romanization*, ACL 2024. https://arxiv.org/abs/2401.14280
