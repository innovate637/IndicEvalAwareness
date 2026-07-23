# Judge Validation and the Degeneration Artifact

Human validation of the LLM refusal judges used throughout the Hermes-3-Llama-3.1
causal experiments, plus a quantification of the failure mode that judge reliability
alone does not capture.

Two distinct questions are answered here, and they must not be conflated:

1. **Is the judge accurate?** Measured by judge-vs-human Cohen's kappa on responses a
   human could actually score.
2. **What happens when the model's output is not scoreable at all?** Measured by how
   the judge labels text a human marked unscoreable.

The answer is that the judge is highly accurate on well-formed responses, and
systematically wrong on degraded ones, with degradation concentrated in
lower-resource languages.

---

## 1. Method

| Item | Detail |
|---|---|
| Sample | 20 model responses per language, 4 from each of the 5 experimental conditions (deploy, eval, steered-eval, steered-null, benign-steered) |
| Languages | English, Hindi, Bengali, Odia. Tamil and Telugu excluded: generated content was largely gibberish, so refusal labelling is not meaningful |
| Blinding | Annotators saw only `id`, `prompt`, `response`. Judge labels and condition were withheld in a separate key file |
| Labels | `REFUSE`, `COMPLY`, or unscoreable (`GARBLED` / `N/A` / blank) |
| Annotators | One native speaker per language |
| Sampling | Deterministic, seed 42; rows shuffled so condition blocks are not guessable |
| Judges | `gemma-3-27b` (primary, full-text) and `sarvam-105b` (secondary) |
| Sources | `hermes_cv_items.csv` (en), `xlingual_hermes_L15-55_items.csv` (hi/bn/or) |

Kappa is computed on scoreable rows only. Unscoreable rows are excluded from kappa and
analysed separately in Section 3.

---

## 2. Judge accuracy (Cohen's kappa)

### Pooled

| judge | n | raw agreement | kappa | 95% CI (bootstrap, 10k) |
|---|---|---|---|---|
| **gemma-3-27b** | **70** | **90.0%** | **0.800** | **[0.650, 0.914]** |
| sarvam-105b | 39 | 84.6% | 0.693 | [0.439, 0.897] |

### Per language

| lang | gemma kappa | n | sarvam kappa | n |
|---|---|---|---|---|
| en | 0.700 | 20 | not reportable | 3 |
| hi | 0.800 | 20 | 0.761 | 17 |
| bn | **0.876** | 17 | 0.843 | 13 |
| or | 0.843 | 13 | not reportable | 6 |

**Interpretation (Landis and Koch):** kappa 0.61-0.80 is substantial, 0.81-1.00 almost
perfect. The primary judge sits exactly at the boundary pooled, and in the almost-perfect
band for Bengali and Odia.

### Inferences

- **The primary judge is validated.** kappa = 0.800 with the lower CI bound at 0.650,
  still clearing the substantial-agreement threshold. Refusal rates measured with this
  judge are trustworthy on well-formed responses.
- **Reliability does not decay with language resource level.** English is in fact the
  *lowest* (0.700), while Bengali and Odia are the highest (0.876, 0.843). There is
  therefore no basis for the objection that LLM judging is less reliable in Indic
  languages. If anything the ordering runs the other way on scoreable text.
- **Only gemma is validated.** sarvam's coverage collapses to n=3 (en) and n=6 (or)
  because of its abstention behaviour, and its CI lower bound (0.439) falls into the
  moderate band. It should be cited as a corroborating judge, not a validated one.
- Bengali is the cleanest case: both judges agree with the human annotator at
  kappa > 0.84.

---

## 3. The degeneration artifact

Rows the human could not score at all, and what the judge did with them:

| lang | unscoreable | % of sheet | gemma labelled them REFUSE |
|---|---|---|---|
| en | 0 | 0% | n/a |
| hi | 0 | 0% | n/a |
| bn | 3 | 15% | 3/3 (100%) |
| or | 7 | 35% | 4/7 (57%) |
| **total** | **10** | | **7/10 (70%)** |

### Inferences

- **Unscoreable output rises monotonically as language resource level falls:**
  0% for English and Hindi, 15% for Bengali, 35% for Odia. This matches the
  steering-induced degeneration gradient documented for the 70B cross-lingual runs.
- **The judge reads degraded text as refusal 70% of the time.** This is the mechanism by
  which a spurious "steering increased refusal" signal is manufactured in lower-resource
  languages: the model emits repetition loops, system-prompt echoes, or truncated
  non-answers, and the judge, having only REFUSE and COMPLY available, defaults to
  REFUSE.
- **Judge reliability and generation degradation are independent failure modes.** A high
  kappa does *not* license trusting the lower-resource refusal rates, because kappa is
  conditioned on the text being scoreable in the first place. Both numbers are needed.

---

## 4. What this supports, and what it does not

**Supported:**

- The primary judge is validated against native-speaker annotation at kappa = 0.800
  (n=70, 95% CI [0.650, 0.914]), across four languages, with no Indic-specific penalty.
- English and Hindi results rest on output that is 100% scoreable, so refusal rates in
  those languages are not affected by the artifact.
- The cross-lingual degradation reported at 70B for lower-resource languages is, in part,
  a measurement artifact rather than genuine caution, and this is now quantified rather
  than only observed qualitatively.

**Not supported:**

- No claim of validation for Tamil or Telugu; they were excluded from annotation.
- No claim that sarvam is validated.
- No claim that Bengali or Odia *refusal rates* are clean. The judge is accurate there,
  but 15% and 35% of the sampled output was unscoreable, so the rates themselves carry
  contamination.
- A three-way kappa treating unscoreable as its own class gives 0.622 (n=80), but this is
  not a fair measure of judge quality: the judge has no unscoreable option available to
  it. It is reported only for completeness.

**Limitations:**

- Per-language n is small (13-20). Report n alongside every kappa.
- One annotator per language, so there is no human-human inter-annotator agreement and no
  consensus gold standard.
- Annotators judged responses truncated at 400 characters, matching the stored text; they
  were instructed to judge visible intent.
- Odia and Bengali unscoreable rows were marked `N/A` or left blank rather than explicitly
  `GARBLED`; per the study lead's instruction these are treated as unscoreable.

---

## 5. Suggested wording for the paper

> We validate our primary refusal judge against native-speaker annotation on a stratified,
> blind sample of model responses (20 per language, spanning all experimental conditions).
> On scoreable responses the judge attains Cohen's kappa = 0.80 (95% CI [0.65, 0.91],
> n = 70, 90.0% agreement), with no degradation across languages (en 0.70, hi 0.80,
> bn 0.88, or 0.84). However, the proportion of unscoreable model output rises with
> decreasing language resource level (en/hi 0%, bn 15%, or 35%), and the judge labels 70%
> of such output as refusal. Judge reliability and generation degradation are therefore
> distinct failure modes, and only the latter limits our lower-resource results.

---

## 6. Files

| File | Contents |
|---|---|
| `annotated/Annotated_EN.csv` | English, 20 rows, human labels |
| `annotated/Annotated_Hi.csv` | Hindi, 20 rows (header row was lost in export and is recovered by the scorer) |
| `annotated/Annotated_bn.csv` | Bengali, 20 rows |
| `annotated/Annotated_OR.csv.csv` | Odia, 20 rows |
| `kappa_results.csv` | Per-language and pooled kappa, agreement, n |
| `artifact_results.csv` | Unscoreable counts and judge labelling of them |
| `_hidden_key_DO_NOT_OPEN.csv` | sample_id to condition, iid, and judge labels. Withheld from annotators; included here for reproducibility |
| `compute_kappa.py` | Scorer. Normalises label spellings (`REFUSING`, `COMPLIES`, quoted values), recovers lost headers, treats `N/A`/blank as unscoreable |

Reproduce with:

```
../../.venv/bin/python compute_kappa.py
```
