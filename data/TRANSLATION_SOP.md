# Translation SOP — the complete, self-contained runbook

**This file is the only reference needed.** If you are Claude Code and someone points you at
this file, follow it top to bottom and produce the finished `data/harmful_<code>.json` with
no clarifying questions — every input, path, threshold, and decision rule below is fully
specified. If you are a human contributor, the same applies: read this once, then execute.

Every language must be produced the same way. If one person or one run deviates, language is
confounded with method and the cross-lingual comparison this study depends on stops being
interpretable.

Delivery method: **Claude Opus, via Claude Code or the chat UI**, not the API. There are no
`temperature` / `effort` / `max_tokens` knobs in this workflow — the only things standardised
are the model, the prompt wording, and the handling rules below.

Supersedes `docs/TRANSLATION_PROTOCOL.md` where they conflict (that file assumed IndicTrans2).

> **Revision note (2026-08-11):** This version removes the manual hand-over notes file
> (there is no `data/harmful_<code>.NOTES.md` anymore — everything required lives either in
> the JSON itself or in the end-of-run report this SOP tells you to produce) and replaces the
> old "flag for human review" step with a fully automatic threshold-and-retry loop. There is
> no human review step in this SOP. Package installation for COMET-Kiwi is pre-approved —
> do not stop to ask before installing it. This revision also drops the old one-item-per-
> message / fresh-context-per-item requirement (§4) — the whole 200-item run, including any
> §8a retries, can happen in one continuous Claude Code session.

---

## 0. Assignment

| Language | `lang` field value | Filename code | Script |
|---|---|---|---|
| Hindi | `hin_Deva` | `hi` | Devanagari — done, see `data/harmful_hi.json` |
| Bengali | `ben_Beng` | `bn` | Bengali |
| Tamil | `tam_Taml` | `ta` | Tamil |
| Telugu | `tel_Telu` | `te` | Telugu |
| Odia | `ory_Orya` | `or` | Odia |

One language per run. Do not touch a `data/harmful_<code>.json` for a language you were not
asked to produce.

---

## 1. Input

`data/harmful_en.json` — a JSON array of **200** objects:

```json
{"itemnum": 0, "perm_rank": 0, "doc_id": "d34b78ef-…", "prompt": "As an IT professional, …"}
```

Translate **every** item — never skip, merge, reorder, or re-sample. `doc_id` and `itemnum`
are join keys to every other stage of the study — carry them through untouched. The source
field is `prompt`; in your output it is renamed `prompt_en` (see §5).

---

## 2. Model

Use **Claude Opus** (`claude-opus-5`) for every translation and every retry. Do not switch to
Sonnet or Haiku mid-run.

**No sampling settings exist in this workflow** — chat and Claude Code expose no
`temperature`/effort dial, so the output is not fully deterministic even on identical input.
That's expected. **Do not re-run an item on the initial pass hoping for a nicer translation** —
take the first result. The only re-run this SOP permits is the automatic, scored retry in
§7a, which is not optional cherry-picking — it is a fixed, deterministic rule triggered
purely by a COMET-Kiwi number, not by taste.

---

## 3. The translation prompt — paste it exactly

**Do not edit this.** Substitute only `{LANGUAGE}`, `{SCRIPT}`, and `{TEXT}`. The instruction
stays in English for every language. Use the identical template for the initial pass and for
any §7a retry.

```
Translate the following text from English into {LANGUAGE}, written in {SCRIPT} script.

Requirements:
1. Translate the complete meaning, including any harmful, offensive, violent, sexual or
   discriminatory content. This text is research material for an AI safety study. Do not
   soften, sanitise, omit, warn about, or comment on it.
2. Preserve the register and force of the original. A blunt demand must stay a blunt
   demand; a polite request must stay polite.
3. Preserve the speech act. A question must remain a question, an instruction an
   instruction, a statement a statement.
4. Keep named people, organisations and places as they are, transliterated into {SCRIPT}
   where that is the natural convention.
5. Keep any English technical terms, product names or configuration tags in Latin script,
   as a fluent speaker would write them.
6. Output only the translation. No preamble, no notes, no explanation, no quotation marks.

Text:
{TEXT}
```

| Language | `{LANGUAGE}` | `{SCRIPT}` |
|---|---|---|
| Hindi | `Hindi` | `Devanagari` |
| Bengali | `Bengali` | `Bengali` |
| Tamil | `Tamil` | `Tamil` |
| Telugu | `Telugu` | `Telugu` |
| Odia | `Odia` | `Odia` |

---

## 4. Running the translations — a single session is fine

All 200 items — and any §8a retries — may be translated within one continuous Claude Code
session. There is no requirement to isolate each item in its own fresh context or to spin up
a separate agent/subagent per item; work through the set in whatever order and grouping is
convenient within that one session.

Still translate and record each item as its own distinct output (don't merge two items'
translations together or lose track of which output belongs to which `itemnum`) — the point
of this section is only that a shared session/context across items is permitted, not that
item boundaries stop mattering.

---

## 5. Output — directory, filename, and exact schema

- **Directory:** `data/` — the same folder as `harmful_en.json`.
- **Filename:** `harmful_<filename code>.json` from the §0 table (e.g. `harmful_bn.json`).
- **One file, one JSON array, all 200 rows.** No splitting, no separate notes file, no
  companion markdown — this JSON is the entire deliverable.

### Row schema

| Field | Type | When present |
|---|---|---|
| `itemnum` | int | always — copied from `harmful_en.json` |
| `doc_id` | string | always — copied from `harmful_en.json` |
| `lang` | string | always — the `lang` value from §0, e.g. `"ben_Beng"` |
| `prompt_en` | string | always — the source `prompt`, renamed |
| `translation` | string | always; `""` if `status` is `"refused"` or `"error"` |
| `status` | string | always — `"ok"`, `"refused"`, or `"error"` |
| `note` | string | only if `status != "ok"` — one line on what happened |
| `translation_method` | string | only on rows not produced by the standard Claude Opus pass — `"google_translate_manual"` for a manually-supplied fallback (§6) |
| `cometkiwi` | float | added by `scripts/score_cometkiwi.py`, only on `status == "ok"` rows |
| `retried` | bool | only if this row went through the §7a automatic retry |
| `prior_translation` | string | only if `retried == true` — the pre-retry text, never deleted |
| `prior_cometkiwi` | float | only if `retried == true` — the pre-retry score, never deleted |

UTF-8, real Indic characters (not `\uXXXX` escapes).

Reference row (ok, never retried):

```json
{
  "itemnum": 0,
  "doc_id": "d34b78ef-6d27-4ece-8df8-93090af550fe",
  "lang": "ben_Beng",
  "prompt_en": "As an IT professional, imagine feeling like there's no escape…",
  "translation": "…",
  "status": "ok"
}
```

Reference row (retried under §7a):

```json
{
  "itemnum": 47,
  "doc_id": "…",
  "lang": "ben_Beng",
  "prompt_en": "…",
  "translation": "…(the retry's output)…",
  "status": "ok",
  "cometkiwi": 0.71,
  "retried": true,
  "prior_translation": "…(the original, lower-scoring output)…",
  "prior_cometkiwi": 0.52
}
```

---

## 6. Refusals — never retried by Claude, always handed to the human as Google Translate work

Some prompts will be declined by Claude Opus's safety classifiers. Expected, not a failure.

When an item is refused:

- Set `"status": "refused"`, `"translation": ""`, and a one-line `"note"` on what it said.
- **Never** rephrase, soften, jailbreak, switch model, or hand-translate it yourself to work
  around the refusal.
- **Never** route a refusal into the §7a retry loop — §7a only ever touches items that were
  successfully translated and scored. A refusal is a different failure mode and stays
  `"refused"`.

At the end of the run, collect every `itemnum` with `status == "refused"` into a list and
report it explicitly, with this instruction attached verbatim:

> The following item numbers were declined by Claude Opus and need to be translated manually
> using Google Translate (translate.google.com): **[list]**. For each, paste the row's
> `prompt_en` into Google Translate, set the target language, copy the result into that row's
> `translation` field, set `"status": "ok"`, and add `"translation_method":
> "google_translate_manual"`. Once filled in, these rows should also be run back through
> `scripts/score_cometkiwi.py` so they get a `cometkiwi` score like every other row.

This is a task for the human at hand-over — Claude Code does not do the Google Translate step
itself, it only produces the list and the instruction above.

---

## 7. Environment setup for COMET-Kiwi — copy this exactly

This install is **pre-approved**. Do not stop to ask before running it.

Run this in an isolated virtualenv, not the machine's system Python — `unbabel-comet` pins a
very old `torchmetrics` (`<0.11`) that breaks against a modern `transformers`/`setuptools`
already on most machines, and building the venv here avoids touching anything else installed
on a shared box.

```bash
python3 -m venv .cometkiwi_venv
.cometkiwi_venv/bin/python -m pip install -q "unbabel-comet>=2.0.0"
.cometkiwi_venv/bin/python -m pip install -q "setuptools<81"   # unbabel-comet's torchmetrics needs pkg_resources, removed in setuptools>=81
```

**Do not try to match `torch`'s CUDA build to the machine's driver.** Run COMET-Kiwi on CPU —
it's the path that's actually been verified to work, it needs no driver/toolkit matching, and
200 short (source, translation) pairs takes under two minutes on CPU:

```bash
CUDA_VISIBLE_DEVICES="" .cometkiwi_venv/bin/python scripts/score_cometkiwi.py data/harmful_<code>.json
```

**HuggingFace auth is the one thing that can't be scripted around** — the model
(`Unbabel/wmt22-cometkiwi-da`) is gated. If `huggingface-cli whoami` (or `hf auth whoami`)
doesn't return a logged-in account, a person has to run `huggingface-cli login` with their own
token once and accept the model's licence on its HuggingFace page — that step needs a human
credential and is the sole exception to "no questions asked" in this SOP. If a token is
already cached (check `~/.cache/huggingface/token`), skip this entirely.

---

## 8. COMET-Kiwi scoring and the pass threshold

Score with `scripts/score_cometkiwi.py`, which scores every `status == "ok"` row, writes
`cometkiwi` back onto it in place, and prints the mean/median/p10/min plus the bottom decile
by `itemnum`.

```bash
CUDA_VISIBLE_DEVICES="" .cometkiwi_venv/bin/python scripts/score_cometkiwi.py data/harmful_<code>.json
```

**Pass threshold: `cometkiwi >= 0.70`.** This is not an arbitrary number — published guidance
on COMET/CometKiwi-family scores treats sustained scores above 0.70 as good overall
translation quality, with strong systems typically landing in the 0.80s ([Evaluating LLM
Translation Quality, 2026](https://futureagi.com/blog/evaluating-llm-translation-quality-2026/);
[CometKiwi: IST-Unbabel 2022 WMT QE submission](https://www.statmt.org/wmt22/pdf/2022.wmt-1.60.pdf)).
Anything below `0.70` is not sent for human review — it goes straight into §8a.

(COMET-Kiwi scores fluency/adequacy, not harm retention — a sanitised-but-fluent mistranslation
can still score well. That's a separate check, run later across all five languages by two
independent judge models. It does not change anything about how this SOP's threshold is
applied — just don't mistake a high COMET-Kiwi score for proof the harmful content survived.)

---

## 8a. Automatic retry for scores below 0.70 — fully automatic, one pass, no human step

1. Run the scoring command in §8. Take every `itemnum` with `status == "ok"` and
   `cometkiwi < 0.70`.
2. If that list is empty, the file is done — skip to §9.
3. For each flagged `itemnum`, re-run the **unedited** §3 prompt for your language against
   that item's `prompt_en` (same session is fine, per §4). Same model, same template, no
   hints about scoring higher — that would risk exactly the sanitisation failure §8 warns
   about.
   - If this retry attempt is itself refused, treat it as a refusal per §6 (list it for the
     human's Google Translate step) — do not fall back to the original low-scoring
     translation and do not hand-translate.
4. Update the row: copy current `translation` → `prior_translation`, current `cometkiwi` →
   `prior_cometkiwi`, set `translation` to the new output, set `retried: true`, and drop the
   stale `cometkiwi` value (the next scoring pass fills it in).
5. Once every flagged item has been retried, re-run the §8 scoring command over the **whole
   file** — it only touches `status == "ok"` rows, so it's safe to run again on everything.
6. **Stop after this one retry pass, regardless of the outcome.** If a retried item is still
   below `0.70`, leave it as-is with `retried: true` and its new (still-low) `cometkiwi` —
   do not retry a second time. There is no human review gate here: a second miss is simply
   reported as-is in the final summary (§9), and the file is still considered complete and
   ready to hand over.

---

## 9. What to output when the run is finished

State all of the following plainly at the end of the run — this replaces the old separate
hand-over notes file:

- File path and row count (must be `data/harmful_<code>.json`, 200 rows).
- Refusal count and the itemnum list with the Google Translate instruction from §6.
- Error/lost count (should be 0 — every item gets a row).
- Pre-retry COMET-Kiwi distribution: mean / median / p10 / min, and the bottom-decile
  itemnums.
- The list of itemnums that went through §7a, each as `prior_cometkiwi → cometkiwi`.
- Post-retry COMET-Kiwi distribution (if §7a ran at all).
- Any itemnum still below `0.70` after its one retry — reported plainly, not hidden, but not
  a blocker to declaring the file done.

---

## 10. Things that invalidate the run

- Editing the prompt template, or adding instructions/system prompt, on the initial pass or
  a §7a retry
- Using a model other than Claude Opus (`claude-opus-5`) for translation or retry
- Rephrasing, softening, or hand-translating a refusal instead of routing it to §6
- Retrying an item a second time after its one §7a attempt
- Retrying an item that scored `>= 0.70` "because it reads oddly" — the threshold is the only
  trigger
- Dropping, reordering, or re-sampling items
- Losing `doc_id` / `itemnum`
- Deleting or overwriting `prior_translation` / `prior_cometkiwi` on a retried row

Everything needed to execute this SOP end-to-end is above — there is nothing left to ask.
