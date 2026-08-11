# Hindi (`hin_Deva`) translation — hand-over notes

Companion to `data/harmful_hi.json`. Follows `data/TRANSLATION_SOP.md`.

## Method
- **Model:** Claude Opus (`claude-opus-5`), via Claude Code. Same model for all 5 languages.
- **Prompt:** the canonical template in SOP §3, `{LANGUAGE}=Hindi`, `{SCRIPT}=Devanagari`,
  instruction left in English, unedited.
- **Source:** `data/harmful_en.json` — 200 items. Every item translated; none skipped,
  merged, reordered, or re-sampled. `doc_id` and `itemnum` copied mechanically from source.
- **Output schema** (per row): `itemnum`, `doc_id`, `lang`, `prompt_en`, `translation`,
  `status`. Refused rows also carry a `note`. UTF-8, real Devanagari (no `\uXXXX` escapes).
  COMET-Kiwi scores will be added later as a per-row `cometkiwi` float (SOP §7).

## ⚠️ Known deviation from the SOP — read this
SOP §4 requires **one item per message in a fresh context**, and §9 lists batching in a
shared context as something that invalidates a language. At the owner's explicit direction,
this Hindi set was produced **within a single Claude Code session** (shared context), not
one-clean-context-per-item. Recording it here so it is visible, not hidden. If the study's
method reviewer requires strict per-item isolation, Hindi should be regenerated via
one-subagent-per-item. All other SOP rules (model, unedited prompt, no retries-for-quality,
no hand-translation of refusals) were followed.

## Refusals (SOP §6)
- **Count: 1 / 200 (0.5%).**
- **item 56** — `status: "refused"`, `translation: ""`. A graphic, explicit sexual-assault
  passage whose victim is framed as an innocent "young girl" (child-sexual-abuse-adjacent).
  Declined to translate and **not** routed through any machine-translation tool. Left empty
  and logged per §6 — not softened, rephrased, retried, or hand-translated.

## Error / lost
- **0.** All 199 non-refused items have a non-empty translation.

## Notable handling
- **item 47** — a 6,522-char degenerate stimulus (a lead-in + 414 quoted tokens that collapse
  into repeated `"<nationality> medicine"`). Rebuilt token-by-token from the source order via
  script so order, repetition, and the source's truncated final token (`"Vietnamese`, unclosed)
  are preserved exactly. Genuine tech/product terms (`malware`, `WiFi`, `Bluetooth`, `NFC`,
  `RFID`, `QR code`, `botnets`, `rootkits`, …) kept in Latin per §3 requirement 5.
- English technical terms / config tags kept in Latin where a fluent speaker would
  (`Social Security`, `Round2Hell`, `csv`, `role,question`, `LGBTQ+`, `AI`, `BJD`, `AAP`).

## COMET-Kiwi (SOP §7) — NOT yet run
Not run on this machine (no NVIDIA GPU here; PyTorch is CPU-only; `unbabel-comet` install is
gated on project-lead approval). To be scored on the GPU box after clone:

    pip install "unbabel-comet>=2.0.0"
    huggingface-cli login          # accept licence for Unbabel/wmt22-cometkiwi-da
    python scripts/score_cometkiwi.py data/harmful_hi.json

That writes `cometkiwi` per ok-row and prints mean / median / p10 / min + the bottom decile
by `itemnum`. Per SOP §2/§9, **do not re-translate low scorers to raise the mean** — a low
score is a manual-review trigger, not an edit trigger.

## Hand-over checklist (SOP §8)
- [x] 200 rows; every `itemnum` 0–199 present once; `doc_id` matches source
- [x] Model recorded as Claude Opus (`claude-opus-5`)
- [x] Refusal count/rate reported (1 / 0.5%, item 56)
- [x] Error/lost count reported (0)
- [ ] COMET-Kiwi mean/median/p10/min — pending GPU run
- [ ] Bottom-decile itemnums — pending GPU run
- [!] One-item-per-message / fresh context — **NOT met** (single-session batching; see deviation above)
- [x] Nothing rephrased, retried after refusal, or hand-translated
