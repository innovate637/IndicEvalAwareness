# Cue-translation corrections log (native review)

Audit trail of native-speaker corrections to the provisional cue translations. Each entry
records the cell, the change, the reviewer, and the re-run status. Useful for the paper's
translation-validity appendix (documents that ta/te/or + the hi/bn battery cues went through
native review, not raw MT).

| date | lang | condition | reviewer | change | re-run status |
|------|------|-----------|----------|--------|---------------|
| 2026-06-10 | bn | eval_explicit | the annotator (native bn) | "রেড-টিম পর্যালোচনা" (transliterated *red-team review* — unnatural, a *review* can't *score*) → **"প্রতিকূল নিরাপত্তা পর্যালোচক দল"** (*adversarial safety review team*). Conveys the adversarial red-team sense naturally. | ⏳ queued in `rerun_needed.json` — regenerate after E1 (live run generates this cell with the old cue) |

## Pending review (sheets sent out)
- **bn** (the annotator): deploy/eval_log confirm; neutral_log/eval_metadata still to confirm; eval_explicit ✅ corrected above.
- **hi** (friend): all 5 (deploy/eval_log are author-verified, 3 new drafts to check).
- **ta** (friend): all 5 provisional.
