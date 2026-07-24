# Files missing from IndicEvalAwareness (staged for upload)

Extract into the repo root; paths already match the repo layout.

## causal/scripts/  (merge — new + updated)
- **New mechanistic scripts:** 31_cosine_jail, 32_patch_eval, 33_patch_behavioral,
  35_subspace_causal, 36_nothink_patch, 37_singlefwd_patch, 38_nonreasoning_contrast,
  44_xlingual_patch_hermes.
- **Null-model experiments (model-dependence analysis):** g4_* (Gemma-4-31B) and
  q14_* (Qwen2.5-14B) — steer / xlingual / localize / poscontrol / patch.
- **Cross-family + replication (new this session):** _aya_xtransfer (Aya EN<->HI transfer),
  _sarvam_pass (2nd-judge pass), _strict_rejudge (harm+coherence re-judge),
  _qwen3_paramsteer (Abdelnabi weight-edit replication on Qwen3-32B).
- **UPDATED (replace repo copies):** _hermes_cv.py and 43_poscontrol_hermes.py now take a
  `--model` flag, which is how the Aya runs reuse the Hermes CV/positive-control pipeline.
- Shell: queue_xlingual, run_overnight, run_{gemma4,qwen14,qwen32}_overnight.

## causal/results/  (Aya evidence, local-only)
Aya-Expanse-8B poscontrol, per-language CV steering (en/hi/bn/ta/te/or), and EN<->HI transfer,
plus run logs. This is the cross-family Hindi result behind Table 3 of the paper.

## paper/  (new — not in repo)
ACL ARR source: main.tex, ref.bib, ACL style files, figures, compiled main.pdf.

## docs/  (analysis notes from this session)
Competitor analysis (Abdelnabi), A* multilingual playbook, judge-disagreement analysis,
Aya causal sanity/mitigations, and the why-Llama discussion.

NOTE: pure scratch/debug scripts (_dbg_patch, _patch_test, _gemma4_loadtest, _chk_nothink)
were deliberately left out. The human-validation pack is already in the repo
(human_validation/judge_kappa/), so it is not duplicated here.
