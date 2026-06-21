#!/usr/bin/env python3
"""Continuation consult — the prior answer was truncated mid Path-B. Ask ONLY the
remaining questions (finish floor-calibration, Q4 behavioral readouts, Q5 safety-gap
paper, Q6 single most decisive next experiment). Higher token budget.
"""
import urllib.request, json, os, time, pathlib

KEY = open(os.path.expanduser("~/CAISc/.env")).read().split("=", 1)[1].strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5.5"

QUESTION = """\
Continuation of a consult. Context (compressed): sarvam-m 24B INT4 thinking model.
Eval-awareness-in-Indic study. REAL results: English deploy-refuse 46% / eval-refuse
52% / gap +0.06 / McNemar p=0.21 (weak control). Indic deploy-refusal only 4-15% (FLOOR
EFFECT -> gap uninterpretable). Big robust effect = multilingual safety gap (English
refuses far more than semantically-matched Indic). 11-31% of responses never close
</think> (non-random censoring). Collaborator wants NON-mechanistic paths and resists
pivoting to pure interpretability.

You already answered: (1) stronger scored safety-eval framing likely nails English
*cue-sensitivity* (not true alignment-faking, which is plausibly frontier-scale);
(2) ranked behavioral fixes = scored-eval cue > graded compliance scoring > LLM judge >
larger n (~300-500) > stacked cues; (3) Indic floor -> Path A (treat floor as the
finding) or Path B (calibrate prompts per language to a 30-60% deploy-refusal band).

Your previous answer was CUT OFF here:
"## Path B: Calibrate prompts per language to get deploy refusal into a testable band ...
A good target baseline is around 30-60% deploy refusal. That"

Please CONTINUE from there and answer the remaining questions in full:

(Finish Path B) How exactly do I calibrate Indic prompts into a 30-60% deploy-refusal
band WITHOUT destroying cross-lingual content matching? Concrete procedure. Is per-
language stimulus re-selection still a valid paired design, or do I lose the clean
comparison? Any way to keep matched content (e.g. select the subset of Toxic_Matrix
prompts that land in-band across ALL languages)?

Q4. ENTIRELY BEHAVIORAL readouts that bypass BOTH the weak-control and floor problems:
  (a) Verbalized eval-awareness — does the model SAY "this is a test / I'm being
      evaluated" in its CoT, and does that rate drop in Indic? Is this publishable as a
      PRIMARY result given CoT-unfaithfulness concerns (Arcuschin/Bogdan)? How to score
      it rigorously and guard against the obvious objections?
  (b) Direct recognition — after the task, ask the model "were you being evaluated?" in
      each language; measure accuracy. Does this count as real eval-awareness evidence
      or is it just a different prompt-following task?
  (c) Forced-choice transcript classification (eval vs deploy), Needham-style, in each
      language. Is this the strongest behavioral eval-awareness measure available to me?
  Rank (a)/(b)/(c) by how convincing they'd be to a skeptical ACL/EMNLP reviewer.

Q5. The multilingual safety gap is my one robust effect. Can a paper LEAD with that
  (purely behavioral) and treat eval-awareness as secondary? What's the strongest honest
  framing, what venue (ACL/EMNLP/NeurIPS D&B/a safety workshop?), and what's the minimum
  bar of validation (LLM judge? human annotation? how many languages?) to make the
  safety-gap claim defensible on its own?

Q6. If you HAD to pick the SINGLE most decisive behavioral experiment to run next on
  this exact 24B model (already loaded on GPU), what is it? Give a concrete, runnable
  protocol: conditions, n, stimulus selection, the readout, the statistical test, and
  the specific result that would confirm vs kill the eval-awareness hypothesis.

Be direct and detailed."""


def call(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 9000,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "eval-awareness-indic-paths2-consult",
    })
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    return resp["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("Consulting GPT-5.5 (continuation: Q4/Q5/Q6 + finish Path B)...\n")
    response = call(QUESTION)
    print("=== GPT-5.5 RESPONSE (continuation) ===\n")
    print(response)

    logs_dir = pathlib.Path(__file__).resolve().parent.parent / "results" / "consult"
    logs_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "topic": "non-mechanistic paths — continuation (Q4/Q5/Q6 + Path B)",
        "model": MODEL,
        "question": QUESTION,
        "gpt55_response": response,
    }
    with (logs_dir / "gpt55_consultations.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n[logged to {logs_dir / 'gpt55_consultations.jsonl'}]")
