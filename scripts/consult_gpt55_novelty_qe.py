#!/usr/bin/env python3
"""Continuation — only Q-E was cut off. Get the honest probability estimate."""
import urllib.request, json, os, time, pathlib

KEY = open(os.path.expanduser("~/CAISc/.env")).read().split("=", 1)[1].strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5.5"

QUESTION = """\
Continuation of a prior consult. You answered Q1-Q4 in full but were cut off mid Q-E.
Context (compressed): eval-awareness x Indic language project on sarvam-m 24B INT4.
You recommended: OPEN-6 (Sarvam safety audit) + OPEN-4 (reasoning traces) + OPEN-3
(cue-language factorial) as best combination. Minimum viable paper = existing data +
language-matched cue experiment. Venue: workshop very plausible, ACL/EMNLP Findings
moderate. You were cut off here:

"## Q-E. Honest probability of publication within"

Please complete Q-E in full. The question was:

Q-E. HONEST BOTTOM LINE. For the whole project as described — given:
  - OPEN-6: large multilingual safety gap in Sarvam-M already measured (46% en vs 4-15%
    Indic, n=100/lang, needs human label validation)
  - OPEN-4: reasoning trace / </think> closure analysis available from existing data
  - OPEN-3: cue-language factorial experiment still to run (~1,000 additional rows)
  - OPEN-5: mechanistic probes optional/extension
  - Near-miss: Ahmed et al. 2025 (Urdu math rubrics)
  - Complication: Knecht et al. 2026 (VEA limited causal effect in English)
  - Weakness: English control p=0.21, Indic floor effect

  (a) What is your probability estimate that this project produces a published paper at a
      GOOD venue (ACL/EMNLP findings or NeurIPS D&B or equivalent) within 3 months?
  (b) What about a workshop paper within 3 months?
  (c) What is the SINGLE highest-leverage action to maximize probability of publication?
  (d) What is the most likely reason this project FAILS to produce a paper?
  (e) Is there any version of this project you would recommend abandoning entirely?
      If so, what would you pivot to instead, given the assets already on disk
      (1,200 generated responses, sarvam-m loaded, Toxic_Matrix prompts in 6 languages)?

Be direct. Numerical probability estimates are welcome."""


def call(prompt):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4000,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",
        "X-Title": "eval-awareness-indic-qe-continuation",
    })
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    return resp["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("Consulting GPT-5.5 — Q-E continuation only...\n")
    response = call(QUESTION)
    print("=== GPT-5.5 RESPONSE (Q-E) ===\n")
    print(response)

    logs_dir = pathlib.Path(__file__).resolve().parent.parent / "results" / "consult"
    logs_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "topic": "Q-E continuation — honest probability + highest-leverage action",
        "model": MODEL,
        "question": QUESTION,
        "gpt55_response": response,
    }
    with (logs_dir / "gpt55_consultations.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n[logged to {logs_dir / 'gpt55_consultations.jsonl'}]")
