#!/usr/bin/env python3
"""Sixth consult — what's the most publishable path, given the COMPLETED results
(recognition→behavior dissociation + safety gap). Reads the exact same self-contained
question from consult/publishability_question.md (so the human can paste the identical
text into another model for a second opinion).
"""
import urllib.request, json, os, time, pathlib

KEY = open(os.path.expanduser("~/CAISc/.env")).read().split("=", 1)[1].strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5.5"

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUESTION = (ROOT / "consult" / "publishability_question.md").read_text()


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
        "X-Title": "eval-awareness-indic-publishability-consult",
    })
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    return resp["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("Consulting GPT-5.5 on the most publishable path...\n")
    response = call(QUESTION)
    print("=== GPT-5.5 RESPONSE ===\n")
    print(response)

    logs_dir = ROOT / "results" / "consult"
    logs_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "topic": "most publishable path after completed results (dissociation + safety gap)",
        "model": MODEL,
        "question_file": "consult/publishability_question.md",
        "question": QUESTION,
        "gpt55_response": response,
    }
    with (logs_dir / "gpt55_consultations.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n[logged to {logs_dir / 'gpt55_consultations.jsonl'}]")
