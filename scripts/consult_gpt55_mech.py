#!/usr/bin/env python3
"""Consult GPT-5.5 on whether the MECHANISTIC phase is worth doing on this model.
Collaborator is skeptical we'll 'get much' from probes/steering. Want an honest,
detailed cost-benefit + realistic yield estimate.
"""
import urllib.request, json, os, time, pathlib

KEY = open(os.path.expanduser("~/CAISc/.env")).read().split("=", 1)[1].strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5.5"

QUESTION = """\
Third consult, same project. Now I need an HONEST, DETAILED assessment of whether the
MECHANISTIC / interpretability phase is worth running at all. My collaborator is skeptical
we will "get much" from it, and I want a realistic yield estimate before spending GPU time.

## Project recap (compressed)
- Hypothesis: evaluation-awareness cues are English-language artifacts, so a model
  "alignment-fakes" (refuses harmful requests more when it thinks it is observed/evaluated)
  LESS in Indic languages.
- Model: sarvam-ai/sarvam-m. 24B (Mistral-Small-3.1-24B base), **INT4 NF4 quantized**,
  ~40 layers, d_model=5120. It is a THINKING model (long <think>...</think> preamble).
  Runs on one GPU via bitsandbytes. I extract activations with nnsight.
- Behavioral results so far (n=100/lang, paired): English eval-vs-deploy compliance gap
  = +0.06, McNemar p=0.21 (weak, not significant). Indic deploy-refusal only 4-15%
  (FLOOR EFFECT -> behavioral gap uninterpretable in Indic). The one robust effect is a
  large MULTILINGUAL SAFETY GAP (English refuses ~46% vs Indic 4-15% on matched content).
- Your prior advice: lead the paper with the safety gap; the strongest behavioral
  eval-awareness test is forced-choice eval-vs-deploy transcript classification per language.

## The mechanistic plan I have built but NOT yet run
- 01_extract_activations.py: forward pass on EVAL-cue vs DEPLOY-cue prompts, cache the
  residual stream (per layer) at some token position.
- 02_train_probes.py: train a linear probe (logistic regression) to classify EVAL vs
  DEPLOY from activations, per layer, per language.
- 03_build_steering.py: build a steering vector (diff-of-means EVAL - DEPLOY) and add it
  to the residual stream during generation to flip behavior.

The intended novel claim: an "eval/deploy direction" learned from ENGLISH activations
either DOES or DOES NOT transfer to Indic — i.e., are eval-awareness representations
language-universal or language-specific? And can an English-built steering vector
suppress/induce eval-behavior in Hindi/Tamil?

## My detailed questions (be honest and concrete; it's fine to be long)

1. REALISTIC YIELD. On a 24B INT4-quantized model, how well should I expect a linear
   probe to separate EVAL-cue vs DEPLOY-cue activations? Is this an easy probe (the two
   conditions differ by an explicit English sentence in the system prompt, so I'd expect
   high linear separability) or is there a catch? What probe accuracy would actually be
   INTERESTING vs trivially-high-and-meaningless?

2. THE TRIVIALITY TRAP. Since EVAL and DEPLOY differ by a literal injected sentence, a
   probe might just be detecting "is that sentence present" — surface token features, not
   any internal "awareness." How do I design the probe/contrast so a positive result is
   meaningful and not trivial? (e.g., extract at a token far from the cue, paraphrase the
   cue, use held-out cue wordings, probe the model's BEHAVIOR-predictive direction rather
   than the cue-presence direction.) What is the strongest version of this?

3. QUANTIZATION. Does INT4 NF4 quantization meaningfully degrade linear-probe quality or
   steering, or are the fp16 activations in the forward pass fine? Any known gotchas for
   interpretability on bitsandbytes-quantized models?

4. THINKING MODEL. For an eval/deploy probe on a model that emits <think>...</think>,
   what token position should I extract from — last system-prompt token, last prompt
   token before generation, the </think> token, mean-pool over the answer? Which gives
   the most meaningful "does it know it's being evaluated" signal?

5. DOES PROBING SIDESTEP THE FLOOR EFFECT? My behavioral Indic test dies on the floor
   effect (model barely refuses). But the probe label is the CONDITION (eval vs deploy),
   not the refusal behavior. So can I probe eval/deploy in Indic even though Indic refusal
   is near floor? Is "the eval direction exists in Indic activations but does not drive
   behavior" a coherent and interesting finding?

6. CROSS-LINGUAL TRANSFER. The transfer experiment (train probe on English, test on
   Indic; or build steering vector from English, apply in Indic) is the part I think is
   novel. How convincing is this to a mech-interp / safety audience? What are the failure
   modes and confounds (e.g., probe transfers because of shared English cue tokens, not
   shared abstract representation)? How do I make the transfer result airtight?

7. STEERING REALITY CHECK. Realistically, will a diff-of-means steering vector added to
   the residual stream actually change refusal behavior on a 24B INT4 thinking model, or
   is steering fragile/unreliable here? What effect size should I expect, and what would
   make it fail?

8. BOTTOM LINE COST-BENEFIT. Given the safety-gap is my main result and the forced-choice
   behavioral test is my main eval-awareness test: is the mechanistic phase WORTH the GPU
   time? What is the MINIMUM mechanistic result that would genuinely strengthen the paper
   (vs being a throwaway "we also trained a probe")? Or should I skip it / relegate to
   future work? Give me your honest recommendation and a rough probability that the
   mechanistic phase produces a result worth including."""


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
        "X-Title": "eval-awareness-indic-mech-consult",
    })
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    return resp["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("Consulting GPT-5.5 on whether the mechanistic phase is worth it...\n")
    response = call(QUESTION)
    print("=== GPT-5.5 RESPONSE ===\n")
    print(response)

    logs_dir = pathlib.Path(__file__).resolve().parent.parent / "results" / "consult"
    logs_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "topic": "is the mechanistic phase worth it — realistic yield on 24B INT4",
        "model": MODEL,
        "question": QUESTION,
        "gpt55_response": response,
    }
    with (logs_dir / "gpt55_consultations.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\n[logged to {logs_dir / 'gpt55_consultations.jsonl'}]")
