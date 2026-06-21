"""
Phase B, Step 1 — Extract residual-stream activations for all eval/deploy prompt pairs.

For each (language, script, label) triple, we extract the last-token residual-stream
activation at EVERY layer, using nnsight forward hooks on sarvam-m (INT4).

Output (per language):
  results/activations/{lang}_{script}.pkl
  → dict keyed by label ("eval" | "deploy"), each entry a list of per-layer arrays.

  Concrete shape per sample:
    activations[label][sample_idx][layer_idx] = np.ndarray [d_model]

Usage:
  python scripts/01_extract_activations.py --lang hi --script native
  python scripts/01_extract_activations.py --lang all   # runs all languages × scripts
"""

import sys
import json
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from nnsight import LanguageModel

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def load_model_and_tokenizer():
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = LanguageModel(
        config.MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="cuda:0",
        dtype=torch.bfloat16,         # nnsight 0.7: dtype= not torch_dtype=
        dispatch=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME, fix_mistral_regex=True
    )
    return model, tokenizer


def load_prompt_pairs(lang: str, script: str) -> list[dict]:
    """
    Load eval vs deploy prompt pairs for a given language/script combination.

    Expected file: data/eval_deploy_pairs/{lang}_{script}.json
    Schema: [{"text": str, "label": "eval" | "deploy"}, ...]
    """
    path = config.EVAL_DEPLOY_DIR / f"{lang}_{script}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No prompt pairs found at {path}.\n"
            f"Run scripts/00b_build_prompt_pairs.py first."
        )
    with open(path) as f:
        return json.load(f)


def format_as_chat(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def extract_all_layers(model, tokenizer, text: str) -> np.ndarray:
    """
    Run one forward pass and extract the last-token residual-stream activation
    at EVERY layer. Returns shape [NUM_LAYERS, D_MODEL].

    nnsight 0.7 notes:
      - model.model  (Envoy proxy) → MistralModel  (NOT model.model.model)
      - .save() returns the tensor directly; access with .float().cpu() after trace
      - All layer saves happen in ONE trace call (one forward pass)
    """
    formatted = format_as_chat(tokenizer, text)

    layer_saves = []
    with model.trace(formatted):
        for layer_idx in range(config.NUM_LAYERS):
            # transformers 5.9: MistralDecoderLayer returns hidden_states directly
            # (not a tuple). .output IS [batch, seq, d_model]. Use [:, -1, :].
            saved = model.model.layers[layer_idx].output[:, -1, :].save()
            layer_saves.append(saved)

    # nnsight 0.7: saved tensors are plain tensors after trace exits — no .value
    acts = np.stack(
        [s.detach().float().cpu().numpy()[0] for s in layer_saves],
        axis=0,
    )   # [NUM_LAYERS, D_MODEL]
    return acts


def run_extraction(lang: str, script: str, model, tokenizer):
    out_path = config.ACT_DIR / f"{lang}_{script}.pkl"
    if out_path.exists():
        print(f"  {lang}/{script}: already extracted, skipping.")
        return

    pairs = load_prompt_pairs(lang, script)

    result = {"eval": [], "deploy": []}
    for item in tqdm(pairs, desc=f"{lang}/{script}"):
        acts = extract_all_layers(model, tokenizer, item["text"])
        result[item["label"]].append(acts)   # list of [NUM_LAYERS, D_MODEL]

    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"  Saved → {out_path}  ({len(result['eval'])} eval, {len(result['deploy'])} deploy samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",   default="all", help="language code or 'all'")
    parser.add_argument("--script", default="all", help="'native', 'roman', or 'all'")
    args = parser.parse_args()

    config.ACT_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer()

    langs   = config.LANG_CODES          if args.lang   == "all" else [args.lang]
    scripts = config.SCRIPTS             if args.script == "all" else [args.script]
    # English has no romanization distinction — skip roman for "en"
    for lang in langs:
        for script in scripts:
            if lang == "en" and script == "roman":
                continue
            print(f"\n→ {lang} / {script}")
            run_extraction(lang, script, model, tokenizer)

    print("\nExtraction complete.")


if __name__ == "__main__":
    main()
