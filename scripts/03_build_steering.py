"""
Phase B, Steps 3a–3c — Build eval/deploy steering vectors and test cross-lingual transfer.

Step 3a: Build a per-language steering vector (difference-of-means in the
         PCA-compressed residual stream, at the best probe layer).

Step 3b: Cross-lingual geometry — cosine similarity matrix between all per-language
         vectors. Tests H3: is there a shared "I'm being evaluated" direction?

Step 3c: Cross-lingual transfer — apply the ENGLISH steering vector during
         inference in Hindi/Bengali/Tamil/Telugu/Odia and measure whether the
         model's eval-awareness changes. Tests H3 (transfer) and H4 (decomposition).

Output:
  results/steering/vectors.pkl            — per-(lang, script, layer) steering vectors
  results/steering/cosine_similarity.csv  — pairwise language cosine sims
  results/steering/transfer_results.csv   — steering-transfer behavioral results

Usage:
  python scripts/03_build_steering.py --mode vectors
  python scripts/03_build_steering.py --mode geometry
  python scripts/03_build_steering.py --mode transfer
"""

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig
from nnsight import LanguageModel
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Helpers ────────────────────────────────────────────────────────────────────

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
        torch_dtype=torch.bfloat16,
        dispatch=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    return model, tokenizer


def best_layer_for(lang: str, script: str) -> int:
    """Return the layer with highest mean AUROC for this language/script."""
    auroc_path = config.PROBE_DIR / "auroc_all.csv"
    df = pd.read_csv(auroc_path)
    sub = df[(df["lang"] == lang) & (df["script"] == script)]
    return int(sub.loc[sub["auroc_mean"].idxmax(), "layer"])


def get_steering_vector(lang: str, script: str, layer: int) -> np.ndarray:
    """
    Difference-of-means steering vector in the PCA-compressed space,
    then projected back to d_model for hook injection.
    Returns ndarray [D_MODEL].
    """
    probe_path = config.PROBE_DIR / f"{lang}_{script}.pkl"
    with open(probe_path, "rb") as f:
        probes = pickle.load(f)

    probe_data = probes[layer]
    pca   = probe_data["pca"]
    probe = probe_data["probe"]

    # eval direction in PCA space (class 1 coefficient − class 0)
    eval_dir_pca   = probe.coef_[config.LABEL_EVAL]
    deploy_dir_pca = probe.coef_[config.LABEL_DEPLOY]
    direction_pca  = eval_dir_pca - deploy_dir_pca   # [PCA_COMPONENTS]

    # project back to d_model via PCA components matrix [PCA_COMPONENTS, D_MODEL]
    direction = direction_pca @ pca.components_        # [D_MODEL]
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction


# ── Step 3a: Build all vectors ─────────────────────────────────────────────────

def build_vectors():
    config.STEER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.STEER_DIR / "vectors.pkl"

    vectors = {}
    for lang in config.LANG_CODES:
        for script in config.SCRIPTS:
            if lang == "en" and script == "roman":
                continue
            layer = best_layer_for(lang, script)
            vec   = get_steering_vector(lang, script, layer)
            vectors[(lang, script)] = {"layer": layer, "vector": vec}
            print(f"  {lang}/{script}: best layer = {layer}, ||v|| = {np.linalg.norm(vec):.4f}")

    with open(out_path, "wb") as f:
        pickle.dump(vectors, f)
    print(f"\nVectors saved → {out_path}")


# ── Step 3b: Cross-lingual geometry ───────────────────────────────────────────

def compute_geometry():
    with open(config.STEER_DIR / "vectors.pkl", "rb") as f:
        vectors = pickle.load(f)

    keys = list(vectors.keys())
    vecs = np.stack([vectors[k]["vector"] for k in keys])   # [N, D_MODEL]
    cos  = cosine_similarity(vecs)

    labels = [f"{lang}_{script}" for lang, script in keys]
    df = pd.DataFrame(cos, index=labels, columns=labels)
    out_path = config.STEER_DIR / "cosine_similarity.csv"
    df.to_csv(out_path)
    print(f"Cosine similarity matrix → {out_path}")
    print(df.round(3).to_string())


# ── Step 3c: Cross-lingual transfer — apply English vector in Indic contexts ──

def run_transfer(model, tokenizer):
    with open(config.STEER_DIR / "vectors.pkl", "rb") as f:
        vectors = pickle.load(f)

    # English steering vector (native script = "native" for English)
    en_entry  = vectors[("en", "native")]
    en_vec    = torch.tensor(en_entry["vector"], dtype=torch.float32, device="cuda:0")
    en_layer  = en_entry["layer"]

    scenarios_path = config.SCENARIOS_DIR / "recognition_items.json"
    with open(scenarios_path) as f:
        items = json.load(f)

    inner = model.model.model
    rows  = []

    for item in tqdm(items, desc="transfer"):
        lang   = item["lang"]
        script = item["script"]
        text   = item["text"]
        label  = item["label"]   # "eval" | "deploy"

        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True
        )

        # Baseline: no steering
        with model.trace(formatted):
            out_base = model.model.generate(
                tokenizer(formatted, return_tensors="pt").input_ids.to("cuda:0"),
                max_new_tokens=50, do_sample=False
            )

        # Steered: apply English "act-deployed" vector (subtract eval direction)
        for strength in config.STEER_STRENGTHS:
            if strength == 0.0:
                continue

            def hook_fn(output, strength=strength):
                # subtract eval direction → nudge toward "I think I'm deployed"
                output[0][:, -1, :] -= strength * en_vec
                return output

            with model.trace(formatted) as _:
                inner.layers[en_layer].output = hook_fn(inner.layers[en_layer].output)
                out_steered_ids = model.model.generate(
                    tokenizer(formatted, return_tensors="pt").input_ids.to("cuda:0"),
                    max_new_tokens=50, do_sample=False
                )

            rows.append({
                "lang":     lang,
                "script":   script,
                "label":    label,
                "strength": strength,
                "response_base":    tokenizer.decode(out_base[0], skip_special_tokens=True),
                "response_steered": tokenizer.decode(out_steered_ids[0], skip_special_tokens=True),
            })

    out_path = config.STEER_DIR / "transfer_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Transfer results → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["vectors", "geometry", "transfer"], required=True
    )
    args = parser.parse_args()

    if args.mode == "vectors":
        build_vectors()
    elif args.mode == "geometry":
        compute_geometry()
    elif args.mode == "transfer":
        model, tokenizer = load_model_and_tokenizer()
        run_transfer(model, tokenizer)


if __name__ == "__main__":
    main()
