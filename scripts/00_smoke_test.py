"""
M0 smoke test — confirms the full stack works before running experiments.

Checks:
  1. sarvam-m loads in INT4 on a single GPU.
  2. GPU VRAM is within budget.
  3. The model generates a coherent Hindi instruction-following response.
  4. nnsight can extract residual-stream activations at a given layer.
  5. Prints actual NUM_LAYERS and D_MODEL (update config.py if they differ).

Run: python scripts/00_smoke_test.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── 1. Load model ──────────────────────────────────────────────────────────────
print(f"Loading {config.MODEL_NAME} in {'INT4' if config.LOAD_IN_4BIT else 'bf16'} ...")

from transformers import AutoTokenizer, BitsAndBytesConfig
from nnsight import LanguageModel

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,   # nested quantisation saves a further ~0.4 bpw
)

model = LanguageModel(
    config.MODEL_NAME,
    quantization_config=bnb_cfg,
    device_map="cuda:0",
    dtype=torch.bfloat16,             # nnsight 0.7: use dtype= not torch_dtype=
    dispatch=True,
)
# nnsight loads its own tokenizer; load separately with the regex fix
tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_NAME, fix_mistral_regex=True
)

# ── 2. Print architecture facts ────────────────────────────────────────────────
# nnsight 0.7 layout:
#   model           = nnsight LanguageModel (Envoy wrapping MistralForCausalLM)
#   model._model    = MistralForCausalLM   (the actual HF model, bypasses proxy)
#   model._model.model = MistralModel      (transformer body; has .layers)
# Use model._model.model for architecture checks outside a trace context.
inner_hf = model._model.model   # actual MistralModel — no proxy overhead
n_layers = len(inner_hf.layers)
# q_proj maps d_model → num_heads*head_dim, so in_features = d_model
d_model  = inner_hf.layers[0].self_attn.q_proj.in_features  # 5120

print(f"\nArchitecture:")
print(f"  layers  : {n_layers}   (config.NUM_LAYERS = {config.NUM_LAYERS})")
print(f"  d_model : {d_model}   (config.D_MODEL    = {config.D_MODEL})")
if n_layers != config.NUM_LAYERS or d_model != config.D_MODEL:
    print("  WARNING  Mismatch — update config.py before running downstream scripts.")
else:
    print("  OK  Matches config.")

# ── 3. VRAM check ──────────────────────────────────────────────────────────────
allocated = torch.cuda.memory_allocated(0) / 1e9
reserved  = torch.cuda.memory_reserved(0)  / 1e9
total     = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"\nGPU VRAM — {torch.cuda.get_device_name(0)}:")
print(f"  Total    : {total:.1f} GB")
print(f"  Allocated: {allocated:.1f} GB  (model weights)")
print(f"  Reserved : {reserved:.1f} GB")
print(f"  Free     : {total - reserved:.1f} GB  (for activations + forward pass)")
if total - reserved < 6:
    print("  ⚠️  Less than 6 GB free — may OOM during activation extraction.")
else:
    print("  ✅ Enough headroom for extraction.")

# ── 4. Hindi generation check ──────────────────────────────────────────────────
hindi_prompt = "भारत की राजधानी क्या है?"   # "What is the capital of India?"
messages = [{"role": "user", "content": hindi_prompt}]
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(formatted, return_tensors="pt").to("cuda:0")

print(f"\nHindi generation test:")
print(f"  Prompt : {hindi_prompt}")
# nnsight 0.7: use model._model.generate() for generation without tracing.
# model.model.generate() proxies through the Envoy and tries to start a trace,
# which is wrong here. model._model is the raw MistralForCausalLM.
with torch.no_grad():
    out = model._model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"  Response: {response.strip()}")
print("  OK  Generation OK" if response.strip() else "  FAIL  Empty response — check model loading.")

# ── 5. nnsight activation extraction check ────────────────────────────────────
# Extract residual-stream activations at the midpoint layer, last token position.
#
# nnsight 0.7 API (differs from 0.3):
#   - Inside trace context: model.model.layers[L]  (Envoy proxy path)
#     NOT model.model.model.layers[L]  (that was wrong; MistralModel has no .model attr)
#   - .save() returns the tensor directly (no .value accessor needed in 0.7)
#   - After `with` exits, the saved tensor IS the real numpy-able tensor.
probe_layer = n_layers // 2
print(f"\nnnsight extraction check (layer {probe_layer}, last token):")

try:
    with model.trace(formatted):
        # transformers 5.9: MistralDecoderLayer returns hidden_states directly
        # (not a tuple). So .output IS [batch, seq, d_model] — not a tuple.
        # output[0] would index the batch dim → 2D wrong. Use [:, -1, :] directly.
        hidden_save = model.model.layers[probe_layer].output[:, -1, :].save()

    # nnsight 0.7: saved tensor is directly usable — no .value attribute
    act = hidden_save.detach().float().cpu().numpy()
    print(f"  Activation shape : {act.shape}   (expected [1, {d_model}])")
    print(f"  Mean / Std       : {act.mean():.4f} / {act.std():.4f}")
    print("  OK  nnsight extraction OK.")
except Exception as e:
    print(f"  FAIL  nnsight extraction failed: {e}")
    import traceback; traceback.print_exc()
    print("  → Check nnsight version and inner module path.")

print("\nSmoke test complete.")
