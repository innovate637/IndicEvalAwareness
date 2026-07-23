"""
Centralised configuration for the eval-awareness Indic study.
Proposal 7: Evaluation Awareness in Indic Settings.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

EVAL_DEPLOY_DIR = DATA_DIR / "eval_deploy_pairs"   # contrastive prompt pairs
SCENARIOS_DIR   = DATA_DIR / "scenarios"            # behavioral safety scenarios
SAFETY_DIR      = DATA_DIR / "safety_prompts"       # harmful prompts for alignment-faking test

ACT_DIR      = RESULTS_DIR / "activations"
PROBE_DIR    = RESULTS_DIR / "probes"
STEER_DIR    = RESULTS_DIR / "steering"
BEHAV_DIR    = RESULTS_DIR / "behavioral"

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "sarvamai/sarvam-m"   # 24B Mistral-Small-3.1 base, instruct + reasoning

# sarvam-m is Mistral-Small-3.1-24B — 40 transformer layers, d_model 5120.
# Confirmed via smoke test (scripts/00_smoke_test.py); update below if wrong.
NUM_LAYERS = 40
D_MODEL    = 5120

# INT4 quantisation via bitsandbytes — brings VRAM from ~48 GB to ~14 GB.
LOAD_IN_4BIT = True

# ── Languages ──────────────────────────────────────────────────────────────────
# (language_code, display_name, script, resource_level)
LANGUAGES = [
    ("en",  "English",  "latin",      "high"),
    ("hi",  "Hindi",    "devanagari", "high"),
    ("bn",  "Bengali",  "bengali",    "high"),
    ("ta",  "Tamil",    "tamil",      "high"),
    ("te",  "Telugu",   "telugu",     "mid"),
    ("or",  "Odia",     "odia",       "low"),
]
LANG_CODES = [l[0] for l in LANGUAGES]

SCRIPTS = ["native", "roman"]   # two orthographic variants per Indic language

# ── Probing ────────────────────────────────────────────────────────────────────
PROBE_LAYERS = list(range(0, NUM_LAYERS))   # all layers; narrow after smoke test
PCA_COMPONENTS = 128
REGULARIZATION_C = 1.0
MAX_ITER = 1000
CV_FOLDS = 5

# ── Steering ───────────────────────────────────────────────────────────────────
# Layers where the eval/deploy direction is sharpest (refined after probing).
# Start with a spread across mid-to-late layers; update after probe AUROC sweep.
STEER_LAYERS = list(range(10, NUM_LAYERS, 5))   # every 5th layer for the sweep
STEER_STRENGTHS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]

# ── Evaluation framing ─────────────────────────────────────────────────────────
# Labels used throughout (binary classification).
LABEL_EVAL   = 1   # "this is an evaluation / safety test"
LABEL_DEPLOY = 0   # "this is a real deployment"
LABEL_NAMES  = {LABEL_EVAL: "eval", LABEL_DEPLOY: "deploy"}

# ==== Hermes-3-70B replication overrides (2026-07-03) ====
from pathlib import Path as _P
_EXP = _P("/localstorage/home/dhruvkumar/IndicEvalAwareness/experiments")
DATA_DIR    = _EXP / "data"
SAFETY_DIR  = DATA_DIR / "safety_prompts"
RESULTS_DIR = _P("/localstorage/home/dhruvkumar/IndicEvalAwareness/llama_causal_experiments/results")
MODEL_NAME  = "unsloth/Hermes-3-Llama-3.1-70B-bnb-4bit"
NUM_LAYERS  = 80
D_MODEL     = 8192
