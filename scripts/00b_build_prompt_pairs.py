"""
Build eval vs deploy prompt pairs from PUBLIC datasets only.
(Publication hygiene: no hand-crafted or proprietary prompts.)

Sources used:
  EVAL-LABELED GROUND TRUTH (English):
    - jjpn2/eval_awareness     — Needham et al. 2025 (CC-BY-NC-SA-4.0)
                                 1000 labeled eval/deploy items

  EVAL-LOOKING ITEMS (Indic — clearly benchmark format):
    - ai4bharat/MILU           — MCQ academic benchmark, 11 Indic langs (CC-BY-4.0, gated)
    - ai4bharat/IndicIFEval    — Instruction-following eval, 15 langs (CC-BY-4.0)

  DEPLOY-LOOKING ITEMS (real organic user queries):
    - allenai/WildChat-1M      — Real ChatGPT conversations, 74 langs (ODC-BY)
    - ai4bharat/indic-align    — Anudesh subset: crowdsourced real prompts (CC-BY-4.0)

  ROMANIZATION:
    - indic-transliteration    — deterministic native→roman conversion (not fabrication)
    - Note: produces ISO-style romanization, not informal Hinglish/Tanglish.
      Label as "systematic_roman" to distinguish from colloquial code-switching.

Output per (lang, script):
  data/eval_deploy_pairs/{lang}_{script}.json
  → [{"id": str, "text": str, "label": "eval"|"deploy",
      "source": str, "lang": str, "script": str}, ...]

Sampling strategy:
  - Target: ~200 eval + 200 deploy items per (lang, script)
  - Balance classes exactly (min of available)
  - Deduplicate by text
  - English: take directly from jjpn2/eval_awareness (already labeled)
  - Indic native: MILU + IndicIFEval → eval; WildChat + Anudesh → deploy
  - Indic roman: transliterate the native items above

USAGE:
  # Install extras first:
  #   pip install datasets indic-transliteration
  #   huggingface-cli login  (for MILU — gated dataset)

  python scripts/00b_build_prompt_pairs.py --lang all
  python scripts/00b_build_prompt_pairs.py --lang hi --script native
  python scripts/00b_build_prompt_pairs.py --dry-run   # print counts, no write
"""

import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

random.seed(42)

TARGET_PER_CLASS = 200   # eval + deploy each (recognition task)
SAFETY_TARGET    = 100   # harmful prompts per language (alignment-faking task)

# ── Language mappings ──────────────────────────────────────────────────────────
# WildChat uses BCP-47; MILU/IndicIFEval use ISO 639-1; Anudesh uses full names.
WILDCHAT_LANG_MAP = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "or": "Odia",
}
MILU_LANG_MAP = {
    "hi": "hi",
    "bn": "bn",
    "ta": "ta",
    "te": "te",
    "or": "or",
}
ANUDESH_LANG_MAP = {
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "or": "Odia",
}


# ── Transliteration ────────────────────────────────────────────────────────────
def to_roman(text: str, lang: str) -> str | None:
    """
    Convert native-script Indic text to systematic romanization (ISO 15919 / ITRANS).
    Returns None if transliteration fails or lang is English.
    Uses indic-transliteration library (pip install indic-transliteration).
    """
    if lang == "en":
        return None
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        script_map = {
            "hi": sanscript.DEVANAGARI,
            "bn": sanscript.BENGALI,
            "ta": sanscript.TAMIL,
            "te": sanscript.TELUGU,
            "or": sanscript.ORIYA,
        }
        src_script = script_map.get(lang)
        if src_script is None:
            return None
        return transliterate(text, src_script, sanscript.ITRANS)
    except Exception:
        return None


# ── Source loaders ─────────────────────────────────────────────────────────────

def load_needham_english() -> list[dict]:
    """
    Load Needham et al. eval_awareness dataset (English, pre-labeled).
    HF: jjpn2/eval_awareness  (CC-BY-NC-SA-4.0)
    Requires HF token if dataset is gated; also requires agreeing not to
    redistribute GAIA content.
    Returns list of {"text", "label", "source"}.
    """
    from datasets import load_dataset
    ds = load_dataset("jjpn2/eval_awareness", split="train")
    items = []
    for row in ds:
        text = row.get("prompt") or row.get("text") or row.get("input") or ""
        if not text.strip():
            continue
        # ground-truth label: metadata.eval_category True → "eval"
        meta = row.get("metadata", {})
        is_eval = meta.get("eval_category", False)
        label = "eval" if is_eval else "deploy"
        items.append({"text": text.strip(), "label": label, "source": "needham_eval_awareness"})
    return items


def load_milu_eval(lang: str, n: int) -> list[dict]:
    """
    Load MILU MCQ benchmark items as eval-context items.
    HF: ai4bharat/MILU  (CC-BY-4.0, gated — run `huggingface-cli login` first)
    Returns [] gracefully if dataset is gated and no HF token is set.
    """
    from datasets import load_dataset
    ds = None
    for split in ["test", "validation"]:
        try:
            ds = load_dataset("ai4bharat/MILU", name=MILU_LANG_MAP[lang], split=split)
            break
        except Exception:
            continue
    if ds is None:
        print(f"    MILU not accessible for {lang} (gated — run huggingface-cli login)")
        return []

    items = []
    for row in ds:
        q = row.get("question", "").strip()
        opts = [row.get(f"option{i}", "") for i in range(1, 5)]
        opts = [o for o in opts if o]
        if not q:
            continue
        text = q + "\n" + "\n".join(f"({chr(64+i+1)}) {o}" for i, o in enumerate(opts))
        items.append({
            "text": text,
            "label": "eval",
            "source": "ai4bharat/MILU",
        })
        if len(items) >= n:
            break
    return items


def load_indic_ifeval_eval(lang: str, n: int) -> list[dict]:
    """
    Load IndicIFEval instruction-following evaluation items.
    HF: ai4bharat/IndicIFEval  (CC-BY-4.0)
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("ai4bharat/IndicIFEval", split="test")
    except Exception:
        return []

    lang_field = None
    # Try to identify the language column
    if ds.features and "language" in ds.features:
        lang_field = "language"
    elif ds.features and "lang" in ds.features:
        lang_field = "lang"

    items = []
    for row in ds:
        if lang_field and row.get(lang_field, "").lower()[:2] != lang:
            continue
        text = row.get("prompt") or row.get("instruction") or row.get("text") or ""
        if not text.strip():
            continue
        items.append({
            "text": text.strip(),
            "label": "eval",
            "source": "ai4bharat/IndicIFEval",
        })
        if len(items) >= n:
            break
    return items


def load_wildchat_deploy(lang: str, n: int) -> list[dict]:
    """
    Load real organic user queries from WildChat.
    HF: allenai/WildChat-1M  (ODC-BY)
    Filter to the target language; extract first user turn (the query itself).
    """
    from datasets import load_dataset
    # stream=True avoids downloading the full ~40GB dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    wc_lang = WILDCHAT_LANG_MAP.get(lang)   # e.g., "Hindi"
    items = []
    for row in ds:
        row_lang = row.get("language", "")
        if row_lang != wc_lang:
            continue
        conv = row.get("conversation", [])
        if not conv:
            continue
        first_user_turn = next(
            (t["content"] for t in conv if t.get("role") == "user"), None
        )
        if not first_user_turn or not first_user_turn.strip():
            continue
        items.append({
            "text": first_user_turn.strip(),
            "label": "deploy",
            "source": "allenai/WildChat-1M",
        })
        if len(items) >= n:
            break
    return items


def load_anudesh_deploy(lang: str, n: int) -> list[dict]:
    """
    Load real crowdsourced prompts from Anudesh subset of ai4bharat/indic-align.
    HF: ai4bharat/indic-align  (CC-BY-4.0)
    Anudesh: crowdsourced human-written prompts — genuine user queries.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset(
            "ai4bharat/indic-align",
            data_files=f"data/Anudesh/*.jsonl",
            split="train",
            streaming=True,
        )
    except Exception:
        try:
            ds = load_dataset("ai4bharat/indic-align", split="train", streaming=True)
        except Exception:
            return []

    anudesh_lang = ANUDESH_LANG_MAP.get(lang)
    items = []
    for row in ds:
        # Anudesh rows have "interactions" list
        interactions = row.get("interactions", [])
        if not interactions:
            continue
        # Language may be in a 'lang' field or inferred from data path
        row_lang = row.get("lang") or row.get("language") or ""
        if anudesh_lang and row_lang and row_lang != anudesh_lang:
            continue
        first_turn = interactions[0]
        text = (
            first_turn.get("human") or
            first_turn.get("user")  or
            (first_turn.get("content") if isinstance(first_turn, dict) else "")
        )
        if not text or not text.strip():
            continue
        items.append({
            "text": text.strip(),
            "label": "deploy",
            "source": "ai4bharat/indic-align/Anudesh",
        })
        if len(items) >= n:
            break
    return items


# ── Safety dataset loaders (alignment-faking arm) ─────────────────────────────

def load_indic_safety_singleturn(lang: str, n: int) -> list[dict]:
    """
    Primary safety source: r1-b/indic-safety-singleturn (CC-BY-4.0).
    ~6.7K harmful prompts across 5 Indic languages with harm taxonomy labels.
    Requires HF login if gated (huggingface-cli login).
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("r1-b/indic-safety-singleturn", split="train", streaming=True)
    except Exception as e:
        print(f"    Could not load r1-b/indic-safety-singleturn: {e}")
        return []

    items = []
    for row in ds:
        row_lang = (row.get("language") or row.get("lang") or "").lower().strip()
        if not (row_lang == lang or row_lang.startswith(lang + "_")):
            continue
        text = (
            row.get("prompt") or row.get("instruction") or
            row.get("query")  or row.get("input") or ""
        ).strip()
        if not text:
            continue
        items.append({
            "text":          text,
            "harm_category": str(row.get("harm_category") or row.get("category") or ""),
            "source":        "r1-b/indic-safety-singleturn",
        })
        if len(items) >= n:
            break
    return items


def load_toxic_matrix(lang: str, n: int) -> list[dict]:
    """
    Fallback: ai4bharat/indic-align Toxic_Matrix (CC-BY-4.0).
    Schema: each row has one column per language variant, e.g.:
      ben_Beng, tam_Taml, tel_Telu, ory_Orya, hin_Deva, eng_Latn ...
    Values are stringified lists: "[['actual text here']]"
    No language filter needed — just pick the right column.
    """
    import ast
    from datasets import load_dataset

    # Map our ISO codes → Toxic_Matrix column names (native script)
    TOXIC_COL = {
        "hi": "hin_Deva",
        "bn": "ben_Beng",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "or": "ory_Orya",
        "en": "eng_Latn",
    }
    col = TOXIC_COL.get(lang)
    if col is None:
        print(f"    No Toxic_Matrix column for lang={lang}")
        return []

    ds = None
    for cfg in ["Toxic_Matrix", "Toxic-Matrix"]:
        try:
            ds = load_dataset("ai4bharat/indic-align", name=cfg,
                              split="train", streaming=True)
            break
        except Exception:
            continue
    if ds is None:
        print(f"    Could not load Toxic_Matrix from ai4bharat/indic-align")
        return []

    def parse_cell(raw) -> str:
        """Extract plain text from stringified [['text']] format."""
        if not raw:
            return ""
        s = str(raw).strip()
        try:
            parsed = ast.literal_eval(s)
            # [[turn1, turn2, ...], ...] or [['text']]
            if isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, list) and first:
                    return str(first[0]).strip()
                return str(first).strip()
        except Exception:
            pass
        # fallback: strip brackets/quotes manually
        return s.strip("[]'\"").split("'")[0].strip()

    items = []
    for row in ds:
        raw = row.get(col, "")
        text = parse_cell(raw)
        if not text or len(text) < 10:
            continue
        items.append({
            "text":          text,
            "harm_category": "toxic",
            "source":        "ai4bharat/indic-align/Toxic_Matrix",
        })
        if len(items) >= n:
            break
    return items


def load_nemotron_safety(lang: str, n: int) -> list[dict]:
    """
    nvidia/Nemotron-Safety-Guard-Dataset-v3 (CC-BY-4.0).
    Hindi and English only; filters to unsafe/harmful rows.
    """
    if lang not in ("hi", "en"):
        return []
    from datasets import load_dataset
    try:
        ds = load_dataset(
            "nvidia/Nemotron-Safety-Guard-Dataset-v3",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"    Could not load Nemotron-Safety-Guard-Dataset-v3: {e}")
        return []

    tgt_name = "Hindi" if lang == "hi" else "English"
    items = []
    for row in ds:
        row_lang = (row.get("language") or row.get("lang") or "").lower()
        if row_lang and tgt_name.lower() not in row_lang and lang not in row_lang:
            continue
        # Filter to unsafe (harmful) items only.
        # Actual column is "prompt_label" with values "safe" / "unsafe".
        prompt_label = str(row.get("prompt_label") or "").strip().lower()
        if prompt_label != "unsafe":
            continue
        text = (row.get("prompt") or row.get("text") or row.get("input") or "").strip()
        if not text:
            continue
        items.append({
            "text":          text,
            "harm_category": str(row.get("violated_categories") or ""),
            "source":        "nvidia/Nemotron-Safety-Guard-Dataset-v3",
        })
        if len(items) >= n:
            break
    return items


def load_advbench_en(n: int) -> list[dict]:
    """
    walledai/AdvBench (MIT): 520 harmful English behaviors.
    Standard red-teaming benchmark used throughout the safety literature.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("walledai/AdvBench", split="train")
    except Exception:
        try:
            ds = load_dataset("walledai/AdvBench", split="test")
        except Exception as e:
            print(f"    Could not load walledai/AdvBench: {e}")
            return []

    items = []
    for row in ds:
        text = (
            row.get("goal") or row.get("behavior") or row.get("prompt") or
            row.get("instruction") or ""
        ).strip()
        if not text:
            continue
        items.append({
            "text":          text,
            "harm_category": str(row.get("category") or "harmful_behavior"),
            "source":        "walledai/AdvBench",
        })
        if len(items) >= n:
            break
    return items


def assemble_safety_language(lang: str, target: int = SAFETY_TARGET) -> list[dict]:
    """
    Build a list of harmful prompts for the alignment-faking behavioral test.

    Primary source for ALL languages: ai4bharat/indic-align Toxic_Matrix.
    Toxic_Matrix contains the SAME harmful prompts translated into every target
    language (eng_Latn, hin_Deva, ben_Beng, tam_Taml, tel_Telu, ory_Orya).
    Using it for all languages guarantees parallel content — the only thing that
    differs between languages is the script, not the harmful scenario. This is
    the cleanest possible design for a between-language compliance-gap comparison.

    Fallback chain (if Toxic_Matrix is insufficient):
      English:  AdvBench → Nemotron (prompt_label == "unsafe")
      Hindi:    Nemotron Hindi (prompt_label == "unsafe")
      Indic:    indic-safety-singleturn (gated, may return [])
    """
    print(f"\n── {lang} (safety prompts) ──────────────────────────────────────────")
    items: list[dict] = []

    # ── Primary: Toxic_Matrix for ALL languages ────────────────────────────────
    print(f"  Loading Toxic_Matrix ({lang})...")
    items = load_toxic_matrix(lang, target)
    print(f"  Got {len(items)} from Toxic_Matrix.")

    # ── Fallbacks ──────────────────────────────────────────────────────────────
    if len(items) < target:
        if lang == "en":
            print("  Supplementing with walledai/AdvBench...")
            items += load_advbench_en(target - len(items))
        if len(items) < target and lang in ("en", "hi"):
            print(f"  Supplementing with Nemotron ({lang}, unsafe only)...")
            items += load_nemotron_safety(lang, target - len(items))
        if len(items) < target and lang != "en":
            print(f"  Supplementing with indic-safety-singleturn ({lang})...")
            items += load_indic_safety_singleturn(lang, target - len(items))

    # Deduplicate, shuffle, trim
    seen, unique = set(), []
    for item in items:
        if item["text"] not in seen:
            seen.add(item["text"])
            unique.append(item)

    random.shuffle(unique)
    unique = unique[:target]

    # Assign IDs and lang field
    for i, item in enumerate(unique):
        item["id"]   = f"{lang}_safety_{i:04d}"
        item["lang"] = lang

    print(f"  Final: {len(unique)} safety prompts")
    return unique


# ── Assembly ───────────────────────────────────────────────────────────────────

def assemble_language(lang: str, dry_run: bool = False) -> dict[str, list[dict]]:
    """
    Build eval + deploy item lists for one language (native script).
    Returns {"native": [...], "roman": [...]} (roman only for non-English).
    """
    print(f"\n── {lang} ──────────────────────────────────────────────")

    if lang == "en":
        print("  Loading Needham et al. eval_awareness (English)...")
        items = load_needham_english()
        eval_items   = [x for x in items if x["label"] == "eval"]
        deploy_items = [x for x in items if x["label"] == "deploy"]
    else:
        # Eval-looking: MILU (primary) + IndicIFEval (supplementary)
        n_milu = TARGET_PER_CLASS
        print(f"  Loading MILU eval items ({lang})...")
        eval_items = load_milu_eval(lang, n_milu)
        if len(eval_items) < TARGET_PER_CLASS:
            remaining = TARGET_PER_CLASS - len(eval_items)
            print(f"  Supplementing with IndicIFEval ({remaining} items)...")
            eval_items += load_indic_ifeval_eval(lang, remaining)

        # Deploy-looking: WildChat (primary) + Anudesh (supplementary)
        print(f"  Loading WildChat deploy items ({lang})...")
        deploy_items = load_wildchat_deploy(lang, TARGET_PER_CLASS)
        if len(deploy_items) < TARGET_PER_CLASS:
            remaining = TARGET_PER_CLASS - len(deploy_items)
            print(f"  Supplementing with Anudesh ({remaining} items)...")
            deploy_items += load_anudesh_deploy(lang, remaining)

    # Balance and deduplicate
    n = min(len(eval_items), len(deploy_items), TARGET_PER_CLASS)
    random.shuffle(eval_items)
    random.shuffle(deploy_items)
    eval_items   = eval_items[:n]
    deploy_items = deploy_items[:n]

    # Dedup by text
    seen = set()
    combined = []
    for item in eval_items + deploy_items:
        if item["text"] not in seen:
            seen.add(item["text"])
            combined.append(item)

    print(f"  Eval: {len(eval_items)}  Deploy: {len(deploy_items)}  "
          f"After dedup: {len(combined)}")

    # Add IDs, lang, script
    for i, item in enumerate(combined):
        item["id"]     = f"{lang}_native_{i:04d}"
        item["lang"]   = lang
        item["script"] = "native"

    result = {"native": combined}

    # Roman variant (Indic only)
    if lang != "en":
        roman_items = []
        for item in combined:
            roman_text = to_roman(item["text"], lang)
            if roman_text:
                roman_items.append({
                    **item,
                    "text":   roman_text,
                    "id":     item["id"].replace("native", "roman"),
                    "script": "systematic_roman",
                })
        print(f"  Romanized: {len(roman_items)} / {len(combined)} items")
        result["roman"] = roman_items

    return result


def write_pairs(lang: str, script: str, items: list[dict]):
    out_path = config.EVAL_DEPLOY_DIR / f"{lang}_{script}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"  → {out_path}  ({len(items)} items)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["pairs", "safety"],
        default="pairs",
        help=(
            "pairs: build eval/deploy contrastive pairs for recognition task; "
            "safety: build harmful prompt lists for alignment-faking task"
        ),
    )
    parser.add_argument("--lang",    default="all", help="language code or 'all'")
    parser.add_argument("--dry-run", action="store_true",
                        help="print counts without writing files (pairs mode only)")
    args = parser.parse_args()

    langs = config.LANG_CODES if args.lang == "all" else [args.lang]

    # ── Mode: pairs ────────────────────────────────────────────────────────────
    if args.mode == "pairs":
        config.EVAL_DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

        for lang in langs:
            try:
                result = assemble_language(lang, dry_run=args.dry_run)
            except Exception as e:
                print(f"  ⚠  {lang} failed: {e}")
                continue

            if not args.dry_run:
                for script, items in result.items():
                    fname_script = "roman" if "roman" in script else "native"
                    write_pairs(lang, fname_script, items)

        print("\nDone.")
        print("\nDataset sources used — recognition task (cite all):")
        print("  Needham et al. (2025) — jjpn2/eval_awareness — CC-BY-NC-SA-4.0")
        print("  AI4Bharat MILU (2024) — ai4bharat/MILU      — CC-BY-4.0 (gated)")
        print("  AI4Bharat IndicIFEval — ai4bharat/IndicIFEval — CC-BY-4.0")
        print("  WildChat-1M (2024)    — allenai/WildChat-1M  — ODC-BY")
        print("  Anudesh               — ai4bharat/indic-align — CC-BY-4.0")
        print("  Romanization: indic-transliteration (ITRANS scheme)")

    # ── Mode: safety ───────────────────────────────────────────────────────────
    elif args.mode == "safety":
        config.SAFETY_DIR.mkdir(parents=True, exist_ok=True)

        for lang in langs:
            try:
                items = assemble_safety_language(lang)
            except Exception as e:
                print(f"  ⚠  {lang} safety failed: {e}")
                continue

            if not args.dry_run:
                out_path = config.SAFETY_DIR / f"{lang}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)
                print(f"  → {out_path}  ({len(items)} items)")

        print("\nDone.")
        print("\nDataset sources used — alignment-faking arm (cite all that contributed):")
        print("  walledai/AdvBench (English)        — MIT")
        print("  r1-b/indic-safety-singleturn       — CC-BY-4.0")
        print("  ai4bharat/indic-align Toxic-Matrix — CC-BY-4.0")
        print("  nvidia/Nemotron-Safety-Guard-v3    — CC-BY-4.0")


if __name__ == "__main__":
    main()
