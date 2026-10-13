"""Per model x language max_tokens. Tokenizer only -- run on the login node.

Two modes (plan §5.2 and the §8.14 extension):

  default        heuristic from tokenizer fertility:
                   r(model,lang) = med_tok_per_char(lang) / med_tok_per_char(en)
                   max_tokens    = clip(ceil(512*r/32)*32, 512, 2048)

  --from-probe   empirical, and preferred once a probe exists:
                   max_tokens = ceil32(p99_observed_completion * 1.25)
                 The probe needs no new script -- it is
                   generate.py --max-tokens-override 16384 --items 40
                 over a 40-item subset, whose shards this mode then reads.

Truncation must never correlate with language (§5.2); that is the cheapest way
to manufacture the headline result. The empirical mode is what defends it.
"""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path
from phase2.config import CONFIG, load_models, load_run
from phase2.assemble import load_items
from phase2.io_jsonl import read_shard

# [LOCAL] CEIL raised 2048 -> 3072. §5.2's ceiling assumed Indic-driven budgets
# under ~2048; measurement (job 450) put ENGLISH at p99=1784, needing 2240. The
# ceiling is bounded by max_model_len(4096) - longest prompt(584) = 3512, so
# 3072 is safe and still leaves >1000 tokens of headroom.
BASE_EN, FLOOR, CEIL, MULT = 512, 512, 3072, 32


def _ceil_mult(x: float) -> int:
    return int(min(CEIL, max(FLOOR, math.ceil(x / MULT) * MULT)))


def tokens_per_char(tok, texts: list[str]) -> float:
    vals = [len(tok(t.strip(), add_special_tokens=False)["input_ids"]) / len(t.strip())
            for t in texts if len(t.strip()) >= 20]
    if not vals:
        raise ValueError("no usable texts")
    return statistics.median(vals)


def from_tokenizer(only: set[str] | None = None) -> dict[str, dict[str, int]]:
    from transformers import AutoTokenizer
    run, models = load_run(), load_models()
    table: dict[str, dict[str, int]] = {}
    for slug, m in models.items():
        if only and slug not in only:
            continue
        tok = AutoTokenizer.from_pretrained(
            m.repo, revision=m.revision, trust_remote_code=m.trust_remote_code)
        per_lang = {lang: tokens_per_char(
                        tok, [r["prompt"] for arm in run.arms
                              for r in load_items(arm, lang)])
                    for lang in run.langs}
        ref = per_lang["en"]
        table[slug] = {lang: _ceil_mult(BASE_EN * (v / ref))
                       for lang, v in per_lang.items()}
        print(slug, table[slug])
    return table


def _p99(xs: list[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    # nearest-rank p99; exact and dependency-free
    k = max(0, math.ceil(0.99 * len(s)) - 1)
    return float(s[k])


def from_probe(root: Path) -> dict[str, dict[str, int]]:
    """Read probe shards and size each cell at ceil32(p99 * 1.25).

    Reports cells where the probe itself hit its ceiling: a saturated probe
    means p99 is a floor, not an estimate, and that cell must be re-probed
    with a larger override before its budget is trusted.
    """
    run = load_run()
    obs: dict[str, dict[str, list[int]]] = {}
    saturated: list[str] = []
    for shard in sorted(Path(root).rglob("*.jsonl")):
        for rec in read_shard(shard):
            slug, lang = rec.get("model_slug"), rec.get("lang")
            if not slug or not lang or rec.get("error"):
                continue
            n = rec.get("n_completion_tokens")
            if n is None:
                continue
            obs.setdefault(slug, {}).setdefault(lang, []).append(int(n))
            if rec.get("finish_reason") == "length":
                tag = f"{slug}/{lang}"
                if tag not in saturated:
                    saturated.append(tag)

    if not obs:
        raise SystemExit(f"no probe rows under {root}")

    table: dict[str, dict[str, int]] = {}
    for slug, per_lang in obs.items():
        table[slug] = {}
        for lang in run.langs:
            xs = per_lang.get(lang) or []
            if not xs:
                print(f"  [warn] {slug}/{lang}: no probe rows, cell omitted")
                continue
            table[slug][lang] = _ceil_mult(_p99(xs) * 1.25)
        print(slug, table[slug], f"(n={sum(len(v) for v in per_lang.values())})")
    if saturated:
        print("\n[WARN] probe hit its own ceiling in these cells -- p99 is a "
              "floor, not an estimate. Re-probe with a larger override "
              "before trusting them:\n  " + "\n  ".join(saturated))
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-probe", dest="probe", default=None,
                    help="root of probe generations "
                         "(generate.py --max-tokens-override 16384 --items 40)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--models", default=None,
                    help="comma-separated slugs; default = every model in "
                         "models.yaml (which would download 6 tokenizers)")
    args = ap.parse_args()

    only = ({m.strip() for m in args.models.split(",") if m.strip()}
            if args.models else None)
    table = from_probe(Path(args.probe)) if args.probe else from_tokenizer(only)
    out = Path(args.out) if args.out else CONFIG / "max_tokens.json"
    out.write_text(json.dumps(table, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
