"""Deterministic language detection by Unicode block.

The six study languages occupy disjoint blocks, so this is exact, needs no
model, and adds nothing to the reproducibility surface.
"""
from __future__ import annotations
from phase2.config import load_languages

_LANGS = load_languages()
_RANGES = {c: [tuple(r) for r in cfg["unicode_ranges"]] for c, cfg in _LANGS.items()}
_SCRIPT = {c: cfg["script"] for c, cfg in _LANGS.items()}


def script_histogram(text: str) -> dict[str, int]:
    hist = {s: 0 for s in set(_SCRIPT.values())}
    hist["other"] = 0
    for ch in text:
        if ch.isspace() or not ch.isprintable():
            continue
        cp, hit = ord(ch), None
        for code, ranges in _RANGES.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                hit = _SCRIPT[code]
                break
        hist[hit or "other"] += 1
    return hist


def dominant_script(text: str, min_chars: int = 12) -> str:
    """Dominant script tag, or 'unknown'/'mixed' for degenerate output."""
    scored = {k: v for k, v in script_histogram(text).items() if k != "other"}
    total = sum(scored.values())
    if total < min_chars:
        return "unknown"
    best = max(scored, key=scored.get)
    return best if scored[best] / total >= 0.60 else "mixed"


def expected_script(lang: str) -> str:
    return _SCRIPT[lang]


def matches_expected(text: str, lang: str) -> bool:
    return dominant_script(text) == _SCRIPT[lang]
