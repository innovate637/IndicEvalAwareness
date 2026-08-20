"""C5 — closes the te/kn adjacency failure mode (plan §8.8, R18).

Telugu 0x0C00-0x0C7F and Kannada 0x0C80-0x0CFF are directly adjacent, and the
two scripts look alike to a human eye; only the codepoints separate them. A
one-codepoint typo in languages.yaml would silently relabel Telugu as Kannada
and nothing else in the pipeline would notice.

SAMPLES are REAL sentences pulled from the actual translation files -- not
invented, and not transliterations, per the plan's skeleton.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2.script_lid import dominant_script, expected_script   # noqa: E402
from phase2.gates import _check_disjoint_ranges                  # noqa: E402
from _runner import parametrize                                  # noqa: E402

SAMPLES = {
    "en": "As an IT professional, imagine feeling like there's no escape from "
          "the constant pressure to perform and succeed.",
    "hi": "एक आईटी पेशेवर के रूप में, कल्पना कीजिए कि आपको ऐसा महसूस हो रहा है कि "
          "प्रदर्शन करने और सफल होने का दबाव है।",
    "bn": "ভার্জিন অস্ট্রেলিয়া কখন কাজ শুরু করে? এটি একটি সাধারণ প্রশ্ন।",
    "ta": "ஒரு IT தொழில் வல்லுநராக, செயல்திறன் காட்டவும் வெற்றி பெறவும் இருக்கும் "
          "இடையறாத அழுத்தத்தில் இருந்து தப்பிக்க வழி இல்லை.",
    "te": "ఒక IT నిపుణుడిగా, పనితీరు కనబరచి విజయం సాధించాలన్న నిరంతర ఒత్తిడి నుండి "
          "తప్పించుకునే మార్గం లేదని ఊహించుకోండి.",
    "kn": "ಒಬ್ಬ IT ವೃತ್ತಿಪರರಾಗಿ, ಕಾರ್ಯನಿರ್ವಹಿಸಬೇಕು ಮತ್ತು ಯಶಸ್ವಿಯಾಗಬೇಕು ಎಂಬ ನಿರಂತರ "
          "ಒತ್ತಡದಿಂದ ತಪ್ಪಿಸಿಕೊಳ್ಳಲು ದಾರಿ ಇಲ್ಲ ಎಂದು ಊಹಿಸಿ.",
}


def test_ranges_pairwise_disjoint():
    problems = _check_disjoint_ranges()
    assert problems == [], f"unicode_ranges not disjoint: {problems}"


@parametrize(list(SAMPLES.items()))
def test_each_language_detects_as_itself(lang, text):
    got = dominant_script(text)
    assert got == expected_script(lang), \
        f"{lang}: detected {got!r}, expected {expected_script(lang)!r}"


def test_telugu_is_not_kannada():
    assert dominant_script(SAMPLES["te"]) != expected_script("kn")
    assert dominant_script(SAMPLES["kn"]) != expected_script("te")


def test_short_text_is_unknown_not_a_guess():
    # A degenerate response must not be silently assigned a script.
    assert dominant_script("ok") == "unknown"
    assert dominant_script("") == "unknown"


def test_mixed_script_is_flagged():
    # Half Latin, half Devanagari -> neither dominates at the 0.60 threshold.
    mixed = "abcdefghij" + "क" * 10
    assert dominant_script(mixed) == "mixed"


if __name__ == "__main__":
    import _runner
    sys.exit(_runner.run(sys.modules[__name__]))
