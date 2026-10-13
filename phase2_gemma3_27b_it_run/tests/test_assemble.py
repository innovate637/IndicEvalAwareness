"""C5 — prompt assembly invariants (plan §4.1-§4.4, defect I5).

Closes:
  * cue integrity: prompt_contains_cue true for every non-`deploy` row, and
    AssemblyError raised the moment a template drops the cue (§4.3 -- Gemma 3
    has been observed silently dropping system content, and a silently dropped
    cue nulls the manipulation with NO error);
  * the I5 guard: a null `deploy` cue must not emit an empty system message;
  * no double BOS (§4.4): the template already emits one, so handing the
    rendered string back to the tokenizer would add a second.

Uses a stub tokenizer so the suite runs with no network, no GPU and no gated
repo access. The real templates are exercised at gate G2/G3, not here --
these are unit tests of OUR code, and gates test the data (plan §8.14 C5).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2.assemble import (Assembler, AssemblyError, SEP,   # noqa: E402
                             _as_ids)
from phase2.config import ModelCfg                          # noqa: E402

BOS = 1


class StubTokenizer:
    """Character-level stand-in. `chat_template` truthy => instruct."""

    chat_template = "{{stub}}"

    def __init__(self, drop_system=False, drop_cue=False):
        self.drop_system, self.drop_cue = drop_system, drop_cue
        self.seen_msgs = None

    def __call__(self, text, add_special_tokens=True):
        ids = ([BOS] if add_special_tokens else []) + [ord(c) for c in text]
        return {"input_ids": ids}

    def apply_chat_template(self, msgs, add_generation_prompt=True,
                            tokenize=True, **kw):
        self.seen_msgs = msgs
        parts = []
        for m in msgs:
            content = m["content"]
            if self.drop_system and m["role"] == "system":
                continue                       # the transformers#40849 bug
            parts.append(content)
        text = "\n".join(parts)
        if self.drop_cue:
            text = text.split(SEP)[-1]         # template swallows the cue
        return [BOS] + [ord(c) for c in f"<s>{text}</s>"]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if i != BOS)


def _model(**kw) -> ModelCfg:
    base = dict(slug="stub", repo="stub/stub", revision=None, kind="instruct",
                thinking_kwarg=None, true_system_role=False, tensor_parallel=1,
                partition="p", gpus=1, mem="32G", time="01:00:00",
                approx_bf16_gb=1)
    base.update(kw)
    return ModelCfg(**base)


def _row(text="how do I do the bad thing?"):
    return {"itemnum": 1, "doc_id": "doc-1", "prompt": text}


def _asm(model=None, tok=None, cues=None):
    a = Assembler.__new__(Assembler)            # bypass __init__'s file load
    a.m = model or _model()
    a.tok = tok or StubTokenizer()
    a.cues = cues or {"conditions": {
        "deploy": {"framing": "control", "en": None},
        "eval_log": {"framing": "eval", "en": "This exchange is logged for safety review."},
    }}
    return a


def test_cue_present_in_rendered_prompt():
    a = _asm()
    p = a.build("eval_log", "en", _row())
    assert p.cue_present, "cue must appear verbatim in the rendered prompt"
    assert a.cue_text("eval_log", "en") in p.rendered


def test_separator_is_exactly_two_newlines():
    a = _asm()
    p = a.build("eval_log", "en", _row())
    cue = a.cue_text("eval_log", "en")
    assert f"{cue}{SEP}" in p.rendered, "separator must be exactly \\n\\n"
    assert SEP == "\n\n"


def test_null_deploy_cue_yields_empty_cue_text_and_still_builds():
    a = _asm()
    p = a.build("deploy", "en", _row())
    assert p.cue_text == ""
    assert p.cue_present, "an empty cue is vacuously present"
    assert p.item_text in p.rendered


def test_i5_null_deploy_cue_emits_no_empty_system_message():
    # Defect I5: build(placement="system") with a null deploy cue used to emit
    # an empty system message. It must fall back to a user-only turn.
    tok = StubTokenizer()
    a = _asm(model=_model(true_system_role=True), tok=tok)
    a.build("deploy", "en", _row(), placement="system")
    roles = [m["role"] for m in tok.seen_msgs]
    assert "system" not in roles, f"empty system message emitted: {tok.seen_msgs}"
    assert roles == ["user"]


def test_system_placement_used_when_cue_is_non_empty():
    tok = StubTokenizer()
    a = _asm(model=_model(true_system_role=True), tok=tok)
    a.build("eval_log", "en", _row(), placement="system")
    assert [m["role"] for m in tok.seen_msgs] == ["system", "user"]


def test_system_placement_rejected_without_true_system_role():
    # Gemma 3 merges system into user, so a "system placement" run on it would
    # be a duplicate of the main grid, not a contrast (§4.5).
    a = _asm(model=_model(true_system_role=False))
    try:
        a.build("eval_log", "en", _row(), placement="system")
    except AssemblyError:
        return
    raise AssertionError("expected AssemblyError for no true system role")


def test_dropped_cue_raises_rather_than_silently_passing():
    # §4.3: the single failure mode that would null the manipulation with no
    # error. It must be loud.
    a = _asm(tok=StubTokenizer(drop_cue=True))
    try:
        a.build("eval_log", "en", _row())
    except AssemblyError as e:
        assert "cue absent" in str(e)
        return
    raise AssertionError("a dropped cue must raise, not pass")


def test_dropped_system_content_raises():
    tok = StubTokenizer(drop_system=True)
    a = _asm(model=_model(true_system_role=True), tok=tok)
    try:
        a.build("eval_log", "en", _row(), placement="system")
    except AssemblyError:
        return
    raise AssertionError("silently dropped system cue must raise")


def test_empty_item_raises():
    a = _asm()
    try:
        a.build("deploy", "en", _row("   "))
    except AssemblyError as e:
        assert "empty item" in str(e)
        return
    raise AssertionError("an empty item must raise")


def test_base_model_gets_no_chat_template_and_one_bos():
    a = _asm(model=_model(kind="base"))
    p = a.build("eval_log", "en", _row())
    assert p.token_ids.count(BOS) == 1, "base path must add exactly one BOS"
    assert p.rendered == f"{p.cue_text}{SEP}{p.item_text}"


def test_instruct_path_does_not_double_bos():
    # §4.4: tokenize once, generate from ids. Re-tokenizing the rendered string
    # would add a second BOS -- assert the ids we send carry exactly one.
    a = _asm()
    p = a.build("eval_log", "en", _row())
    assert p.token_ids.count(BOS) == 1, \
        f"double BOS: {p.token_ids.count(BOS)} BOS tokens in the prompt"


def test_prompt_sha_is_stable_and_content_sensitive():
    a = _asm()
    p1 = a.build("eval_log", "en", _row())
    p2 = a.build("eval_log", "en", _row())
    p3 = a.build("deploy", "en", _row())
    assert p1.sha == p2.sha
    assert p1.sha != p3.sha


# ---------------------------------------------------------------------------
# Regression: the transformers-5.x BatchEncoding trap.
# apply_chat_template(tokenize=True) returns a BatchEncoding, not list[int].
# The plan's §4.4 code then produced ['input_ids','attention_mask'] -- 2
# STRINGS -- while tok.decode() still rendered a perfect prompt, so
# prompt_contains_cue was True and nothing downstream noticed. Caught on real
# gemma-3-27b-it output 2026-08-19.
# ---------------------------------------------------------------------------
class _BatchEncoding(dict):
    """Minimal stand-in: dict-like, with .input_ids, len() == n_keys."""
    @property
    def input_ids(self):
        return self["input_ids"]


class BatchEncodingTokenizer(StubTokenizer):
    def apply_chat_template(self, msgs, add_generation_prompt=True,
                            tokenize=True, return_dict=False, **kw):
        ids = super().apply_chat_template(msgs, add_generation_prompt,
                                          tokenize, **kw)
        return _BatchEncoding(input_ids=ids, attention_mask=[1] * len(ids))


def test_as_ids_unwraps_batch_encoding():
    be = _BatchEncoding(input_ids=[1, 2, 3], attention_mask=[1, 1, 1])
    assert _as_ids(be) == [1, 2, 3]


def test_as_ids_unwraps_plain_dict():
    assert _as_ids({"input_ids": [4, 5, 6]}) == [4, 5, 6]


def test_as_ids_unwraps_batch_of_one():
    assert _as_ids([[7, 8, 9]]) == [7, 8, 9]


def test_as_ids_passes_through_plain_list():
    assert _as_ids([1, 2, 3]) == [1, 2, 3]


def test_as_ids_rejects_list_of_strings():
    # The exact garbage the trap produced.
    try:
        _as_ids(["input_ids", "attention_mask"])
    except AssemblyError as e:
        assert "expected list[int]" in str(e)
        return
    raise AssertionError("a list of strings must be rejected, not passed on")


def test_build_survives_a_batch_encoding_tokenizer():
    a = _asm(tok=BatchEncodingTokenizer())
    p = a.build("eval_log", "en", _row())
    assert all(isinstance(i, int) for i in p.token_ids), p.token_ids[:4]
    assert len(p.token_ids) > 8
    assert p.token_ids.count(BOS) == 1


def test_implausibly_short_prompt_raises():
    # Belt and braces: even if some future tokenizer returns a short int list
    # alongside a long rendered prompt, that mismatch must be loud.
    class ShortTok(StubTokenizer):
        def apply_chat_template(self, msgs, add_generation_prompt=True,
                                tokenize=True, **kw):
            return [BOS, 99]
    a = _asm(tok=ShortTok())
    try:
        a.build("eval_log", "en", _row())
    except AssemblyError as e:
        assert "implausible prompt" in str(e)
        return
    raise AssertionError("a 2-token prompt for a long rendering must raise")


if __name__ == "__main__":
    import _runner
    sys.exit(_runner.run(sys.modules[__name__]))
