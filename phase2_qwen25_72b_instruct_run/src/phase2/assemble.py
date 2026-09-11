from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from phase2.config import REPO, ModelCfg

SEP = "\n\n"          # invariant across every model, language and condition


class AssemblyError(RuntimeError):
    pass


@dataclass
class Prompt:
    itemnum: int
    doc_id: str
    cue_text: str
    item_text: str
    rendered: str
    token_ids: list[int]

    @property
    def sha(self) -> str:
        return hashlib.sha1(self.rendered.encode()).hexdigest()

    @property
    def cue_present(self) -> bool:
        return (not self.cue_text) or (self.cue_text in self.rendered)


def _as_ids(out) -> list[int]:
    """Return a flat list[int] of token ids from whatever the tokenizer gave us.

    §4.4 assumes `apply_chat_template(tokenize=True)` returns a list of ids.
    **In transformers 5.x it returns a `BatchEncoding`** (`return_dict` now
    defaults to True), so the plan's code yields `['input_ids',
    'attention_mask']` -- a 2-element list of STRINGS -- while `tok.decode()`
    still renders the prompt perfectly. Every human-readable check passes,
    `prompt_contains_cue` is True, and the model is handed garbage.

    That is the worst failure mode in this pipeline: silent, invisible in the
    logs, and it would have corrupted every generation. So this helper is
    deliberately paranoid, and `build()` asserts on its output.
    """
    if hasattr(out, "input_ids"):          # BatchEncoding / dict-like
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    out = list(out)
    if out and isinstance(out[0], (list, tuple)):    # batch of one
        out = list(out[0])
    if not out or not all(isinstance(i, int) for i in out):
        raise AssemblyError(
            f"tokenizer returned {type(out).__name__} of "
            f"{[type(i).__name__ for i in out[:3]]} -- expected list[int]. "
            f"This is the transformers-5.x BatchEncoding trap; see _as_ids.")
    return out


def load_items(arm: str, lang: str) -> list[dict]:
    fn = (f"final_harmful_200_{lang}.json" if arm == "harmful"
          else f"benign_200_{lang}.json")
    p = REPO / "data" / "final_set" / fn
    if not p.exists():
        raise FileNotFoundError(p)
    rows = json.loads(p.read_text())
    rows.sort(key=lambda r: r["itemnum"])          # determinism (§5.3)
    return rows


def load_cues() -> dict:
    return json.loads((REPO / "data" / "cues" / "cue_battery.json").read_text())


class Assembler:
    """Builds the exact token sequence sent to the model."""

    def __init__(self, model: ModelCfg, tokenizer):
        self.m, self.tok = model, tokenizer
        self.cues = load_cues()
        if not model.is_base and not getattr(tokenizer, "chat_template", None):
            raise AssemblyError(f"{model.slug} declared instruct but has no "
                                f"chat_template")

    def cue_text(self, cue: str, lang: str) -> str:
        node = self.cues["conditions"].get(cue)
        if node is None:
            return ""
        val = node.get(lang)
        return "" if val is None else val.strip()

    def build(self, cue: str, lang: str, row: dict,
              placement: str = "user_prepend") -> Prompt:
        item_text = (row.get("prompt") or "").strip()
        if not item_text:
            raise AssemblyError(f"empty item {row.get('itemnum')} [{lang}]")
        cue_text = self.cue_text(cue, lang)

        # ---- system placement (sub-run B only) --------------------------
        if placement == "system":
            if self.m.is_base or not self.m.true_system_role:
                raise AssemblyError(f"{self.m.slug} has no true system role")
            if not cue_text:
                # I5: never emit an empty system message; fall back to user-only
                msgs = [{"role": "user", "content": item_text}]
            else:
                msgs = [{"role": "system", "content": cue_text},
                        {"role": "user", "content": item_text}]
            ids = self._templ(msgs)
            rendered = self.tok.decode(ids, skip_special_tokens=False)

        # ---- user_prepend (the main grid) --------------------------------
        else:
            content = f"{cue_text}{SEP}{item_text}" if cue_text else item_text
            if self.m.is_base:
                rendered = content
                ids = self.tok(rendered, add_special_tokens=True)["input_ids"]
            else:
                ids = self._templ([{"role": "user", "content": content}])
                rendered = self.tok.decode(ids, skip_special_tokens=False)

        ids = _as_ids(ids)
        # A rendered prompt this long cannot tokenize to a handful of ids. The
        # BatchEncoding trap produced exactly 2, with a perfect `rendered`.
        if len(ids) < 8:
            raise AssemblyError(
                f"implausible prompt: {len(ids)} token ids for "
                f"{len(rendered)} rendered chars ({self.m.slug}/{cue}/{lang})")
        p = Prompt(itemnum=row["itemnum"], doc_id=row["doc_id"],
                   cue_text=cue_text, item_text=item_text,
                   rendered=rendered, token_ids=ids)

        # §4.3 — asserted on EVERY row, because a silently dropped cue nulls
        # the manipulation with no error (and Gemma 3 is known to do this).
        if not p.cue_present:
            raise AssemblyError(
                f"cue absent from rendered prompt: {self.m.slug}/{cue}/{lang}")
        return p

    def _templ(self, msgs: list[dict]) -> list[int]:
        kw = {}
        if self.m.thinking_kwarg:
            kw[self.m.thinking_kwarg] = False          # §5.1
        out = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=False, **kw)
        return _as_ids(out)

    def shard(self, arm: str, cue: str, lang: str,
              placement: str = "user_prepend") -> list[Prompt]:
        return [self.build(cue, lang, r, placement) for r in load_items(arm, lang)]
