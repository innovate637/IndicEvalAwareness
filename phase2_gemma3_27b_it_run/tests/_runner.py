"""Minimal pytest shim so the C5 suite runs on a machine without pytest.

pytest is the intended runner (see pyproject.toml [dev]); it is not installed
here and installing is not authorised (CLAUDE.md rule 2). These tests are
written pytest-style -- `pytest tests/` works unchanged once it is available.
"""
from __future__ import annotations
import sys, traceback


def run(module) -> int:
    tests = [(n, f) for n, f in vars(module).items()
             if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in tests:
        cases = getattr(fn, "_params", [(())])
        for args in cases:
            label = f"{name}{args if args != () else ''}"
            try:
                fn(*args)
                print(f"  PASS  {label}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL  {label}: {e}")
                traceback.print_exc(limit=2)
                failed += 1
            except Exception as e:
                print(f"  ERROR {label}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=3)
                failed += 1
    print(f"\n{module.__name__}: {passed} passed, {failed} failed")
    return 1 if failed else 0


def parametrize(cases):
    """Stand-in for @pytest.mark.parametrize with a single tuple arg list."""
    def deco(fn):
        fn._params = [c if isinstance(c, tuple) else (c,) for c in cases]
        return fn
    return deco
