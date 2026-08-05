"""xarray call conventions enforced across the package.

``xr.concat``/``xr.merge`` are alignment-sensitive: the default ``join`` decides
whether mismatched coordinates union (NaN-filling the gaps) or raise. xarray is
changing ``concat``'s default from ``"outer"`` to ``"exact"``, so a call that
leaves it implicit changes behaviour on a routine dependency bump — silently
today, as a ValueError afterwards.

Checked by parsing the source: the failure is a latent one that no runtime test
exercises until a real coordinate mismatch reaches the call.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent.parent / "h2mare"
ALIGNING_CALLS = {"concat", "merge"}


def _implicit_join_calls(path: Path) -> list[str]:
    """``xr.concat``/``xr.merge`` calls in *path* that omit ``join=``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr in ALIGNING_CALLS):
            continue
        # Only xr.<call> / xarray.<call> — not DataFrame.merge and friends.
        if not (isinstance(fn.value, ast.Name) and fn.value.id in {"xr", "xarray"}):
            continue
        if not any(kw.arg == "join" for kw in node.keywords):
            found.append(f"{path.name}:{node.lineno} xr.{fn.attr}(...)")
    return found


def test_concat_and_merge_pass_join_explicitly():
    """Regression: three xr.concat calls relied on the default join. The one in
    storage.py already emitted xarray's FutureWarning about that default
    flipping to "exact"; the other two carried the same latent break with no
    warning because no test fed them mismatched coordinates."""
    violations = [
        v for path in sorted(PKG_DIR.rglob("*.py")) for v in _implicit_join_calls(path)
    ]
    assert violations == [], (
        "xr.concat/xr.merge must pass join= explicitly, since xarray is changing "
        "concat's default from 'outer' to 'exact': " + "; ".join(violations)
    )
