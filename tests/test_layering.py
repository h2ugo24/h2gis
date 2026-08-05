"""Package layering invariants.

``utils/`` is the leaf of the pipeline for everything ``utils/__init__`` pulls
in, so no module reachable from that import may depend on ``storage`` — at
module scope or inside a function. Enforced by parsing the source rather than by
importing, so a violation is reported as the offending file and line instead of
an ImportError raised from wherever the cycle happens to close.

``plot.py`` is the deliberate exception: it imports ``storage.zarr_catalog`` at
module scope and is therefore *not* re-exported from ``utils/__init__``. That is
what keeps the dependency one-way — storage may import utils, and plot may
import storage, because importing ``h2mare.utils`` never reaches plot.
"""

from __future__ import annotations

import ast
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent.parent / "h2mare" / "utils"

# Every utils module except plot.py, which is intentionally outside the package's
# eager import graph (see module docstring).
LEAF_MODULES = sorted(p for p in UTILS_DIR.glob("*.py") if p.name != "plot.py")


def _any_storage_imports(path: Path) -> list[tuple[int, str]]:
    """Every ``h2mare.storage`` import in *path*, at any nesting depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "h2mare.storage"
        ):
            found.append((node.lineno, node.module or ""))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("h2mare.storage"):
                    found.append((node.lineno, alias.name))
    return found


class TestUtilsIsALeafPackage:
    def test_no_storage_imports_at_all(self):
        """Regression: utils/date_range.py imported storage.coverage and
        utils/plot.py imported storage.parquet_helpers, while storage imported
        utils in eight places — a genuine cycle that had already forced one
        function-level import to work around it.

        Function-level imports count as violations here: a deferred import is a
        cycle that still exists, just one that fails at call time instead of
        import time. plot.py is excluded because it is not reachable from
        ``utils/__init__``."""
        violations = [
            f"{path.name}:{line} imports {mod}"
            for path in LEAF_MODULES
            for line, mod in _any_storage_imports(path)
        ]
        assert violations == [], "utils must not import storage: " + (
            "; ".join(violations)
        )

    def test_plot_is_not_reachable_from_the_utils_package(self):
        """plot.py may import storage only because ``utils/__init__`` does not
        import plot. Re-exporting it would close the cycle again, so the
        exclusion is asserted rather than left as a comment."""
        init = UTILS_DIR / "__init__.py"
        tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))

        imported_submodules = {
            (node.module or "").lstrip(".")
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }

        assert "plot" not in imported_submodules
