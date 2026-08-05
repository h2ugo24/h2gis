"""Package layering invariants.

``utils/`` is the leaf of the pipeline: every other package may import it, and
it may import none of them at module scope. Enforced by parsing the source
rather than by importing, so a violation is reported as the offending file and
line instead of an ImportError raised from wherever the cycle happens to close.
"""

from __future__ import annotations

import ast
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent.parent / "h2mare" / "utils"

# plot_records_on_field opens a ZarrCatalog, which is a genuine one-way call-time
# dependency: storage.zarr_catalog imports utils.labels, and utils/__init__ eagerly
# imports plot, so a module-level import here would close the cycle again. Deferring
# it into the function body is the intended escape hatch, not an oversight.
ALLOWED_FUNCTION_LEVEL = {("plot.py", "ZarrCatalog")}


def _module_level_imports(path: Path) -> list[tuple[int, str]]:
    """Every ``h2mare.storage`` import at module scope in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in tree.body:  # module scope only — nested imports are not walked
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
    def test_no_module_level_storage_imports(self):
        """Regression: utils/date_range.py imported storage.coverage and
        utils/plot.py imported storage.parquet_helpers, while storage imported
        utils in eight places — a genuine cycle that had already forced one
        function-level import to work around it."""
        violations = [
            f"{path.name}:{line} imports {mod}"
            for path in sorted(UTILS_DIR.glob("*.py"))
            for line, mod in _module_level_imports(path)
        ]
        assert violations == [], "utils must not import storage at module scope: " + (
            "; ".join(violations)
        )

    def test_function_level_exceptions_are_the_known_ones(self):
        """Deferred storage imports are allowed, but each one is a cycle waiting
        to be re-opened, so the set is pinned rather than left to grow."""
        found = set()
        for path in sorted(UTILS_DIR.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "h2mare.storage"
                ):
                    if (node.lineno, node.module) not in _module_level_imports(path):
                        for alias in node.names:
                            found.add((path.name, alias.name))
        assert found == ALLOWED_FUNCTION_LEVEL
