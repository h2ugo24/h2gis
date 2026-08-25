"""
Importing h2mare must not mutate the interpreter's global warning filters.

Six modules used to call ``warnings.filterwarnings("ignore")`` at import time,
so importing the CLI silenced *every* warning in the process. The casualty that
mattered: ``config.load_app_config`` raises a ``RuntimeWarning`` when AVISO
variables are configured without credentials, and the downloader module that
installed the filter was always imported first — so the warning never reached
anyone. It also defeated the message-pinned ``filterwarnings`` in pyproject.toml,
whose whole premise is that a warning which shows up is new and worth reading.

These run in a subprocess on purpose. pytest wraps each test in
``catch_warnings()`` and re-applies the ini filters, which discards anything a
module installed at import — so an in-process assertion would pass either way.
"""

import subprocess
import sys

# Modules that owned a blanket filter, plus the CLI that transitively imports them.
_IMPORT_TARGETS = [
    "h2mare.cli",
    "h2mare.format_converters.netcdf2zarr",
    "h2mare.downloader.cmems_downloader",
    "h2mare.downloader.cds_downloader",
    "h2mare.downloader.aviso_downloader",
    "h2mare.processing.core.cds",
    "h2mare.processing.compiler",
]

_PROBE = """
import warnings, importlib, sys
importlib.import_module({module!r})
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    # Re-apply whatever the import left behind, then check a warning survives.
    for entry in warnings.filters:
        if entry[0] == "ignore" and entry[2] is Warning and entry[1] is None:
            print("BLANKET_IGNORE_INSTALLED")
            sys.exit(0)
print("OK")
"""


def _run_probe(module: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


class TestNoGlobalWarningSuppression:
    def test_importing_cli_leaves_warnings_deliverable(self):
        code = (
            "import warnings, h2mare.cli\n"
            "with warnings.catch_warnings(record=True) as caught:\n"
            "    warnings.warn('canary', RuntimeWarning)\n"
            "print('DELIVERED' if caught else 'SWALLOWED')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "DELIVERED"

    def test_no_module_installs_a_blanket_ignore(self):
        offenders = [m for m in _IMPORT_TARGETS if "BLANKET" in _run_probe(m)]
        assert not offenders, (
            f"These modules install a global ignore-everything warning filter "
            f"at import: {offenders}. Suppress specific warnings by message in "
            f"pyproject.toml, or fix the code that raises them."
        )
