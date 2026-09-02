"""H2GIS command-line interface."""

import sys
import warnings

import typer

from h2mare.cli.audit import audit
from h2mare.cli.catalog import catalog
from h2mare.cli.compile import compile
from h2mare.cli.main import run
from h2mare.cli.nc2zarr import convert
from h2mare.cli.parquet2zarr import parquet2zarr
from h2mare.cli.zarr2parquet import parquet
from h2mare.utils.logging import configure_logging

app = typer.Typer(
    name="h2mare",
    help="Climate and ocean data pipeline — download, convert, and inspect.",
    no_args_is_help=True,
)


def _use_utf8_console() -> None:
    """Switch stdout/stderr to UTF-8 so command output can carry non-ASCII.

    Windows hands Python a cp1252 console, which raises UnicodeEncodeError on
    the arrows and dashes the summaries print — killing the command mid-output.
    Reconfigure in place so sinks already bound to these streams follow along.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        encoding = (getattr(stream, "encoding", "") or "").lower()
        if reconfigure is not None and encoding.replace("-", "") != "utf8":
            reconfigure(encoding="utf-8", errors="replace")


def _silence_known_benign_warnings() -> None:
    """Filter warnings that are noise for every h2mare command, by message.

    Installed here rather than at module import: importing h2mare as a library
    must leave the interpreter's filters untouched, which
    ``tests/test_warning_filters.py`` enforces after six modules once installed
    blanket ignores at import and swallowed the AVISO missing-credentials
    warning. Each entry is pinned to an exact message so a *new* warning still
    reaches the console. ``pyproject.toml``'s ``filterwarnings`` covers the same
    ground under pytest; keep the two lists in step.
    """
    from zarr.errors import ZarrUserWarning

    # zarr 3.x warns that consolidated metadata sits outside the Zarr v3 spec.
    # It is raised from `consolidate_metadata`, so it is a write-path warning
    # only: every `to_zarr` triggers it (xarray consolidates by default) and no
    # read ever does. h2mare cannot be bitten by what it warns about, because
    # every `open_zarr` call site passes `consolidated=False` and so never reads
    # the consolidated block. Appends re-consolidate, so it cannot go stale for
    # an outside reader either. Drops out if xarray stops consolidating by
    # default or zarr folds this into the spec.
    warnings.filterwarnings(
        "ignore",
        message="Consolidated metadata is currently not part in the Zarr format 3 specification",
        category=ZarrUserWarning,
    )


@app.callback()
def _configure() -> None:
    """Configure the console and logging once for every h2mare command."""
    _use_utf8_console()
    _silence_known_benign_warnings()
    configure_logging()


app.command("run", help="Download and convert data for one or more variable keys.")(run)
app.command(
    "convert", help="Convert downloaded NetCDF/GRIB files to Zarr (no download)."
)(convert)
app.command("catalog", help="Inspect ZarrCatalog metadata for a variable.")(catalog)
app.command(
    "audit",
    help="Report days missing from the middle of a store's own time span.",
)(audit)
app.command(
    "compile",
    help="Merge per-variable Zarr stores into the unified h2ds compiled dataset.",
)(compile)
app.command(
    "parquet",
    help="Convert compiled Zarr stores to Hive-partitioned Parquet.",
)(parquet)
app.command(
    "parquet2zarr",
    help="Rebuild per-period Zarr files from a Parquet store.",
)(parquet2zarr)
