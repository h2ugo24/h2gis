"""H2GIS command-line interface."""

import sys

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


@app.callback()
def _configure() -> None:
    """Configure the console and logging once for every h2mare command."""
    _use_utf8_console()
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
