"""
h2mare parquet2zarr — rebuild per-period Zarr files from a Parquet store.

Config-free inverse of the ``parquet`` command. Reads the long-format Parquet
rows, pivots each time chunk back to a gridded dataset, and writes per-period
``.zarr`` files (one per year by default) into the output directory using the
pipeline's standard snap → chunk → append path.

Examples
--------
Rebuild the whole store into per-year Zarr files:

    uv run h2mare parquet2zarr D:/parquet_store/h2ds D:/rebuilt_zarr --name h2ds

Restrict to a date range and a subset of variables:

    uv run h2mare parquet2zarr D:/parquet_store/h2ds D:/rebuilt_zarr \
        --name h2ds --start-date 2021-01-01 --end-date 2021-12-31 -v sst -v ssh

Write one file per month instead of per year:

    uv run h2mare parquet2zarr D:/parquet_store/h2ds D:/out --date-format yearmonth
"""

from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger

app = typer.Typer(help="Rebuild per-period Zarr files from a Parquet store.")


@app.command()
def parquet2zarr(
    parquet_root: Path = typer.Argument(..., help="Root of the Parquet store to read."),
    out_dir: Path = typer.Argument(
        ..., help="Directory to write per-period .zarr files into."
    ),
    name: str = typer.Option(
        "data",
        "--name",
        help="Identity label used in the output filename ({name}_{label}.zarr).",
    ),
    start_date: Optional[str] = typer.Option(
        None, "--start-date", help="Start date (YYYY-MM-DD). Defaults to store start."
    ),
    end_date: Optional[str] = typer.Option(
        None, "--end-date", help="End date (YYYY-MM-DD). Defaults to store end."
    ),
    var_keys: Optional[List[str]] = typer.Option(
        None,
        "--vars",
        "-v",
        help="Variable column(s) to read (repeat for multiple). Defaults to all.",
    ),
    file_period: str = typer.Option(
        "month",
        "--file-period",
        # Kept so existing scripts and cron entries keep working; --file-period
        # is the name shown in help.
        "--time-resolution",
        help="Read/append chunk granularity: 'year' or 'month'. Default 'month'.",
    ),
    date_format: str = typer.Option(
        "year",
        "--date-format",
        help="Output file granularity: 'year', 'yearmonth', or 'date'. Default 'year'.",
    ),
    layout: str = typer.Option(
        "timeseries",
        "--layout",
        help="Zarr chunk layout: 'timeseries' (extraction) or 'map' (display).",
    ),
) -> None:
    """Rebuild per-period Zarr files from a Hive-partitioned Parquet store."""

    if bool(start_date) ^ bool(end_date):
        typer.echo(
            "Error: --start-date and --end-date must be provided together.", err=True
        )
        raise typer.Exit(code=1)

    if start_date and end_date:
        import pandas as pd

        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            typer.echo(
                f"Error: --start-date ({start_date}) must not be after "
                f"--end-date ({end_date}).",
                err=True,
            )
            raise typer.Exit(code=1)

    if date_format not in ("year", "yearmonth", "date"):
        typer.echo(
            "Error: --date-format must be 'year', 'yearmonth', or 'date'.", err=True
        )
        raise typer.Exit(code=1)

    if layout not in ("timeseries", "map"):
        typer.echo("Error: --layout must be 'timeseries' or 'map'.", err=True)
        raise typer.Exit(code=1)

    from h2mare.format_converters.parquet2zarr import convert_parquet_to_zarr

    with logger.contextualize(var=name):
        convert_parquet_to_zarr(
            parquet_root,
            out_dir,
            name=name,
            start_date=start_date,
            end_date=end_date,
            file_period=file_period,
            date_format=date_format,  # type: ignore[arg-type]
            variables=list(var_keys) if var_keys else None,
            layout=layout,  # type: ignore[arg-type]
        )


if __name__ == "__main__":
    app()
