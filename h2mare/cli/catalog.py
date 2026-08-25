"""
h2mare catalog — inspect ZarrCatalog metadata for a variable.

Shows coverage, file count, variables, and per-dataset breakdown from the
local Parquet index without opening any Zarr files.

Examples
--------
    # Summary for SST
    uv run h2mare catalog sst

    # Summary for all configured variables
    uv run h2mare catalog --all

    # Show individual catalog rows
    uv run h2mare catalog sst --rows
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from h2mare.config import get_settings
from h2mare.types import BBox

app = typer.Typer()


# Half a degree: wider than the half-cell offset between a requested edge and
# the first cell centre on any grid the pipeline handles (coarsest is 0.25°),
# narrower than any mismatch worth reporting — those run to whole degrees.
_BBOX_TOL_DEG = 0.5


def _bbox_differs(a: BBox, b: BBox) -> bool:
    """Whether two bboxes differ by more than grid-cell rounding."""
    return any(abs(x - y) > _BBOX_TOL_DEG for x, y in zip(a.to_tuple(), b.to_tuple()))


def _fmt_bbox(bbox: BBox) -> str:
    """Render a bbox as numeric bounds plus the label used in filenames."""
    return (
        f"{bbox.xmin:g}, {bbox.ymin:g} → {bbox.xmax:g}, {bbox.ymax:g}"
        f"  ({bbox.to_label()})"
    )


def _print_catalog(var_key: str, show_rows: bool) -> None:
    from h2mare.storage.zarr_catalog import ZarrCatalog

    try:
        # warn_if_missing is for callers about to write; this one only reads,
        # and its "will be created when data is added" advice is false here —
        # `moon` is computed at compile time and never gets a store. The
        # absence is reported on the Store line instead, where it belongs to
        # the variable being inspected rather than scrolling past as a log line.
        cat = ZarrCatalog(var_key, warn_if_missing=False)
    except Exception as e:
        typer.echo(f"  [{var_key}] Could not load catalog: {e}", err=True)
        return

    df = cat.df
    summary = cat.summary()
    cov = summary.get("time_coverage")

    typer.echo(f"\nZarrCatalog — {var_key.upper()}")
    typer.echo(f"  Files      : {summary['num_files']}")

    if cov and cov != "No data":
        typer.echo(f"  Coverage   : {cov.start.date()} → {cov.end.date()}")
    else:
        typer.echo("  Coverage   : No data")

    store_bbox = summary.get("store_bbox")
    config_bbox = summary.get("bbox")
    config_bbox = config_bbox if isinstance(config_bbox, BBox) else None

    if store_bbox is not None:
        typer.echo(f"  BBox       : {_fmt_bbox(store_bbox)}")
        # The configured bbox is what was *asked for*; the store reports cell
        # centres, so it sits a half-cell inside the request on every variable.
        # Only a difference beyond that is a request the store didn't honour.
        if config_bbox is not None and _bbox_differs(config_bbox, store_bbox):
            typer.echo(f"  BBox (cfg) : {_fmt_bbox(config_bbox)}")
    elif config_bbox is not None:
        typer.echo(f"  BBox (cfg) : {_fmt_bbox(config_bbox)}")
    else:
        typer.echo("  BBox       : —")

    variables = summary.get("variables") or set()
    typer.echo(f"  Variables  : {', '.join(sorted(variables)) if variables else '—'}")
    typer.echo(f"  Timesteps  : {summary.get('total_timesteps', '—')}")
    store_root = summary.get("store_root")
    missing = (
        "  (does not exist)" if store_root and not Path(store_root).exists() else ""
    )
    typer.echo(f"  Store      : {store_root or '—'}{missing}")
    typer.echo(f"  Catalog    : {summary.get('catalog_path', '—')}")
    last = summary.get("last_scanned")
    last_str = (
        last.strftime("%Y-%m-%d %H:%M:%S")
        if last is not None and pd.notna(last)
        else "—"
    )
    typer.echo(f"  Scanned    : {last_str}")

    if not df.empty and "dataset" in df.columns:
        typer.echo("\n  Dataset breakdown:")
        for dataset, group in df.groupby("dataset", sort=True):
            start = group["start_date"].min()
            end = group["end_date"].max()
            n_ts = (
                group["num_timesteps"].sum()
                if "num_timesteps" in group.columns
                else "—"
            )
            typer.echo(f"    {dataset}")
            typer.echo(f"      {start.date()} → {end.date()}  ({n_ts} timesteps)")

    if show_rows and not df.empty:
        cols = [
            c
            for c in [
                "filename",
                "dataset",
                "start_date",
                "end_date",
                "num_timesteps",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ]
            if c in df.columns
        ]
        typer.echo(f"\n  Rows:\n{df[cols].to_string(index=False)}")


def catalog(
    var_key: Optional[str] = typer.Argument(
        None,
        help="Variable key to inspect (e.g. sst, ssh). Omit with --all to show every variable.",
    ),
    all_vars: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Show catalog summary for all variables configured in config.yaml.",
    ),
    show_rows: bool = typer.Option(
        False,
        "--rows",
        "-r",
        help="Print individual catalog rows (filename, dataset, dates, timesteps).",
    ),
) -> None:
    """Inspect ZarrCatalog metadata: coverage, file count, and per-dataset breakdown."""

    if not var_key and not all_vars:
        typer.echo("Provide a variable key or use --all.", err=True)
        raise typer.Exit(code=1)

    keys = list(get_settings().app_config.variables.keys()) if all_vars else [var_key]

    for key in keys:
        if key not in get_settings().app_config.variables:
            typer.echo(
                f"Unknown variable key '{key}'. Available: {', '.join(get_settings().app_config.variables)}.",
                err=True,
            )
            continue
        _print_catalog(key, show_rows)


app.command()(catalog)
