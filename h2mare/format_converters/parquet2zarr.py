"""
Rebuild a Zarr store from a Hive-partitioned Parquet store.

Config-free inverse of ``zarr2parquet``: read the long-format Parquet rows
(time/lat/lon + variable columns), pivot each time chunk back to a gridded
dataset, and write per-period Zarr files through the pipeline's standard
snap → chunk → append path.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.storage.coverage import split_time_range
from h2mare.storage.parquet_indexer import ParquetIndexer
from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import chunk_dataset, snap_grid_coords
from h2mare.types import DateLike, DateRange, TimeResolution
from h2mare.utils.labels import create_label_from_dataset
from h2mare.validators import validate_time_resolution

# Hive partition columns the store derives from the time column; they are
# storage bookkeeping, not data variables, so they never become Zarr arrays.
_PARTITION_COLS = ("year", "month", "day")


def convert_parquet_to_zarr(
    parquet_root: Path | str,
    out_dir: Path | str,
    *,
    name: str = "data",
    start_date: DateLike | None = None,
    end_date: DateLike | None = None,
    time_resolution: TimeResolution | str = TimeResolution.MONTH,
    date_format: Literal["year", "date", "yearmonth"] = "year",
    variables: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    layout: Literal["timeseries", "map"] = "timeseries",
    indexer_kwargs: Optional[dict] = None,
) -> list[Path]:
    """
    Rebuild per-period Zarr files from a Parquet store, without a configured
    ``var_key``.

    Config-free inverse of :func:`convert_zarr_to_parquet`. It reads the Parquet
    rows through :class:`ParquetIndexer` — so date/bbox/column filtering and
    partition-column handling match the rest of the pipeline — pivots each time
    chunk from long format back to a gridded ``time × lat × lon`` dataset, and
    writes it through the same Zarr-prep the pipeline uses:
    ``snap_grid_coords`` → ``chunk_dataset`` → ``write_append_zarr``.

    Output is split into per-period Zarr files matching the pipeline store
    layout. ``date_format`` controls the *file* granularity (one ``.zarr`` per
    year by default); ``time_resolution`` controls how much is read and pivoted
    *at once* (monthly by default, to bound memory). With the defaults each
    month is read, pivoted, and appended into its year's ``.zarr`` file.

    Args:
        parquet_root: Root of the Parquet store to read.
        out_dir: Directory the per-period ``.zarr`` files are written into
            (created if absent).
        name: Identity label used in the output filename
            (``{name}_{label}.zarr``) and in write/append logs and overlap
            resolution. It need **not** exist in config.
        start_date: Start of the window. Defaults to the store's first time step.
        end_date: End of the window. Defaults to the store's last time step.
        time_resolution: Granularity of each read/pivot batch. Defaults to
            ``TimeResolution.MONTH`` so a single batch fits in memory.
        date_format: Output file granularity / filename date label — ``"year"``
            (one file per year), ``"yearmonth"`` (per month), or ``"date"``
            (explicit start–end range).
        variables: Subset of data columns to read. ``None`` reads all. The
            time/lat/lon coordinate columns are always included regardless.
        bbox: Optional ``(min_lon, min_lat, max_lon, max_lat)`` spatial filter
            forwarded to the indexer scan.
        layout: Zarr chunk layout — ``"timeseries"`` (default, extraction) or
            ``"map"`` (interactive display). See :func:`chunk_dataset`.
        indexer_kwargs: Extra keyword arguments forwarded to
            :class:`ParquetIndexer` (e.g. non-canonical ``time_col`` /
            ``lon_col`` / ``lat_col``, or ``partition_by``).

    Returns:
        The list of distinct ``.zarr`` paths written (one per output period).

    Raises:
        ValueError: If the Parquet store is empty, or ``start_date`` is after
            ``end_date``.
    """
    time_resolution = validate_time_resolution(time_resolution)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    indexer = ParquetIndexer(Path(parquet_root), **(indexer_kwargs or {}))
    coverage = (
        indexer.get_time_coverage() if indexer._dataset_meta_initialized else None
    )
    if coverage is None:
        raise ValueError(
            f"Parquet store at {parquet_root} is empty — nothing to convert."
        )

    window = DateRange(
        start=pd.Timestamp(start_date) if start_date is not None else coverage.start,
        end=pd.Timestamp(end_date) if end_date is not None else coverage.end,
    )
    if window.start > window.end:
        raise ValueError(
            f"start_date ({window.start.date()}) must be before "
            f"end_date ({window.end.date()})"
        )

    periods = split_time_range(window, time_resolution)
    logger.info(
        f"Parquet → Zarr conversion: {window.start.date()} → {window.end.date()} "
        f"({len(periods)} chunk(s), name={name!r}) → {out_dir}"
    )

    coord_cols = (indexer.time_col, indexer.lat_col, indexer.lon_col)
    written: list[Path] = []

    for period in periods:
        ds: xr.Dataset | None = None
        try:
            df = indexer.load(
                dates=(period.start, period.end),
                bbox=bbox,
                columns=variables,
            )
            if df.is_empty():
                logger.debug(
                    f"  no rows for {period.start.date()} → {period.end.date()} "
                    "— skipping"
                )
                continue

            ds = _long_df_to_grid(df, coord_cols)
            ds = snap_grid_coords(ds)
            ds = chunk_dataset(ds, layout=layout)

            path = out_dir / f"{name}_{create_label_from_dataset(ds, date_format)}.zarr"
            write_append_zarr(name, ds, path)
            if path not in written:
                written.append(path)
        finally:
            if ds is not None:
                ds.close()
            del ds
            gc.collect()

    logger.success(
        f"Parquet → Zarr conversion complete: {len(written)} file(s) in {out_dir}"
    )
    return written


def _long_df_to_grid(df, coord_cols: tuple[str, str, str]) -> xr.Dataset:
    """
    Pivot a long-format ``(time, lat, lon, *vars)`` frame back to a gridded
    dataset.

    Drops Hive partition bookkeeping columns, sets the time/lat/lon coordinate
    index, and expands to the Cartesian grid via ``to_xarray`` — absent cells
    surface as NaN, reconstructing the gridded layout. Coordinate dims are
    renamed to the canonical ``time``/``lat``/``lon`` when they differ.
    """
    time_col, lat_col, lon_col = coord_cols
    drop = [c for c in _PARTITION_COLS if c in df.columns and c not in coord_cols]
    pdf = (df.drop(drop) if drop else df).to_pandas()
    pdf[time_col] = pd.to_datetime(pdf[time_col])

    ds = pdf.set_index([time_col, lat_col, lon_col]).to_xarray()
    rename = {time_col: "time", lat_col: "lat", lon_col: "lon"}
    rename = {k: v for k, v in rename.items() if k != v and k in ds.dims}
    return ds.rename(rename) if rename else ds
