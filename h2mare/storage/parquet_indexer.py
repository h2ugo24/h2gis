"""
Parquet files handling and manipulation.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Union

import polars as pl

if TYPE_CHECKING:
    from h2mare.storage.parquet_plotter import ParquetPlotter

from h2mare.types import BBox, DateRange

from .parquet_catalog import ParquetCatalog
from .parquet_store import ParquetStore


class ParquetIndexer:
    """
    Unified facade over ``ParquetStore`` (writes) and ``ParquetCatalog`` (reads).

    Public API is identical to the previous monolithic class; existing call-sites
    continue to work without modification. Construct ``ParquetStore`` or
    ``ParquetCatalog`` directly when only one concern is needed.
    """

    def __init__(
        self,
        parquet_root: str | Path,
        *,
        time_col: str = "time",
        lon_col: str = "lon",
        lat_col: str = "lat",
        target_file_mb: int = 256,
        partition_by: list[str] | None = None,
        column_groups: dict[str, str] | None = None,
    ):
        """
        Parquet data indexer.

        Args:
            parquet_root (str | Path): Root directory for parquet data
            time_col (str, optional): Time column name. Defaults to "time".
            lon_col (str, optional): Longitude column name. Defaults to "lon".
            lat_col (str, optional): Latitude column name. Defaults to "lat".
            target_file_mb (int, optional): Target size per Parquet file in MB. Defaults to 256.
            partition_by (list[str] | None, optional): Hive partition column names. Temporal
                components ("year", "month", "day") are auto-derived from the time column; all
                other columns must be present in the DataFrame passed to add_data().
                None means ["year", "month"].
            column_groups (dict[str, str] | None, optional): Maps a column to the
                name of whatever produces it (a var_key, for the pipeline). Used
                only to report missing columns by their producer instead of one
                by one. None means report raw column names.

        Raises:
            ValueError: If time, lat, lon cols not in data.
        """
        self._store = ParquetStore(
            parquet_root,
            time_col=time_col,
            lon_col=lon_col,
            lat_col=lat_col,
            target_file_mb=target_file_mb,
            partition_by=partition_by,
            column_groups=column_groups,
        )
        self._catalog = ParquetCatalog(self._store)

    def __repr__(self) -> str:
        if self._store.physical_schema is None:
            return ""

        time_cov = self._store.get_time_coverage()
        bbox = self._store.get_geoextent()

        return (
            f"ParquetIndexer(\n"
            f"  path={self._store.parquet_root},\n"
            f"  coverage={time_cov if time_cov is not None else None},\n"
            f"  bbox={bbox.to_label() if bbox is not None else None},\n"
            f"  n_columns={len(self._store.get_schema().keys())},\n"
            f")"
        )

    # --- Delegated attributes ---

    @property
    def parquet_root(self) -> Path:
        """Root directory of the Hive-partitioned store."""
        return self._store.parquet_root

    @property
    def time_col(self) -> str:
        """Name of the time column, as given to the constructor."""
        return self._store.time_col

    @property
    def lon_col(self) -> str:
        """Name of the longitude column, as given to the constructor."""
        return self._store.lon_col

    @property
    def lat_col(self) -> str:
        """Name of the latitude column, as given to the constructor."""
        return self._store.lat_col

    @property
    def physical_schema(self):
        """
        Column → polars dtype for the store as written, or None before the
        first write. Excludes the Hive partition columns, which live in the
        directory names rather than the files.
        """
        return self._store.physical_schema

    @property
    def physical_cols(self) -> set[str]:
        """Column names present in the parquet files themselves."""
        return self._store.physical_cols

    @property
    def partition_cols(self) -> set[str]:
        """Column names encoded in the Hive directory layout, not in the files."""
        return self._store.partition_cols

    @property
    def _partition_by(self) -> list[str]:
        return self._store._partition_by

    @property
    def _dataset_meta_initialized(self) -> bool:
        return self._store._dataset_meta_initialized

    # --- Write API ---

    def add_data(
        self,
        df,
        time_mode: Literal["date", "datetime"] = "date",
        fmt: str | None = None,
    ) -> None:
        """Write df into the store and invalidate the plot cache."""
        self._store.add_data(df, time_mode=time_mode, fmt=fmt)
        self._catalog._clear_plot_cache()

    # --- Read API ---

    def scan(
        self,
        dates: Optional[Union[list, tuple]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        columns: Optional[Union[str, list[str]]] = None,
    ) -> pl.LazyFrame:
        """
        Lazily query the store, filtered by date, bounding box and columns.

        Nothing is read until the frame is collected, and only the partitions
        the date filter selects are opened.

        Args:
            dates: Discrete ``list`` of dates, or a ``(start, end)`` tuple for a
                range. None reads every partition.
            bbox: ``(xmin, ymin, xmax, ymax)`` filter on the lon/lat columns.
            columns: Column or columns to select, *in addition to* time/lon/lat,
                which are always returned.

        Returns:
            A LazyFrame over the matching rows.

        Raises:
            RuntimeError: If the store has never been written to.
            FileNotFoundError: If no parquet files match.
        """
        return self._catalog.scan(dates=dates, bbox=bbox, columns=columns)

    def load(
        self,
        dates: Optional[Union[list, tuple]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
        columns: Optional[Union[str, list[str]]] = None,
    ) -> pl.DataFrame:
        """
        Eagerly read the store into memory. Same arguments as :meth:`scan`.

        Prefer :meth:`scan` for anything large or further filtered — this
        collects the whole result.
        """
        return self._catalog.load(dates=dates, bbox=bbox, columns=columns)

    def get_schema(self) -> dict:
        """Union of every partition's schema, as column → polars dtype."""
        return self._store.get_schema()

    def get_time_coverage(self) -> DateRange | None:
        """
        Span of the store's time column, or None before the first write.

        A frontier — first and last date — not a completeness check: a store
        holding only January 1st and December 31st reports the whole year.
        """
        return self._store.get_time_coverage()

    def get_var_coverage(
        self, columns: list[str] | None = None
    ) -> dict[str, DateRange]:
        """
        Per-column span of *non-null* values, as column → DateRange.

        Distinct from :meth:`get_time_coverage`, which spans rows regardless of
        content. Use it to spot a lagging variable whose trailing dates exist as
        rows but hold only nulls.

        Args:
            columns: Columns to report on. None means every data column.
        """
        return self._store.get_var_coverage(columns)

    def get_var_coverage_end(self, columns: list[str]) -> dict:
        """
        Last non-null date per column, as column → datetime.

        The same end dates :meth:`get_var_coverage` reports, found by reading
        year- or year/month-partitioned stores newest-first and stopping once
        every column has been seen non-null — usually one partition rather than
        the whole store. Columns never found non-null are omitted.
        """
        return self._store.get_var_coverage_end(columns)

    def get_var_backfill_start(
        self, columns: list[str], not_before: dict | None = None
    ) -> dict:
        """
        Earliest date each column has rows but only nulls, as column → datetime.

        The signature of an append that landed while a column's source lagged.
        :meth:`get_var_coverage_end` cannot see these: once a later append
        arrives with the column populated, the last non-null date jumps past the
        gap and an end-based backfill window strands it permanently.

        Args:
            columns: Columns to check.
            not_before: Per-column floor, to stop the scan walking back past a
                date a column is known never to have covered.
        """
        return self._store.get_var_backfill_start(columns, not_before=not_before)

    def get_geoextent(self) -> BBox | None:
        """Bounding box of the store's lon/lat columns, or None before the first write."""
        return self._store.get_geoextent()

    def _resolve_files(self, dates) -> list[Path]:
        return self._catalog._resolve_files(dates)

    # --- Visualization ---

    @cached_property
    def plot(self) -> "ParquetPlotter":
        """Visualization accessor. Use ``indexer.plot.time_series(...)``."""
        return self._catalog.plot
