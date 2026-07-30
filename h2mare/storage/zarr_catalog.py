"""
Zarr catalog management for tracking processed zarr datasets.

Opens Zarr data and builds store paths, delegating catalog bookkeeping to
:class:`~h2mare.storage.zarr_index.ZarrIndex`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.config import AppConfig, get_settings
from h2mare.storage.zarr_index import ZarrIndex, _variables_list
from h2mare.types import BBox, DateLike, DateRange, TimeResolution
from h2mare.utils.datetime_utils import normalize_dates
from h2mare.utils.labels import create_label_from_dataset
from h2mare.utils.paths import resolve_store_path
from h2mare.utils.spatial import sel_padded_bbox
from h2mare.validators import validate_time_resolution, validate_var_key


# ================ Convenience functions for quick access ==========================
def get_zarr_time_coverage(var_key: str) -> DateRange | None:
    catalog = ZarrCatalog(var_key)
    return catalog.get_time_coverage()


class ZarrCatalog:
    def __init__(
        self,
        var_key: str,
        *,
        time_resolution: TimeResolution = TimeResolution.YEAR,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        metadata_root: Optional[Path] = None,
        auto_refresh: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Manages a Parquet catalog of processed zarr datasets.
        The catalog tracks metadata about zarr files including spatial extent,
        temporal coverage, variables, and file locations.

        Args:
            var_key (str): Variable key that must exist in app_config.variables
            time_resolution: Temporal granularity for file storage ('year' or 'month'). Defaults to 'year'.
            app_config (Optional[AppConfig], optional): Application configuration. If None, loads from get_settings().
            store_root (Optional[Path]): Root directory for zarr files. If None, uses get_settings().STORE_ROOT or get_settings().ZARR_DIR.
            metadata_root (Optional[Path]): Root directory for catalog parquet files. If None, uses get_settings().METADATA_DIR.
            auto_refresh (bool): Automatically check for changes on access. Defaults to True.

        Raises:
            ValueError: If var_key not found in configuration or invalid period
            ValueError: If store_root doesn't exist

        Example:
            >>> catalog = ZarrCatalog("ssh")
        """
        # Load config
        self.app_config = app_config or get_settings().app_config
        self.var_key = validate_var_key(var_key, self.app_config)
        self.var_config = self.app_config.variables[var_key]

        self.time_resolution = validate_time_resolution(time_resolution)

        # Setup directories
        self.store_root = resolve_store_path(self.var_config, store_root)
        self.metadata_root = metadata_root or get_settings().METADATA_DIR

        # The catalog DataFrame and its refresh lifecycle live in ZarrIndex;
        # everything below either opens Zarr data or delegates to it.
        # auto_refresh/verbose are exposed as properties over the index rather
        # than copied here, so setting them on the facade cannot silently
        # diverge from the flags the catalog actually reads.
        self._index = ZarrIndex(
            self.var_key,
            self.var_config,
            store_root=self.store_root,
            metadata_root=self.metadata_root,
            time_resolution=self.time_resolution,
            auto_refresh=auto_refresh,
            verbose=verbose,
        )

    def __repr__(self) -> str:
        """String representation with useful info."""
        df = self.df
        time_cov = self.get_time_coverage()

        time_str = (
            f"{time_cov.start.date()} to {time_cov.end.date()}"
            if time_cov
            else "No data"
        )

        bbox = self.get_bbox()

        return (
            f"ZarrCatalog(\n"
            f"  var_key={self.var_key},\n"
            f"  time_resolution={self.time_resolution.value},\n"
            f"  files={df['path'].nunique() if not df.empty else 0},\n"
            f"  coverage={time_str},\n"
            f"  bbox={bbox.to_label() if bbox is not None else None},\n"
            f")"
        )

    # ==================== Index delegation ====================
    # The catalog DataFrame, its refresh lifecycle and every query answered
    # from it live in ZarrIndex. These forwards keep the public surface
    # unchanged for the call sites that construct a ZarrCatalog.

    @property
    def auto_refresh(self) -> bool:
        """Whether each ``.df`` access re-checks the store for changes."""
        return self._index.auto_refresh

    @auto_refresh.setter
    def auto_refresh(self, value: bool) -> None:
        self._index.auto_refresh = value

    @property
    def verbose(self) -> bool:
        """Whether catalog bookkeeping logs at all."""
        return self._index.verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self._index.verbose = value

    def _log(self, level: str, msg: str) -> None:
        """Log at *level* when verbose, silent otherwise."""
        self._index._log(level, msg)

    @property
    def catalog_path(self) -> Path:
        """Path to the catalog parquet file."""
        return self._index.catalog_path

    def exists(self) -> bool:
        """Check if catalog file exists."""
        return self._index.exists()

    def has_changes(self) -> bool:
        """Check if the store directory has changed since the last scan."""
        return self._index.has_changes()

    def refresh(self, force: bool = False) -> pd.DataFrame:
        """Refresh the catalog if needed. See :meth:`ZarrIndex.refresh`."""
        return self._index.refresh(force=force)

    @property
    def df(self) -> pd.DataFrame:
        """Catalog DataFrame. See :attr:`ZarrIndex.df`."""
        return self._index.df

    def reload(self) -> pd.DataFrame:
        """Force reload from disk or rescan."""
        return self._index.reload()

    def get_change_summary(self) -> dict[str, Any]:
        """Return a summary of added / removed / modified zarr files."""
        return self._index.get_change_summary()

    def map_dates_to_paths(
        self, dates: Union[DateLike, Sequence[DateLike]]
    ) -> dict[str, list[pd.Timestamp]]:
        """Map zarr file paths to their corresponding dates."""
        return self._index.map_dates_to_paths(dates)

    def get_paths_in_range(
        self,
        start_date: DateLike,
        end_date: DateLike,
    ) -> list[str]:
        """Get all zarr file paths that overlap with a date range."""
        return self._index.get_paths_in_range(start_date, end_date)

    def open_dataset(
        self,
        dates: DateLike | Sequence[DateLike] | None = None,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
        bbox: BBox | Sequence[float] | None = None,
        variables: str | Sequence[str] | None = None,
        chunks: dict | str | None = "auto",
    ) -> xr.Dataset:
        """
        Open zarr dataset(s) with flexible date and spatial selection.

        Supports two modes:
        1. Sparse dates: Provide `dates` for specific dates
        2. Date range: Provide `start_date` and/or `end_date`

        Args:
            dates: Specific dates to select (sparse mode)
            start_date: Start of date range (range mode)
            end_date: End of date range (range mode)
            bbox: Bounding box [xmin, ymin, xmax, ymax]
            variables: Variables to select (None = all)
            chunks: Chunking strategy for dask ('auto' or dict)

        Returns:
            Lazy xarray Dataset, or None if no data found

        Raises:
            ValueError: If neither dates nor start_date/end_date provided,
                       or if bbox format is invalid

        Example:
            >>> # Sparse dates
            >>> ds = catalog.open_dataset(dates=['2020-01-15', '2020-06-20'])
            >>>
            >>> # Date range
            >>> ds = catalog.open_dataset(
            ...     start_date='2020-01-01',
            ...     end_date='2020-12-31',
            ...     bbox=(-180, -90, 180, 90),
            ...     variables=['mld']
            ... )
        """
        # Validate inputs
        if dates is None and start_date is None and end_date is None:
            date_range = self.get_time_coverage()
            if date_range:
                start_date, end_date = date_range.start, date_range.end
            else:
                raise ValueError(
                    "Please provide sparse 'dates' or 'start_date/end_date' range"
                )

        if dates is not None and (start_date is not None or end_date is not None):
            raise ValueError(
                "Cannot use both 'dates' and 'start_date'/'end_date'. "
                "Use one mode or the other."
            )

        if bbox is not None:
            bbox = bbox if isinstance(bbox, BBox) else BBox.from_tuple(bbox)

        # Route to appropriate method
        if dates is not None:
            return self._open_sparse_dates(
                dates=dates,
                bbox=bbox,
                variables=variables,
                chunks=chunks,
            )
        else:
            return self._open_date_range(
                start_date=start_date,
                end_date=end_date,
                bbox=bbox,
                variables=variables,
                chunks=chunks,
            )

    def _open_sparse_dates(
        self,
        dates: DateLike | Sequence[DateLike],
        bbox: BBox | None,
        variables: str | Sequence[str] | None,
        chunks: dict | str | None,
    ) -> xr.Dataset:
        """Open dataset for specific sparse dates."""
        date_list = normalize_dates(dates)
        if not date_list:
            raise ValueError("No valid dates provided")

        # Get file paths
        path_mapping = self.map_dates_to_paths(date_list)

        if not path_mapping:
            raise FileNotFoundError(f"No zarr files contain dates: {date_list}")

        paths = list(path_mapping.keys())

        # Open datasets
        try:
            ds = xr.open_mfdataset(
                paths,
                engine="zarr",
                combine="by_coords",
                parallel=True,
                data_vars="minimal",
                coords="minimal",  # type: ignore[arg-type]
                compat="override",
                chunks=chunks,
                preprocess=lambda d: self._preprocess_dataset(d, bbox, variables),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open zarr files: {e}") from e

        # Normalize time coordinates
        ds = self._normalize_time(ds)

        # Select only requested dates
        requested_dates = pd.DatetimeIndex(date_list).normalize()
        available_dates = pd.DatetimeIndex(ds.time.values).normalize()

        valid_dates = requested_dates.intersection(available_dates)

        if len(valid_dates) == 0:
            raise FileNotFoundError(
                f"None of the requested dates found in dataset. "
                f"Requested: {requested_dates.tolist()}, "
                f"Available: {available_dates.tolist()}"
            )

        if len(valid_dates) < len(requested_dates):
            missing = requested_dates.difference(available_dates)
            self._log("warning", f"Missing dates: {missing.tolist()}")

        return ds.sel(time=valid_dates.tolist())

    def _open_date_range(
        self,
        start_date: DateLike | None,
        end_date: DateLike | None,
        bbox: BBox | None,
        variables: str | Sequence[str] | None,
        chunks: dict | str | None,
    ) -> xr.Dataset:
        """Open dataset for continuous date range."""

        # Get catalog
        df = self.df

        if df.empty:
            raise FileNotFoundError("Catalog is empty")

        # Default to full range if not specified
        start: pd.Timestamp = (
            df["start_date"].min()
            if start_date is None
            else pd.to_datetime(start_date).normalize()
        )
        end: pd.Timestamp = (
            df["end_date"].max()
            if end_date is None
            else pd.to_datetime(end_date).normalize()
        )

        if pd.isna(start) or pd.isna(end):
            raise ValueError(
                "Date range contains NaT — check catalog date columns for missing values"
            )

        # Warn if requested range extends beyond what the catalog covers
        available_start = pd.Timestamp(df["start_date"].min()).normalize()
        available_end = pd.Timestamp(df["end_date"].max()).normalize()

        if start_date is not None and start < available_start:
            self._log(
                "warning",
                f"[{self.var_key}] Requested start {start.date()} not available — "
                f"opening from {available_start.date()}",
            )
            start = available_start

        if end_date is not None and end > available_end:
            self._log(
                "warning",
                f"[{self.var_key}] Requested end {end.date()} not available — "
                f"opening until {available_end.date()}",
            )
            end = available_end

        # Get overlapping paths
        paths = self.get_paths_in_range(start, end)

        if not paths:
            raise FileNotFoundError(
                f"No zarr files found for range: {start.date()} to {end.date()}"
            )

        # Open datasets
        try:
            ds = xr.open_mfdataset(
                paths,
                engine="zarr",
                combine="by_coords",
                parallel=True,
                data_vars="minimal",
                coords="minimal",  # type: ignore[arg-type]
                compat="override",
                chunks=chunks,
                preprocess=lambda d: self._preprocess_dataset(d, bbox, variables),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open zarr files: {e}") from e

        # Normalize time
        ds = self._normalize_time(ds)

        # Select time range
        return ds.sel(time=slice(start, end))

    def build_file_path(
        self,
        ds: xr.Dataset,
        date_format: Literal["year", "date", "yearmonth"],
        store_root: Optional[Path] = None,
        name_key: Optional[str] = None,
    ) -> Path:
        """
        Build zarr file path as: store_root / {source}_{name_key}_{geo_extent}_{dates}.zarr

        name_key defaults to var_key; pass dataset_id_rep (or any string) to override.
        If store_root not provided, uses self.store_root.
        """
        label = create_label_from_dataset(ds, date_format=date_format)
        store_root = store_root or self.store_root
        key = name_key if name_key is not None else self.var_key
        return store_root / f"{self.var_config.source}_{key}_{label}.zarr"

    # ==================== Helper Methods ====================
    def _normalize_time(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Normalize time coordinates to midnight (00:00:00).

        Args:
            ds: Dataset with time coordinate

        Returns:
            Dataset with normalized time
        """
        if "time" not in ds.coords:
            return ds

        normalized_time = pd.to_datetime(ds["time"].values).normalize()
        return ds.assign_coords(time=normalized_time)

    def _preprocess_dataset(
        self,
        ds: xr.Dataset,
        bbox: BBox | None,
        variables: str | Sequence[str] | None,
    ) -> xr.Dataset:
        """
        Preprocess dataset: apply bbox and variable selection.

        This runs BEFORE datasets are combined in open_mfdataset.

        Args:
            ds: Input dataset
            bbox: Bounding box to apply
            variables: Variables to select

        Returns:
            Preprocessed dataset
        """
        # Select variables
        if variables is not None:
            var_list = [variables] if isinstance(variables, str) else list(variables)
            # Only select variables that exist
            available = set(ds.data_vars.keys())
            to_select = [v for v in var_list if v in available]

            if not to_select:
                self._log(
                    "warning",
                    f"None of requested variables found. "
                    f"Requested: {var_list}, Available: {list(available)}",
                )
            else:
                ds = ds[to_select]

        # Ensure lat is monotonically increasing (ERA5 comes north→south)
        if "lat" in ds.coords and ds.lat.values[0] > ds.lat.values[-1]:
            ds = ds.sortby("lat")

        # Apply spatial subset
        if bbox is not None:
            ds = self._apply_bbox(ds, bbox)

        return ds

    def _apply_bbox(
        self,
        ds: xr.Dataset,
        bbox: BBox,
    ) -> xr.Dataset:
        """
        Apply bounding box selection.

        Args:
            ds: Input dataset
            bbox: [xmin, ymin, xmax, ymax]

        Returns:
            Spatially subset dataset

        Raises:
            ValueError: If bbox format is invalid
        """
        # Determine coordinate names (support lat/lon or y/x)
        lat_coord = "lat" if "lat" in ds.coords else "y"
        lon_coord = "lon" if "lon" in ds.coords else "x"

        if lat_coord not in ds.coords or lon_coord not in ds.coords:
            self._log(
                "warning",
                f"Cannot apply bbox: missing coordinates. "
                f"Available: {list(ds.coords.keys())}",
            )
            return ds

        try:
            return sel_padded_bbox(
                ds, bbox.to_tuple(), lat_coord=lat_coord, lon_coord=lon_coord
            )
        except Exception as e:
            logger.error(f"Failed to apply bbox: {e}")
            return ds

    # ==================== Metadata Queries ====================
    def get_var_time_coverage(self, var_name: str) -> DateRange | None:
        """Time coverage for zarr files that contain *var_name* as a data variable."""
        return self._index.get_var_time_coverage(var_name)

    def get_vars_nonnull_end(self, var_names: Sequence[str]) -> dict[str, pd.Timestamp]:
        """
        Last date at which each variable holds a non-null value in the store.

        Unlike :meth:`get_var_time_coverage` — which reports the *file's* date
        span for any file listing the variable — this reflects where the variable
        actually has data. That distinction matters for compiled stores where a
        lagging variable is NaN-padded out to the global end by ``xr.merge``: the
        file end overstates its real coverage, whereas this method returns the
        last date with genuine data.

        Files are opened newest-first and scanning stops once every requested
        variable has been found, so the common case (all variables current)
        touches only the most recent file. Variables never found non-null — or
        absent from the store — are omitted from the result.

        Args:
            var_names: Variables to look up (typically one representative column
                per source var_key).

        Returns:
            Mapping of variable name to its last non-null date (normalised to
            midnight). Empty when the store has no data.
        """
        df = self.df
        if df.empty or "variables" not in df.columns or "path" not in df.columns:
            return {}

        files = df.sort_values("end_date", ascending=False).drop_duplicates(
            subset="path"
        )
        remaining = set(var_names)
        result: dict[str, pd.Timestamp] = {}

        for _, row in files.iterrows():
            if not remaining:
                break
            file_vars = _variables_list(row["variables"])
            cols_here = [c for c in remaining if c in file_vars]
            if not cols_here:
                continue
            try:
                ds = xr.open_zarr(row["path"], consolidated=False)
            except Exception as e:
                logger.warning(f"Could not open {row['path']} for non-null scan: {e}")
                continue
            try:
                if "time" not in ds.coords:
                    continue
                times = pd.to_datetime(ds["time"].values)
                # Reduce each variable to a per-time "has any data" mask, batched
                # into one lazy compute so the file is read in a single pass.
                lazy = {
                    c: ds[c].notnull().any(dim=[d for d in ds[c].dims if d != "time"])
                    for c in cols_here
                }
                mask_ds = xr.Dataset(lazy).compute()
                for c in cols_here:
                    m = mask_ds[c].values
                    if m.any():
                        result[c] = pd.Timestamp(times[m].max()).normalize()
                        remaining.discard(c)
            finally:
                ds.close()

        return result

    def get_variables(self) -> set[str]:
        """Get all unique variables across all zarr files."""
        return self._index.get_variables()

    def get_time_coverage(self) -> DateRange | None:
        """Get overall time coverage across all files."""
        return self._index.get_time_coverage()

    def get_bbox(self) -> BBox | None:
        """
        Get overall geographic extent (bbox) across all files.
        """
        bbox = self.var_config.bbox
        if bbox is None:
            return None
        if not isinstance(bbox, BBox):
            return BBox.from_tuple(bbox)
        return bbox

    def summary(self) -> dict:
        """
        Get summary statistics about the catalog.

        Returns:
            Dictionary with catalog statistics
        """
        df = self.df

        if df.empty:
            return {
                "num_files": 0,
                "time_coverage": None,
                "bbox": "No data",
                "period": self.time_resolution,
                "variables": set(),
                "total_timesteps": 0,
                "store_root": str(self.store_root),
                "catalog_path": str(self.catalog_path),
                "last_scanned": None,
            }

        time_cov = self.get_time_coverage()
        bbox = self.get_bbox()

        return {
            "num_files": df["path"].nunique(),
            "time_coverage": time_cov if time_cov is not None else "No data",
            "bbox": bbox if bbox is not None else "No data",
            "period": self.time_resolution,
            "variables": self.get_variables(),
            "total_timesteps": (
                df["num_timesteps"].sum().item()
                if "num_timesteps" in df.columns
                else None
            ),
            "store_root": str(self.store_root),
            "catalog_path": str(self.catalog_path),
            "last_scanned": (
                df["scanned_at"].max() if "scanned_at" in df.columns else None
            ),
        }

    # ==================== Migration Helpers ====================

    def backfill_provenance(self, rep_end_date: DateLike) -> int:
        """
        One-time migration: retroactively write provenance for Zarr files
        that pre-date automatic tracking by Netcdf2Zarr. Returns the number
        of files updated. See :func:`h2mare.storage.provenance.backfill_provenance`.
        """
        from h2mare.storage.provenance import backfill_provenance

        return backfill_provenance(self, rep_end_date)
