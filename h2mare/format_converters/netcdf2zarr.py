"""
Process downloaded netcdf/grib raw data to zarr files
"""

from __future__ import annotations

import json
import re
import time
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.config import AppConfig, get_settings
from h2mare.format_converters.base import BaseConverter
from h2mare.processing.registry import PROCESSORS
from h2mare.storage.recovery import recover_zarr_store
from h2mare.storage.storage import write_append_zarr
from h2mare.storage.xarray_helpers import chunk_dataset, rename_dims, snap_grid_coords
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import DateLike, DateRange, TimeResolution
from h2mare.utils.files_io import filter_raw_files, safe_move_files, safe_rmtree
from h2mare.utils.paths import resolve_download_path
from h2mare.validators import validate_time_resolution, validate_var_key

warnings.filterwarnings("ignore")


def _dataset_dates(ds: xr.Dataset) -> pd.DatetimeIndex:
    """Unique calendar days on a dataset's time axis, sorted. Empty if time-less."""
    if "time" not in ds.coords:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(ds.time.values).normalize().unique().sort_values()


def _format_date_blocks(dates: pd.DatetimeIndex, max_blocks: int = 8) -> str:
    """Render a date index as contiguous blocks: '2025-06-02, 2025-07-10→2025-07-14'."""
    if len(dates) == 0:
        return "none"
    blocks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = prev = dates[0]
    for day in dates[1:]:
        if (day - prev).days > 1:
            blocks.append((start, prev))
            start = day
        prev = day
    blocks.append((start, prev))

    shown = [
        str(a.date()) if a == b else f"{a.date()}→{b.date()}"
        for a, b in blocks[:max_blocks]
    ]
    if len(blocks) > max_blocks:
        shown.append(f"… and {len(blocks) - max_blocks} more block(s)")
    return ", ".join(shown)


def convert_netcdf_to_zarr(
    paths: Path | str | Iterable[Path | str],
    out_path: Path | str,
    *,
    name: str = "data",
    processor: Callable[[xr.Dataset], xr.Dataset] | None = None,
    apply_rename: bool = True,
    open_kwargs: Optional[dict] = None,
) -> Path:
    """
    Convert one or more NetCDF/GRIB files to a single Zarr store, without a
    configured ``var_key``.

    This is the config-free counterpart to :class:`Netcdf2Zarr`. It applies the
    same generic Zarr-prep the pipeline uses — open → (optional ``rename_dims``)
    → (optional ``processor``) → ``snap_grid_coords`` → ``chunk_dataset`` →
    ``write_append_zarr`` — but driven entirely by arguments instead of
    ``config.yaml``. Use it to convert arbitrary files (a single path or a list)
    that are not registered as a variable.

    Args:
        paths: One path, or an iterable of paths, to ``.nc``/``.grib`` files.
            NetCDF and GRIB may be mixed; the engine is auto-detected from the
            first file.
        out_path: Destination ``.zarr`` path. If it already exists, the data is
            appended using the store's standard overlap semantics.
        name: Identity label used in the write/append logs and overlap
            resolution. It need **not** exist in config.
        processor: Optional callable applied after rename and before snap/chunk —
            the same slot a registry processor occupies in
            ``Netcdf2Zarr.process_dataset``. To reuse a registered processor,
            wrap it: ``processor=lambda ds: PROCESSORS["sst"](ds, cfg, "sst")``.
        apply_rename: Apply ``rename_dims`` (``longitude/latitude/valid_time`` →
            ``lon/lat/time``). Set ``False`` when the files already use canonical
            dim names.
        open_kwargs: Extra keyword arguments forwarded to ``xr.open_mfdataset``.

    Returns:
        The ``out_path`` that was written.

    Raises:
        FileNotFoundError: If no input paths are given.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = sorted(Path(p) for p in paths)
    if not files:
        raise FileNotFoundError("convert_netcdf_to_zarr: no input files given")

    out_path = Path(out_path)
    engine = "cfgrib" if files[0].suffix.lower() in {".grib", ".grb"} else "netcdf4"

    logger.info(
        f"Converting {len(files)} file(s) → {out_path.name} "
        f"(engine={engine}, name={name!r})"
    )

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        engine=engine,
        decode_timedelta=True,
        chunks={"time": 1, "depth": 1},
        **(open_kwargs or {}),
    )
    try:
        if apply_rename:
            ds = rename_dims(ds)
        if processor is not None:
            ds = processor(ds)
        ds = snap_grid_coords(ds)
        ds = chunk_dataset(ds)
        write_append_zarr(name, ds, out_path)
    finally:
        ds.close()

    return out_path


class Netcdf2Zarr(BaseConverter):
    def __init__(
        self,
        var_key: str,
        *,
        app_config: Optional[AppConfig] = None,
        store_root: Optional[Path] = None,
        download_root: Optional[Path] = None,
        time_resolution: TimeResolution = TimeResolution.YEAR,
        date_format: Literal["year", "date", "yearmonth"] = "year",
    ) -> None:
        """
        Initializes the setup to process downloaded raw data into Zarr files.

        Args:
            var_key: The key for the variable to be processed
            app_config: The application's configuration object.
            store_root: The root directory where the processed Zarr files will be stored.
            download_root: The root directory containing the downloaded raw data files to be processed.
            time_resolution: Temporal granularity ('year' or 'month') for file storage. Defaults to 'year'.
            date_format: string date format for output file name.
        """

        self.app_config = app_config or get_settings().app_config
        self.var_key = validate_var_key(var_key, self.app_config)
        self.var_config = self.app_config.variables[self.var_key]

        # Single source of truth for the archive-vs-delete decision, read by
        # both _archive_raw_files and _cleanup_period_files so they cannot drift.
        self.archive_raw = self.var_config.archive_raw

        self.download_root = resolve_download_path(self.var_config, download_root)

        self.time_resolution = validate_time_resolution(time_resolution)
        self.date_format: Literal["year", "date", "yearmonth"] = date_format

        self.catalog = ZarrCatalog(self.var_key, store_root=store_root)
        self.store_root = self.catalog.store_root

    def run(
        self,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
    ) -> bool:
        """
        Convert downloaded raw files to Zarr.

        Args:
            start_date: Restrict the conversion to this window. Both dates must
                be given together; when omitted every downloaded file is
                converted, which is the default resume behaviour.
            end_date: End of the conversion window.

        Restricting the window is what makes it possible to re-convert one
        period from raw files already on disk, without re-downloading.
        """
        t0 = time.perf_counter()
        logger.info(
            f"Initializing Netcdf -> Zarr conversion for variable key: {self.var_key.upper()}"
        )

        window = (
            DateRange(pd.to_datetime(start_date), pd.to_datetime(end_date))
            if start_date is not None and end_date is not None
            else None
        )
        if window is not None:
            logger.info(f"Restricting conversion to {window}")

        # Reconcile any zarr write interrupted by a hard kill before gap
        # detection reads the store, so a half-written store is not trusted and
        # data stranded in a backup is restored.
        recover_zarr_store(self.store_root)

        # Trajectory-format variables (e.g. eddies) require spatial binning
        # before zarr storage — bypass the standard open_mfdataset pipeline.
        if self.var_config.trajectory_format:
            self._process_eddies(start_date, end_date)
            self.catalog.refresh(force=True)
            logger.success(
                f"Conversion complete (trajectory) in {time.perf_counter() - t0:.1f}s"
            )
            return True

        file_groups = self._group_map(groupby=self.time_resolution, window=window)

        for period, paths in file_groups.items():
            self._process_period(period, paths)

        self.catalog.refresh(force=True)
        self._cleanup_downloads()
        logger.success(
            f"Conversion complete: {len(file_groups)} period(s) "
            f"in {time.perf_counter() - t0:.1f}s"
        )
        return True

    # ========= DATA PREPARATION FUNCTIONS =========

    def _group_map(
        self, groupby: TimeResolution, window: Optional[DateRange] = None
    ) -> dict[int | tuple[int, int], list[Path]]:
        """
        Group paths of nc files by period i.e groupby str.
        Returns a dictionary where the key is either an integer year or a
        (year, month) tuple and the value is a lis of paths to open in xr.open_mfdataset.

        Args:
            groupby: How to aggregate files before opening.

        Conventions:
            * GRIBs are read with ``engine="cfgrib"``.
            * NetCDFs are read with the default ``netcdf4`` engine.
            * The function automatically detects which engine to use per file,
              so you can mix formats in the same directory without trouble.

        Returns:
            Mapping of group key/period → files path for the selected period.
        """

        file_map = self._get_file_date_series()
        if file_map.empty:
            return {}

        if window is not None:
            file_map = file_map[
                (file_map.index >= window.start) & (file_map.index <= window.end)
            ]
            if file_map.empty:
                logger.warning(
                    f"[{self.var_key}] No downloaded files fall within {window}"
                )
                return {}

        if groupby == TimeResolution.YEAR:
            groups = file_map.groupby(file_map.index.year)  # type: ignore
        elif groupby == TimeResolution.MONTH:
            groups = file_map.groupby([file_map.index.year, file_map.index.month])  # type: ignore
        else:
            raise ValueError("groupby must be 'year' or 'month'")

        out = {}
        for key, group in groups:
            paths = sorted(set(group))
            out[key] = [Path(p) for p in paths]
        return out

    def _get_file_date_series(self) -> pd.Series:
        files = self._get_downloaded_files()
        records = []

        for f in files:
            for d in self._parse_file_dates(f):
                records.append((d, f))

        if not records:
            return pd.Series(dtype="object")

        dates, paths = zip(*records)
        return pd.Series(paths, index=pd.DatetimeIndex(dates)).sort_index()

    def _get_downloaded_files(self) -> list[Path]:
        """
        Return all downloaded files matching the pattern (nc and grib)

         Raises:
        FileNotFoundError: If no ``*.nc`` or ``*.grib`` files are present in
                           :pyattr:`self.down_dir`.
        """
        files = list(self.download_root.rglob("*.nc")) + list(
            self.download_root.rglob("*.grib")
        )
        if not files:
            raise FileNotFoundError(
                f"No downloaded NetCDF/GRIB files found in {self.download_root!s}"
            )

        files = filter_raw_files(files, self.var_config)
        if not files:
            raise FileNotFoundError(
                f"No downloaded files in {self.download_root!s} match "
                f"raw_include={self.var_config.raw_include!r}"
            )
        return sorted(files)

    def _parse_file_dates(self, file: Path) -> list[pd.Timestamp]:
        if self.var_config.pattern is None:
            return []
        match = re.search(self.var_config.pattern, file.name)
        if not match:
            return []

        if self.var_config.filename_date_range:
            start, end = map(pd.to_datetime, match.groups())
            return list(pd.date_range(start, end, freq="D"))

        return [pd.to_datetime("-".join(match.groups()))]

    def _get_file_date_bounds(
        self, file: Path
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Return (start, end) date for a raw file without expanding the full date range."""
        if self.var_config.pattern is None:
            return None
        match = re.search(self.var_config.pattern, file.name)
        if not match:
            return None
        if self.var_config.filename_date_range:
            start, end = map(pd.to_datetime, match.groups())
            return start, end
        date = pd.to_datetime("-".join(match.groups()))
        return date, date

    def _read_manifest(self) -> list[dict]:
        """Read the download manifest written by CMEMSDownloader, or return [] if absent."""
        manifest_path = self.download_root / "h2mare_manifest.json"
        if not manifest_path.exists():
            return []
        try:
            return json.loads(manifest_path.read_text())
        except Exception as e:
            logger.warning(f"Could not read download manifest: {e}")
            return []

    def _write_provenance(self, zarr_path: Path, paths: list[Path]) -> None:
        """
        Write source provenance as a zarr root attribute (``source_datasets``).
        Skipped silently when no download manifest is available.
        """
        manifest = self._read_manifest()
        if not manifest:
            return

        tasks = [
            {
                "dataset_id": e["dataset_id"],
                "dataset_type": e["dataset_type"],
                "start": pd.to_datetime(e["start"]),
                "end": pd.to_datetime(e["end"]),
            }
            for e in manifest
        ]

        # For each raw file, find its date bounds and match to the correct task
        dataset_info: dict[str, dict] = {}
        for file_path in paths:
            bounds = self._get_file_date_bounds(file_path)
            if bounds is None:
                continue
            f_start, f_end = bounds

            matched = next(
                (t for t in tasks if t["start"] <= f_start <= t["end"]),
                None,
            )
            if matched is None:
                logger.warning(
                    f"Could not match {file_path.name} (start={f_start.date()}) "
                    "to any manifest task — skipping provenance for this file"
                )
                continue

            did = matched["dataset_id"]
            if did not in dataset_info:
                dataset_info[did] = {
                    "dataset_type": matched["dataset_type"],
                    "start_date": f_start,
                    "end_date": f_end,
                }
            else:
                dataset_info[did]["start_date"] = min(
                    dataset_info[did]["start_date"], f_start
                )
                dataset_info[did]["end_date"] = max(
                    dataset_info[did]["end_date"], f_end
                )

        if not dataset_info:
            return

        records = sorted(
            [
                {
                    "dataset_id": did,
                    "dataset_type": info["dataset_type"],
                    "start_date": info["start_date"].strftime("%Y-%m-%d"),
                    "end_date": info["end_date"].strftime("%Y-%m-%d"),
                }
                for did, info in dataset_info.items()
            ],
            key=lambda r: r["start_date"],
        )

        import zarr

        root = zarr.open_group(str(zarr_path), mode="r+")
        root.attrs["source_datasets"] = json.dumps(records)
        logger.debug(f"Wrote provenance to zarr attrs: {zarr_path.name}")

    # ========= PROCESSING FUNCTIONS =========
    def _process_eddies(
        self,
        start_date: DateLike | None = None,
        end_date: DateLike | None = None,
    ) -> None:
        import h2mare.processing.core.aviso as aviso

        try:
            ed_processor = aviso.EDDIESProcessor(
                store_root=self.store_root,
                download_root=self.download_root,
                time_resolution=self.time_resolution,
                date_format=self.date_format,
            )
            ed_processor.run(start_date, end_date)
            # Staging is download cleanup: it files freshly downloaded raw data
            # into the store. A windowed convert re-reads files that are
            # already wherever they belong, so there is nothing to file away.
            if start_date is None and end_date is None:
                self._stage_eddies_to_store(self.download_root)
            else:
                logger.debug(
                    f"[{self.var_key}] Windowed conversion — leaving raw files "
                    "where they are"
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed processing data for var_key {self.var_key}"
            ) from e

    def _stage_eddies_to_store(self, download_root: Path) -> None:
        """
        Move raw eddies NetCDF files from download subfolders to the store.

        REP files (``download_root/rep/``) are moved additively — existing
        store files are preserved alongside new ones.

        NRT files (``download_root/nrt/``) replace the store entirely — all
        previous NRT files are deleted before the new ones are moved in.
        This ensures only the most recent NRT snapshot is kept.

        Falls back to a flat move to ``store_root`` when no rep/nrt subfolders
        are present (backward compatibility).

        Does nothing when the raw files already live in the store, i.e. when
        ``download_root`` and ``store_root`` are the same directory — which is
        the case whenever ``convert --in-dir`` is pointed at an ``archive_raw``
        store. Staging there would move every file onto itself. The generic
        path guards this the same way in ``_archive_raw_files``.
        """
        if download_root.resolve() == self.store_root.resolve():
            logger.debug(
                f"[{self.var_key}] Raw files already in the store "
                f"({self.store_root}) — nothing to stage"
            )
            return

        rep_src = download_root / "rep"
        nrt_src = download_root / "nrt"

        if rep_src.exists() or nrt_src.exists():
            if rep_src.exists():
                rep_dst = self.store_root / "rep"
                rep_dst.mkdir(parents=True, exist_ok=True)
                safe_move_files(list(rep_src.glob("*.nc")), rep_dst)
                logger.debug(f"Moved REP eddies files to {rep_dst}")

            if nrt_src.exists():
                nrt_dst = self.store_root / "nrt"
                nrt_dst.mkdir(parents=True, exist_ok=True)

                # Move the new snapshot in first, then drop whatever it did not
                # replace. Clearing the destination up front means a move that
                # fails half way leaves neither copy.
                incoming = {p.name for p in nrt_src.glob("*.nc")}
                safe_move_files(list(nrt_src.glob("*.nc")), nrt_dst)
                logger.info(f"Moved new NRT eddies files to {nrt_dst}")

                stale = [p for p in nrt_dst.glob("*.nc") if p.name not in incoming]
                for old_file in stale:
                    old_file.unlink()
                if stale:
                    logger.info(
                        f"Cleared {len(stale)} stale NRT eddies file(s) from {nrt_dst}"
                    )
        else:
            # No subfolders — flat move (legacy layout)
            paths = list(download_root.rglob("*.nc"))
            if paths:
                safe_move_files(paths, self.store_root)

    def _process_period(self, period, paths: list[Path]) -> None:
        logger.info(f"Processing period (year/year-month): {period}")

        ds = self._open_dataset(paths)

        try:
            ds = self.process_dataset(ds)
            path = self.catalog.build_file_path(ds, self.date_format)
            # Read the axis before the write: write_append_zarr closes ds.
            written = _dataset_dates(ds)
            write_append_zarr(self.var_key, ds, path)
            # Deliberately before _archive_raw_files and _cleanup_period_files:
            # a failure here must leave the raw files where they are, so the
            # period can simply be re-converted once the gap is understood.
            self._verify_written_dates(
                path, written, self._expected_dates(period, paths), period
            )
            try:
                self._write_provenance(path, paths)
            except Exception as e:
                logger.warning(f"Could not write provenance for {path.name}: {e}")
            ds.close()
            del ds
            self._archive_raw_files(period, paths)
            self._cleanup_period_files(paths)

        except Exception as e:
            raise RuntimeError(
                f"Failed processing data for var_key {self.var_key}"
            ) from e
        # finally:
        #    ds.close()

    # ========= WRITE VERIFICATION =========

    def _period_bounds(self, period: int | tuple[int, int]) -> DateRange:
        """Calendar span of a group key produced by :meth:`_group_map`."""
        if isinstance(period, tuple):
            year, month = period
            start = pd.Timestamp(year=year, month=month, day=1)
            return DateRange(start=start, end=start + pd.offsets.MonthEnd(0))
        start = pd.Timestamp(year=period, month=1, day=1)
        return DateRange(start=start, end=pd.Timestamp(year=period, month=12, day=31))

    def _expected_dates(
        self, period: int | tuple[int, int], paths: list[Path]
    ) -> pd.DatetimeIndex:
        """
        Daily calendar this period was *asked* to deliver, for the lag warning.

        Taken from the download manifest rather than the files on disk, and the
        distinction is the whole point: a file that failed to download is
        absent from disk, so an expectation derived from disk moves to match
        the loss and reports success. The manifest states what was requested,
        and is written before conversion runs.

        Falls back to the dates the raw filenames claim when no manifest exists
        (a re-convert from archived files). That fallback cannot see a missing
        file — which is why it only ever drives a warning, never the error.
        """
        bounds = self._period_bounds(period)
        spans = [
            (pd.to_datetime(e["start"]), pd.to_datetime(e["end"]))
            for e in self._read_manifest()
            if e.get("start") and e.get("end")
        ]
        if not spans:
            dates = pd.DatetimeIndex(
                sorted({d for p in paths for d in self._parse_file_dates(p)})
            )
            return dates[(dates >= bounds.start) & (dates <= bounds.end)]

        start = max(min(s for s, _ in spans), bounds.start)
        end = min(max(e for _, e in spans), bounds.end)
        if start > end:
            return pd.DatetimeIndex([])
        return pd.date_range(start, end, freq="D")

    def _stored_dates(self, path: Path) -> pd.DatetimeIndex:
        """Time axis of the Zarr just written. Coordinate read only — no data."""
        try:
            with xr.open_zarr(path, consolidated=False) as stored:
                return _dataset_dates(stored)
        except Exception as e:
            logger.warning(f"Could not read back the time axis of {path.name}: {e}")
            return pd.DatetimeIndex([])

    def _verify_written_dates(
        self,
        path: Path,
        written: pd.DatetimeIndex,
        expected: pd.DatetimeIndex,
        period: int | tuple[int, int],
    ) -> None:
        """
        Reconcile the dates now in the store against the dates asked for.

        Raises when a day is missing from the *middle* of the span this
        conversion just wrote: the provider published the days on either side,
        both are in the store, and the one between them is not. There is no
        benign reading of that, and it is the one shape of loss that nothing
        downstream can recover — store coverage is a min/max watermark that an
        interior hole cannot move, so the year goes on reporting itself full.

        A shortfall at either end is only warned about, because provider lag
        legitimately produces one every time the requested window runs to today.

        Deliberately silent on days that are present but entirely null. That is
        what a genuine source gap looks like (chl has three in 1999 alone), and
        a check that fires on those every year is one people learn to ignore —
        at which point it protects nothing.

        Raises:
            RuntimeError: If a day is missing from inside the written span.
        """
        if not self.var_config.expect_daily:
            logger.debug(f"[{self.var_key}] expect_daily=False — skipping date check")
            return
        if len(written) < 2:
            return

        stored = self._stored_dates(path)
        if len(stored) == 0:
            return

        span_start, span_end = written[0], written[-1]
        missing = pd.date_range(span_start, span_end, freq="D").difference(stored)
        if len(missing):
            raise RuntimeError(
                f"[{self.var_key}] period {period}: {len(missing)} day(s) missing "
                f"from {path.name} inside the range just written "
                f"({span_start.date()} → {span_end.date()}): "
                f"{_format_date_blocks(missing)}. Raw files were left in place — "
                f"re-run the download for those dates, then re-convert."
            )

        shortfall = expected[
            (expected < span_start) | (expected > span_end)
        ].difference(stored)
        if len(shortfall):
            logger.warning(
                f"[{self.var_key}] period {period}: {len(shortfall)} requested day(s) "
                f"not delivered outside the written span: "
                f"{_format_date_blocks(shortfall)}. Ordinary provider lag at the "
                f"tail; anything else is a gap `h2mare audit` will keep reporting."
            )

    def _open_dataset(self, paths: list[Path]) -> xr.Dataset:
        """Open a group of files as a single dataset."""
        first_ext = paths[0].suffix.lower()
        engine = "cfgrib" if first_ext in {".grib", ".grb"} else "netcdf4"

        def preprocess(ds: xr.Dataset) -> xr.Dataset:
            import h2mare.processing.core.cds as cds

            ds = rename_dims(ds)
            ds = cds.merge_time_step(ds)
            return cds._get_ds_for_month(ds)

        return xr.open_mfdataset(
            sorted(paths),
            combine="by_coords",
            engine=engine,
            decode_timedelta=True,
            chunks={"time": 1, "depth": 1},
            preprocess=(preprocess if self.var_config.merge_time_step else None),
        )

    def process_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply dataset-specific processing depending on downloader and variable key."""
        if self.var_config.source != "cds":
            ds = rename_dims(ds)

        processor = PROCESSORS.get(self.var_key)
        if processor:
            ds = processor(ds, self.var_config, self.var_key)

        # Snap lon/lat to a canonical grid so float-noise drift between a source's
        # reprocessed periods can't union into a doubled axis on read/append.
        ds = snap_grid_coords(ds)

        return chunk_dataset(ds)

    # ========= CLEANUP FUNCTIONS =========

    def _archive_raw_files(
        self, period: int | tuple[int, int], paths: list[Path], retries=10, delay=0.5
    ) -> None:
        """
        Move files for period folders (year or month as defined in period) if store_root is different from download root.
        Runs when ``archive_raw`` is set (by default aviso (fsle only) and cds data).
        """
        if not self.archive_raw:
            return None

        if self.download_root != self.store_root:
            logger.info(
                f"Archiving raw files from {self.download_root} to {self.store_root}"
            )
            dest_dir = self.store_root / self._resolve_string(period)
            dest_dir.mkdir(parents=True, exist_ok=True)

            safe_move_files(paths, dest_dir, retries=retries, delay=delay)

    def _cleanup_period_files(self, paths: list[Path]) -> None:
        """
        Delete a period's raw files once it has been converted successfully.

        Without this, a failure on a later period aborts ``run()`` with every
        raw file still on disk, so the next run re-globs and reprocesses *all*
        periods (correct, via the overlap resolver, but wasteful — and it
        rewrites already-good stores, widening the crash-exposure window).
        Removing each period's inputs as it completes makes the convert step
        incrementally resumable.

        Variables whose raw files are archived into the store (``archive_raw``;
        by default ``cds``/``aviso``) are handled by :meth:`_archive_raw_files`
        and skipped here; so is the in-place layout where downloads live in the
        store itself.
        """
        if self.archive_raw:
            return
        if self.download_root == self.store_root:
            return
        for p in paths:
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                logger.debug(f"Could not remove raw file {p}: {e}")

    def _cleanup_downloads(self) -> None:
        """Remove raw data from dowloads folder if download_root is different from store_root to avoid cluttering downloads with raw files."""
        if self.download_root != self.store_root:
            try:
                logger.debug(f"Removing raw files from {self.download_root}")
                safe_rmtree(self.download_root)
            except OSError:
                logger.exception(f"Could not remove {self.download_root}")

    def _resolve_string(self, period: int | tuple[int, int]) -> str:
        """
        Resolve year/yearmonth folders for file move.

        Args:
            period (int | tuple[int, int]): year (int) of tuple (year, month) derived from _group_map function.

        Raises:
            ValueError: If period is not int nor tuple

        Returns:
            str: with year or year/month
        """
        if isinstance(period, int):
            return str(period)
        elif isinstance(period, tuple) and len(period) == 2:
            return rf"{str(period[0])}\{str(period[1])}"
        raise ValueError("Input must be a int or a 2-tuple of ints")
