"""
Store integrity checks: does what is on disk match what it claims to cover?

Every coverage mechanism in the pipeline is a *frontier*, not a *density*.
``get_store_coverage`` returns a ``DateRange``; ``_h2ds_nonnull_ends`` tracks the
last non-null date per variable. Both answer "how far have we got", never "is
what we have solid". A store holding 1999-01-01 and 1999-12-31 with 237 days
missing in between reports itself healthy to all of them.

This module asks the other question. It is deliberately cheap and deliberately
narrow:

* **Cheap** — the Zarr side reads coordinates only, never data. Scanning all
  435 files of the production store takes ~55s, which is why there is no cache
  and no incremental mode to get wrong. The Parquet side reads footer
  statistics, so it reads no data either.
* **Narrow** — it flags days missing from a *time axis*, not days whose values
  are null. Those are two different failures that look identical downstream
  (both become all-null days in ``h2ds``) but are cleanly separable here:

  =============  ==========================================  ==================
  fsle_max 1999  128/365 steps — days absent from the axis    pipeline defect
  chl 1999       365/365 steps, 3 days null-valued            genuine source gap
  =============  ==========================================  ==================

  A check on axis length catches the defect and is structurally silent on chl.
  A check on values fires on both — and a check that flags chl every year is
  one people learn to ignore, at which point it protects nothing.

``check_slice_health`` is the value-based check, for the cases where a store is
present and complete but degenerate. It is the second line of defence and is
run only on request (``--values``), because it is the expensive one.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.config import AppConfig, get_settings
from h2mare.utils.paths import resolve_store_path
from h2mare.validators import validate_var_key


class AxisGap(NamedTuple):
    """Days absent from a store's time axis, strictly inside its own span."""

    path: Path
    span: tuple[pd.Timestamp, pd.Timestamp]
    missing: pd.DatetimeIndex


class SliceIssue(NamedTuple):
    """A time slice that is present but carries no usable data."""

    path: Path
    variable: str
    date: pd.Timestamp
    kind: str  # "empty" | "degenerate"
    detail: str


class VarAudit(NamedTuple):
    """Everything found for one ``var_key``."""

    var_key: str
    store_root: Path
    n_files: int
    gaps: list[AxisGap]
    slices: list[SliceIssue]
    errors: list[str]
    # Days excluded because config records the provider never published them.
    n_known_gaps: int = 0

    @property
    def n_missing_days(self) -> int:
        return sum(len(g.missing) for g in self.gaps)

    @property
    def ok(self) -> bool:
        return not (self.gaps or self.slices or self.errors)


def known_gap_days(var_config) -> pd.DatetimeIndex:
    """
    Expand a variable's ``known_gaps`` config entries into individual days.

    Accepts ``"YYYY-MM-DD"`` and the closed interval ``"YYYY-MM-DD/YYYY-MM-DD"``.
    Malformed entries are warned about and skipped rather than raising: a typo
    in a suppression list must not be able to stop a pipeline run.

    These are days the provider never published. A source shipping one file per
    day produces an *axis* hole when it skips one, which is otherwise
    indistinguishable from data the pipeline lost, so without this the checks
    would report the same unfixable day on every run — and a check that cries
    wolf stops being read.
    """
    entries = getattr(var_config, "known_gaps", None) or []
    days: list[pd.DatetimeIndex] = []
    for entry in entries:
        try:
            if "/" in str(entry):
                start, end = str(entry).split("/", 1)
                days.append(pd.date_range(start.strip(), end.strip(), freq="D"))
            else:
                days.append(pd.DatetimeIndex([pd.to_datetime(str(entry))]))
        except Exception as e:
            logger.warning(f"Ignoring malformed known_gaps entry {entry!r}: {e}")

    if not days:
        return pd.DatetimeIndex([])
    return (
        pd.DatetimeIndex(np.concatenate([d.values for d in days]))
        .normalize()
        .unique()
        .sort_values()
    )


def contiguous_blocks(
    dates: pd.DatetimeIndex,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Compress a date index into ``(first, last)`` runs of consecutive days."""
    if len(dates) == 0:
        return []
    dates = pd.DatetimeIndex(dates).sort_values()
    runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = prev = dates[0]
    for day in dates[1:]:
        if (day - prev).days > 1:
            runs.append((start, prev))
            start = day
        prev = day
    runs.append((start, prev))
    return runs


def format_date_blocks(dates: pd.DatetimeIndex, max_blocks: int = 8) -> str:
    """Render a date index inline: ``'2025-06-02, 2025-07-10→2025-07-14'``."""
    runs = contiguous_blocks(dates)
    if not runs:
        return "none"
    shown = [
        str(a.date()) if a == b else f"{a.date()}→{b.date()}"
        for a, b in runs[:max_blocks]
    ]
    if len(runs) > max_blocks:
        shown.append(f"… and {len(runs) - max_blocks} more block(s)")
    return ", ".join(shown)


def interior_gaps(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Days absent from *dates*, strictly between its own first and last entry.

    Restricting to the interior is what makes this usable without an allowlist.
    A store whose tail stops short of today is ordinary provider lag; a day the
    provider published, surrounded on both sides by days that are present, is
    not. Across all 435 files of the production store this rule produced two
    findings and no false positives — chl's null days included, since they are
    on the axis.
    """
    if len(dates) < 2:
        return pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(dates).normalize().unique().sort_values()
    return pd.date_range(dates[0], dates[-1], freq="D").difference(dates)


def check_slice_health(
    ds: xr.Dataset, *, sample: Optional[int] = None
) -> list[tuple[str, pd.Timestamp, str, str]]:
    """
    Per-(variable, time) reduction returning every empty or degenerate slice.

    Replaces the old ``have_vars_unique_values``, which was never wired into any
    code path and could not have caught an interior hole even if it had been:
    it inspected ``isel(time=-1)`` only, the one position that cannot reveal
    one. It also used ``np.unique`` — a sort, plus full materialisation of a
    dask slice — to answer what ``min == max`` answers in one pass, and it
    conflated "all missing" with "constant" because NaN collapses to a single
    unique value.

    Here one lazy pass yields ``(n_finite, min, max)`` per slice, from which
    both signals fall out separately:

    * ``n_finite == 0``            → ``"empty"``      (no usable data at all)
    * ``min == max``, n_finite > 0 → ``"degenerate"`` (a single repeated value)

    Returns:
        ``(variable, date, kind, detail)`` tuples. Empty when everything is fine.
    """
    issues: list[tuple[str, pd.Timestamp, str, str]] = []

    for name, da in ds.data_vars.items():
        if "time" not in da.dims:
            continue
        spatial = [d for d in da.dims if d != "time"]
        if not spatial:
            continue

        sub = da.isel(time=slice(None, sample)) if sample else da
        finite = np.isfinite(sub)
        n_finite = finite.sum(dim=spatial)
        lo = sub.min(dim=spatial, skipna=True)
        hi = sub.max(dim=spatial, skipna=True)

        n_finite, lo, hi = xr.align(n_finite, lo, hi, join="exact")
        n_vals, lo_vals, hi_vals = (
            np.asarray(n_finite.values),
            np.asarray(lo.values),
            np.asarray(hi.values),
        )
        times = pd.DatetimeIndex(sub.time.values)

        for i, when in enumerate(times):
            if n_vals[i] == 0:
                issues.append((str(name), when, "empty", "no finite values"))
            elif lo_vals[i] == hi_vals[i]:
                issues.append(
                    (str(name), when, "degenerate", f"single value {lo_vals[i]:.6g}")
                )

    return issues


def audit_zarr_file(
    path: Path,
    *,
    check_values: bool = False,
    known_gaps: Optional[pd.DatetimeIndex] = None,
) -> tuple[
    Optional[AxisGap],
    list[SliceIssue],
    Optional[str],
]:
    """
    Audit one Zarr store. Coordinates only unless *check_values* is set.

    Days listed in *known_gaps* are dropped from the result — the provider
    never published them, so reporting them every run would train the reader to
    ignore the output.
    """
    try:
        ds = xr.open_zarr(path, consolidated=False)
    except Exception as e:
        return None, [], f"{path.name}: could not open ({e})"

    try:
        if "time" not in ds.coords:
            return None, [], None

        dates = pd.DatetimeIndex(ds.time.values).normalize().unique().sort_values()
        missing = interior_gaps(dates)
        if known_gaps is not None and len(missing):
            missing = missing.difference(known_gaps)
        gap = (
            AxisGap(path=path, span=(dates[0], dates[-1]), missing=missing)
            if len(missing)
            else None
        )

        slices: list[SliceIssue] = []
        if check_values:
            slices = [
                SliceIssue(path=path, variable=v, date=d, kind=k, detail=detail)
                for v, d, k, detail in check_slice_health(ds)
            ]
        return gap, slices, None
    except Exception as e:
        return None, [], f"{path.name}: audit failed ({e})"
    finally:
        ds.close()


def audit_var_key(
    var_key: str,
    *,
    app_config: Optional[AppConfig] = None,
    store_root: Optional[Path] = None,
    check_values: bool = False,
) -> VarAudit:
    """
    Audit every Zarr store belonging to *var_key*.

    Reads the stores directly rather than the catalog index: the index records
    ``num_timesteps`` and a start/end per file, which is exactly the frontier
    view that cannot see an interior hole. It is also what a hard kill leaves
    stale, and a hard kill is the most likely origin of a ragged store.
    """
    config = app_config or get_settings().app_config
    var_key = validate_var_key(var_key, config)
    root = resolve_store_path(config.variables[var_key], store_root)

    gaps: list[AxisGap] = []
    slices: list[SliceIssue] = []
    errors: list[str] = []

    suppressed = known_gap_days(config.variables[var_key])

    files = sorted(root.glob("*.zarr")) if root.exists() else []
    for path in files:
        gap, found, error = audit_zarr_file(
            path, check_values=check_values, known_gaps=suppressed
        )
        if gap:
            gaps.append(gap)
        slices.extend(found)
        if error:
            errors.append(error)

    return VarAudit(
        var_key=var_key,
        store_root=root,
        n_files=len(files),
        gaps=gaps,
        slices=slices,
        errors=errors,
        n_known_gaps=len(suppressed),
    )


def audit_parquet_nulls(
    parquet_root: Path, columns: Optional[list[str]] = None
) -> list[tuple[Path, str]]:
    """
    Columns that are entirely null in a Parquet file, from footer metadata.

    Costs no data read: every Parquet footer carries a per-column
    ``null_count`` per row group, so "is this column all null in this file"
    is a metadata comparison. With ``year=/month=`` partitioning that gives
    month granularity for free.

    Worth having as a parity check even though the Zarr side is authoritative,
    because the Parquet representation is the more dangerous of the two: a
    missing time step is detectable by counting an axis, but a null column
    inside complete-looking rows is not.

    Returns:
        ``(file, column)`` pairs. Empty when nothing is wholly null.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from h2mare.storage.parquet_store import iter_store_parquet_files

    findings: list[tuple[Path, str]] = []
    skip = {"time", "lon", "lat", "year", "month", "day"}

    for path in iter_store_parquet_files(parquet_root):
        try:
            pf = pq.ParquetFile(path)
            meta, arrow_schema = pf.metadata, pf.schema_arrow
        except Exception as e:
            logger.warning(f"Could not read parquet footer for {path.name}: {e}")
            continue
        if meta.num_rows == 0:
            continue

        names = meta.schema.names
        for col, name in enumerate(names):
            if name in skip or (columns is not None and name not in columns):
                continue

            # A null-typed column holds nothing by construction and carries no
            # statistics to consult.
            field = arrow_schema.field(name) if name in arrow_schema.names else None
            if field is not None and pa.types.is_null(field.type):
                findings.append((path, name))
                continue

            nulls = 0
            for rg in range(meta.num_row_groups):
                stats = meta.row_group(rg).column(col).statistics
                if stats is None or stats.null_count is None:
                    nulls = -1
                    break
                nulls += stats.null_count
            if nulls == meta.num_rows:
                findings.append((path, name))

    return findings
