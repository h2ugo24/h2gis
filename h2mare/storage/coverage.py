"""
Helpers for ZarrCatalog data coverage and store management.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from h2mare.storage.zarr_catalog import get_zarr_time_coverage
from h2mare.types import DateLike, DateRange, TimeResolution
from h2mare.utils.datetime_utils import normalize_date


def split_time_range(date_range: DateRange, split: TimeResolution) -> list[DateRange]:
    """
    Split date range into chunks.

    Args:
        date_range: Range to split
        split: Split strategy

    Returns:
        List of (start, end) tuples
    """
    chunks = []
    current = date_range.start

    while current <= date_range.end:
        if split == TimeResolution.MONTH:
            # End of current month
            chunk_end = current + pd.offsets.MonthEnd(0)
        elif split == TimeResolution.YEAR:
            # End of current year
            chunk_end = pd.Timestamp(year=current.year, month=12, day=31)
        else:
            raise ValueError("Invalid split value. Options are: 'month' or 'year'")

        # Don't exceed requested end
        chunk_end = min(chunk_end, date_range.end)

        chunks.append(DateRange(start=current, end=chunk_end))

        # Move to next period
        current = chunk_end + pd.Timedelta(days=1)

    return chunks


def get_store_coverage(var_key: str) -> Optional[DateRange]:
    """
    Get time coverage of existing data in store.

    Returns:
        DateRange of stored data, or None if no data exists
    """
    try:
        coverage = get_zarr_time_coverage(var_key)

        if coverage is None:
            logger.debug(f"No existing data found for {var_key}")
            return None

        return DateRange(start=coverage.start, end=coverage.end)

    except Exception as e:
        logger.warning(f"Could not read store coverage: {e}")
        return None


def get_store_missing_days(var_key: str) -> pd.DatetimeIndex:
    """
    Days absent from a store's time axis, inside each file's own span.

    The density counterpart to :func:`get_store_coverage`, which returns a
    ``DateRange`` and so cannot express a hole: a store holding 1999-01-01 and
    1999-12-31 with 237 days missing between them reports the same coverage as
    a complete one.

    Reads coordinates only — no data — so this is cheap enough to call freely.
    Note it sees days missing from the *axis*, not days present with null
    values; the latter is a genuine source gap and is deliberately not reported
    (see :mod:`h2mare.storage.audit`).

    Returns:
        The missing days, empty when the store is dense or unreadable.
    """
    from h2mare.storage.audit import audit_var_key

    try:
        result = audit_var_key(var_key)
    except Exception as e:
        logger.warning(f"Could not scan '{var_key}' for missing days: {e}")
        return pd.DatetimeIndex([])

    if not result.gaps:
        return pd.DatetimeIndex([])
    return (
        pd.DatetimeIndex(
            [day for gap in result.gaps for day in gap.missing],
        )
        .unique()
        .sort_values()
    )


def resolve_date_range(
    var_key: str,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
) -> Optional[DateRange]:
    """
    Resolve a date range for download or processing.

    When *start* or *end* are omitted they are inferred from the store:
    start = store_end + 1 day, end = today.  If the store is already
    up to date (inferred start > end) ``None`` is returned so callers
    can skip gracefully without treating it as an error.

    When both dates are supplied explicitly and start > end a
    ``ValueError`` is raised (caller error).

    Lives beside :func:`get_store_coverage` rather than in ``utils`` because
    the store lookup is its whole purpose: importing it the other way round
    made a leaf package depend on ``storage`` and closed an import cycle.

    Args:
        var_key: Variable key used to look up the store.
        start: Explicit start date, or ``None`` to infer.
        end: Explicit end date, or ``None`` to infer.

    Returns:
        Resolved :class:`DateRange`, or ``None`` if the store is already
        up to date.

    Raises:
        ValueError: If no store exists and dates cannot be inferred, or
            if explicitly supplied dates produce start > end.
    """
    start = normalize_date(start) if start else None
    end = normalize_date(end) if end else None
    inferred = start is None or end is None

    if inferred:
        store_coverage = get_store_coverage(var_key)

        if store_coverage is None:
            raise ValueError(
                f"No existing data found for '{var_key}'. "
                f"Please provide start and end dates explicitly."
            )

        if start is None:
            start = store_coverage.end + pd.Timedelta(days=1)
        if end is None:
            end = pd.Timestamp.now().normalize()

        logger.info(
            f"Date range in store: {store_coverage.start.date()} -> {store_coverage.end.date()}"
        )

    assert start is not None and end is not None
    if start > end:
        if inferred:
            return None
        raise ValueError(
            f"Invalid date range: start ({start.date()}) > end ({end.date()})"
        )

    return DateRange(start=start, end=end)
