"""
CMEMS API utilities with caching.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import copernicusmarine
import pandas as pd
from loguru import logger


class CMEMSAPIError(Exception):
    """Raised when CMEMS API operations fail."""

    pass


@lru_cache(maxsize=128)
def get_dataset_time_range(dataset_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get time coverage of a CMEMS dataset (cached).

    Results are cached to avoid repeated API calls for the same dataset.

    Args:
        dataset_id: CMEMS dataset identifier

    Returns:
        Tuple of (start_time, end_time) as Timestamps

    Raises:
        CMEMSAPIError: If dataset not found or time range unavailable

    Example:
        >>> start, end = get_dataset_time_range("cmems_mod_glo_phy_my_0.083deg_P1D-m")
        >>> # Second call uses cache
        >>> start, end = get_dataset_time_range("cmems_mod_glo_phy_my_0.083deg_P1D-m")
    """
    # logger.debug(f"Fetching time range for dataset: {dataset_id}")

    try:
        metadata = copernicusmarine.describe(
            contains=[dataset_id],
            disable_progress_bar=True,
        )
    except Exception as e:
        raise CMEMSAPIError(
            f"Failed to fetch metadata for dataset '{dataset_id}': {e}"
        ) from e

    # Try to find time coordinate
    time_info = _find_time_coordinate(metadata)

    if time_info is None:
        raise CMEMSAPIError(
            f"No time coordinate found in dataset '{dataset_id}'. "
            f"Dataset may not have temporal coverage."
        )

    # Parse time bounds
    tmin, tmax = _parse_time_values(time_info, dataset_id)

    logger.info(f"Dataset '{dataset_id}' coverage: {tmin.date()} -> {tmax.date()}")

    return tmin, tmax


def _find_time_coordinate(metadata) -> Optional[dict]:
    """
    Search metadata for time coordinate information.

    Returns:
        Dictionary with 'minimum_value' and 'maximum_value' or None
    """
    # Flatten the nested structure
    for product in getattr(metadata, "products", []):
        for dataset in getattr(product, "datasets", []):
            for version in getattr(dataset, "versions", []):
                for part in getattr(version, "parts", []):
                    for service in getattr(part, "services", []):
                        for variable in getattr(service, "variables", []):
                            for coord in getattr(variable, "coordinates", []):
                                if getattr(coord, "coordinate_id", None) == "time":
                                    return {
                                        "minimum_value": getattr(
                                            coord, "minimum_value", None
                                        ),
                                        "maximum_value": getattr(
                                            coord, "maximum_value", None
                                        ),
                                    }

    return None


def _parse_time_values(
    time_info: dict,
    dataset_id: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Parse and validate time values.

    Args:
        time_info: Dictionary with minimum_value and maximum_value
        dataset_id: Dataset ID for error messages

    Returns:
        The dataset's first and last stamps, exactly as the catalogue reports
        them — time of day included.

        Deliberately not floored to midnight. Every consumer builds a
        ``DateRange``, which floors both ends itself, so flooring here changed
        nothing except to destroy the one thing a caller cannot recover: whether
        the last day is a full day of data or a few hours of one still being
        published. :meth:`CMEMSDownloader._get_dataset_availability` needs that
        to decide whether the trailing day is safe to ingest.

    Raises:
        CMEMSAPIError: If values are invalid
    """
    min_val = time_info["minimum_value"]
    max_val = time_info["maximum_value"]

    # Validate presence
    if min_val is None or max_val is None:
        raise CMEMSAPIError(
            f"Missing time bounds for dataset '{dataset_id}': "
            f"min={min_val}, max={max_val}"
        )

    # Convert to float (CMEMS uses milliseconds since epoch)
    try:
        tmin_ms = float(min_val)
        tmax_ms = float(max_val)
    except (TypeError, ValueError) as e:
        raise CMEMSAPIError(
            f"Invalid time values for dataset '{dataset_id}': "
            f"min={min_val}, max={max_val}. Error: {e}"
        ) from e

    # Convert milliseconds to datetime
    try:
        tmin = pd.Timestamp(tmin_ms, unit="ms")
        tmax = pd.Timestamp(tmax_ms, unit="ms")
    except (ValueError, pd.errors.OutOfBoundsDatetime) as e:
        raise CMEMSAPIError(
            f"Failed to parse timestamps for dataset '{dataset_id}': "
            f"{tmin_ms=}, {tmax_ms=}. Error: {e}"
        ) from e

    # Sanity check
    if tmin > tmax:
        raise CMEMSAPIError(
            f"Invalid time range for dataset '{dataset_id}': "
            f"start ({tmin.date()}) > end ({tmax.date()})"
        )

    return tmin, tmax


def clear_dataset_cache():
    """Clear the dataset time range cache."""
    get_dataset_time_range.cache_clear()
    logger.debug("Cleared dataset time range cache")
