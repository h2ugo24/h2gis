"""
Validation utilities for h2mare.

Provides common validation functions used across multiple modules.
"""

from __future__ import annotations

from typing import Sequence

import polars as pl

from h2mare.config import AppConfig
from h2mare.types import TimeResolution


def validate_columns(
    data: pl.LazyFrame | pl.DataFrame, cols: str | Sequence[str]
) -> None:
    """
    Check that *cols* are present in *data*.

    Lives here rather than beside its Parquet callers because both
    ``storage.parquet_helpers`` and ``utils.plot`` need it: importing it from
    storage into utils inverted the package layering (utils is a leaf) and
    closed an import cycle, since ``storage.zarr_catalog`` imports
    ``utils.labels``.

    Args:
        data (pl.LazyFrame | pl.DataFrame): Input data
        cols (str | Sequence[str]): Columns to check.

    Raises:
        TypeError: If cols are not a string or sequence of strings.
        ValueError: If columns are not in data.
    """
    if isinstance(cols, str):
        required = {cols}
    elif isinstance(cols, Sequence):
        required = set(cols)
    else:
        raise TypeError("`cols` must a string or a sequence of strings")

    if isinstance(data, pl.LazyFrame):
        existing = set(data.collect_schema().names())
    else:
        existing = set(data.schema.names())

    missing = required - existing
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def validate_var_key(var_key: str, config: AppConfig) -> str:
    """
    Validate that a variable key exists in config.

    Args:
        var_key: Variable identifier to validate
        config: Application configuration

    Raises:
        ValueError: If var_key not found in config.variables

    Example:
        >>> validate_var_key("sea_surface_height", app_config)
        >>> # Raises ValueError if not found
    """
    if var_key not in config.variables:
        available = ", ".join(config.variables.keys())
        raise ValueError(
            f"Variable key '{var_key}' not found in config. Available keys: {available}"
        )
    return var_key


def validate_var_keys(var_keys: Sequence[str], config: AppConfig) -> None:
    """
    Validate multiple variable keys.

    Args:
        var_keys: Variable identifiers to validate
        config: Application configuration

    Raises:
        ValueError: If any var_key not found
    """
    invalid = [key for key in var_keys if key not in config.variables]

    if invalid:
        available = ", ".join(config.variables.keys())
        raise ValueError(
            f"Variable key(s) not found in config: {', '.join(invalid)}. "
            f"Available keys: {available}"
        )


def validate_time_resolution(period: str | TimeResolution) -> TimeResolution:
    """
    Validate supported period granularity for data storage.

    Raises:
        ValueError: if period not 'year' or 'month'
    """
    if isinstance(period, TimeResolution):
        return period

    # Validate string value
    if not isinstance(period, str):
        raise ValueError(
            f"Period must be a string or Period enum, got {type(period).__name__}"
        )
    # Normalize to lowercase for case-insensitive comparison
    period_lower = period.lower()

    try:
        return TimeResolution(period_lower)
    except ValueError:
        valid_values = ", ".join(p.value for p in TimeResolution)
        raise ValueError(f"Invalid period '{period}'. Must be one of: {valid_values}")


# def validate_depth_range(depth_range: Sequence[float]) -> None:
#    """ Validate depth range for vertical data variables"""
#    if depth_range and len(depth_range) != 2:
#            raise ValueError("Depth range requires a min and max values.")
