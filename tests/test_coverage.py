"""Tests for storage/coverage.py — split_time_range and get_store_coverage."""

from unittest.mock import patch

import pandas as pd
import pytest

from h2mare.storage.coverage import get_store_coverage, split_time_range
from h2mare.types import DateRange, FilePeriod

# ---------------------------------------------------------------------------
# split_time_range
# ---------------------------------------------------------------------------


class TestSplitTimeRange:
    def test_single_month_stays_as_one_chunk(self):
        dr = DateRange("2021-03-01", "2021-03-31")
        chunks = split_time_range(dr, FilePeriod.MONTH)
        assert len(chunks) == 1
        assert chunks[0].start == pd.Timestamp("2021-03-01")
        assert chunks[0].end == pd.Timestamp("2021-03-31")

    def test_cross_month_boundary_splits_in_two(self):
        dr = DateRange("2021-01-15", "2021-02-28")
        chunks = split_time_range(dr, FilePeriod.MONTH)
        assert len(chunks) == 2
        assert chunks[0].end == pd.Timestamp("2021-01-31")
        assert chunks[1].start == pd.Timestamp("2021-02-01")

    def test_year_split_within_single_year(self):
        dr = DateRange("2021-03-01", "2021-09-30")
        chunks = split_time_range(dr, FilePeriod.YEAR)
        assert len(chunks) == 1
        assert chunks[0].start == pd.Timestamp("2021-03-01")
        assert chunks[0].end == pd.Timestamp("2021-09-30")

    def test_year_split_across_two_years(self):
        dr = DateRange("2020-06-01", "2021-03-31")
        chunks = split_time_range(dr, FilePeriod.YEAR)
        assert len(chunks) == 2
        assert chunks[0].end == pd.Timestamp("2020-12-31")
        assert chunks[1].start == pd.Timestamp("2021-01-01")
        assert chunks[1].end == pd.Timestamp("2021-03-31")

    def test_chunk_end_does_not_exceed_range_end(self):
        dr = DateRange("2021-01-01", "2021-06-15")
        chunks = split_time_range(dr, FilePeriod.YEAR)
        assert len(chunks) == 1
        assert chunks[0].end == pd.Timestamp("2021-06-15")

    def test_invalid_split_raises_value_error(self):
        dr = DateRange("2021-01-01", "2021-12-31")
        with pytest.raises(ValueError, match="Invalid split"):
            split_time_range(dr, "invalid")  # type: ignore


# ---------------------------------------------------------------------------
# get_store_coverage
# ---------------------------------------------------------------------------


class TestGetStoreCoverage:
    def test_returns_none_when_no_coverage(self):
        with patch("h2mare.storage.coverage.get_zarr_time_coverage", return_value=None):
            assert get_store_coverage("sst") is None

    def test_returns_date_range_when_coverage_exists(self):
        coverage = DateRange("2020-01-01", "2021-12-31")
        with patch(
            "h2mare.storage.coverage.get_zarr_time_coverage", return_value=coverage
        ):
            result = get_store_coverage("sst")
        assert result is not None
        assert result.start == pd.Timestamp("2020-01-01")
        assert result.end == pd.Timestamp("2021-12-31")

    def test_returns_none_on_exception(self):
        with patch(
            "h2mare.storage.coverage.get_zarr_time_coverage",
            side_effect=OSError("no store"),
        ):
            assert get_store_coverage("sst") is None


class TestUnreadableStoreIsNotAnEmptyStore:
    """
    Both return None from get_store_coverage, but they are different problems.
    resolve_date_range used to report either as "No existing data found",
    sending the reader to look for missing files while the real cause — a
    locked or damaged store — sat on a warning line above it.
    """

    def test_read_failure_is_reported_with_its_cause(self):
        from h2mare.storage.coverage import resolve_date_range

        with patch(
            "h2mare.storage.coverage.get_zarr_time_coverage",
            side_effect=OSError("[WinError 32] file in use"),
        ):
            with pytest.raises(ValueError) as excinfo:
                resolve_date_range("sst")

        msg = str(excinfo.value)
        assert "Could not read the store" in msg
        assert "WinError 32" in msg, "the underlying cause is not in the message"
        assert excinfo.value.__cause__ is not None, "cause not chained"

    def test_genuinely_empty_store_keeps_its_own_message(self):
        from h2mare.storage.coverage import resolve_date_range

        with patch("h2mare.storage.coverage.get_zarr_time_coverage", return_value=None):
            with pytest.raises(ValueError, match="No existing data found"):
                resolve_date_range("sst")

    def test_a_later_clean_read_clears_the_recorded_failure(self):
        """
        The breadcrumb must describe the most recent attempt. Without clearing,
        a var_key that failed once would keep reporting that stale error for
        every later genuinely-empty lookup.
        """
        from h2mare.storage.coverage import resolve_date_range

        with patch(
            "h2mare.storage.coverage.get_zarr_time_coverage",
            side_effect=OSError("transient"),
        ):
            with pytest.raises(ValueError, match="Could not read the store"):
                resolve_date_range("sst")

        # Same var_key, store now readable but empty.
        with patch("h2mare.storage.coverage.get_zarr_time_coverage", return_value=None):
            with pytest.raises(ValueError, match="No existing data found"):
                resolve_date_range("sst")

    def test_explicit_dates_skip_the_lookup_entirely(self):
        from h2mare.storage.coverage import resolve_date_range

        with patch(
            "h2mare.storage.coverage.get_zarr_time_coverage",
            side_effect=OSError("would fail if consulted"),
        ):
            dr = resolve_date_range("sst", start="2021-01-01", end="2021-01-31")
        assert dr is not None
        assert dr.start == pd.Timestamp("2021-01-01")
