"""Unit tests for ParquetCatalog — the read half of the ParquetIndexer split.

The existing suite drives ParquetCatalog only through the ``ParquetIndexer``
facade, so its own contract (partition pruning in ``_resolve_files``, the
mandatory-column rule in ``scan``, plot-cache invalidation) is unpinned. These
exercise the class directly.
"""

from datetime import date

import polars as pl
import pytest
from conftest import make_grid_df

from h2mare.storage.parquet_catalog import ParquetCatalog
from h2mare.storage.parquet_store import ParquetStore


def _catalog(tmp_path, dates=None, **kwargs) -> ParquetCatalog:
    """A catalog over a store primed with *dates* (empty when None)."""
    store = ParquetStore(tmp_path / "store", **kwargs)
    if dates:
        store.add_data(make_grid_df(dates))
    return ParquetCatalog(store)


class TestScanGuards:
    def test_scan_on_empty_store_raises(self, tmp_path):
        cat = _catalog(tmp_path)
        with pytest.raises(RuntimeError, match="No data in parquet store"):
            cat.scan()

    def test_columns_always_include_time_lon_lat(self, tmp_path):
        """A caller asking for one variable still needs the coordinates that
        locate it, so the mandatory trio is unioned in rather than replaced."""
        cat = _catalog(tmp_path, [date(2021, 6, 1)])

        got = set(cat.scan(columns="sst").collect_schema().names())

        assert got == {"time", "lon", "lat", "sst"}

    def test_unknown_column_is_dropped_not_raised(self, tmp_path):
        """Selection intersects the physical schema, so a stale column name in a
        caller's list degrades to fewer columns instead of a scan error."""
        cat = _catalog(tmp_path, [date(2021, 6, 1)])

        got = set(cat.scan(columns=["sst", "not_a_column"]).collect_schema().names())

        assert got == {"time", "lon", "lat", "sst"}


class TestResolveFiles:
    """``_resolve_files`` prunes to candidate partition dirs; walking the whole
    tree would scale with store size instead of query size."""

    def test_range_prunes_to_the_requested_months(self, tmp_path):
        cat = _catalog(
            tmp_path,
            [date(2021, 1, 15), date(2021, 2, 15), date(2021, 3, 15)],
        )

        files = cat._resolve_files(("2021-02-01", "2021-02-28"))

        assert files, "expected the February partition"
        assert all("month=2" in str(f) for f in files)

    def test_range_spanning_a_year_boundary_wraps_the_month(self, tmp_path):
        """Regression risk: the month counter in the range loop has to roll over
        to January and bump the year, or December-to-January windows return
        nothing."""
        cat = _catalog(tmp_path, [date(2020, 12, 15), date(2021, 1, 15)])

        files = cat._resolve_files(("2020-12-01", "2021-01-31"))

        assert any("year=2020" in str(f) and "month=12" in str(f) for f in files)
        assert any("year=2021" in str(f) and "month=1" in str(f) for f in files)

    def test_date_list_selects_only_those_partitions(self, tmp_path):
        cat = _catalog(tmp_path, [date(2021, 1, 15), date(2021, 6, 15)])

        files = cat._resolve_files([date(2021, 6, 15)])

        assert files
        assert all("month=6" in str(f) for f in files)

    def test_none_returns_every_file(self, tmp_path):
        cat = _catalog(tmp_path, [date(2021, 1, 15), date(2021, 6, 15)])

        assert len(cat._resolve_files(None)) >= 2

    def test_bad_dates_type_raises(self, tmp_path):
        cat = _catalog(tmp_path, [date(2021, 6, 1)])
        with pytest.raises(ValueError, match="list or"):
            cat._resolve_files("2021-06-01")


class TestFiltering:
    def test_bbox_filters_rows(self, tmp_path):
        cat = _catalog(tmp_path, [date(2021, 6, 1)])

        df = cat.load(bbox=(-11.0, 29.0, -6.0, 36.0))

        assert df["lon"].max() <= -6.0
        assert df["lat"].max() <= 36.0

    def test_date_range_filters_rows(self, tmp_path):
        cat = _catalog(tmp_path, [date(2021, 6, 1), date(2021, 6, 2), date(2021, 6, 3)])

        df = cat.load(dates=("2021-06-02", "2021-06-03"))

        assert set(df["time"].cast(pl.Utf8).to_list()) == {"2021-06-02", "2021-06-03"}


class TestPlotAccessor:
    def test_plot_is_cached(self, tmp_path):
        cat = _catalog(tmp_path, [date(2021, 6, 1)])
        assert cat.plot is cat.plot

    def test_clear_plot_cache_is_safe_before_first_access(self, tmp_path):
        """Called on every write, including before anything has plotted."""
        cat = _catalog(tmp_path, [date(2021, 6, 1)])
        cat._clear_plot_cache()  # must not raise
        assert "plot" not in cat.__dict__
