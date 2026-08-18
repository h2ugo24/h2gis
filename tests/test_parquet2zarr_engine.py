"""Tests for the config-free convert_parquet_to_zarr engine function.

Mirrors test_zarr2parquet_engine.py: the point is rebuilding a Zarr store from
an arbitrary Parquet store *without* a configured var_key, so these tests write
Parquet straight to disk via ParquetIndexer and never go through ZarrCatalog /
app_config.

The headline test is a full round trip: Zarr → Parquet → Zarr must preserve the
grid and values (modulo the store's Date-precision time and Float32 downcast).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.format_converters.parquet2zarr import convert_parquet_to_zarr
from h2mare.format_converters.zarr2parquet import convert_zarr_to_parquet
from h2mare.storage.parquet_indexer import ParquetIndexer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ds(start: str = "2020-01-01", n_days: int = 40, seed: int = 0) -> xr.Dataset:
    """Varied data spanning >1 month so monthly chunking exercises >1 period."""
    times = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    data = rng.uniform(10.0, 30.0, size=(n_days, 2, 2)).astype(np.float32)
    return xr.Dataset(
        {"adhoc_var": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": [30.0, 35.0],
            "lon": [-10.0, -5.0],
        },
    )


def _make_parquet(tmp_path, ds: xr.Dataset, name: str = "store"):
    """Write a dataset to a Parquet store and return its root path."""
    idx = ParquetIndexer(tmp_path / name)
    idx.add_data(__import__("polars").from_pandas(ds.to_dataframe().reset_index()))
    return tmp_path / name


# ---------------------------------------------------------------------------
# Round trip — the contract this converter exists to honour
# ---------------------------------------------------------------------------


def test_roundtrip_zarr_parquet_zarr_preserves_grid_and_values(tmp_path):
    """Zarr → Parquet → Zarr reproduces the grid shape and values."""
    ds = _make_ds(n_days=10)
    src = tmp_path / "src.zarr"
    ds.to_zarr(src)

    parquet = tmp_path / "parquet"
    convert_zarr_to_parquet(src, parquet)

    out_dir = tmp_path / "rebuilt"
    written = convert_parquet_to_zarr(parquet, out_dir, name="adhoc")

    assert len(written) == 1  # single year → one file
    rebuilt = xr.open_zarr(written[0])
    try:
        assert dict(rebuilt.sizes) == {"time": 10, "lat": 2, "lon": 2}
        assert list(rebuilt.data_vars) == ["adhoc_var"]

        a = ds.sortby(["time", "lat", "lon"])
        b = rebuilt.sortby(["time", "lat", "lon"])
        np.testing.assert_array_equal(a.lat.values, b.lat.values)
        np.testing.assert_array_equal(a.lon.values, b.lon.values)
        np.testing.assert_array_equal(
            pd.to_datetime(a.time.values), pd.to_datetime(b.time.values)
        )
        np.testing.assert_allclose(
            a["adhoc_var"].values, b["adhoc_var"].values, rtol=1e-5
        )
    finally:
        rebuilt.close()


def test_rebuilt_values_are_float32(tmp_path):
    """chunk_dataset downcasts; the rebuilt store stays Float32."""
    parquet = _make_parquet(tmp_path, _make_ds(n_days=5))
    written = convert_parquet_to_zarr(parquet, tmp_path / "out", name="adhoc")

    rebuilt = xr.open_zarr(written[0])
    try:
        assert rebuilt["adhoc_var"].dtype == np.float32
    finally:
        rebuilt.close()


# ---------------------------------------------------------------------------
# Core conversion — no config / no var_key
# ---------------------------------------------------------------------------


def test_converts_without_config(tmp_path):
    parquet = _make_parquet(tmp_path, _make_ds(n_days=5))
    out_dir = tmp_path / "out"

    written = convert_parquet_to_zarr(parquet, out_dir, name="adhoc")

    assert len(written) == 1
    assert written[0].parent == out_dir
    assert written[0].name.startswith("adhoc_")
    assert written[0].suffix == ".zarr"


def test_accepts_str_paths(tmp_path):
    parquet = _make_parquet(tmp_path, _make_ds(n_days=5))
    out_dir = tmp_path / "out"

    written = convert_parquet_to_zarr(str(parquet), str(out_dir), name="adhoc")

    assert xr.open_zarr(written[0]).sizes["time"] == 5


def test_empty_store_raises(tmp_path):
    """An empty Parquet root has no coverage and cannot be converted."""
    ParquetIndexer(tmp_path / "empty")  # never written to
    with pytest.raises(ValueError, match="(?i)empty"):
        convert_parquet_to_zarr(tmp_path / "empty", tmp_path / "out")


# ---------------------------------------------------------------------------
# Per-period file layout
# ---------------------------------------------------------------------------


def test_year_format_groups_months_into_one_file(tmp_path):
    """date_format='year' appends monthly chunks into a single per-year file."""
    # 40 days from Jan 1 spans Jan + Feb → two monthly read chunks, one year file.
    parquet = _make_parquet(tmp_path, _make_ds("2020-01-01", n_days=40))
    written = convert_parquet_to_zarr(
        parquet, tmp_path / "out", name="adhoc", date_format="year"
    )

    assert len(written) == 1
    rebuilt = xr.open_zarr(written[0])
    try:
        assert rebuilt.sizes["time"] == 40  # both months landed in the year file
    finally:
        rebuilt.close()


def test_yearmonth_format_writes_one_file_per_month(tmp_path):
    """date_format='yearmonth' produces a distinct file per month."""
    parquet = _make_parquet(tmp_path, _make_ds("2020-01-01", n_days=40))
    written = convert_parquet_to_zarr(
        parquet, tmp_path / "out", name="adhoc", date_format="yearmonth"
    )

    assert len(written) == 2  # Jan + Feb
    names = sorted(p.name for p in written)
    assert "2020-01" in names[0]
    assert "2020-02" in names[1]


def test_spans_two_years_yields_two_files(tmp_path):
    """A Dec→Jan window writes one per-year file on each side of the boundary."""
    parquet = _make_parquet(tmp_path, _make_ds("2020-12-20", n_days=30))
    written = convert_parquet_to_zarr(parquet, tmp_path / "out", name="adhoc")

    assert len(written) == 2
    names = sorted(p.name for p in written)
    assert names[0].endswith("_2020.zarr")
    assert names[1].endswith("_2021.zarr")


# ---------------------------------------------------------------------------
# Date window + variable subset
# ---------------------------------------------------------------------------


def test_explicit_date_window_subsets(tmp_path):
    parquet = _make_parquet(tmp_path, _make_ds("2020-01-01", n_days=40))
    written = convert_parquet_to_zarr(
        parquet,
        tmp_path / "out",
        name="adhoc",
        start_date="2020-01-05",
        end_date="2020-01-09",
    )

    rebuilt = xr.open_zarr(written[0])
    try:
        times = pd.to_datetime(rebuilt.time.values)
        assert times.min() == pd.Timestamp("2020-01-05")
        assert times.max() == pd.Timestamp("2020-01-09")
    finally:
        rebuilt.close()


def test_variables_subset(tmp_path):
    ds = _make_ds(n_days=5)
    ds = ds.assign(other=ds["adhoc_var"] * 0.5)
    parquet = _make_parquet(tmp_path, ds)

    written = convert_parquet_to_zarr(
        parquet, tmp_path / "out", name="adhoc", variables=["adhoc_var"]
    )

    rebuilt = xr.open_zarr(written[0])
    try:
        assert "adhoc_var" in rebuilt.data_vars
        assert "other" not in rebuilt.data_vars
    finally:
        rebuilt.close()


def test_bbox_subsets_grid(tmp_path):
    # 3×3 grid so a bbox can drop the outer ring yet still leave a labelable
    # (≥2×2) extent — BBox.from_dataset, used for the filename, requires
    # xmin < xmax and ymin < ymax.
    times = pd.date_range("2020-01-01", periods=5, freq="D")
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "adhoc_var": (
                ["time", "lat", "lon"],
                rng.random((5, 3, 3)).astype(np.float32),
            )
        },
        coords={
            "time": times,
            "lat": [30.0, 35.0, 40.0],
            "lon": [-10.0, -5.0, 0.0],
        },
    )
    parquet = _make_parquet(tmp_path, ds)

    # Keep the lower-left 2×2 block (lon ∈ {-10, -5}, lat ∈ {30, 35}).
    written = convert_parquet_to_zarr(
        parquet, tmp_path / "out", name="adhoc", bbox=(-11.0, 29.0, -4.0, 36.0)
    )

    rebuilt = xr.open_zarr(written[0])
    try:
        assert rebuilt.lon.values.tolist() == [-10.0, -5.0]
        assert rebuilt.lat.values.tolist() == [30.0, 35.0]
    finally:
        rebuilt.close()


# ---------------------------------------------------------------------------
# Append semantics (reuses write_append_zarr)
# ---------------------------------------------------------------------------


def test_second_call_extends_year_file(tmp_path):
    """A later monthly chunk appends into the existing per-year store."""
    out_dir = tmp_path / "out"

    a = _make_parquet(tmp_path, _make_ds("2020-01-01", 5, seed=1), name="a")
    first = convert_parquet_to_zarr(a, out_dir, name="adhoc")

    b = _make_parquet(tmp_path, _make_ds("2020-02-01", 5, seed=2), name="b")
    second = convert_parquet_to_zarr(b, out_dir, name="adhoc")

    assert first[0] == second[0]  # same per-year file
    rebuilt = xr.open_zarr(second[0])
    try:
        times = pd.to_datetime(rebuilt.time.values)
        assert times.min() == pd.Timestamp("2020-01-01")
        assert times.max() == pd.Timestamp("2020-02-05")
    finally:
        rebuilt.close()


# ---------------------------------------------------------------------------
# file_period accepts a plain string / validation
# ---------------------------------------------------------------------------


def test_file_period_accepts_plain_string(tmp_path):
    parquet = _make_parquet(tmp_path, _make_ds(n_days=40))
    written = convert_parquet_to_zarr(
        parquet, tmp_path / "out", name="adhoc", file_period="year"
    )
    assert xr.open_zarr(written[0]).sizes["time"] == 40


def test_invalid_file_period_raises(tmp_path):
    parquet = _make_parquet(tmp_path, _make_ds(n_days=5))
    with pytest.raises(ValueError, match="(?i)period|month|year"):
        convert_parquet_to_zarr(
            parquet, tmp_path / "out", name="adhoc", file_period="monthly"
        )
