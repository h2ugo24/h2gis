"""Tests for format_converters/zarr_map_export.py (export_map_zarr).

These use the config-free ``source_root`` path so no var_key / config.yaml is
needed: the export is exercised against tiny on-disk Zarr stores built here.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.format_converters.zarr_map_export import export_map_zarr


def _make_ds(year: int, n_time: int = 20, seed: int = 0) -> xr.Dataset:
    """Varied data (so values can be compared) on a small lat/lon grid."""
    times = pd.date_range(f"{year}-01-01", periods=n_time, freq="D")
    rng = np.random.default_rng(seed)
    data = rng.uniform(10.0, 30.0, size=(n_time, 6, 8)).astype(np.float32)
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": np.linspace(30, 40, 6),
            "lon": np.linspace(-10, 0, 8),
        },
    )


def _write_source(src_dir, year: int, n_time: int = 20, seed: int = 0):
    """Write a per-year source store chunked in a large time block (extraction)."""
    src_dir.mkdir(parents=True, exist_ok=True)
    path = src_dir / f"h2mare_h2ds_{year}.zarr"
    ds = _make_ds(year, n_time=n_time, seed=seed)
    # Large time chunk = the extraction layout the export is meant to fix.
    ds.chunk({"time": n_time, "lat": 6, "lon": 8}).to_zarr(path, mode="w")
    return path, ds


class TestExportMapZarr:
    def test_creates_sibling_map_store_with_time_chunk_one(self, tmp_path):
        src_dir = tmp_path / "h2ds"
        _write_source(src_dir, 2020)

        written = export_map_zarr(source_root=src_dir, map_time_chunk=1)

        dst_dir = tmp_path / "h2ds_map"
        assert dst_dir.is_dir()
        assert len(written) == 1
        assert written[0] == dst_dir / "h2mare_h2ds_2020.zarr"

        out = xr.open_zarr(written[0], consolidated=False)
        try:
            # Map layout: one timestep per chunk, spatial dims contiguous.
            assert all(c == 1 for c in out.chunks["time"])
            assert out.chunks["lat"] == (6,)
            assert out.chunks["lon"] == (8,)
        finally:
            out.close()

    def test_default_map_time_chunk_is_14(self, tmp_path):
        src_dir = tmp_path / "h2ds"
        _write_source(src_dir, 2020, n_time=20)

        written = export_map_zarr(source_root=src_dir)

        out = xr.open_zarr(written[0], consolidated=False)
        try:
            # 20 days -> 14, 6
            assert out.chunks["time"] == (14, 6)
        finally:
            out.close()

    def test_values_preserved(self, tmp_path):
        src_dir = tmp_path / "h2ds"
        _, src_ds = _write_source(src_dir, 2020, seed=42)

        written = export_map_zarr(source_root=src_dir)

        out = xr.open_zarr(written[0], consolidated=False)
        try:
            np.testing.assert_array_equal(out["sst"].values, src_ds["sst"].values)
        finally:
            out.close()

    def test_source_store_untouched(self, tmp_path):
        src_dir = tmp_path / "h2ds"
        src_path, _ = _write_source(src_dir, 2020)

        export_map_zarr(source_root=src_dir)

        # Source keeps its large-time-block layout — export never writes back.
        src = xr.open_zarr(src_path, consolidated=False)
        try:
            assert src.chunks["time"] == (20,)
        finally:
            src.close()

    def test_map_time_chunk_passthrough(self, tmp_path):
        src_dir = tmp_path / "h2ds"
        _write_source(src_dir, 2020, n_time=20)

        written = export_map_zarr(source_root=src_dir, map_time_chunk=7)

        out = xr.open_zarr(written[0], consolidated=False)
        try:
            # 20 -> 7,7,6
            assert out.chunks["time"] == (7, 7, 6)
        finally:
            out.close()

    def test_years_filter(self, tmp_path):
        src_dir = tmp_path / "h2ds"
        _write_source(src_dir, 2020)
        _write_source(src_dir, 2021)

        written = export_map_zarr(source_root=src_dir, years=[2021])

        assert len(written) == 1
        assert written[0].name == "h2mare_h2ds_2021.zarr"
        assert not (tmp_path / "h2ds_map" / "h2mare_h2ds_2020.zarr").exists()

    def test_custom_dest_root(self, tmp_path):
        src_dir = tmp_path / "h2ds"
        _write_source(src_dir, 2020)
        dest = tmp_path / "elsewhere"

        written = export_map_zarr(source_root=src_dir, dest_root=dest)

        assert written[0] == dest / "h2mare_h2ds_2020.zarr"
        assert written[0].is_dir()

    def test_skips_existing_without_overwrite(self, tmp_path):
        src_dir = tmp_path / "h2ds"
        _write_source(src_dir, 2020)

        first = export_map_zarr(source_root=src_dir)
        assert len(first) == 1
        # Second run skips the already-exported file.
        second = export_map_zarr(source_root=src_dir)
        assert second == []

    def test_overwrite_reexports(self, tmp_path):
        src_dir = tmp_path / "h2ds"
        _write_source(src_dir, 2020)

        export_map_zarr(source_root=src_dir)
        again = export_map_zarr(source_root=src_dir, overwrite=True)
        assert len(again) == 1

    def test_raises_when_no_zarr_files(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No .zarr files"):
            export_map_zarr(source_root=empty)
