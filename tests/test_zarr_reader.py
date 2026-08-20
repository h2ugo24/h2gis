"""Unit tests for ZarrReader — the opening half of the ZarrCatalog split.

The suite drives ZarrReader only through the ``ZarrCatalog`` facade, leaving its
own contract untested: the two-mode argument guard, time normalisation, and the
preprocessing applied per-file before ``open_mfdataset`` combines them. Those
last two run inside a callback during the open, so they are exercised directly
here rather than inferred from a combined dataset.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.storage.zarr_reader import ZarrReader, snap_axes_to_reference
from h2mare.types import BBox


def _reader(coverage=None) -> ZarrReader:
    """Reader over a stubbed index — these paths never touch the catalog."""
    index = MagicMock()
    index.get_time_coverage.return_value = coverage
    index.var_key = "sst"
    return ZarrReader(index)


def _ds(times, lats=(30.0, 35.0, 40.0), lons=(-10.0, -5.0, 0.0), variables=("sst",)):
    """Small time × lat × lon dataset, one distinct value per cell."""
    index = pd.DatetimeIndex(times)
    shape = (len(index), len(lats), len(lons))
    size = int(np.prod(shape))
    return xr.Dataset(
        {
            v: (("time", "lat", "lon"), np.arange(size, dtype=float).reshape(shape))
            for v in variables
        },
        coords={"time": index, "lat": list(lats), "lon": list(lons)},
    )


class TestOpenDatasetGuards:
    def test_both_modes_at_once_raises(self):
        reader = _reader()
        with pytest.raises(ValueError, match="Cannot use both"):
            reader.open_dataset(dates="2021-01-01", start_date="2021-01-01")

    def test_no_dates_and_no_coverage_raises(self):
        """With nothing requested the reader falls back to full coverage; an
        empty store leaves it with no range to open."""
        reader = _reader(coverage=None)
        with pytest.raises(ValueError, match="sparse 'dates' or"):
            reader.open_dataset()


class TestNormalizeTime:
    def test_times_are_truncated_to_midnight(self):
        reader = _reader()
        ds = _ds(["2021-01-01T13:30:00", "2021-01-02T06:15:00"])

        out = reader._normalize_time(ds)

        assert list(pd.DatetimeIndex(out.time.values)) == [
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2021-01-02"),
        ]

    def test_dataset_without_time_passes_through(self):
        """Time-less statics (bathy) go through the same open path."""
        reader = _reader()
        ds = xr.Dataset(
            {"elevation": (("lat",), [1.0, 2.0])}, coords={"lat": [0.0, 1.0]}
        )

        assert reader._normalize_time(ds) is ds


class TestPreprocessDataset:
    def test_selects_requested_variables(self):
        reader = _reader()
        ds = _ds(["2021-01-01"], variables=("sst", "chl"))

        out = reader._preprocess_dataset(ds, bbox=None, variables=["sst"])

        assert set(out.data_vars) == {"sst"}

    def test_accepts_a_bare_string_variable(self):
        reader = _reader()
        ds = _ds(["2021-01-01"], variables=("sst", "chl"))

        out = reader._preprocess_dataset(ds, bbox=None, variables="sst")

        assert set(out.data_vars) == {"sst"}

    def test_unknown_variable_keeps_everything(self):
        """Selecting nothing would silently empty the dataset, so a fully
        unmatched request warns and leaves the file untouched instead."""
        reader = _reader()
        ds = _ds(["2021-01-01"], variables=("sst",))

        out = reader._preprocess_dataset(ds, bbox=None, variables=["nope"])

        assert set(out.data_vars) == {"sst"}

    def test_descending_latitude_is_sorted_ascending(self):
        """ERA5 ships north-to-south; leaving it that way makes every later
        slice(lat) selection silently empty."""
        reader = _reader()
        ds = _ds(["2021-01-01"], lats=(40.0, 35.0, 30.0))

        out = reader._preprocess_dataset(ds, bbox=None, variables=None)

        assert list(out.lat.values) == [30.0, 35.0, 40.0]

    def test_bbox_subsets_the_grid_keeping_one_cell_of_padding(self):
        """The subset is padded by one cell on each side, so interpolating onto
        the requested extent later has neighbours at the edges instead of NaN.
        A grid only as wide as the request would hide the padding entirely."""
        reader = _reader()
        ds = _ds(
            ["2021-01-01"],
            lats=(0.0, 10.0, 20.0, 30.0, 40.0, 50.0),
            lons=(-20.0, -10.0, 0.0, 10.0, 20.0),
        )

        out = reader._preprocess_dataset(
            ds, bbox=BBox.from_tuple((-10.0, 10.0, 0.0, 20.0)), variables=None
        )

        # requested lats 10-20, plus one cell either side
        assert list(out.lat.values) == [0.0, 10.0, 20.0, 30.0]
        # requested lons -10-0, plus one cell either side
        assert list(out.lon.values) == [-20.0, -10.0, 0.0, 10.0]


class TestApplyBbox:
    def test_missing_coordinates_returns_dataset_unchanged(self):
        """A store without lat/lon (or y/x) cannot be subset; that is a warning,
        not a failure, so the caller still gets its data."""
        reader = _reader()
        ds = xr.Dataset({"v": (("depth",), [1.0, 2.0])}, coords={"depth": [0.0, 10.0]})

        out = reader._apply_bbox(ds, BBox.from_tuple((-10.0, 30.0, 0.0, 40.0)))

        assert out is ds


class TestSnapAxesToReference:
    """
    The same grid written on different occasions can disagree in the last
    float bits. combine="by_coords" compares coordinate arrays exactly, so it
    stops treating the axis as shared and starts treating it as one to
    concatenate along — surfacing as "does not have monotonic global indexes
    along dimension lon", which names neither the cause nor the file.
    """

    @staticmethod
    def _drifted(values, scale: float = 1e-12) -> np.ndarray:
        """The same axis as it might be rebuilt: equal to within float noise."""
        arr = np.asarray(values, dtype=float)
        return arr + np.linspace(0.0, scale, arr.size)

    def test_float_noise_is_snapped_to_the_reference(self):
        ref = np.array([-10.0, -5.0, 0.0])
        ds = _ds(["2021-01-01"], lons=self._drifted(ref))

        out, snapped = snap_axes_to_reference(ds, {"lon": ref})

        assert set(snapped) == {"lon"}
        assert snapped["lon"] == pytest.approx(1e-12, rel=0.1)
        np.testing.assert_array_equal(out.lon.values, ref)

    def test_an_identical_axis_is_left_alone(self):
        ref = np.array([-10.0, -5.0, 0.0])
        ds = _ds(["2021-01-01"], lons=ref)

        out, snapped = snap_axes_to_reference(ds, {"lon": ref})

        assert snapped == {}
        np.testing.assert_array_equal(out.lon.values, ref)

    def test_a_genuinely_different_grid_is_refused(self):
        """0.25 vs 0.1 spacing is not drift; it must still reach xarray."""
        ref = np.array([-10.0, -5.0, 0.0])
        shifted = ref + 0.25
        ds = _ds(["2021-01-01"], lons=shifted)

        out, snapped = snap_axes_to_reference(ds, {"lon": ref})

        assert snapped == {}
        np.testing.assert_array_equal(out.lon.values, shifted)

    def test_a_different_length_is_refused(self):
        ds = _ds(["2021-01-01"], lons=(-10.0, -5.0, 0.0))

        out, snapped = snap_axes_to_reference(ds, {"lon": np.array([-10.0, 0.0])})

        assert snapped == {}
        assert out.lon.size == 3

    def test_both_axes_snap_independently(self):
        ref_lat = np.array([30.0, 35.0, 40.0])
        ref_lon = np.array([-10.0, -5.0, 0.0])
        ds = _ds(["2021-01-01"], lats=self._drifted(ref_lat), lons=ref_lon)

        _, snapped = snap_axes_to_reference(ds, {"lat": ref_lat, "lon": ref_lon})

        assert set(snapped) == {"lat"}  # lon already matched

    def test_a_coord_absent_from_the_dataset_is_skipped(self):
        ds = _ds(["2021-01-01"]).drop_vars("lat")

        _, snapped = snap_axes_to_reference(ds, {"lat": np.array([1.0, 2.0, 3.0])})

        assert snapped == {}

    def test_tolerance_is_the_boundary(self):
        ref = np.array([0.0, 1.0, 2.0])
        just_over = ref + np.array([0.0, 0.0, 2e-9])

        _, snapped = snap_axes_to_reference(
            _ds(["2021-01-01"], lons=just_over), {"lon": ref}, tol=1e-9
        )
        assert snapped == {}

        _, snapped = snap_axes_to_reference(
            _ds(["2021-01-01"], lons=just_over), {"lon": ref}, tol=1e-8
        )
        assert set(snapped) == {"lon"}


class TestDriftedStoreIsReadable:
    """
    Regression, end to end through xarray: two files whose axes differ only by
    float noise must combine. Without the snap this raises "Resulting object
    does not have monotonic global indexes along dimension lon".
    """

    @staticmethod
    def _write_pair(tmp_path, drift: float):
        lons = np.array([-10.0, -5.0, 0.0])
        a = _ds(["2021-01-01", "2021-01-02"], lons=lons)
        b = _ds(["2021-01-03", "2021-01-04"], lons=lons + np.linspace(0, drift, 3))
        pa, pb = tmp_path / "a.zarr", tmp_path / "b.zarr"
        a.to_zarr(pa, consolidated=False)
        b.to_zarr(pb, consolidated=False)
        return [str(pa), str(pb)]

    def _open(self, paths, reference):
        reader = _reader()
        return xr.open_mfdataset(
            paths,
            engine="zarr",
            combine="by_coords",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            preprocess=lambda d: reader._preprocess_dataset(d, None, None, reference),
        )

    def test_float_drifted_files_combine(self, tmp_path):
        paths = self._write_pair(tmp_path, drift=1e-12)
        reference = ZarrReader._reference_axes(paths)

        with self._open(paths, reference) as ds:
            assert ds.sizes["time"] == 4
            assert ds.sizes["lon"] == 3  # not doubled by an outer join

    def test_the_same_files_are_damaged_without_the_snap(self, tmp_path):
        """
        Pins what the snap prevents. The damage is version-dependent: this
        xarray still defaults to join='outer' and unions the two axes into a
        doubled one full of near-duplicate points, warning that a future
        release will raise instead. Across a real store of many files the same
        mismatch surfaces as "Resulting object does not have monotonic global
        indexes along dimension lon". Either way the axis is not the grid.
        """
        paths = self._write_pair(tmp_path, drift=1e-12)

        with self._open(paths, reference={}) as ds:
            assert ds.sizes["lon"] > 3  # 3 real points, unioned into more
            assert np.diff(ds.lon.values).min() < 1e-11  # near-duplicate steps

    def test_reference_axes_needs_more_than_one_file(self, tmp_path):
        """A lone file is never combined against anything; it defines its axes."""
        paths = self._write_pair(tmp_path, drift=1e-12)

        assert ZarrReader._reference_axes(paths[:1]) == {}
        assert set(ZarrReader._reference_axes(paths)) == {"lat", "lon"}

    def test_reference_axes_survives_an_unreadable_path(self):
        """Advisory only — a failure here must fall back, not break the read."""
        assert ZarrReader._reference_axes(["/no/such/a.zarr", "/no/such/b.zarr"]) == {}
