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

from h2mare.storage.zarr_reader import ZarrReader
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
