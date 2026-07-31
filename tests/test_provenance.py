"""Tests for storage/provenance.py — manifest-to-provenance helpers.

These back the eddies path, which writes its own Zarr and so never reaches
Netcdf2Zarr._write_provenance. Two behaviours matter and neither is covered by
the generic converter path:

* a written window must be attributed by intersecting it with the manifest,
  because one eddies trajectory file spans years and both rep and nrt;
* repeated appends to the same period must widen the recorded coverage rather
  than replace it.
"""

import json

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from h2mare.storage.provenance import (
    merge_records,
    records_for_window,
    write_provenance_for_window,
)
from h2mare.types import DateRange

_MANIFEST = [
    {
        "dataset_id": "/delayed-time",
        "dataset_type": "rep",
        "start": "2020-01-01",
        "end": "2020-06-30",
    },
    {
        "dataset_id": "/near-real-time",
        "dataset_type": "nrt",
        "start": "2020-07-01",
        "end": "2020-12-31",
    },
]


def _window(start: str, end: str) -> DateRange:
    return DateRange(pd.Timestamp(start), pd.Timestamp(end))


def _write_zarr(tmp_path, start="2020-01-01", n_days=5):
    times = pd.date_range(start, periods=n_days, freq="D")
    ds = xr.Dataset(
        {"amp": (["time", "lat", "lon"], np.ones((n_days, 2, 2)))},
        coords={"time": times, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    path = tmp_path / "eddies_2020.zarr"
    ds.to_zarr(path, consolidated=True)
    return path


def _read_records(path) -> list[dict]:
    raw = zarr.open_group(str(path), mode="r").attrs.get("source_datasets")
    return json.loads(raw) if raw else []


class TestRecordsForWindow:
    def test_window_inside_one_dataset(self):
        got = records_for_window(_MANIFEST, _window("2020-02-01", "2020-03-01"))
        assert [r["dataset_type"] for r in got] == ["rep"]

    def test_window_clipped_to_the_written_span(self):
        """A whole-year manifest entry must not claim more than was written."""
        got = records_for_window(_MANIFEST, _window("2020-02-01", "2020-03-01"))
        assert got[0]["start_date"] == "2020-02-01"
        assert got[0]["end_date"] == "2020-03-01"

    def test_window_spanning_the_boundary_splits(self):
        got = records_for_window(_MANIFEST, _window("2020-06-01", "2020-08-31"))
        assert [r["dataset_type"] for r in got] == ["rep", "nrt"]
        assert got[0]["end_date"] == "2020-06-30"
        assert got[1]["start_date"] == "2020-07-01"

    def test_non_overlapping_entries_are_dropped(self):
        got = records_for_window(_MANIFEST, _window("2020-08-01", "2020-09-01"))
        assert [r["dataset_type"] for r in got] == ["nrt"]

    def test_window_outside_the_manifest_yields_nothing(self):
        assert records_for_window(_MANIFEST, _window("2019-01-01", "2019-02-01")) == []

    def test_records_are_sorted_by_start(self):
        got = records_for_window(_MANIFEST, _window("2020-01-01", "2020-12-31"))
        assert [r["start_date"] for r in got] == sorted(r["start_date"] for r in got)


class TestMergeRecords:
    def test_new_dataset_is_added(self):
        existing = [
            {
                "dataset_id": "a",
                "dataset_type": "rep",
                "start_date": "2020-01-01",
                "end_date": "2020-06-30",
            }
        ]
        new = [
            {
                "dataset_id": "b",
                "dataset_type": "nrt",
                "start_date": "2020-07-01",
                "end_date": "2020-12-31",
            }
        ]
        assert len(merge_records(existing, new)) == 2

    def test_same_dataset_widens_rather_than_replaces(self):
        """Regression for the overwrite the generic converter path performs."""
        existing = [
            {
                "dataset_id": "a",
                "dataset_type": "nrt",
                "start_date": "2020-01-01",
                "end_date": "2020-04-20",
            }
        ]
        new = [
            {
                "dataset_id": "a",
                "dataset_type": "nrt",
                "start_date": "2020-04-21",
                "end_date": "2020-07-13",
            }
        ]

        merged = merge_records(existing, new)

        assert len(merged) == 1
        assert merged[0]["start_date"] == "2020-01-01"
        assert merged[0]["end_date"] == "2020-07-13"

    def test_empty_existing_returns_new(self):
        new = [
            {
                "dataset_id": "a",
                "dataset_type": "nrt",
                "start_date": "2020-01-01",
                "end_date": "2020-02-01",
            }
        ]
        assert merge_records([], new) == new


class TestWriteProvenanceForWindow:
    def test_writes_records_to_zarr_attrs(self, tmp_path):
        path = _write_zarr(tmp_path)

        write_provenance_for_window(
            path, _MANIFEST, _window("2020-01-01", "2020-01-05")
        )

        got = _read_records(path)
        assert [r["dataset_type"] for r in got] == ["rep"]

    def test_second_append_widens_coverage(self, tmp_path):
        """Two runs over the same period must not lose the first one's span."""
        path = _write_zarr(tmp_path)
        write_provenance_for_window(
            path, _MANIFEST, _window("2020-07-01", "2020-08-31")
        )
        write_provenance_for_window(
            path, _MANIFEST, _window("2020-09-01", "2020-10-31")
        )

        got = _read_records(path)

        assert len(got) == 1
        assert got[0]["start_date"] == "2020-07-01"
        assert got[0]["end_date"] == "2020-10-31"

    def test_window_outside_manifest_writes_nothing(self, tmp_path):
        path = _write_zarr(tmp_path)

        assert (
            write_provenance_for_window(
                path, _MANIFEST, _window("2019-01-01", "2019-02-01")
            )
            == []
        )
        assert _read_records(path) == []
