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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from h2mare.storage.provenance import (
    COMPILED_PROVENANCE_ATTR,
    annotate_covered,
    collect_source_datasets,
    merge_records,
    read_source_datasets,
    records_for_window,
    refresh_root_attrs,
    write_compiled_provenance,
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
        assert got[0]["requested_start"] == "2020-02-01"
        assert got[0]["requested_end"] == "2020-03-01"

    def test_only_the_requested_window_is_emitted(self):
        """The covered pair is read off the store, so it cannot be set here."""
        [got] = records_for_window(_MANIFEST, _window("2020-02-01", "2020-03-01"))
        assert "start_date" not in got and "end_date" not in got

    def test_window_spanning_the_boundary_splits(self):
        got = records_for_window(_MANIFEST, _window("2020-06-01", "2020-08-31"))
        assert [r["dataset_type"] for r in got] == ["rep", "nrt"]
        assert got[0]["requested_end"] == "2020-06-30"
        assert got[1]["requested_start"] == "2020-07-01"

    def test_non_overlapping_entries_are_dropped(self):
        got = records_for_window(_MANIFEST, _window("2020-08-01", "2020-09-01"))
        assert [r["dataset_type"] for r in got] == ["nrt"]

    def test_window_outside_the_manifest_yields_nothing(self):
        assert records_for_window(_MANIFEST, _window("2019-01-01", "2019-02-01")) == []

    def test_records_are_sorted_by_start(self):
        got = records_for_window(_MANIFEST, _window("2020-01-01", "2020-12-31"))
        starts = [r["requested_start"] for r in got]
        assert starts == sorted(starts)


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
                "requested_start": "2020-01-01",
                "requested_end": "2020-02-01",
            }
        ]
        assert merge_records([], new) == new

    def test_a_legacy_record_gains_its_requested_pair(self):
        """The old layout meant requested by start_date/end_date, so that is
        how a file written under it is read forward."""
        legacy = [
            {
                "dataset_id": "a",
                "dataset_type": "nrt",
                "start_date": "2020-01-01",
                "end_date": "2020-02-01",
            }
        ]
        [got] = merge_records([], legacy)

        assert got["requested_start"] == "2020-01-01"
        assert got["requested_end"] == "2020-02-01"


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
        assert got[0]["requested_start"] == "2020-07-01"
        assert got[0]["requested_end"] == "2020-10-31"

    def test_covered_reflects_the_store_not_the_request(self, tmp_path):
        """
        The fixture holds 2020-01-01..2020-01-05 and the request asks for July,
        so nothing was covered — which the record has to say rather than
        repeating the window back.
        """
        path = _write_zarr(tmp_path)
        write_provenance_for_window(
            path, _MANIFEST, _window("2020-07-01", "2020-08-31")
        )

        [got] = _read_records(path)

        assert got["days"] == 0
        assert "start_date" not in got

    def test_covered_is_the_overlap_with_the_store(self, tmp_path):
        path = _write_zarr(tmp_path)
        write_provenance_for_window(
            path, _MANIFEST, _window("2020-01-01", "2020-06-30")
        )

        [got] = _read_records(path)

        assert got["start_date"] == "2020-01-01"
        assert got["end_date"] == "2020-01-05"
        assert got["days"] == 5
        assert got["requested_end"] == "2020-06-30"

    def test_window_outside_manifest_writes_nothing(self, tmp_path):
        path = _write_zarr(tmp_path)

        assert (
            write_provenance_for_window(
                path, _MANIFEST, _window("2019-01-01", "2019-02-01")
            )
            == []
        )
        assert _read_records(path) == []


# ---------------------------------------------------------------------------
# annotate_covered
#
# start_date/end_date are what the store actually holds, so a reader can answer
# "which product covers which part of the archive, and where does rep hand over
# to nrt" from metadata alone. The requested window stays alongside, because
# comparing the two is what turns "did this product deliver what it was asked
# for" into a metadata comparison rather than a re-read of the data.
# ---------------------------------------------------------------------------


class TestAnnotateCovered:
    _RECORD = {
        "dataset_id": "ds-rep",
        "dataset_type": "rep",
        "requested_start": "2020-01-01",
        "requested_end": "2020-01-10",
    }

    def test_counts_the_days_inside_the_span(self):
        stored = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        [out] = annotate_covered([self._RECORD], stored)
        assert out["days"] == 10

    def test_a_missing_day_shows_up_as_a_shortfall(self):
        stored = pd.date_range("2020-01-01", "2020-01-10", freq="D").drop(
            pd.Timestamp("2020-01-05")
        )
        [out] = annotate_covered([self._RECORD], stored)
        assert out["days"] == 9

    def test_covered_dates_are_what_the_store_holds(self):
        """Not the request: the store starts on the 3rd and stops on the 8th."""
        stored = pd.date_range("2020-01-03", "2020-01-08", freq="D")
        [out] = annotate_covered([self._RECORD], stored)
        assert out["start_date"] == "2020-01-03"
        assert out["end_date"] == "2020-01-08"

    def test_days_outside_the_span_are_not_counted(self):
        stored = pd.date_range("2019-01-01", "2021-12-31", freq="D")
        [out] = annotate_covered([self._RECORD], stored)
        assert out["days"] == 10
        assert out["start_date"] == "2020-01-01"
        assert out["end_date"] == "2020-01-10"

    def test_empty_span_gets_zero_and_no_covered_dates(self):
        """Asked and received nothing, rather than a span that does not exist."""
        [out] = annotate_covered([self._RECORD], pd.DatetimeIndex([]))
        assert out["days"] == 0
        assert "start_date" not in out and "end_date" not in out

    def test_requested_span_is_preserved(self):
        [out] = annotate_covered([self._RECORD], pd.date_range("2020-01-01", periods=3))
        assert out["requested_start"] == "2020-01-01"
        assert out["requested_end"] == "2020-01-10"

    def test_a_legacy_record_upgrades_in_place(self):
        """
        Records written before the switch carry the requested window under
        start_date/end_date. Reading them as such is what lets a file correct
        itself on the next write instead of needing a migration.
        """
        legacy = {
            "dataset_id": "ds-rep",
            "dataset_type": "rep",
            "start_date": "2020-01-01",
            "end_date": "2020-01-10",
            "delivered_days": 4,
            "delivered_start": "2020-01-01",
            "delivered_end": "2020-01-04",
        }
        [out] = annotate_covered([legacy], pd.date_range("2020-01-03", "2020-01-08"))

        assert out["requested_start"] == "2020-01-01"
        assert out["requested_end"] == "2020-01-10"
        assert out["start_date"] == "2020-01-03"
        assert out["end_date"] == "2020-01-08"
        assert not [k for k in out if k.startswith("delivered_")]

    def test_is_idempotent(self):
        """Recomputed from the store, so re-running must not accumulate."""
        stored = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        once = annotate_covered([self._RECORD], stored)
        twice = annotate_covered(once, stored)
        assert twice == once


class TestMergeRecordsDropsDeliveredFields:
    def test_delivered_fields_are_not_carried_through_merge(self):
        """They cannot be combined arithmetically, so they are recomputed."""
        existing = [
            {
                "dataset_id": "ds-rep",
                "dataset_type": "rep",
                "start_date": "2020-01-01",
                "end_date": "2020-01-10",
                "delivered_days": 10,
                "delivered_start": "2020-01-01",
                "delivered_end": "2020-01-10",
            }
        ]
        new = [
            {
                "dataset_id": "ds-rep",
                "dataset_type": "rep",
                "start_date": "2020-01-11",
                "end_date": "2020-01-20",
                "delivered_days": 10,
            }
        ]

        [merged] = merge_records(existing, new)

        assert "delivered_days" not in merged
        assert (merged["start_date"], merged["end_date"]) == (
            "2020-01-01",
            "2020-01-20",
        )

    def test_span_still_widens(self):
        existing = [
            {
                "dataset_id": "ds",
                "dataset_type": "rep",
                "start_date": "2020-02-01",
                "end_date": "2020-02-10",
            }
        ]
        new = [
            {
                "dataset_id": "ds",
                "dataset_type": "rep",
                "start_date": "2020-01-01",
                "end_date": "2020-01-15",
            }
        ]

        [merged] = merge_records(existing, new)

        assert merged["start_date"] == "2020-01-01"
        assert merged["end_date"] == "2020-02-10"


# ---------------------------------------------------------------------------
# Compiled (h2ds) provenance
# ---------------------------------------------------------------------------


def _seed_source_datasets(path, records: list[dict]) -> None:
    """Put a per-variable store's provenance in place, as the converter would."""
    root = zarr.open_group(str(path), mode="r+")
    root.attrs["source_datasets"] = json.dumps(records)


def _rec(dataset_id, kind, start, end) -> dict:
    return {
        "dataset_id": dataset_id,
        "dataset_type": kind,
        "start_date": start,
        "end_date": end,
    }


def _read_compiled(path) -> dict:
    raw = zarr.open_group(str(path), mode="r").attrs.get(COMPILED_PROVENANCE_ATTR)
    return json.loads(raw) if raw else {}


class TestReadSourceDatasets:
    """Never raise: provenance is metadata, and losing it must not fail a compile."""

    def test_reads_what_the_converter_wrote(self, tmp_path):
        path = _write_zarr(tmp_path)
        _seed_source_datasets(path, [_rec("ds-a", "rep", "2020-01-01", "2020-01-05")])

        assert read_source_datasets(path) == [
            _rec("ds-a", "rep", "2020-01-01", "2020-01-05")
        ]

    def test_store_without_provenance_is_empty_not_an_error(self, tmp_path):
        assert read_source_datasets(_write_zarr(tmp_path)) == []

    def test_missing_store_is_empty_not_an_error(self, tmp_path):
        assert read_source_datasets(tmp_path / "absent.zarr") == []

    def test_malformed_json_is_empty_not_an_error(self, tmp_path):
        path = _write_zarr(tmp_path)
        zarr.open_group(str(path), mode="r+").attrs["source_datasets"] = "{not json"

        assert read_source_datasets(path) == []


class TestCollectSourceDatasets:
    def test_merges_across_every_file_in_the_window(self, tmp_path):
        """A window spans several period files, and rep can turn into nrt inside it."""
        a = _write_zarr(tmp_path / "a", start="2020-01-01")
        b = _write_zarr(tmp_path / "b", start="2021-01-01")
        _seed_source_datasets(a, [_rec("ds-rep", "rep", "2020-01-01", "2020-12-31")])
        _seed_source_datasets(b, [_rec("ds-nrt", "nrt", "2021-01-01", "2021-12-31")])

        catalog = SimpleNamespace(get_paths_in_range=lambda s, e: [str(a), str(b)])
        got = collect_source_datasets(catalog, _window("2020-01-01", "2021-12-31"))

        assert [r["dataset_id"] for r in got] == ["ds-rep", "ds-nrt"]

    def test_no_files_yields_no_records(self, tmp_path):
        catalog = SimpleNamespace(get_paths_in_range=lambda s, e: [])
        assert (
            collect_source_datasets(catalog, _window("2020-01-01", "2020-12-31")) == []
        )


class TestWriteCompiledProvenance:
    def test_keyed_by_variable_not_a_flat_list(self, tmp_path):
        """h2ds merges many sources, so a record has to say which one it came from."""
        path = _write_zarr(tmp_path)

        write_compiled_provenance(
            path,
            {
                "sst": [_rec("sst-rep", "rep", "2020-01-01", "2020-01-05")],
                "ssh": [_rec("ssh-rep", "rep", "2020-01-01", "2020-01-05")],
            },
        )

        got = _read_compiled(path)
        assert set(got) == {"sst", "ssh"}
        assert got["sst"][0]["dataset_id"] == "sst-rep"

    def test_uses_its_own_attribute_name(self, tmp_path):
        """Reusing source_datasets would hand a reader a dict where it wants a list."""
        path = _write_zarr(tmp_path)
        write_compiled_provenance(
            path, {"sst": [_rec("sst-rep", "rep", "2020-01-01", "2020-01-05")]}
        )

        assert COMPILED_PROVENANCE_ATTR != "source_datasets"
        assert _read_records(path) == []

    def test_partial_compile_keeps_the_other_variables(self, tmp_path):
        """`run -v sst` must not erase every other variable's provenance."""
        path = _write_zarr(tmp_path)
        write_compiled_provenance(
            path,
            {
                "sst": [_rec("sst-rep", "rep", "2020-01-01", "2020-01-05")],
                "ssh": [_rec("ssh-rep", "rep", "2020-01-01", "2020-01-05")],
            },
        )

        write_compiled_provenance(
            path, {"sst": [_rec("sst-rep", "rep", "2020-01-06", "2020-01-10")]}
        )

        got = _read_compiled(path)
        assert set(got) == {"sst", "ssh"}, "a one-variable compile dropped the rest"
        assert got["ssh"][0]["end_date"] == "2020-01-05"

    def test_recompiling_widens_a_variable_span(self, tmp_path):
        path = _write_zarr(tmp_path)
        write_compiled_provenance(
            path, {"sst": [_rec("sst-rep", "rep", "2020-07-01", "2020-08-31")]}
        )
        write_compiled_provenance(
            path, {"sst": [_rec("sst-rep", "rep", "2020-09-01", "2020-10-31")]}
        )

        got = _read_compiled(path)["sst"]
        assert len(got) == 1
        assert (got[0]["start_date"], got[0]["end_date"]) == (
            "2020-07-01",
            "2020-10-31",
        )

    def test_rep_and_nrt_stay_separate_records(self, tmp_path):
        """The whole point is being able to tell which dates came from which."""
        path = _write_zarr(tmp_path)
        write_compiled_provenance(
            path,
            {
                "ssh": [
                    _rec("ssh-rep", "rep", "2025-01-01", "2025-10-18"),
                    _rec("ssh-nrt", "nrt", "2025-10-19", "2025-12-31"),
                ]
            },
        )

        got = _read_compiled(path)["ssh"]
        assert [r["dataset_type"] for r in got] == ["rep", "nrt"]


class TestRefreshRootAttrs:
    """
    Root attributes must come from config, not from whatever the file was first
    created with. xr.concat keeps the first dataset's attrs, so an append leaves
    the existing globals in place — h2ds advertised a products ID block for a
    fortnight after the key was deleted from config.
    """

    def test_a_key_removed_from_config_is_removed_from_the_store(self, tmp_path):
        path = _write_zarr(tmp_path)
        zarr.open_group(str(path), mode="r+").attrs.put(
            {"title": "old", "products ID": {"sst": "stale"}}
        )

        refresh_root_attrs(path, {"title": "new"})

        got = dict(zarr.open_group(str(path), mode="r").attrs)
        assert "products ID" not in got, "updating alone would leave it forever"
        assert got["title"] == "new"

    def test_a_stale_value_is_corrected(self, tmp_path):
        path = _write_zarr(tmp_path)
        zarr.open_group(str(path), mode="r+").attrs.put({"source": "typo"})

        refresh_root_attrs(path, {"source": "corrected"})

        assert dict(zarr.open_group(str(path), mode="r").attrs)["source"] == "corrected"

    def test_provenance_survives_the_refresh(self, tmp_path):
        """It is written by the pipeline and has no counterpart in config."""
        path = _write_zarr(tmp_path)
        write_compiled_provenance(
            path, {"sst": [_rec("sst-rep", "rep", "2020-01-01", "2020-01-05")]}
        )

        refresh_root_attrs(path, {"title": "h2ds"})

        got = dict(zarr.open_group(str(path), mode="r").attrs)
        assert COMPILED_PROVENANCE_ATTR in got, "a refresh erased the provenance"
        assert json.loads(got[COMPILED_PROVENANCE_ATTR])["sst"][0]["dataset_id"] == (
            "sst-rep"
        )
        assert got["title"] == "h2ds"

    def test_returns_what_it_wrote(self, tmp_path):
        path = _write_zarr(tmp_path)
        assert refresh_root_attrs(path, {"title": "h2ds"}) == {"title": "h2ds"}
