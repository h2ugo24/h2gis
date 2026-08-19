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
    MODIFIED_ATTR,
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
        starts = [r["start_date"] for r in got]
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
                "start_date": "2020-01-01",
                "end_date": "2020-02-01",
            }
        ]
        assert merge_records([], new) == new

    def test_a_republished_window_takes_the_days_from_its_old_owner(self):
        """
        CMEMS extends its reprocessed product periodically, so re-downloading
        the newly-reprocessed days means rep now supplies dates nrt supplied
        before. Widening alone would leave both claiming them.
        """
        existing = [
            {
                "dataset_id": "REP",
                "dataset_type": "rep",
                "start_date": "2025-01-01",
                "end_date": "2025-12-18",
            },
            {
                "dataset_id": "NRT",
                "dataset_type": "nrt",
                "start_date": "2025-12-19",
                "end_date": "2025-12-31",
            },
        ]
        republished = [
            {
                "dataset_id": "REP",
                "dataset_type": "rep",
                "start_date": "2025-12-19",
                "end_date": "2025-12-31",
            }
        ]

        merged = merge_records(existing, republished)

        assert [r["dataset_id"] for r in merged] == ["REP"]
        assert merged[0]["end_date"] == "2025-12-31"

    def test_a_partly_republished_window_moves_the_handover(self):
        existing = [
            {
                "dataset_id": "REP",
                "dataset_type": "rep",
                "start_date": "2025-01-01",
                "end_date": "2025-06-30",
            },
            {
                "dataset_id": "NRT",
                "dataset_type": "nrt",
                "start_date": "2025-07-01",
                "end_date": "2025-12-31",
            },
        ]
        republished = [
            {
                "dataset_id": "REP",
                "dataset_type": "rep",
                "start_date": "2025-07-01",
                "end_date": "2025-09-30",
            }
        ]

        rep, nrt = merge_records(existing, republished)

        assert rep["dataset_id"] == "REP"
        assert nrt["start_date"] == "2025-10-01", "nrt keeps only what rep left"

    def test_an_older_record_cannot_take_days_back(self):
        """Only the run that just wrote gets to supersede."""
        existing = [
            {
                "dataset_id": "REP",
                "dataset_type": "rep",
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
            }
        ]
        new = [
            {
                "dataset_id": "NRT",
                "dataset_type": "nrt",
                "start_date": "2025-12-19",
                "end_date": "2025-12-31",
            }
        ]

        merged = merge_records(existing, new)

        assert [r["dataset_id"] for r in merged] == ["REP", "NRT"]
        assert merged[0]["start_date"] == "2025-01-01"

    def test_fields_from_a_previous_layout_do_not_survive(self):
        """
        h2ds only ever merges — nothing at that level recomputes a record — so
        a merge that copied unknown keys through kept every field any past
        version wrote. It carried `requested_*` long after that layout was gone.
        """
        stale = [
            {
                "dataset_id": "a",
                "dataset_type": "rep",
                "start_date": "2020-01-01",
                "end_date": "2020-06-30",
                "requested_start": "2020-01-01",
                "requested_end": "2020-12-31",
                "delivered_days": 180,
            }
        ]

        [got] = merge_records(stale, [])

        assert set(got) == {"dataset_id", "dataset_type", "start_date", "end_date"}

    def test_per_write_fields_are_not_carried_through(self):
        """days and updated describe a write, not a span, so a merge drops
        them for annotate_covered to set again."""
        existing = [
            {
                "dataset_id": "a",
                "dataset_type": "nrt",
                "start_date": "2020-01-01",
                "end_date": "2020-04-20",
                "days": 111,
                "updated": "2020-04-20",
            }
        ]
        [got] = merge_records(existing, [])

        assert "days" not in got and "updated" not in got


class TestWriteProvenanceForWindow:
    def test_writes_records_to_zarr_attrs(self, tmp_path):
        path = _write_zarr(tmp_path)

        write_provenance_for_window(
            path, _MANIFEST, _window("2020-01-01", "2020-01-05")
        )

        got = _read_records(path)
        assert [r["dataset_type"] for r in got] == ["rep"]

    def test_two_runs_still_describe_one_file(self, tmp_path):
        """
        Two runs over the same period must not leave the file claiming only the
        second one. The fixture holds 2020-01-01..2020-01-05 throughout, and
        both runs name the same dataset, so the file says so once.
        """
        path = _write_zarr(tmp_path)
        write_provenance_for_window(
            path, _MANIFEST, _window("2020-07-01", "2020-08-31")
        )
        write_provenance_for_window(
            path, _MANIFEST, _window("2020-09-01", "2020-10-31")
        )

        got = _read_records(path)

        assert len(got) == 1
        assert got[0]["start_date"] == "2020-01-01"
        assert got[0]["end_date"] == "2020-01-05"

    def test_dates_come_from_the_file_not_the_window(self, tmp_path):
        """The window asked for July; the file holds the first five days of
        January, and that is what it has to report."""
        path = _write_zarr(tmp_path)
        write_provenance_for_window(
            path, _MANIFEST, _window("2020-07-01", "2020-08-31")
        )

        [got] = _read_records(path)

        assert got["start_date"] == "2020-01-01"
        assert got["end_date"] == "2020-01-05"
        assert got["days"] == 5

    def test_the_write_is_stamped(self, tmp_path):
        path = _write_zarr(tmp_path)
        write_provenance_for_window(
            path, _MANIFEST, _window("2020-01-01", "2020-06-30")
        )

        [got] = _read_records(path)

        assert got["updated"] == pd.Timestamp.today().strftime("%Y-%m-%d")

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


def _rec(dataset_id, kind, start, end) -> dict:
    return {
        "dataset_id": dataset_id,
        "dataset_type": kind,
        "start_date": start,
        "end_date": end,
    }


class TestAnnotateCovered:
    _RECORD = _rec("ds-rep", "rep", "2020-01-01", "2020-01-10")
    _YEAR = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    def test_the_file_is_accounted_for_end_to_end(self):
        [out] = annotate_covered([self._RECORD], self._YEAR)

        assert out["start_date"] == "2020-01-01"
        assert out["end_date"] == "2020-12-31"
        assert out["days"] == 366

    def test_a_record_naming_one_week_still_claims_the_whole_file(self):
        """
        The shape that made 2026 unreadable: a store filled week by week kept a
        record naming the most recent week, so the year appeared to hold seven
        days. One dataset means the file is made of that dataset.
        """
        last_week = _rec("ds-nrt", "nrt", "2020-12-25", "2020-12-31")

        [out] = annotate_covered([last_week], self._YEAR)

        assert out["start_date"] == "2020-01-01"
        assert out["end_date"] == "2020-12-31"

    def test_a_handover_splits_at_the_second_dataset(self):
        records = [
            _rec("REP", "rep", "2020-01-01", "2020-06-30"),
            _rec("NRT", "nrt", "2020-07-01", "2020-12-31"),
        ]

        rep, nrt = annotate_covered(records, self._YEAR)

        assert (rep["start_date"], rep["end_date"]) == ("2020-01-01", "2020-06-30")
        assert (nrt["start_date"], nrt["end_date"]) == ("2020-07-01", "2020-12-31")
        assert rep["days"] + nrt["days"] == len(self._YEAR)

    def test_days_counts_what_is_present_not_the_span(self):
        gappy = self._YEAR.drop(pd.date_range("2020-03-01", "2020-03-10"))

        [out] = annotate_covered([self._RECORD], gappy)

        assert out["start_date"] == "2020-01-01"
        assert out["end_date"] == "2020-12-31"
        assert out["days"] == 356

    def test_the_write_is_stamped(self):
        [out] = annotate_covered([self._RECORD], self._YEAR)
        assert out["updated"] == pd.Timestamp.today().strftime("%Y-%m-%d")

    def test_a_file_with_no_axis_yields_nothing(self):
        assert annotate_covered([self._RECORD], pd.DatetimeIndex([])) == []

    def test_legacy_delivered_fields_do_not_survive(self):
        legacy = {
            **self._RECORD,
            "delivered_days": 4,
            "delivered_start": "2020-01-01",
            "delivered_end": "2020-01-04",
        }
        [out] = annotate_covered([legacy], self._YEAR)

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
    def test_the_file_is_stamped_once(self, tmp_path):
        """
        One stamp for the file, not one per variable: how current a variable is
        already reads off its own end_date, and what that cannot say is when the
        file itself last changed.
        """
        path = _write_zarr(tmp_path)

        write_compiled_provenance(
            path,
            {
                "sst": [_rec("sst-rep", "rep", "2020-01-01", "2020-01-05")],
                "ssh": [_rec("ssh-rep", "rep", "2020-01-01", "2020-01-05")],
            },
        )

        attrs = zarr.open_group(str(path), mode="r").attrs
        assert attrs[MODIFIED_ATTR] == pd.Timestamp.today().strftime("%Y-%m-%d")
        for records in _read_compiled(path).values():
            assert all("updated" not in r for r in records)

    def test_the_stamp_is_rewritten_not_kept(self, tmp_path):
        path = _write_zarr(tmp_path)
        zarr.open_group(str(path), mode="r+").attrs[MODIFIED_ATTR] = "2020-01-06"

        write_compiled_provenance(
            path, {"sst": [_rec("sst-rep", "rep", "2020-01-01", "2020-01-05")]}
        )

        attrs = zarr.open_group(str(path), mode="r").attrs
        assert attrs[MODIFIED_ATTR] == pd.Timestamp.today().strftime("%Y-%m-%d")

    def test_a_config_refresh_does_not_wipe_the_stamp(self, tmp_path):
        """refresh_root_attrs replaces the root wholesale, and the stamp has no
        counterpart in config to restore it from."""
        path = _write_zarr(tmp_path)
        write_compiled_provenance(
            path, {"sst": [_rec("sst-rep", "rep", "2020-01-01", "2020-01-05")]}
        )

        refresh_root_attrs(path, {"title": "h2ds"})

        attrs = zarr.open_group(str(path), mode="r").attrs
        assert attrs[MODIFIED_ATTR] == pd.Timestamp.today().strftime("%Y-%m-%d")

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
