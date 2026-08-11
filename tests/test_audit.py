"""Tests for storage/audit.py — interior gap and slice-health detection.

The load-bearing property is not "finds gaps" but "finds gaps and nothing
else". A check that also fires on chl's legitimate all-null days would be
switched off within a season, at which point it protects nothing — so the
false-positive cases below matter as much as the true-positive ones.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.storage.audit import (
    audit_zarr_file,
    check_slice_health,
    contiguous_blocks,
    format_date_blocks,
    interior_gaps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ds(dates, null_days=(), constant_days=(), seed: int = 0) -> xr.Dataset:
    times = pd.DatetimeIndex(dates)
    nulls, flats = pd.DatetimeIndex(null_days), pd.DatetimeIndex(constant_days)
    rng = np.random.default_rng(seed)
    data = rng.uniform(10.0, 30.0, size=(len(times), 3, 3))
    for i, t in enumerate(times):
        if t in nulls:
            data[i, :, :] = np.nan
        elif t in flats:
            data[i, :, :] = 7.5
    return xr.Dataset(
        {"testvar": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": [30.0, 35.0, 40.0],
            "lon": [-10.0, -5.0, 0.0],
        },
    )


def _write(ds: xr.Dataset, path):
    ds.to_zarr(path)
    return path


_JAN = pd.date_range("2020-01-01", "2020-01-10", freq="D")


# ---------------------------------------------------------------------------
# interior_gaps
# ---------------------------------------------------------------------------


class TestInteriorGaps:
    def test_finds_a_missing_middle_day(self):
        missing = interior_gaps(_JAN.drop(pd.Timestamp("2020-01-05")))
        assert list(missing) == [pd.Timestamp("2020-01-05")]

    def test_complete_range_has_no_gaps(self):
        assert len(interior_gaps(_JAN)) == 0

    def test_ignores_a_short_tail(self):
        """A store stopping short of today is provider lag, not a defect."""
        assert len(interior_gaps(pd.date_range("2020-01-01", periods=5))) == 0

    def test_finds_a_multi_day_block(self):
        dropped = _JAN.drop(pd.date_range("2020-01-04", "2020-01-07"))
        assert len(interior_gaps(dropped)) == 4

    def test_single_date_is_never_a_gap(self):
        assert len(interior_gaps(pd.DatetimeIndex(["2020-01-01"]))) == 0

    def test_empty_index_is_never_a_gap(self):
        assert len(interior_gaps(pd.DatetimeIndex([]))) == 0

    def test_duplicate_dates_do_not_create_gaps(self):
        doubled = _JAN.append(_JAN).sort_values()
        assert len(interior_gaps(doubled)) == 0

    def test_unsorted_input_is_handled(self):
        assert len(interior_gaps(_JAN[::-1])) == 0


# ---------------------------------------------------------------------------
# audit_zarr_file
# ---------------------------------------------------------------------------


class TestAuditZarrFile:
    def test_reports_the_missing_day(self, tmp_path):
        path = _write(_ds(_JAN.drop(pd.Timestamp("2020-01-05"))), tmp_path / "a.zarr")

        gap, slices, error = audit_zarr_file(path)

        assert error is None
        assert gap is not None
        assert list(gap.missing) == [pd.Timestamp("2020-01-05")]

    def test_records_the_span_it_checked(self, tmp_path):
        path = _write(_ds(_JAN.drop(pd.Timestamp("2020-01-05"))), tmp_path / "a.zarr")

        gap, _, _ = audit_zarr_file(path)

        assert gap.span == (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10"))

    def test_complete_store_reports_nothing(self, tmp_path):
        path = _write(_ds(_JAN), tmp_path / "a.zarr")

        gap, slices, error = audit_zarr_file(path)

        assert (gap, slices, error) == (None, [], None)

    def test_all_null_day_is_not_an_axis_gap(self, tmp_path):
        """chl's 1999 null days: on the axis, so structurally invisible here."""
        path = _write(
            _ds(_JAN, null_days=[pd.Timestamp("2020-01-05")]), tmp_path / "a.zarr"
        )

        gap, slices, error = audit_zarr_file(path)

        assert gap is None
        assert slices == []

    def test_all_null_day_is_found_when_values_are_checked(self, tmp_path):
        path = _write(
            _ds(_JAN, null_days=[pd.Timestamp("2020-01-05")]), tmp_path / "a.zarr"
        )

        _, slices, _ = audit_zarr_file(path, check_values=True)

        assert [s.kind for s in slices] == ["empty"]
        assert slices[0].date == pd.Timestamp("2020-01-05")

    def test_unreadable_store_is_reported_not_raised(self, tmp_path):
        broken = tmp_path / "broken.zarr"
        broken.mkdir()

        gap, slices, error = audit_zarr_file(broken)

        assert error is not None
        assert gap is None

    def test_timeless_store_is_skipped(self, tmp_path):
        ds = xr.Dataset(
            {"bathy": (["lat", "lon"], np.ones((3, 3)))},
            coords={"lat": [30.0, 35.0, 40.0], "lon": [-10.0, -5.0, 0.0]},
        )
        path = _write(ds, tmp_path / "static.zarr")

        assert audit_zarr_file(path) == (None, [], None)


# ---------------------------------------------------------------------------
# check_slice_health
#
# Replaces have_vars_unique_values, which inspected isel(time=-1) only — the
# one position that cannot reveal an interior hole — and conflated "all
# missing" with "constant", since NaN collapses to a single unique value.
# ---------------------------------------------------------------------------


class TestCheckSliceHealth:
    def test_empty_slice_is_reported_as_empty(self):
        issues = check_slice_health(_ds(_JAN, null_days=[pd.Timestamp("2020-01-05")]))
        assert [(k, d) for _, d, k, _ in issues] == [
            ("empty", pd.Timestamp("2020-01-05"))
        ]

    def test_constant_slice_is_reported_as_degenerate(self):
        issues = check_slice_health(
            _ds(_JAN, constant_days=[pd.Timestamp("2020-01-05")])
        )
        assert [k for _, _, k, _ in issues] == ["degenerate"]

    def test_empty_and_degenerate_are_not_conflated(self):
        """The old np.unique check could not tell these apart."""
        issues = check_slice_health(
            _ds(
                _JAN,
                null_days=[pd.Timestamp("2020-01-03")],
                constant_days=[pd.Timestamp("2020-01-07")],
            )
        )
        kinds = {d: k for _, d, k, _ in issues}
        assert kinds[pd.Timestamp("2020-01-03")] == "empty"
        assert kinds[pd.Timestamp("2020-01-07")] == "degenerate"

    def test_healthy_dataset_reports_nothing(self):
        assert check_slice_health(_ds(_JAN)) == []

    def test_interior_position_is_inspected_not_only_the_last(self):
        """The old check looked at isel(time=-1) and would have missed this."""
        issues = check_slice_health(_ds(_JAN, null_days=[pd.Timestamp("2020-01-04")]))
        assert issues and issues[0][1] == pd.Timestamp("2020-01-04")

    def test_timeless_variable_is_skipped(self):
        ds = xr.Dataset(
            {"bathy": (["lat", "lon"], np.full((3, 3), 5.0))},
            coords={"lat": [30.0, 35.0, 40.0], "lon": [-10.0, -5.0, 0.0]},
        )
        assert check_slice_health(ds) == []


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestContiguousBlocks:
    def test_single_run(self):
        assert contiguous_blocks(pd.date_range("2020-01-01", periods=3)) == [
            (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))
        ]

    def test_two_runs(self):
        dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-09"])
        assert len(contiguous_blocks(dates)) == 2

    def test_empty(self):
        assert contiguous_blocks(pd.DatetimeIndex([])) == []


class TestFormatDateBlocks:
    def test_single_day_renders_bare(self):
        assert format_date_blocks(pd.DatetimeIndex(["2025-06-02"])) == "2025-06-02"

    def test_run_renders_as_a_range(self):
        out = format_date_blocks(pd.date_range("2020-01-01", periods=3))
        assert out == "2020-01-01→2020-01-03"

    def test_empty_renders_as_none(self):
        assert format_date_blocks(pd.DatetimeIndex([])) == "none"

    def test_long_lists_are_truncated(self):
        out = format_date_blocks(
            pd.date_range("2020-01-01", periods=40, freq="3D"), max_blocks=3
        )
        assert "more block(s)" in out


# ---------------------------------------------------------------------------
# audit_parquet_nulls
# ---------------------------------------------------------------------------


class TestAuditParquetNulls:
    @pytest.fixture
    def store(self, tmp_path):
        from h2mare.storage.parquet_store import ParquetStore

        return ParquetStore(tmp_path / "pq")

    def test_wholly_null_column_is_found_from_footer_stats(self, store, tmp_path):
        """The realistic shape: a real float column whose every value is null."""
        import polars as pl
        from conftest import make_grid_df

        from h2mare.storage.audit import audit_parquet_nulls

        df = make_grid_df(
            [pd.Timestamp("2020-01-01").date()], variables={"sst": 20.0, "chl": 1.0}
        ).with_columns(pl.lit(None, dtype=pl.Float64).alias("chl"))
        store.add_data(df)

        findings = audit_parquet_nulls(store.parquet_root)

        assert [c for _, c in findings] == ["chl"]

    def test_null_typed_column_is_found_without_statistics(self, store):
        """A null-typed column carries no stats, so it needs the schema path."""
        from conftest import make_grid_df

        from h2mare.storage.audit import audit_parquet_nulls

        df = make_grid_df(
            [pd.Timestamp("2020-01-01").date()], variables={"sst": 20.0, "chl": 1.0}
        ).with_columns(chl=None)
        store.add_data(df)

        assert [c for _, c in audit_parquet_nulls(store.parquet_root)] == ["chl"]

    def test_populated_columns_are_not_flagged(self, store):
        from conftest import make_grid_df

        from h2mare.storage.audit import audit_parquet_nulls

        store.add_data(
            make_grid_df([pd.Timestamp("2020-01-01").date()], variables={"sst": 20.0})
        )

        assert audit_parquet_nulls(store.parquet_root) == []

    def test_coordinate_columns_are_never_flagged(self, store):
        from conftest import make_grid_df

        from h2mare.storage.audit import audit_parquet_nulls

        store.add_data(
            make_grid_df([pd.Timestamp("2020-01-01").date()], variables={"sst": 20.0})
        )

        assert not any(
            c in {"time", "lon", "lat"}
            for _, c in audit_parquet_nulls(store.parquet_root)
        )


# ---------------------------------------------------------------------------
# known_gaps
#
# A source shipping one file per day produces an *axis* hole when it skips one,
# which is indistinguishable from data the pipeline lost. AVISO has no fsle
# file for 2025-06-02 — its remote listing jumps 20250601 → 20250603 — so that
# day can never be filled. Without somewhere to record it, the checks would
# report the same unfixable day forever, and a check that cries wolf is one
# people stop reading.
# ---------------------------------------------------------------------------


class _Cfg:
    def __init__(self, known_gaps=None):
        self.known_gaps = known_gaps


class TestKnownGapDays:
    def test_single_date(self):
        from h2mare.storage.audit import known_gap_days

        assert list(known_gap_days(_Cfg(["2025-06-02"]))) == [
            pd.Timestamp("2025-06-02")
        ]

    def test_closed_interval_expands(self):
        from h2mare.storage.audit import known_gap_days

        assert len(known_gap_days(_Cfg(["2025-06-02/2025-06-05"]))) == 4

    def test_dates_and_intervals_mix(self):
        from h2mare.storage.audit import known_gap_days

        out = known_gap_days(_Cfg(["2025-01-01", "2025-06-02/2025-06-03"]))
        assert len(out) == 3

    def test_none_is_empty(self):
        from h2mare.storage.audit import known_gap_days

        assert len(known_gap_days(_Cfg(None))) == 0

    def test_malformed_entry_is_skipped_not_raised(self):
        """A typo in a suppression list must not stop a pipeline run."""
        from h2mare.storage.audit import known_gap_days

        out = known_gap_days(_Cfg(["not-a-date", "2025-06-02"]))
        assert list(out) == [pd.Timestamp("2025-06-02")]

    def test_result_is_deduplicated_and_sorted(self):
        from h2mare.storage.audit import known_gap_days

        out = known_gap_days(_Cfg(["2025-06-03", "2025-06-02", "2025-06-03"]))
        assert list(out) == [pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-03")]


class TestKnownGapsSuppressReporting:
    def test_a_known_gap_is_not_reported(self, tmp_path):
        path = _write(_ds(_JAN.drop(pd.Timestamp("2020-01-05"))), tmp_path / "a.zarr")

        gap, _, _ = audit_zarr_file(
            path, known_gaps=pd.DatetimeIndex([pd.Timestamp("2020-01-05")])
        )

        assert gap is None

    def test_an_unlisted_gap_is_still_reported(self, tmp_path):
        path = _write(_ds(_JAN.drop(pd.Timestamp("2020-01-05"))), tmp_path / "a.zarr")

        gap, _, _ = audit_zarr_file(
            path, known_gaps=pd.DatetimeIndex([pd.Timestamp("2020-01-08")])
        )

        assert list(gap.missing) == [pd.Timestamp("2020-01-05")]

    def test_only_the_listed_days_are_dropped(self, tmp_path):
        dropped = _JAN.drop(pd.DatetimeIndex(["2020-01-05", "2020-01-08"]))
        path = _write(_ds(dropped), tmp_path / "a.zarr")

        gap, _, _ = audit_zarr_file(
            path, known_gaps=pd.DatetimeIndex([pd.Timestamp("2020-01-05")])
        )

        assert list(gap.missing) == [pd.Timestamp("2020-01-08")]


class TestVarAuditKnownGaps:
    def test_count_derives_from_the_dates(self, tmp_path):
        from h2mare.storage.audit import VarAudit

        v = VarAudit(
            var_key="fsle",
            store_root=tmp_path,
            n_files=1,
            gaps=[],
            slices=[],
            errors=[],
            known_gaps=pd.DatetimeIndex(["2025-06-02", "2025-06-03"]),
        )

        assert v.n_known_gaps == 2

    def test_defaults_to_none_suppressed(self, tmp_path):
        from h2mare.storage.audit import VarAudit

        v = VarAudit(
            var_key="sst",
            store_root=tmp_path,
            n_files=1,
            gaps=[],
            slices=[],
            errors=[],
        )

        assert v.n_known_gaps == 0
        assert len(v.known_gaps) == 0

    def test_suppressed_days_do_not_make_a_var_fail(self, tmp_path):
        from h2mare.storage.audit import VarAudit

        v = VarAudit(
            var_key="fsle",
            store_root=tmp_path,
            n_files=1,
            gaps=[],
            slices=[],
            errors=[],
            known_gaps=pd.DatetimeIndex(["2025-06-02"]),
        )

        assert v.ok is True
