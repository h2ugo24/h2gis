"""Tests for ZarrCatalog: build_file_path, dataset column, and provenance sidecars."""

import json
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import MagicMock

import msgspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger

from h2mare.models import AppConfig
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import BBox, DateRange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTRY = {
    "local_folder": "sst",
    "source_vars": ["analysed_sst"],
    "dataset_id_rep": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    "source": "cmems",
    "archive_raw": False,
    "pattern": r".*\.nc",
}


def _make_app_config(time_step: str = "daily") -> AppConfig:
    return msgspec.convert(
        {"variables": {"sst": {**_ENTRY, "time_step": time_step}}, "secrets": {}},
        AppConfig,
    )


def _make_catalog(tmp_path, time_step: str = "daily") -> ZarrCatalog:
    return ZarrCatalog(
        "sst",
        app_config=_make_app_config(time_step),
        store_root=tmp_path,
        auto_refresh=False,
    )


def _make_ds(start: str = "2020-01-01", n_days: int = 5) -> xr.Dataset:
    times = pd.date_range(start, periods=n_days, freq="D")
    data = np.ones((n_days, 3, 3))
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": [30.0, 35.0, 40.0],
            "lon": [-10.0, -5.0, 0.0],
        },
    )


# ---------------------------------------------------------------------------
# build_file_path
# ---------------------------------------------------------------------------


class TestBuildFilePath:
    def test_default_uses_var_key(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds(), "year")
        assert "sst" in path.name

    def test_default_includes_source(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds(), "year")
        assert "cmems" in path.name

    def test_default_ends_with_zarr(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds(), "year")
        assert path.suffix == ".zarr"

    def test_name_key_replaces_var_key(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(
            _make_ds(), "year", name_key="cmems_mod_glo_phy_my"
        )
        assert "cmems_mod_glo_phy_my" in path.name

    def test_name_key_excludes_var_key(self, tmp_path):
        """When name_key is provided, var_key (sst) must not appear in the stem."""
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds(), "year", name_key="override-id")
        # stem: "cmems_override-id_<label>" — sst should not be in it
        assert "sst" not in path.stem

    def test_store_root_override(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        other_root = tmp_path / "other"
        path = catalog.build_file_path(_make_ds(), "year", store_root=other_root)
        assert path.parent == other_root

    def test_year_format_contains_year(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds("2021-03-01"), "year")
        assert "2021" in path.name

    def test_yearmonth_format_contains_month(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        path = catalog.build_file_path(_make_ds("2021-03-01"), "yearmonth")
        assert "2021" in path.name
        assert "03" in path.name


# ---------------------------------------------------------------------------
# Helpers for dataset / provenance tests
# ---------------------------------------------------------------------------

_ENTRY_WITH_NRT = {
    **_ENTRY,
    "dataset_id_nrt": "cmems_obs-sl_glo_phy-ssh_nrt",
}


def _make_app_config_with_nrt() -> AppConfig:
    return msgspec.convert(
        {"variables": {"sst": _ENTRY_WITH_NRT}, "secrets": {}},
        AppConfig,
    )


def _make_catalog_with_nrt(tmp_path) -> ZarrCatalog:
    return ZarrCatalog(
        "sst",
        app_config=_make_app_config_with_nrt(),
        store_root=tmp_path,
        auto_refresh=False,
    )


def _write_zarr(store_root, ds, name="test.zarr"):
    """Write a consolidated zarr so xr.open_zarr(consolidated=True) works in tests."""
    path = store_root / name
    ds.to_zarr(path, consolidated=True)
    return path


def _two_row_df(zarr_path) -> pd.DataFrame:
    """Minimal DataFrame with rep and nrt rows for the same zarr path."""
    p = str(zarr_path)
    return pd.DataFrame(
        [
            {
                "path": p,
                "filename": zarr_path.name,
                "start_date": pd.Timestamp("2023-01-01"),
                "end_date": pd.Timestamp("2023-06-30"),
            },
            {
                "path": p,
                "filename": zarr_path.name,
                "start_date": pd.Timestamp("2023-07-01"),
                "end_date": pd.Timestamp("2023-12-31"),
            },
        ]
    )


# ---------------------------------------------------------------------------
# dataset column and provenance sidecar tests
# ---------------------------------------------------------------------------


class TestDatasetColumn:
    def test_no_sidecar_returns_single_row_with_rep_id(self, tmp_path):
        ds = _make_ds("2023-01-01", n_days=10)
        zarr_path = _write_zarr(tmp_path, ds)
        catalog = _make_catalog(tmp_path)

        rows = catalog._index._scanner._extract_zarr_metadata(zarr_path)

        assert len(rows) == 1
        assert rows[0]["dataset"] == _ENTRY["dataset_id_rep"]

    def test_zarr_attrs_two_sources_returns_two_rows(self, tmp_path):
        import zarr

        ds = _make_ds("2023-01-01", n_days=365)
        zarr_path = _write_zarr(tmp_path, ds)
        sources = [
            {
                "dataset_id": "REP_ID",
                "dataset_type": "rep",
                "start_date": "2023-01-01",
                "end_date": "2023-06-30",
            },
            {
                "dataset_id": "NRT_ID",
                "dataset_type": "nrt",
                "start_date": "2023-07-01",
                "end_date": "2023-12-31",
            },
        ]
        root = zarr.open_group(str(zarr_path), mode="r+")
        root.attrs["source_datasets"] = json.dumps(sources)
        zarr.consolidate_metadata(str(zarr_path))
        catalog = _make_catalog(tmp_path)

        rows = catalog._index._scanner._extract_zarr_metadata(zarr_path)

        assert len(rows) == 2
        assert rows[0]["dataset"] == "REP_ID"
        assert rows[1]["dataset"] == "NRT_ID"
        assert rows[0]["end_date"] < rows[1]["start_date"]

    def test_sidecar_fallback_still_works_for_old_files(self, tmp_path):
        ds = _make_ds("2023-01-01", n_days=365)
        zarr_path = _write_zarr(tmp_path, ds)
        sidecar = zarr_path.parent / (zarr_path.stem + "_prov.json")
        sidecar.write_text(
            json.dumps(
                [
                    {
                        "dataset_id": "REP_ID",
                        "dataset_type": "rep",
                        "start_date": "2023-01-01",
                        "end_date": "2023-06-30",
                    },
                    {
                        "dataset_id": "NRT_ID",
                        "dataset_type": "nrt",
                        "start_date": "2023-07-01",
                        "end_date": "2023-12-31",
                    },
                ]
            )
        )
        catalog = _make_catalog(tmp_path)

        rows = catalog._index._scanner._extract_zarr_metadata(zarr_path)

        assert len(rows) == 2
        assert rows[0]["dataset"] == "REP_ID"
        assert rows[1]["dataset"] == "NRT_ID"

    def test_get_paths_in_range_deduplicates(self, tmp_path):
        zarr_path = tmp_path / "dummy.zarr"
        catalog = _make_catalog(tmp_path)
        catalog._index._df_cache = _two_row_df(zarr_path)

        result = catalog.get_paths_in_range("2023-01-01", "2023-12-31")

        assert result == [str(zarr_path)]

    def test_map_dates_to_paths_with_split_rows(self, tmp_path):
        zarr_path = tmp_path / "dummy.zarr"
        catalog = _make_catalog(tmp_path)
        catalog._index._df_cache = _two_row_df(zarr_path)

        result = catalog.map_dates_to_paths(["2023-03-15", "2023-09-20"])

        assert set(result.keys()) == {str(zarr_path)}
        assert len(result[str(zarr_path)]) == 2

    def test_load_from_disk_adds_dataset_column_for_old_parquet(self, tmp_path):
        old_df = pd.DataFrame(
            [
                {
                    "path": "/p/a.zarr",
                    "filename": "a.zarr",
                    "start_date": pd.Timestamp("2020-01-01"),
                    "end_date": pd.Timestamp("2020-12-31"),
                }
            ]
        )
        catalog = _make_catalog(tmp_path)
        catalog.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        old_df.to_parquet(catalog.catalog_path, index=False)

        df = catalog._index._load_from_disk()

        assert "dataset" in df.columns
        assert (df["dataset"] == _ENTRY["dataset_id_rep"]).all()

    def test_backfill_provenance_splits_boundary_file(self, tmp_path):
        import zarr

        ds = _make_ds("2023-01-01", n_days=365)
        zarr_path = _write_zarr(tmp_path, ds)
        catalog = _make_catalog_with_nrt(tmp_path)

        n = catalog.backfill_provenance("2023-06-30")

        assert n == 1
        # Provenance must be in zarr attrs, not a sidecar file
        root = zarr.open_group(str(zarr_path), mode="r")
        sources = json.loads(root.attrs["source_datasets"])
        assert len(sources) == 2
        assert sources[0]["dataset_type"] == "rep"
        assert sources[1]["dataset_type"] == "nrt"
        prov_file = zarr_path.parent / (zarr_path.stem + "_prov.json")
        assert not prov_file.exists()
        df = catalog.df
        assert df["path"].nunique() == 1
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Variable coverage: non-null end + ndarray-cell handling
# ---------------------------------------------------------------------------


def _make_padded_ds(start: str, n_days: int, n_valid: int, var: str) -> xr.Dataset:
    """Dataset where *var* has real data for the first *n_valid* days, NaN after."""
    times = pd.date_range(start, periods=n_days, freq="D")
    data = np.ones((n_days, 3, 3))
    data[n_valid:] = np.nan
    return xr.Dataset(
        {var: (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": [30.0, 35.0, 40.0], "lon": [-10.0, -5.0, 0.0]},
    )


def _df_for_zarr(zarr_path, variables, end_date) -> pd.DataFrame:
    """Single-row catalog df. ``variables`` may be a list or ndarray on purpose."""
    return pd.DataFrame(
        [
            {
                "path": str(zarr_path),
                "filename": zarr_path.name,
                "variables": variables,
                "start_date": pd.Timestamp("2026-01-01"),
                "end_date": pd.Timestamp(end_date),
            }
        ]
    )


class TestVarsNonnullEnd:
    def test_nonnull_end_ignores_nan_tail(self, tmp_path):
        # ac_amp has data for 10 days then NaN-padded to 15 (mimics a lagging
        # variable padded out to the global end by the compiler's outer merge).
        ds = _make_padded_ds("2026-05-01", n_days=15, n_valid=10, var="ac_amp")
        zarr_path = _write_zarr(tmp_path, ds, name="h2ds_2026.zarr")
        catalog = _make_catalog(tmp_path)
        catalog._index._df_cache = _df_for_zarr(zarr_path, ["ac_amp"], "2026-05-15")

        result = catalog.get_vars_nonnull_end(["ac_amp"])

        # Real extent is day 10 (2026-05-10), not the file end (2026-05-15).
        assert result["ac_amp"] == pd.Timestamp("2026-05-10")

    def test_nonnull_end_handles_ndarray_variables_cell(self, tmp_path):
        # Catalogs loaded from Parquet hold the variables cell as an ndarray.
        ds = _make_padded_ds("2026-05-01", n_days=15, n_valid=12, var="ac_amp")
        zarr_path = _write_zarr(tmp_path, ds, name="h2ds_2026.zarr")
        catalog = _make_catalog(tmp_path)
        catalog._index._df_cache = _df_for_zarr(
            zarr_path, np.array(["ac_amp"]), "2026-05-15"
        )

        result = catalog.get_vars_nonnull_end(["ac_amp"])

        assert result["ac_amp"] == pd.Timestamp("2026-05-12")

    def test_absent_variable_omitted(self, tmp_path):
        ds = _make_padded_ds("2026-05-01", n_days=5, n_valid=5, var="ac_amp")
        zarr_path = _write_zarr(tmp_path, ds, name="h2ds_2026.zarr")
        catalog = _make_catalog(tmp_path)
        catalog._index._df_cache = _df_for_zarr(zarr_path, ["ac_amp"], "2026-05-05")

        assert catalog.get_vars_nonnull_end(["not_a_var"]) == {}

    def test_get_var_time_coverage_handles_ndarray_cell(self, tmp_path):
        # Regression: ndarray cells previously made membership checks always
        # false, so this silently returned None for disk-loaded catalogs.
        catalog = _make_catalog(tmp_path)
        catalog._index._df_cache = _df_for_zarr(
            tmp_path / "h2ds_2026.zarr", np.array(["ac_amp", "c_amp"]), "2026-05-15"
        )

        cov = catalog.get_var_time_coverage("ac_amp")

        assert cov is not None
        assert cov.end == pd.Timestamp("2026-05-15")


# ---------------------------------------------------------------------------
# get_bbox
# ---------------------------------------------------------------------------


class TestGetBbox:
    def test_returns_none_when_unset(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        assert catalog.get_bbox() is None

    def test_converts_tuple_to_bbox(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        catalog.var_config.bbox = (-10.0, 30.0, 0.0, 40.0)
        assert catalog.get_bbox() == BBox(-10.0, 30.0, 0.0, 40.0)

    def test_returns_bbox_instance_unchanged(self, tmp_path):
        # Regression: a BBox-typed config bbox used to fall through the
        # isinstance check and silently return None.
        catalog = _make_catalog(tmp_path)
        catalog.var_config.bbox = BBox(-10.0, 30.0, 0.0, 40.0)
        assert catalog.get_bbox() == BBox(-10.0, 30.0, 0.0, 40.0)


# ---------------------------------------------------------------------------
# get_store_bbox
# ---------------------------------------------------------------------------


def _df_with_extent(rows: Sequence[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Catalog df carrying only what the extent union reads."""
    return pd.DataFrame(
        [
            {"path": f"f{i}.zarr", "xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1}
            for i, (x0, y0, x1, y1) in enumerate(rows)
        ]
    )


class TestGetStoreBbox:
    def test_returns_none_for_an_empty_catalog(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        catalog._index._df_cache = pd.DataFrame()
        assert catalog.get_store_bbox() is None

    def test_unions_the_extents_of_every_file(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        catalog._index._df_cache = _df_with_extent(
            [(-10.0, 30.0, 0.0, 40.0), (-5.0, 25.0, 10.0, 35.0)]
        )
        assert catalog.get_store_bbox() == BBox(-10.0, 25.0, 10.0, 40.0)

    def test_reports_the_store_not_the_configured_bbox(self, tmp_path):
        # The point of the method: config says what was asked for, the files
        # say what arrived. A wider store must not be reported as the request.
        catalog = _make_catalog(tmp_path)
        catalog.var_config.bbox = BBox(-10.0, 30.0, 0.0, 40.0)
        catalog._index._df_cache = _df_with_extent([(-20.0, 20.0, 10.0, 50.0)])

        assert catalog.get_store_bbox() == BBox(-20.0, 20.0, 10.0, 50.0)
        assert catalog.get_bbox() == BBox(-10.0, 30.0, 0.0, 40.0)

    def test_returns_none_when_the_extent_columns_are_absent(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        catalog._index._df_cache = _df_for_zarr(
            tmp_path / "h2ds_2026.zarr", ["sst"], "2026-05-15"
        )
        assert catalog.get_store_bbox() is None

    def test_returns_none_for_a_degenerate_extent(self, tmp_path):
        # A single-cell store has xmin == xmax, which BBox rejects; the
        # inspector must degrade to "unknown" rather than raise.
        catalog = _make_catalog(tmp_path)
        catalog._index._df_cache = _df_with_extent([(0.0, 0.0, 0.0, 0.0)])
        assert catalog.get_store_bbox() is None

    def test_returns_none_when_extents_are_nan(self, tmp_path):
        catalog = _make_catalog(tmp_path)
        catalog._index._df_cache = _df_with_extent([(float("nan"), 30.0, 0.0, 40.0)])
        assert catalog.get_store_bbox() is None


# ---------------------------------------------------------------------------
# Catalog cache semantics (_df_cache / auto_refresh)
#
# Characterization tests over ZarrIndex, driven through the ZarrCatalog facade
# so both the delegation and the underlying contract stay covered. ``_df_cache``
# is the only mutable state involved, written solely by refresh/df/reload.
# They are not regression tests for a bug — they pin the refresh/scan contract
# so it stays observable as this class is split up.
# ---------------------------------------------------------------------------


class _FakeScanner:
    """Counting stand-in for ZarrDirectoryScanner.

    Records how often the store is scanned and how often it is interrogated
    for changes, so tests can assert on rescan behaviour rather than on IO.
    """

    def __init__(self, records=None, changes: bool = False):
        self._records = records if records is not None else []
        self.changes = changes
        self.scan_calls = 0
        self.has_changes_calls = 0
        self.reset_calls = 0

    def scan(self):
        self.scan_calls += 1
        return list(self._records)

    def has_changes(self) -> bool:
        self.has_changes_calls += 1
        return self.changes

    def reset(self) -> None:
        self.reset_calls += 1


def _record(path, start="2020-01-01", end="2020-12-31") -> dict:
    return {
        "path": str(path),
        "filename": path.name,
        "start_date": pd.Timestamp(start),
        "end_date": pd.Timestamp(end),
    }


def _catalog_with_scanner(tmp_path, scanner, *, auto_refresh: bool = False):
    """Catalog wired to *scanner*, with metadata kept inside tmp_path.

    Built with auto_refresh=False so no scan happens during __init__; the flag
    is applied afterwards, once the fake scanner is in place.
    """
    catalog = ZarrCatalog(
        "sst",
        app_config=_make_app_config(),
        store_root=tmp_path,
        metadata_root=tmp_path / "metadata",
        auto_refresh=False,
    )
    catalog._index._scanner = scanner
    catalog.auto_refresh = auto_refresh
    return catalog


class TestCacheSemantics:
    def test_auto_refresh_false_scans_once(self, tmp_path):
        """With auto_refresh=False the store is scanned once and cached."""
        scanner = _FakeScanner([_record(tmp_path / "a.zarr")], changes=True)
        catalog = _catalog_with_scanner(tmp_path, scanner)

        catalog.df
        catalog.df

        assert scanner.scan_calls == 1

    def test_auto_refresh_true_rechecks_on_every_access(self, tmp_path):
        """With auto_refresh=True every .df access re-interrogates the store.

        This is the documented cost behind the snapshot comment in
        map_dates_to_paths — one directory stat per access, not per call site.
        """
        scanner = _FakeScanner(changes=False)
        catalog = _catalog_with_scanner(tmp_path, scanner, auto_refresh=True)
        catalog._index._df_cache = pd.DataFrame([_record(tmp_path / "a.zarr")])

        catalog.df
        catalog.df

        assert scanner.has_changes_calls == 2
        assert scanner.scan_calls == 0

    def test_refresh_uses_cache_when_no_changes(self, tmp_path):
        scanner = _FakeScanner(changes=False)
        catalog = _catalog_with_scanner(tmp_path, scanner)
        catalog._index._df_cache = pd.DataFrame([_record(tmp_path / "a.zarr")])

        catalog.refresh()

        assert scanner.scan_calls == 0

    def test_refresh_force_rescans_when_no_changes(self, tmp_path):
        scanner = _FakeScanner([_record(tmp_path / "a.zarr")], changes=False)
        catalog = _catalog_with_scanner(tmp_path, scanner)
        catalog._index._df_cache = pd.DataFrame([_record(tmp_path / "a.zarr")])

        catalog.refresh(force=True)

        assert scanner.scan_calls == 1

    def test_reload_clears_cache_and_resets_scanner(self, tmp_path):
        scanner = _FakeScanner([_record(tmp_path / "a.zarr")], changes=False)
        catalog = _catalog_with_scanner(tmp_path, scanner)
        catalog._index._df_cache = pd.DataFrame([_record(tmp_path / "a.zarr")])

        catalog.reload()

        assert scanner.reset_calls == 1
        assert scanner.scan_calls == 1

    def test_missing_catalog_file_triggers_initial_scan(self, tmp_path):
        """No catalog parquet on disk: the store is scanned to build one."""
        scanner = _FakeScanner([_record(tmp_path / "a.zarr")], changes=False)
        catalog = _catalog_with_scanner(tmp_path, scanner)
        assert not catalog.exists()

        catalog.refresh()

        assert scanner.scan_calls == 1
        assert catalog.exists()

    def test_stale_catalog_on_disk_triggers_rescan(self, tmp_path):
        """A catalog that disagrees with the store directory is rebuilt.

        The scanner reports no changes, so only the disk-vs-catalog filename
        comparison can catch that b.zarr was added out of band.
        """
        (tmp_path / "a.zarr").mkdir()
        (tmp_path / "b.zarr").mkdir()
        scanner = _FakeScanner(
            [_record(tmp_path / "a.zarr"), _record(tmp_path / "b.zarr")],
            changes=False,
        )
        catalog = _catalog_with_scanner(tmp_path, scanner)
        # Catalog on disk knows about a.zarr only.
        catalog.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([_record(tmp_path / "a.zarr")]).to_parquet(
            catalog.catalog_path, index=False
        )

        df = catalog.refresh()

        assert scanner.scan_calls == 1
        assert set(df["filename"]) == {"a.zarr", "b.zarr"}

    def test_fresh_catalog_on_disk_is_loaded_without_rescan(self, tmp_path):
        """Catalog and store agree: load from parquet, do not rescan."""
        (tmp_path / "a.zarr").mkdir()
        scanner = _FakeScanner([_record(tmp_path / "a.zarr")], changes=False)
        catalog = _catalog_with_scanner(tmp_path, scanner)
        catalog.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([_record(tmp_path / "a.zarr")]).to_parquet(
            catalog.catalog_path, index=False
        )

        catalog.refresh()

        assert scanner.scan_calls == 0

    def test_has_changes_delegates_to_scanner(self, tmp_path):
        scanner = _FakeScanner(changes=True)
        catalog = _catalog_with_scanner(tmp_path, scanner)

        assert catalog.has_changes() is True
        assert scanner.has_changes_calls == 1


# ---------------------------------------------------------------------------
# open_dataset routing
#
# Characterization tests for the dispatch contract only: the sparse/range
# split, the mutually-exclusive-modes guard, the no-argument fallback and
# bbox coercion. The zarr IO inside each branch is stubbed out on purpose.
# ---------------------------------------------------------------------------


@pytest.fixture
def routed_catalog(tmp_path, monkeypatch):
    """Catalog whose two open paths are replaced by call recorders."""
    catalog = ZarrCatalog(
        "sst",
        app_config=_make_app_config(),
        store_root=tmp_path,
        metadata_root=tmp_path / "metadata",
        auto_refresh=False,
    )
    calls: dict = {}

    def _sparse(**kwargs):
        calls["mode"] = "sparse"
        calls["kwargs"] = kwargs
        return "sparse-ds"

    def _range(**kwargs):
        calls["mode"] = "range"
        calls["kwargs"] = kwargs
        return "range-ds"

    monkeypatch.setattr(catalog._reader, "_open_sparse_dates", _sparse)
    monkeypatch.setattr(catalog._reader, "_open_date_range", _range)
    return catalog, calls


class TestOpenDatasetRouting:
    def test_dates_route_to_sparse(self, routed_catalog):
        catalog, calls = routed_catalog

        result = catalog.open_dataset(dates=["2020-01-15", "2020-06-20"])

        assert result == "sparse-ds"
        assert calls["mode"] == "sparse"
        assert calls["kwargs"]["dates"] == ["2020-01-15", "2020-06-20"]

    def test_start_end_route_to_date_range(self, routed_catalog):
        catalog, calls = routed_catalog

        result = catalog.open_dataset(start_date="2020-01-01", end_date="2020-12-31")

        assert result == "range-ds"
        assert calls["mode"] == "range"
        assert calls["kwargs"]["start_date"] == "2020-01-01"
        assert calls["kwargs"]["end_date"] == "2020-12-31"

    def test_start_only_routes_to_date_range(self, routed_catalog):
        catalog, calls = routed_catalog

        catalog.open_dataset(start_date="2020-01-01")

        assert calls["mode"] == "range"
        assert calls["kwargs"]["end_date"] is None

    def test_both_modes_raise(self, routed_catalog):
        catalog, calls = routed_catalog

        with pytest.raises(ValueError, match="Cannot use both"):
            catalog.open_dataset(dates=["2020-01-15"], start_date="2020-01-01")

        assert calls == {}

    def test_no_args_falls_back_to_time_coverage(self, routed_catalog, tmp_path):
        """With no dates at all, the full catalog extent becomes the window."""
        catalog, calls = routed_catalog
        catalog._index._df_cache = pd.DataFrame(
            [
                _record(tmp_path / "a.zarr", "2020-01-01", "2020-06-30"),
                _record(tmp_path / "b.zarr", "2020-07-01", "2020-12-31"),
            ]
        )

        catalog.open_dataset()

        assert calls["mode"] == "range"
        assert calls["kwargs"]["start_date"] == pd.Timestamp("2020-01-01")
        assert calls["kwargs"]["end_date"] == pd.Timestamp("2020-12-31")

    def test_no_args_without_coverage_raises(self, routed_catalog):
        catalog, calls = routed_catalog
        catalog._index._df_cache = pd.DataFrame()

        with pytest.raises(ValueError, match="Please provide sparse"):
            catalog.open_dataset()

        assert calls == {}

    def test_bbox_tuple_is_coerced(self, routed_catalog):
        """A raw tuple is normalised to BBox before reaching the open path."""
        catalog, calls = routed_catalog

        catalog.open_dataset(dates=["2020-01-15"], bbox=(-10.0, 30.0, 0.0, 40.0))

        assert calls["kwargs"]["bbox"] == BBox(-10.0, 30.0, 0.0, 40.0)

    def test_bbox_instance_passes_through(self, routed_catalog):
        catalog, calls = routed_catalog
        bbox = BBox(-10.0, 30.0, 0.0, 40.0)

        catalog.open_dataset(dates=["2020-01-15"], bbox=bbox)

        assert calls["kwargs"]["bbox"] is bbox


# ---------------------------------------------------------------------------
# open_dataset — integration
#
# These drive the real opening paths against real Zarr stores on disk:
# _open_sparse_dates, _open_date_range, _preprocess_dataset, _apply_bbox and
# _normalize_time. Nothing is mocked — the catalog scans the files it is given
# and xr.open_mfdataset actually reads them.
#
# The routing tests above cover which branch runs; these cover what each branch
# does. Together they are the safety net for moving this logic into its own
# class.
# ---------------------------------------------------------------------------

# 2-degree grid, so a one-cell bbox pad is unambiguous in assertions.
_LATS = [30.0, 32.0, 34.0, 36.0, 38.0, 40.0]
_LONS = [-10.0, -8.0, -6.0, -4.0, -2.0, 0.0]


def _grid_ds(
    start: str,
    n_days: int,
    *,
    variables: Sequence[str] = ("sst",),
    hour: int = 0,
    lat_descending: bool = False,
) -> xr.Dataset:
    """Gridded dataset on the 6x6 test grid, one value per (time, lat, lon)."""
    times = pd.date_range(f"{start} {hour:02d}:00", periods=n_days, freq="D")
    lats = list(reversed(_LATS)) if lat_descending else list(_LATS)
    data = {
        name: (
            ["time", "lat", "lon"],
            np.full((n_days, len(lats), len(_LONS)), float(i + 1)),
        )
        for i, name in enumerate(variables)
    }
    return xr.Dataset(
        data,
        coords={"time": times, "lat": lats, "lon": list(_LONS)},
    )


def _scanned_catalog(tmp_path, datasets: dict, *, verbose: bool = False) -> ZarrCatalog:
    """Write *datasets* as zarr stores, then return a catalog that scanned them.

    Built with auto_refresh=True and a tmp metadata root, so the catalog rows
    come from the real scanner rather than an injected DataFrame.
    """
    for name, ds in datasets.items():
        _write_zarr(tmp_path, ds, name=name)
    return ZarrCatalog(
        "sst",
        app_config=_make_app_config(),
        store_root=tmp_path,
        metadata_root=tmp_path / "metadata",
        auto_refresh=True,
        verbose=verbose,
    )


@pytest.fixture
def warnings_logged():
    """Collect loguru WARNING messages (the catalog logs via loguru, not stdlib)."""
    messages: list[str] = []
    sink_id = logger.add(
        lambda m: messages.append(m.record["message"]), level="WARNING"
    )
    yield messages
    logger.remove(sink_id)


@pytest.fixture
def two_file_catalog(tmp_path):
    """Catalog over two zarr files: 2020-01-01..01-10 and 2020-02-01..02-10."""
    return _scanned_catalog(
        tmp_path,
        {
            "sst_2020a.zarr": _grid_ds("2020-01-01", 10),
            "sst_2020b.zarr": _grid_ds("2020-02-01", 10),
        },
    )


class TestOpenDatasetIntegration:
    def test_catalog_scanned_both_files(self, two_file_catalog):
        """Guard for the tests below: the fixture really did index two stores."""
        assert two_file_catalog.df["path"].nunique() == 2

    # ---- date range mode ----

    def test_range_spans_both_files(self, two_file_catalog):
        ds = two_file_catalog.open_dataset(
            start_date="2020-01-01", end_date="2020-02-10"
        )
        assert ds.sizes["time"] == 20
        assert pd.Timestamp(ds.time.values[0]) == pd.Timestamp("2020-01-01")
        assert pd.Timestamp(ds.time.values[-1]) == pd.Timestamp("2020-02-10")

    def test_range_within_one_file(self, two_file_catalog):
        ds = two_file_catalog.open_dataset(
            start_date="2020-01-03", end_date="2020-01-05"
        )
        assert ds.sizes["time"] == 3

    def test_range_start_before_coverage_returns_available_data(self, two_file_catalog):
        """A too-early start opens from what exists rather than failing."""
        ds = two_file_catalog.open_dataset(
            start_date="2015-01-01", end_date="2020-01-05"
        )
        assert pd.Timestamp(ds.time.values[0]) == pd.Timestamp("2020-01-01")

    def test_range_end_after_coverage_returns_available_data(self, two_file_catalog):
        ds = two_file_catalog.open_dataset(
            start_date="2020-02-01", end_date="2030-01-01"
        )
        assert pd.Timestamp(ds.time.values[-1]) == pd.Timestamp("2020-02-10")

    def test_out_of_range_request_warns_about_clamping(self, tmp_path, warnings_logged):
        """The clamp itself is only observable in the log.

        ``ds.sel(time=slice(...))`` clips to what exists regardless, so asserting
        on the returned times cannot distinguish a clamped request from an
        unclamped one — only the warning can.
        """
        catalog = _scanned_catalog(
            tmp_path, {"sst_2020.zarr": _grid_ds("2020-01-01", 10)}, verbose=True
        )

        catalog.open_dataset(start_date="2015-01-01", end_date="2030-01-01")

        assert sum("not available" in m for m in warnings_logged) == 2

    def test_in_range_request_does_not_warn(self, tmp_path, warnings_logged):
        catalog = _scanned_catalog(
            tmp_path, {"sst_2020.zarr": _grid_ds("2020-01-01", 10)}, verbose=True
        )

        catalog.open_dataset(start_date="2020-01-02", end_date="2020-01-05")

        assert not any("not available" in m for m in warnings_logged)

    def test_range_entirely_outside_coverage_raises(self, two_file_catalog):
        with pytest.raises(FileNotFoundError, match="No zarr files found"):
            two_file_catalog.open_dataset(
                start_date="2030-01-01", end_date="2030-02-01"
            )

    def test_no_dates_opens_full_extent(self, two_file_catalog):
        ds = two_file_catalog.open_dataset()
        assert ds.sizes["time"] == 20

    # ---- sparse dates mode ----

    def test_sparse_dates_across_files(self, two_file_catalog):
        ds = two_file_catalog.open_dataset(dates=["2020-01-05", "2020-02-03"])
        got = [pd.Timestamp(t) for t in ds.time.values]
        assert got == [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-02-03")]

    def test_sparse_dates_partially_missing_returns_the_rest(self, two_file_catalog):
        """A date inside the store span but absent from any file is dropped."""
        ds = two_file_catalog.open_dataset(dates=["2020-01-05", "2020-01-20"])
        got = [pd.Timestamp(t) for t in ds.time.values]
        assert got == [pd.Timestamp("2020-01-05")]

    def test_sparse_dates_none_present_raises(self, two_file_catalog):
        with pytest.raises(FileNotFoundError, match="No zarr files contain dates"):
            two_file_catalog.open_dataset(dates=["1999-01-01"])

    def test_sparse_empty_date_list_raises(self, two_file_catalog):
        with pytest.raises(ValueError, match="No valid dates provided"):
            two_file_catalog.open_dataset(dates=[])

    # ---- variable selection (_preprocess_dataset) ----

    def test_variables_subset_is_applied(self, tmp_path):
        catalog = _scanned_catalog(
            tmp_path,
            {"multi.zarr": _grid_ds("2020-01-01", 3, variables=("sst", "chl"))},
        )
        ds = catalog.open_dataset(
            start_date="2020-01-01", end_date="2020-01-03", variables=["sst"]
        )
        assert set(ds.data_vars) == {"sst"}

    def test_variables_accepts_a_plain_string(self, tmp_path):
        catalog = _scanned_catalog(
            tmp_path,
            {"multi.zarr": _grid_ds("2020-01-01", 3, variables=("sst", "chl"))},
        )
        ds = catalog.open_dataset(
            start_date="2020-01-01", end_date="2020-01-03", variables="chl"
        )
        assert set(ds.data_vars) == {"chl"}

    def test_unknown_variable_falls_back_to_all(self, tmp_path):
        """None of the requested variables exist: warn and keep the dataset whole."""
        catalog = _scanned_catalog(
            tmp_path,
            {"multi.zarr": _grid_ds("2020-01-01", 3, variables=("sst", "chl"))},
        )
        ds = catalog.open_dataset(
            start_date="2020-01-01", end_date="2020-01-03", variables=["not_a_var"]
        )
        assert set(ds.data_vars) == {"sst", "chl"}

    # ---- bbox (_apply_bbox) ----

    def test_bbox_subsets_with_one_cell_padding(self, two_file_catalog):
        """sel_padded_bbox keeps one grid cell beyond each requested edge."""
        ds = two_file_catalog.open_dataset(
            start_date="2020-01-01",
            end_date="2020-01-03",
            bbox=(-6.0, 34.0, -4.0, 36.0),
        )
        assert list(ds.lat.values) == [32.0, 34.0, 36.0, 38.0]
        assert list(ds.lon.values) == [-8.0, -6.0, -4.0, -2.0]

    def test_bbox_accepts_a_bbox_instance(self, two_file_catalog):
        ds = two_file_catalog.open_dataset(
            start_date="2020-01-01",
            end_date="2020-01-03",
            bbox=BBox(-6.0, 34.0, -4.0, 36.0),
        )
        assert list(ds.lat.values) == [32.0, 34.0, 36.0, 38.0]

    def test_bbox_without_lat_lon_coords_returns_dataset_unchanged(self, tmp_path):
        """Degenerate grid: log and pass through rather than raise."""
        catalog = _make_catalog(tmp_path)
        ds = xr.Dataset(
            {"sst": (["row", "col"], np.ones((2, 2)))},
            coords={"row": [0, 1], "col": [0, 1]},
        )

        out = catalog._reader._apply_bbox(ds, BBox(-6.0, 34.0, -4.0, 36.0))

        assert out.sizes == ds.sizes

    # ---- lat ordering + time normalisation ----

    def test_descending_lat_is_sorted_ascending(self, tmp_path):
        """ERA5-style north-to-south grids are flipped so slicing works."""
        catalog = _scanned_catalog(
            tmp_path,
            {"desc.zarr": _grid_ds("2020-01-01", 3, lat_descending=True)},
        )
        ds = catalog.open_dataset(start_date="2020-01-01", end_date="2020-01-03")
        assert list(ds.lat.values) == sorted(_LATS)

    def test_time_of_day_is_normalised_to_midnight(self, tmp_path):
        catalog = _scanned_catalog(
            tmp_path, {"noon.zarr": _grid_ds("2020-01-01", 3, hour=12)}
        )
        ds = catalog.open_dataset(start_date="2020-01-01", end_date="2020-01-03")
        assert all(pd.Timestamp(t).hour == 0 for t in ds.time.values)

    def test_hourly_store_keeps_its_sub_daily_stamps(self, tmp_path):
        """
        time_step=hourly must skip the midnight snap. Normalizing an hourly axis
        maps all 24 steps of a day onto one stamp — a year of 8784 steps becomes
        366 duplicated timestamps, silently.
        """
        catalog = _make_catalog(tmp_path, time_step="hourly")
        ds = xr.Dataset(
            {"sst": (["time"], np.arange(48.0))},
            coords={"time": pd.date_range("2020-01-01", periods=48, freq="h")},
        )

        out = catalog._reader._normalize_time(ds)
        times = pd.DatetimeIndex(out.time.values)
        assert len(times) == 48
        assert times.is_unique, "hourly stamps were collapsed"
        assert sorted({t.hour for t in times}) == list(range(24))

    def test_daily_store_is_still_normalised(self, tmp_path):
        """The default cadence keeps the existing midnight snap."""
        catalog = _make_catalog(tmp_path)  # time_step defaults to daily
        out = catalog._reader._normalize_time(_grid_ds("2020-01-01", 3, hour=12))
        assert all(pd.Timestamp(t).hour == 0 for t in out.time.values)

    def test_var_config_without_time_step_defaults_to_daily(self, tmp_path):
        """A stand-in config predating the field must keep the old behaviour."""
        catalog = _make_catalog(tmp_path)
        catalog._reader._index.var_config = SimpleNamespace()  # no time_step

        out = catalog._reader._normalize_time(_grid_ds("2020-01-01", 3, hour=12))
        assert all(pd.Timestamp(t).hour == 0 for t in out.time.values)

    def test_normalize_time_passes_through_without_time_coord(self, tmp_path):
        """Static fields (e.g. bathy) have no time axis and must survive intact."""
        catalog = _make_catalog(tmp_path)
        ds = xr.Dataset(
            {"bathy": (["lat", "lon"], np.ones((2, 2)))},
            coords={"lat": [30.0, 32.0], "lon": [-10.0, -8.0]},
        )

        assert catalog._reader._normalize_time(ds) is ds

    def test_range_on_empty_catalog_raises(self, tmp_path):
        """Explicit dates against a store with no zarr files at all."""
        catalog = _scanned_catalog(tmp_path, {})

        with pytest.raises(FileNotFoundError, match="Catalog is empty"):
            catalog.open_dataset(start_date="2020-01-01", end_date="2020-01-03")


# ---------------------------------------------------------------------------
# get_nonnull_days
#
# The set-valued counterpart to get_vars_nonnull_end. That one answers "how far
# has this variable got", which cannot see a null day *inside* an
# otherwise-compiled span: the frontier moves past it and nothing asks again.
# ---------------------------------------------------------------------------


class TestGetNonnullDays:
    def _store(self, tmp_path, null_days=(), name="v_2020.zarr"):
        times = pd.date_range("2020-01-01", "2020-01-10", freq="D")
        nulls = pd.DatetimeIndex(null_days)
        data = np.random.default_rng(0).uniform(1, 2, size=(len(times), 2, 2))
        for i, t in enumerate(times):
            if t in nulls:
                data[i, :, :] = np.nan
        ds = xr.Dataset(
            {"sst": (["time", "lat", "lon"], data)},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        root = tmp_path / "store"
        root.mkdir(exist_ok=True)
        ds.to_zarr(root / name)
        return root

    def _call(self, root, window, var_names, tmp_path):
        from h2mare.storage.zarr_catalog import ZarrCatalog

        cat = ZarrCatalog.__new__(ZarrCatalog)
        cat._index = MagicMock()
        df = pd.DataFrame(
            [
                {
                    "path": str(p),
                    "variables": ["sst"],
                    "start_date": pd.Timestamp("2020-01-01"),
                    "end_date": pd.Timestamp("2020-01-10"),
                }
                for p in sorted(root.glob("*.zarr"))
            ]
        )
        type(cat._index).df = property(lambda self, _df=df: _df)
        return ZarrCatalog.get_nonnull_days(cat, window, var_names)

    def test_returns_the_days_with_data(self, tmp_path):
        root = self._store(tmp_path)
        window = DateRange(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10"))

        out = self._call(root, window, ["sst"], tmp_path)

        assert len(out["sst"]) == 10

    def test_a_null_day_is_excluded(self, tmp_path):
        root = self._store(tmp_path, null_days=[pd.Timestamp("2020-01-05")])
        window = DateRange(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10"))

        out = self._call(root, window, ["sst"], tmp_path)

        assert pd.Timestamp("2020-01-05") not in out["sst"]
        assert len(out["sst"]) == 9

    def test_window_bounds_the_scan(self, tmp_path):
        root = self._store(tmp_path)
        window = DateRange(pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-05"))

        out = self._call(root, window, ["sst"], tmp_path)

        assert len(out["sst"]) == 3

    def test_none_reduces_across_every_variable(self, tmp_path):
        root = self._store(tmp_path)
        window = DateRange(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10"))

        out = self._call(root, window, None, tmp_path)

        assert len(out["__any__"]) == 10

    def test_a_variable_with_no_data_is_omitted(self, tmp_path):
        root = self._store(tmp_path, null_days=pd.date_range("2020-01-01", periods=10))
        window = DateRange(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10"))

        assert self._call(root, window, ["sst"], tmp_path) == {}
