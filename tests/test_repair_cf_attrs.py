"""
Tests for scripts/repair_cf_attrs.py.

The other repair script has none, and for a relabel that is defensible. This
one *deletes an array*, so the guard that decides whether to is worth pinning:
a ``valid_time`` that duplicates ``time`` is redundancy, and a ``valid_time``
that does not is data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "repair_cf_attrs.py"
_spec = importlib.util.spec_from_file_location("repair_cf_attrs", _SCRIPT)
assert _spec and _spec.loader
repair_cf_attrs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair_cf_attrs)


def _store(tmp_path: Path, *, stray: np.ndarray | None) -> Path:
    """A waves-shaped store, optionally carrying a valid_time coordinate."""
    time = pd.date_range("2020-01-01", periods=4, freq="h")
    ds = xr.Dataset(
        {"swh": (["time", "lat", "lon"], np.ones((4, 2, 2), dtype="float32"))},
        coords={"time": time, "lat": [0.0, 0.25], "lon": [0.0, 0.25]},
    )
    if stray is not None:
        ds = ds.assign_coords(valid_time=("time", stray))
    path = tmp_path / "waves_2020.zarr"
    ds.to_zarr(path, consolidated=True)
    # What the real stores carry: a coordinates list naming five coordinates
    # that never existed here, plus time, which is a dimension coordinate.
    root = zarr.open_group(str(path), mode="r+")
    root["swh"].attrs["coordinates"] = (
        "number time step meanSea latitude longitude valid_time"
    )
    zarr.consolidate_metadata(root.store)
    return path


class TestStrayCoordinateGuard:
    def test_a_duplicate_of_time_is_recognised(self, tmp_path):
        time = pd.date_range("2020-01-01", periods=4, freq="h").values
        path = _store(tmp_path, stray=time)
        assert repair_cf_attrs._stray_is_a_duplicate(path) is True

    def test_a_differing_valid_time_is_not_a_duplicate(self, tmp_path):
        """
        The guard that keeps this from deleting data. An hour's offset is enough
        to make it a distinct axis rather than redundancy.
        """
        time = pd.date_range("2020-01-01", periods=4, freq="h").values
        path = _store(tmp_path, stray=time + np.timedelta64(1, "h"))
        assert repair_cf_attrs._stray_is_a_duplicate(path) is False

    def test_absent_is_reported_as_absent_not_as_differing(self, tmp_path):
        path = _store(tmp_path, stray=None)
        assert repair_cf_attrs._stray_is_a_duplicate(path) is None


class TestCoordinatesAttrCleanup:
    def test_only_real_auxiliary_coordinates_survive(self, tmp_path):
        """
        Of 'number time step meanSea latitude longitude valid_time', five never
        existed in the store, valid_time has been removed by this point, and
        time is a dimension coordinate — which CF says must not be listed. That
        leaves nothing, so the attribute goes rather than being left empty.
        """
        path = _store(tmp_path, stray=None)
        root = zarr.open_group(str(path), mode="r+")

        changes = repair_cf_attrs._clean_coordinates_attr(root, {"swh"})

        assert changes == {"swh": None}

    def test_a_genuine_auxiliary_coordinate_is_kept(self, tmp_path):
        path = _store(tmp_path, stray=None)
        # A real auxiliary coordinate: along time, but not named for it. Added
        # through xarray so it lands with dimension_names set and reaches the
        # consolidated metadata, which is how a real store would carry it.
        with xr.open_zarr(path, consolidated=False) as ds:
            ds = ds.assign_coords(reference_time=("time", np.arange(4.0)))
            ds.load()
        ds.to_zarr(path, mode="w", consolidated=True)

        root = zarr.open_group(str(path), mode="r+")
        root["swh"].attrs["coordinates"] = "time reference_time"

        changes = repair_cf_attrs._clean_coordinates_attr(root, {"swh"})

        assert changes["swh"] == "reference_time", (
            "time is a dimension coordinate and must be dropped; "
            "reference_time is auxiliary and must be kept"
        )


class TestRepair:
    @pytest.fixture(autouse=True)
    def _point_at_the_temp_store(self, tmp_path, monkeypatch):
        path = _store(
            tmp_path, stray=pd.date_range("2020-01-01", periods=4, freq="h").values
        )
        monkeypatch.setattr(repair_cf_attrs, "store_paths", lambda var_key: [path])
        monkeypatch.setattr(repair_cf_attrs, "is_compiled", lambda var_key: False)
        self.path = path

    def test_a_dry_run_writes_nothing(self, capsys):
        repair_cf_attrs.repair("waves", apply=False)

        root = zarr.open_group(str(self.path), mode="r")
        assert "valid_time" in {name for name, _ in root.arrays()}
        assert "standard_name" not in dict(root["lat"].attrs)
        assert "WOULD FIX" in capsys.readouterr().out

    def test_apply_removes_the_duplicate_and_labels_the_axes(self):
        repair_cf_attrs.repair("waves", apply=True)

        root = zarr.open_group(str(self.path), mode="r")
        assert "valid_time" not in {name for name, _ in root.arrays()}
        assert root["lat"].attrs["standard_name"] == "latitude"
        assert root["lat"].attrs["units"] == "degrees_north"
        assert "coordinates" not in dict(root["swh"].attrs)

    def test_the_data_survives(self):
        before = xr.open_zarr(self.path, consolidated=False)["swh"].values.copy()

        repair_cf_attrs.repair("waves", apply=True)

        with xr.open_zarr(self.path, consolidated=False) as after:
            assert np.array_equal(after["swh"].values, before)
            assert after.sizes["time"] == 4

    def test_consolidated_metadata_stops_advertising_the_removed_array(self):
        """
        Every edit goes to the arrays' own metadata. Left unconsolidated, a
        reader that trusts the consolidated copy still sees valid_time.
        """
        repair_cf_attrs.repair("waves", apply=True)

        with xr.open_zarr(self.path, consolidated=True) as ds:
            assert "valid_time" not in ds.variables

    def test_running_twice_changes_nothing_the_second_time(self):
        assert repair_cf_attrs.repair("waves", apply=True) == 1
        assert repair_cf_attrs.repair("waves", apply=True) == 0
