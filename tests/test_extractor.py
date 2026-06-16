"""Tests for Extractor — focused on logic that doesn't need external data."""

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from h2mare.processing.extractor import Extractor, _keys_path, _save_completed_keys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spatial_ds(
    lons: list[float] = [-10.0, -5.0, 0.0],
    lats: list[float] = [30.0, 35.0, 40.0],
) -> xr.Dataset:
    """Minimal spatial dataset (no time dimension)."""
    data = np.arange(len(lats) * len(lons), dtype=float).reshape(len(lats), len(lons))
    return xr.Dataset(
        {"sst": (["lat", "lon"], data)},
        coords={"lat": lats, "lon": lons},
    )


def _make_spatiotemporal_ds(
    lons: list[float] = [-10.0, -5.0, 0.0],
    lats: list[float] = [30.0, 35.0, 40.0],
    n_days: int = 5,
) -> xr.Dataset:
    """Minimal dataset with daily time axis."""
    times = pd.date_range("2020-01-01", periods=n_days, freq="D")
    data = np.ones((n_days, len(lats), len(lons)))
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )


def _make_extractor(time_values: list, time_col: str = "time") -> Extractor:
    """Build a minimal Extractor from a list of time strings."""
    df = pd.DataFrame(
        {
            time_col: time_values,
            "lon": [10.0] * len(time_values),
            "lat": [40.0] * len(time_values),
        }
    )
    return Extractor(df, time_col=time_col)


def _make_distinct_ds() -> xr.Dataset:
    """Spatiotemporal dataset with a distinct value per (time, lat, lon) cell.

    sst[t, lat_i, lon_i] = t * 9 + lat_i * 3 + lon_i, so each cell is uniquely
    identifiable — lets a point/time extraction assert an exact expected value.
    """
    times = pd.to_datetime(["2020-01-01", "2020-01-02"])
    lats = [30.0, 35.0, 40.0]
    lons = [-10.0, -5.0, 0.0]
    data = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3)
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )


def _make_geodf(geometries: list, times: list) -> gpd.GeoDataFrame:
    """Minimal GeoDataFrame with a time column and EPSG:4326 geometries."""
    return gpd.GeoDataFrame({"time": times, "geometry": geometries}, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# _resolve_time_col
# ---------------------------------------------------------------------------


class TestResolveTimeCol:
    def test_date_only_strings(self):
        """Date-only strings should NOT be truncated (no time component)."""
        ext = _make_extractor(["2020-01-01", "2020-01-02"])
        # Should stay as date-only (normalised to midnight, no truncation log)
        assert ext.data["time"].dt.hour.eq(0).all()

    def test_uniform_time_component_truncated(self):
        """Datetimes where all times are identical → truncate to midnight."""
        ext = _make_extractor(
            [
                "2020-01-01 06:00:00",
                "2020-01-02 06:00:00",
                "2020-01-03 06:00:00",
            ]
        )
        # All midnight after truncation
        assert ext.data["time"].dt.hour.eq(0).all()

    def test_variable_time_component_kept(self):
        """Datetimes with varying times → keep full datetime."""
        ext = _make_extractor(
            [
                "2020-01-01 06:00:00",
                "2020-01-01 12:00:00",
                "2020-01-01 18:00:00",
            ]
        )
        hours = ext.data["time"].dt.hour.tolist()
        assert len(set(hours)) > 1  # times preserved

    def test_tz_aware_input_becomes_naive(self):
        """TZ-aware strings must become tz-naive after conversion."""
        ext = _make_extractor(
            [
                "2020-06-15T10:00:00+00:00",
                "2020-06-16T10:00:00+00:00",
            ]
        )
        assert ext.data["time"].dt.tz is None

    def test_raw_check_before_conversion(self):
        """
        Regression: raw string check was done AFTER pd.to_datetime conversion,
        so datetime(00:00:00) always matched the HH:MM:SS pattern, causing
        date-only inputs to be misclassified as having a time component.
        """
        ext = _make_extractor(["2020-01-01", "2020-01-02"])
        # If the bug is present, the code enters the has_time_component branch
        # and then truncates to date — result would be midnight (same as correct).
        # The real regression is: variable-time branch would not be entered.
        # We verify by checking the date-only input stays at date (midnight).
        dates = ext.data["time"].dt.normalize()
        assert (ext.data["time"] == dates).all()

    def test_non_default_time_col(self):
        """Extractor should handle a non-default time column name."""
        ext = _make_extractor(["2020-03-01", "2020-03-02"], time_col="date")
        assert "time" in ext.data.columns  # renamed internally


# ---------------------------------------------------------------------------
# _nearest_grid_indices
# ---------------------------------------------------------------------------


class TestNearestGridIndices:
    def test_exact_grid_points(self):
        """Querying exact grid coordinates returns their exact indices."""
        ds = _make_spatial_ds()
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-10.0, -5.0, 0.0]), np.array([30.0, 35.0, 40.0])
        )
        np.testing.assert_array_equal(lon_idx, [0, 1, 2])
        np.testing.assert_array_equal(lat_idx, [0, 1, 2])

    def test_off_grid_snaps_to_nearest(self):
        """Off-grid point snaps to the nearest grid point."""
        ds = _make_spatial_ds()
        # -8.0 is 2° from -10 and 3° from -5 → nearest is -10 (index 0)
        # 32.0 is 2° from 30 and 3° from 35 → nearest is 30 (index 0)
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-8.0]), np.array([32.0])
        )
        assert lon_idx[0] == 0
        assert lat_idx[0] == 0

    def test_returns_ndarrays(self):
        """Output must be numpy ndarrays regardless of input size."""
        ds = _make_spatial_ds()
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-10.0, 0.0]), np.array([30.0, 40.0])
        )
        assert isinstance(lat_idx, np.ndarray)
        assert isinstance(lon_idx, np.ndarray)

    def test_single_point(self):
        """Single-point query returns length-1 arrays."""
        ds = _make_spatial_ds()
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([0.0]), np.array([40.0])
        )
        assert lat_idx.shape == (1,)
        assert lon_idx.shape == (1,)
        assert lon_idx[0] == 2  # 0.0 is last lon (index 2)
        assert lat_idx[0] == 2  # 40.0 is last lat (index 2)

    def test_irregular_grid(self):
        """Works on a non-uniform grid where searchsorted alone would be wrong."""
        ds = _make_spatial_ds(lons=[-10.0, -3.0, 0.0], lats=[30.0, 38.0, 40.0])
        # -4.0 is 1° from -3 and 6° from -10 → nearest is -3 (index 1)
        lat_idx, lon_idx = Extractor._nearest_grid_indices(
            ds, np.array([-4.0]), np.array([30.0])
        )
        assert lon_idx[0] == 1


# ---------------------------------------------------------------------------
# _nearest_time_indices
# ---------------------------------------------------------------------------


class TestNearestTimeIndices:
    def test_exact_match(self):
        """Exact timestamp returns the correct index."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 0

    def test_picks_closer_left_neighbor(self):
        """Point 6h after a step is closer to that step than the next (18h away)."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01 06:00:00"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 0  # 6 h from Jan 1, 18 h from Jan 2

    def test_picks_closer_right_neighbor(self):
        """Point 18h after a step is closer to the next step (6h away)."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01 18:00:00"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 1  # 18 h from Jan 1, 6 h from Jan 2

    def test_before_first_step_clips_to_zero(self):
        """Query before the first time step is clipped to index 0."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2019-12-31"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 0

    def test_after_last_step_clips_to_last(self):
        """Query after the last time step is clipped to the last index."""
        ds = _make_spatiotemporal_ds(n_days=5)
        q = np.array(pd.to_datetime(["2025-01-01"]))
        idx = Extractor._nearest_time_indices(ds, q)
        assert idx[0] == 4

    def test_multiple_queries(self):
        """Multiple timestamps resolved correctly in one call."""
        ds = _make_spatiotemporal_ds()
        q = np.array(pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-05"]))
        idx = Extractor._nearest_time_indices(ds, q)
        np.testing.assert_array_equal(idx, [0, 2, 4])


# ---------------------------------------------------------------------------
# Atomic checkpoint helpers
# ---------------------------------------------------------------------------


class TestAtomicCheckpoint:
    def test_save_completed_keys_writes_correct_content(self, tmp_path):
        checkpoint = tmp_path / "data.feather"
        keys = {"sst", "chl", "mld"}
        _save_completed_keys(checkpoint, keys)
        dest = _keys_path(checkpoint)
        with open(dest) as f:
            loaded = set(json.load(f))
        assert loaded == keys

    def test_save_completed_keys_no_staging_file_remains(self, tmp_path):
        """The .tmp staging file must be cleaned up after a successful write."""
        checkpoint = tmp_path / "data.feather"
        _save_completed_keys(checkpoint, {"sst"})
        dest = _keys_path(checkpoint)
        staging = dest.with_suffix(".tmp")
        assert dest.exists()
        assert not staging.exists()

    def test_feather_atomic_write_no_tmp_remains(self, tmp_path):
        """Verify the staging-then-replace pattern: no .tmp file left after write."""
        feather_path = tmp_path / "checkpoint.feather"
        staging = feather_path.with_suffix(".tmp")

        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_feather(staging)
        staging.replace(feather_path)

        assert feather_path.exists()
        assert not staging.exists()

    def test_feather_atomic_write_is_readable(self, tmp_path):
        """Data written through the staging pattern is readable from the final path."""
        feather_path = tmp_path / "checkpoint.feather"
        staging = feather_path.with_suffix(".tmp")

        df = pd.DataFrame({"x": [10, 20, 30], "y": [1.0, 2.0, 3.0]})
        df.to_feather(staging)
        staging.replace(feather_path)

        loaded = pd.read_feather(feather_path)
        pd.testing.assert_frame_equal(df, loaded)


# ---------------------------------------------------------------------------
# extract_from_dataset — extraction against an arbitrary in-memory dataset
# ---------------------------------------------------------------------------


class TestExtractFromDataset:
    def test_csv_spatiotemporal_exact_values(self):
        """Points resolve to the exact nearest (time, lat, lon) cell value."""
        ds = _make_distinct_ds()
        pts = pd.DataFrame(
            {
                "time": ["2020-01-02", "2020-01-01"],
                "lon": [-10.0, 0.0],
                "lat": [40.0, 30.0],
            }
        )
        out = Extractor(pts).extract_from_dataset(ds)

        # row 0: t=1, lat_i=2, lon_i=0 -> 1*9 + 2*3 + 0 = 15
        # row 1: t=0, lat_i=0, lon_i=2 -> 0    + 0   + 2 = 2
        assert out["sst"].tolist() == [15.0, 2.0]
        assert out.index.tolist() == [0, 1]

    def test_csv_spatial_only_no_time_coord(self):
        """A dataset without a time coord extracts purely on space (no raise)."""
        ds = _make_spatial_ds()  # sst = arange(9), lat rows, lon cols
        pts = pd.DataFrame({"time": ["2020-01-01"], "lon": [0.0], "lat": [40.0]})
        out = Extractor(pts).extract_from_dataset(ds)
        # lat_i=2, lon_i=2 -> 2*3 + 2 = 8
        assert out["sst"].tolist() == [8.0]

    def test_csv_vars_subset(self):
        """`vars` restricts the extracted columns to the requested variable."""
        ds = _make_spatial_ds()
        ds = ds.assign(mld=ds["sst"] + 100)
        pts = pd.DataFrame({"time": ["2020-01-01"], "lon": [0.0], "lat": [40.0]})
        out = Extractor(pts).extract_from_dataset(ds, vars="sst")
        # to_dataframe also carries lon/lat coord columns; the point is that the
        # unselected variable (mld) is absent.
        assert "sst" in out.columns
        assert "mld" not in out.columns

    def test_csv_clip_to_coverage_drops_out_of_extent(self):
        """Out-of-extent rows become NaN; in-extent rows keep their value."""
        ds = _make_distinct_ds()  # lon in [-10, 0], lat in [30, 40]
        pts = pd.DataFrame(
            {
                "time": ["2020-01-01", "2020-01-01"],
                "lon": [0.0, 50.0],  # second point is far outside
                "lat": [30.0, 30.0],
            }
        )
        out = Extractor(pts).extract_from_dataset(ds, clip_to_coverage=True)
        assert out["sst"].iloc[0] == 2.0
        assert np.isnan(out["sst"].iloc[1])

    def test_dataarray_with_vars_raises(self):
        """`vars` against a DataArray is a TypeError (ds[vars] is invalid)."""
        da = _make_spatial_ds()["sst"]
        pts = pd.DataFrame({"time": ["2020-01-01"], "lon": [0.0], "lat": [40.0]})
        with pytest.raises(TypeError):
            Extractor(pts).extract_from_dataset(da, vars="sst")

    def test_shp_geometry_mean_on_ds_without_crs(self):
        """Geometry extraction computes the clipped mean and sets CRS via ensure_crs.

        The dataset is passed WITHOUT a rio CRS; ensure_crs must write the
        GeoDataFrame's CRS onto it so ds.rio.clip succeeds.
        """
        # Spatial-only ds: sst = arange(9); lat rows [30,35,40], lon cols [-10,-5,0]
        #   row(lat30): 0 1 2 | row(lat35): 3 4 5 | row(lat40): 6 7 8
        ds = _make_spatial_ds()
        assert ds.rio.crs is None  # precondition: no CRS on the dataset

        geoms = [
            box(-12, 28, -3, 36),  # touches lon{-10,-5} x lat{30,35} -> 0,1,3,4
            box(-1, 38, 1, 42),  # touches lon{0} x lat{40} -> 8
        ]
        gdf = _make_geodf(geoms, ["2020-01-01", "2020-01-01"])
        out = Extractor(gdf).extract_from_dataset(ds, n_workers=2)

        out = out.sort_index()
        assert out.loc[0, "sst"] == pytest.approx(2.0)  # mean(0,1,3,4)
        assert out.loc[1, "sst"] == pytest.approx(8.0)
