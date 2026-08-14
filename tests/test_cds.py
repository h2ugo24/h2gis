"""Tests for processing/core/cds.py pure transformation functions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from h2mare.models import TimeStep
from h2mare.processing.core import cds
from h2mare.processing.core.cds import (
    _get_ds_for_month,
    daily_cloud_cover,
    daily_sea_level_pressure,
    daily_total_rain,
    daily_waves,
    daily_wind,
    direction_to_uv,
    drop_dims,
    hourly_radiation,
    resample_daily_mean,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_ds(n_days: int = 3, **data_vars) -> xr.Dataset:
    """Minimal hourly dataset (time × lat × lon)."""
    n = n_days * 24
    times = pd.date_range("2020-01-01", periods=n, freq="h")
    lat, lon = [30.0, 35.0], [-10.0, -5.0]
    return xr.Dataset(
        {k: (["time", "lat", "lon"], v) for k, v in data_vars.items()},
        coords={"time": times, "lat": lat, "lon": lon},
    )


def _rad_da(values: list[float], name: str = "ssrd") -> xr.DataArray:
    """Radiation DataArray with lat/lon dims (needed for hourly_radiation transpose)."""
    times = pd.date_range("2020-01-01", periods=len(values), freq="h")
    arr = np.array(values)[:, None, None] * np.ones((1, 2, 2))
    return xr.DataArray(
        arr,
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        name=name,
    )


# ---------------------------------------------------------------------------
# _get_ds_for_month
# ---------------------------------------------------------------------------


class TestGetDsForMonth:
    def test_returns_dominant_month_only(self):
        # 17 Jan days + 10 Feb days → January is dominant
        times = pd.date_range("2020-01-15", periods=27, freq="D")
        ds = xr.Dataset({"x": ("time", np.zeros(27))}, coords={"time": times})
        result = _get_ds_for_month(ds)
        assert all(pd.Timestamp(t).month == 1 for t in result.time.values)

    def test_single_month_is_unchanged(self):
        times = pd.date_range("2020-03-01", "2020-03-31", freq="D")
        ds = xr.Dataset({"x": ("time", np.zeros(31))}, coords={"time": times})
        assert len(_get_ds_for_month(ds).time) == 31


# ---------------------------------------------------------------------------
# drop_dims
# ---------------------------------------------------------------------------


class TestDropDims:
    def test_removes_listed_variables(self):
        ds = xr.Dataset({"a": 1.0, "b": 2.0, "c": 3.0})
        result = drop_dims(ds, dims_to_drop=["a", "b"])
        assert "a" not in result
        assert "c" in result

    def test_ignores_absent_names_without_error(self):
        ds = xr.Dataset({"x": 1.0})
        result = drop_dims(ds, dims_to_drop=["x", "does_not_exist"])
        assert "x" not in result


# ---------------------------------------------------------------------------
# resample_daily_mean
# ---------------------------------------------------------------------------


class TestResampleDailyMean:
    def test_collapses_24_steps_to_one_day(self):
        times = pd.date_range("2020-01-01", periods=48, freq="h")
        ds = xr.Dataset({"x": ("time", np.ones(48))}, coords={"time": times})
        assert len(resample_daily_mean(ds).time) == 2

    def test_mean_value_is_correct(self):
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"x": ("time", np.arange(24, dtype=float))}, coords={"time": times}
        )
        np.testing.assert_allclose(resample_daily_mean(ds)["x"].values, [11.5])


# ---------------------------------------------------------------------------
# daily_wind
# ---------------------------------------------------------------------------


class TestDailyWind:
    def test_all_output_variables_present(self):
        ds = _hourly_ds(2, u10=np.ones((48, 2, 2)), v10=np.ones((48, 2, 2)))
        result = daily_wind(ds)
        for v in ("wind_mean", "wind_std", "wind_max", "u10", "v10"):
            assert v in result

    def test_daily_resolution(self):
        ds = _hourly_ds(3, u10=np.ones((72, 2, 2)), v10=np.zeros((72, 2, 2)))
        assert len(daily_wind(ds).time) == 3

    def test_wind_speed_magnitude(self):
        # u=3, v=4 → |w|=5
        ds = _hourly_ds(1, u10=np.full((24, 2, 2), 3.0), v10=np.full((24, 2, 2), 4.0))
        np.testing.assert_allclose(daily_wind(ds)["wind_mean"].values, 5.0, rtol=1e-5)

    def test_raises_without_time_dim(self):
        with pytest.raises(ValueError, match="time"):
            daily_wind(xr.Dataset({"u10": 1.0, "v10": 1.0}))


# ---------------------------------------------------------------------------
# daily_cloud_cover
# ---------------------------------------------------------------------------


class TestDailyCloudCover:
    def test_daily_resolution(self):
        ds = _hourly_ds(3, tcc=np.ones((72, 2, 2)) * 0.5)
        assert len(daily_cloud_cover(ds).time) == 3

    def test_mean_value_preserved(self):
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"tcc": (["time", "lat", "lon"], np.full((24, 2, 2), 0.7))},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        np.testing.assert_allclose(daily_cloud_cover(ds)["tcc"].values, 0.7, rtol=1e-5)


# ---------------------------------------------------------------------------
# daily_sea_level_pressure
# ---------------------------------------------------------------------------


class TestDailySeaLevelPressure:
    def test_pa_to_hpa_conversion(self):
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"msl": (["time", "lat", "lon"], np.full((24, 2, 2), 101325.0))},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        np.testing.assert_allclose(
            daily_sea_level_pressure(ds)["msl"].values, 1013.25, rtol=1e-5
        )

    def test_units_attribute_set_to_hpa(self):
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"msl": (["time", "lat", "lon"], np.ones((24, 2, 2)))},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        assert daily_sea_level_pressure(ds)["msl"].attrs.get("units") == "hPa"


# ---------------------------------------------------------------------------
# hourly_radiation
# ---------------------------------------------------------------------------


class TestHourlyRadiation:
    def test_accumulated_to_watt_rate(self):
        # 3600 J/m² per hour → 1 W/m²
        da = _rad_da([0.0, 3600.0, 7200.0, 10800.0])
        result = hourly_radiation(da)
        assert result.shape[0] == 3  # diff reduces by 1
        np.testing.assert_allclose(result.values, 1.0, rtol=1e-5)

    def test_clips_large_negative_rates_to_zero(self):
        # diff = −36000 J in one step → rate = −10 W/m² → clipped to 0
        da = _rad_da([0.0, -36000.0, 0.0])
        result = hourly_radiation(da, clip_small_negatives=True)
        assert float(result.values[0, 0, 0]) == pytest.approx(0.0, abs=1e-9)

    def test_preserves_negative_when_clip_disabled(self):
        da = _rad_da([0.0, -36000.0, 0.0])
        result = hourly_radiation(da, clip_small_negatives=False)
        assert float(result.values[0, 0, 0]) == pytest.approx(-10.0, rel=1e-5)


# ---------------------------------------------------------------------------
# daily_total_rain
# ---------------------------------------------------------------------------


class TestDailyTotalRain:
    def test_m_to_mm_and_daily_sum(self):
        # 0.001 m/h × 24 h = 24 mm/day
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"tp": (["time", "lat", "lon"], np.full((24, 2, 2), 0.001))},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        np.testing.assert_allclose(daily_total_rain(ds)["tp"].values, 24.0, rtol=1e-5)

    def test_units_attribute_is_mm(self):
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        ds = xr.Dataset(
            {"tp": (["time", "lat", "lon"], np.zeros((24, 2, 2)))},
            coords={"time": times, "lat": [30.0, 35.0], "lon": [-10.0, -5.0]},
        )
        assert daily_total_rain(ds)["tp"].attrs.get("units") == "mm"


# ---------------------------------------------------------------------------
# direction_to_uv
# ---------------------------------------------------------------------------


class TestDirectionToUv:
    def test_east_zero_degrees(self):
        # 0° → u=cos(0)=1, v=sin(0)=0
        da = xr.DataArray([0.0], dims=["time"], name="mdts")
        r = direction_to_uv(da)
        np.testing.assert_allclose(r["u_ts"].values, 1.0, atol=1e-7)
        np.testing.assert_allclose(r["v_ts"].values, 0.0, atol=1e-7)

    def test_north_ninety_degrees(self):
        # 90° → u=0, v=1
        da = xr.DataArray([90.0], dims=["time"], name="mdts")
        r = direction_to_uv(da)
        np.testing.assert_allclose(r["u_ts"].values, 0.0, atol=1e-7)
        np.testing.assert_allclose(r["v_ts"].values, 1.0, atol=1e-7)

    def test_output_variable_names(self):
        da = xr.DataArray([45.0, 135.0], dims=["time"], name="mdts")
        r = direction_to_uv(da)
        assert "u_ts" in r and "v_ts" in r


# ---------------------------------------------------------------------------
# daily_waves
# ---------------------------------------------------------------------------


class TestDailyWaves:
    def test_daily_resolution(self):
        ds = _hourly_ds(3, swh=np.ones((72, 2, 2)), mdts=np.zeros((72, 2, 2)))
        assert len(daily_waves(ds).time) == 3

    def test_both_variables_in_output(self):
        ds = _hourly_ds(2, swh=np.ones((48, 2, 2)), mdts=np.zeros((48, 2, 2)))
        r = daily_waves(ds)
        assert "swh" in r and "mdts" in r

    def test_raises_without_time_dim(self):
        with pytest.raises(ValueError, match="time"):
            daily_waves(xr.Dataset({"swh": 1.0, "mdts": 0.0}))


# ---------------------------------------------------------------------------
# Integration: process_atm_instante
# ---------------------------------------------------------------------------


class TestProcessAtmInstante:
    def test_output_has_all_expected_variables(self):
        from h2mare.processing.core.cds import process_atm_instante

        ds = _hourly_ds(
            2,
            u10=np.ones((48, 2, 2)),
            v10=np.ones((48, 2, 2)),
            tcc=np.ones((48, 2, 2)) * 0.5,
            msl=np.full((48, 2, 2), 101325.0),
        )
        result = process_atm_instante(ds)
        for v in ("wind_mean", "wind_std", "wind_max", "u10", "v10", "tcc", "msl"):
            assert v in result

    def test_lat_is_reversed(self):
        from h2mare.processing.core.cds import process_atm_instante

        ds = _hourly_ds(
            1,
            u10=np.ones((24, 2, 2)),
            v10=np.ones((24, 2, 2)),
            tcc=np.ones((24, 2, 2)),
            msl=np.ones((24, 2, 2)),
        )
        result = process_atm_instante(ds)
        # isel(lat=slice(None, None, -1)) reverses the lat order
        assert list(result.lat.values) == list(reversed(ds.lat.values))


# ---------------------------------------------------------------------------
# Integration: process_waves
# ---------------------------------------------------------------------------


class TestProcessWaves:
    def test_daily_output_with_reversed_lat(self):
        from h2mare.processing.core.cds import process_waves

        ds = _hourly_ds(2, swh=np.ones((48, 2, 2)), mdts=np.zeros((48, 2, 2)))
        result = process_waves(ds)
        assert len(result.time) == 2
        assert list(result.lat.values) == list(reversed(ds.lat.values))


# ---------------------------------------------------------------------------
# process_waves — cadence selected by config
# ---------------------------------------------------------------------------


class TestProcessWavesCadence:
    """
    time_step decides whether convert aggregates. The signal reaches the
    processor through var_config, which every processor already receives.
    """

    @staticmethod
    def _ds():
        shape = (48, 2, 2)
        return _hourly_ds(
            2,
            swh=np.full(shape, 1.5),
            mdts=np.full(shape, 180.0),
        )

    def test_daily_config_aggregates_as_before(self):
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_waves

        cfg = SimpleNamespace(time_step=TimeStep.DAILY)
        out = process_waves(self._ds(), cfg, "waves")

        assert out.sizes["time"] == 2, "daily store must still be one step per day"
        assert set(out.data_vars) == {"swh", "mdts"}

    def test_hourly_config_keeps_the_native_axis(self):
        from h2mare.models import TimeStep
        from h2mare.processing.core.cds import process_waves

        cfg = SimpleNamespace(time_step=TimeStep.HOURLY)
        out = process_waves(self._ds(), cfg, "waves")

        assert out.sizes["time"] == 48, "hourly store must keep every step"
        assert set(out.data_vars) == {"swh", "mdts"}

    def test_missing_config_defaults_to_daily(self):
        """A processor called without config keeps the historical behaviour."""
        from h2mare.processing.core.cds import process_waves

        out = process_waves(self._ds(), None, "waves")
        assert out.sizes["time"] == 2


# ---------------------------------------------------------------------------
# mdts is a direction — circular, not linear
# ---------------------------------------------------------------------------


class TestWaveDirectionIsCircular:
    """
    Averaging degrees arithmetically is wrong across the 0/360 wrap: 350° and
    10° are 20° apart and average to 0°, not 180°.
    """

    @staticmethod
    def _wrapping_ds():
        """One day whose directions straddle north: half at 350°, half at 10°."""
        shape = (24, 2, 2)
        mdts = np.full(shape, 350.0)
        mdts[12:] = 10.0
        return _hourly_ds(1, swh=np.full(shape, 2.0), mdts=mdts)

    def test_daily_mean_direction_does_not_flip_across_north(self):
        out = daily_waves(self._wrapping_ds())
        got = float(out["mdts"].isel(time=0, lat=0, lon=0))

        # 0/360 are the same heading, so accept either end.
        assert min(got, 360.0 - got) < 1e-6, (
            f"expected ~0/360 (north), got {got} — arithmetic mean would give 180"
        )

    def test_height_still_averages_arithmetically(self):
        out = daily_waves(self._wrapping_ds())
        assert float(out["swh"].isel(time=0, lat=0, lon=0)) == pytest.approx(2.0)

    def test_uv_round_trip_preserves_direction(self):
        from h2mare.processing.core.cds import uv_to_direction

        da = xr.DataArray([0.0, 10.0, 90.0, 180.0, 350.0], dims="time")
        comp = direction_to_uv(da)
        back = uv_to_direction(comp["u_ts"], comp["v_ts"])
        np.testing.assert_allclose(back.values, da.values, atol=1e-9)


# ---------------------------------------------------------------------------
# get_previous_dates_da — rolling-feature warm-up
# ---------------------------------------------------------------------------


class TestEkmanSeedWindow:
    """
    The seed prepended before a range decides whether the deep rolling features
    start warm. Nothing downstream can detect a short seed: ``min_periods=1``
    accepts a partial window and emits a plausible wrong number instead of
    raising, so these assertions are the only guard.
    """

    @staticmethod
    def _capture_requested_span(monkeypatch, t0: pd.Timestamp) -> dict:
        seen: dict[str, pd.Timestamp] = {}

        class _FakeCatalog:
            def __init__(self, var_key):
                pass

            def open_dataset(self, start_date, end_date):
                seen["start"] = pd.Timestamp(start_date)
                seen["end"] = pd.Timestamp(end_date)
                return None  # early-return path; only the request matters here

            def close(self):
                pass

        monkeypatch.setattr(cds, "ZarrCatalog", _FakeCatalog)

        da = xr.DataArray(
            np.zeros(5),
            dims=["time"],
            coords={"time": pd.date_range(t0, periods=5, freq="D")},
        )
        cds.get_previous_dates_da(da, "atm-accum-avg")
        return seen

    def test_seed_covers_the_deepest_lagged_feature(self, monkeypatch):
        """
        ekman_anom_lag14 reads the anomaly 14 days back, and that anomaly is
        itself a 7-day rolling mean — so the first output day needs 14 + 7 - 1
        days of real history behind it. The original 15-day seed was 5 short,
        which silently corrupted lag14 for the first weeks of every isolated
        range.
        """
        t0 = pd.Timestamp("2020-01-01")
        seen = self._capture_requested_span(monkeypatch, t0)

        needed = max(cds._EKMAN_LAGS) + cds._EKMAN_ROLL_DAYS - 1
        assert (t0 - seen["start"]).days >= needed, (
            f"seed spans {(t0 - seen['start']).days} days but lag"
            f"{max(cds._EKMAN_LAGS)} needs {needed}"
        )

    def test_seed_covers_the_deepest_event_window(self, monkeypatch):
        """n_upwell_events_14d sums exceedances back to t-13, each needing 7."""
        t0 = pd.Timestamp("2020-01-01")
        seen = self._capture_requested_span(monkeypatch, t0)

        needed = max(cds._EKMAN_EVENT_WINDOWS) - 1 + cds._EKMAN_ROLL_DAYS - 1
        assert (t0 - seen["start"]).days >= needed

    def test_seed_stops_the_day_before_the_range(self, monkeypatch):
        """The seed must abut the range, never overlap it."""
        t0 = pd.Timestamp("2020-01-01")
        seen = self._capture_requested_span(monkeypatch, t0)

        assert seen["end"] == t0 - pd.Timedelta(days=1)

    def test_hourly_store_keeps_the_native_axis_and_raw_fields_only(self):
        """
        With ``time_step: hourly`` the convert step stores raw ERA5 and derives
        nothing — the ekman chain is daily by construction and moves to compile.
        """
        n_days = 3
        ds = _hourly_ds(
            n_days=n_days,
            avg_iews=np.ones((n_days * 24, 2, 2)),
            avg_inss=np.ones((n_days * 24, 2, 2)),
            tp=np.ones((n_days * 24, 2, 2)),
        )
        cfg = SimpleNamespace(time_step=TimeStep.HOURLY)

        out = cds.process_atm_accum_avg(ds, cfg, "atm-accum-avg")

        assert out.sizes["time"] == n_days * 24, "hourly axis must not be resampled"
        assert set(out.data_vars) == {"avg_iews", "avg_inss", "tp"}
        assert not [v for v in out.data_vars if "ekman" in v or "upwell" in v]

    def test_hourly_store_drops_fields_the_pipeline_does_not_publish(self):
        ds = _hourly_ds(
            n_days=1,
            avg_iews=np.ones((24, 2, 2)),
            avg_inss=np.ones((24, 2, 2)),
            tp=np.ones((24, 2, 2)),
            stray=np.ones((24, 2, 2)),
        )
        out = cds.process_atm_accum_avg(
            ds, SimpleNamespace(time_step=TimeStep.HOURLY), "atm-accum-avg"
        )

        assert "stray" not in out.data_vars

    def test_hourly_store_matches_the_daily_stores_lat_orientation(self):
        """Both cadences must write ascending lat, or compile regrids upside down."""
        ds = _hourly_ds(
            n_days=1,
            avg_iews=np.ones((24, 2, 2)),
            avg_inss=np.ones((24, 2, 2)),
            tp=np.ones((24, 2, 2)),
        )
        out = cds.process_atm_accum_avg(
            ds, SimpleNamespace(time_step=TimeStep.HOURLY), "atm-accum-avg"
        )

        assert list(out.lat.values) == list(ds.lat.values)[::-1]

    def test_warmup_is_derived_from_the_declared_depths(self):
        """
        Adding a deeper lag must widen the seed automatically — the constant is
        computed from the tuples, so this fails if someone reverts it to a
        literal and then adds lag21.
        """
        assert cds._EKMAN_WARMUP_DAYS >= max(cds._EKMAN_LAGS) + cds._EKMAN_ROLL_DAYS - 1
        assert (
            cds._EKMAN_WARMUP_DAYS
            >= max(cds._EKMAN_EVENT_WINDOWS) - 1 + cds._EKMAN_ROLL_DAYS - 1
        )


# ---------------------------------------------------------------------------
# compute_curl_and_ekman — spatial stencil chunking
# ---------------------------------------------------------------------------


class TestCurlStencilChunking:
    """The curl rolls along lat/lon, so those axes must not be tiled."""

    def test_curl_runs_on_lat_lon_contiguous_chunks(self):
        """
        The curl is a spatial stencil, so lat/lon must sit in one chunk each.

        A store written with the "timeseries" layout is spatially tiled; rolling
        across those tiles shuffles every chunk and the circular wrap couples
        opposite ends of the axis, which is what stalled a real compile. Values
        are unaffected either way, so only the chunking can catch it.
        """
        n_lat, n_lon = 12, 16
        times = pd.date_range("2020-01-01", periods=48, freq="h")
        shape = (len(times), n_lat, n_lon)
        ds = xr.Dataset(
            {
                "avg_iews": (["time", "lat", "lon"], np.ones(shape)),
                "avg_inss": (["time", "lat", "lon"], np.ones(shape)),
            },
            coords={
                "time": times,
                "lat": np.linspace(30.0, 41.0, n_lat),
                "lon": np.linspace(-10.0, 5.0, n_lon),
            },
        )
        # Tile the spatial dims the way the on-disk timeseries layout does.
        tiled = ds.chunk({"time": 24, "lat": n_lat // 2, "lon": n_lon // 2})
        assert len(tiled["avg_iews"].chunks[1]) > 1, "fixture must be tiled"

        out = cds.compute_curl_and_ekman(tiled)

        lat_chunks = out["ekman_pumping"].chunks[out["ekman_pumping"].dims.index("lat")]
        lon_chunks = out["ekman_pumping"].chunks[out["ekman_pumping"].dims.index("lon")]
        assert len(lat_chunks) == 1, f"lat still tiled: {lat_chunks}"
        assert len(lon_chunks) == 1, f"lon still tiled: {lon_chunks}"

    def test_curl_leaves_unchunked_input_unchunked(self):
        """A numpy-backed dataset must not be forced into dask by the rechunk."""
        n_lat, n_lon = 12, 16
        times = pd.date_range("2020-01-01", periods=24, freq="h")
        shape = (len(times), n_lat, n_lon)
        ds = xr.Dataset(
            {
                "avg_iews": (["time", "lat", "lon"], np.ones(shape)),
                "avg_inss": (["time", "lat", "lon"], np.ones(shape)),
            },
            coords={
                "time": times,
                "lat": np.linspace(30.0, 41.0, n_lat),
                "lon": np.linspace(-10.0, 5.0, n_lon),
            },
        )

        out = cds.compute_curl_and_ekman(ds)

        assert out["ekman_pumping"].chunks is None
