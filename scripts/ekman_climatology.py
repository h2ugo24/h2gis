"""
Compute and save the Ekman day-of-year (DOY) mean and monthly 90th-percentile
climatology from the 1998-2017 baseline period.

Outputs (saved to CLIMATOLOGY_DIR):
    - cds_ekman-doy-mean_80W-10E-0N-70N_1998-2017.nc
    - cds_ekman-monthly-90thquantile_80W-10E-0N-70N_1998-2017.nc

Run once before **compiling** `atm-accum-avg`: since that store went hourly the
ekman chain runs at compile time (`compiler_registry._daily_features_for_slab`),
and `add_engineered_ekman` reads both files on every compile and raises if they
are missing.

Source
------
The hourly `atm-accum-avg` store, which holds ERA5's raw stress components
(`avg_iews`, `avg_inss`) and nothing derived. `ekman_pumping` no longer exists
on disk anywhere, so it is recomputed here through the very same
`compute_curl_and_ekman` the compile path calls.

That identity is the point. The climatology is subtracted from the compile-time
ekman *before* it is regridded to the h2ds base grid, so it has to sit on this
store's native grid and carry this store's int16 quantisation. Reading
`ekman_pumping` back from h2ds instead would be a grid and a rounding away from
what it is subtracted from.

Cost
----
The baseline is 142 GB of decoded hourly source, against ~3 GB of daily ekman
coming out. It is reduced a year at a time and each year is cached under
CLIMATOLOGY_DIR, so a run that dies at year twelve resumes at year twelve.
Expect a long run — the old version read a daily store and took seconds.
"""

from __future__ import annotations

import warnings

import xarray as xr
from loguru import logger
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from h2mare.config import get_settings
from h2mare.processing.core.cds import calendar_doy, compute_curl_and_ekman
from h2mare.storage.zarr_catalog import ZarrCatalog

warnings.simplefilter("ignore", UserWarning)

VAR_KEY = "atm-accum-avg"
BASELINE_YEARS = range(1998, 2018)
#: Rolling window the climatology describes. The compile-time anomaly is
#: (7-day mean − climatology), so the climatology must be of the 7-day mean too.
ROLL_DAYS = 7

#: Attempts per year before giving up on a read.
#:
#: A year is ~7 GB of decoded source fetched as thousands of chunk files, and
#: zarr reads them concurrently. That is enough to draw the occasional transient
#: OSError out of the store's drive — one arrived as EINVAL on a chunk that read
#: back perfectly a minute later. Only OSError is retried: a KeyError or a shape
#: mismatch means the store is not what this script thinks it is, and waiting
#: will not change that.
READ_ATTEMPTS = 4


def _compute_daily_ekman(catalog: ZarrCatalog, year: int) -> xr.Dataset:
    """
    Daily-mean ekman_pumping for one year, materialised.

    Read through ZarrCatalog rather than open_mfdataset for two reasons: it
    undoes the int16 decode widening (float32 out, not float64 — half the 142 GB)
    and its read slice covers the whole final day, which a bare
    ``slice(start, end)`` against an hourly axis would cut at 00:00.
    """
    ds = catalog.open_dataset(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
    try:
        # Already returns daily means; the curl stencil needs whole lat/lon,
        # which compute_curl_and_ekman rechunks for itself.
        return compute_curl_and_ekman(ds)[["ekman_pumping"]].compute()
    finally:
        ds.close()


def _daily_ekman_year(catalog: ZarrCatalog, year: int, cache_dir) -> xr.Dataset:
    """One year of daily ekman, computed once and cached under *cache_dir*."""
    cached = cache_dir / f"ekman_daily_{year}.zarr"
    if cached.exists():
        logger.info(f"{year}: reusing cached daily ekman")
        return xr.open_zarr(cached)

    def _log_retry(state) -> None:
        exc = state.outcome.exception()
        logger.warning(
            f"{year}: attempt {state.attempt_number} failed "
            f"({type(exc).__name__}: {exc}). Retrying in "
            f"{state.next_action.sleep:.0f}s."
        )

    logger.info(f"{year}: reading hourly stress and computing ekman")

    def _attempt() -> xr.Dataset:
        # Mirrors BaseDownloader._retry_call: returning from inside the
        # attempt is what ends the loop, and reraise=True means the last
        # failure comes back out rather than a RetryError.
        for attempt in Retrying(
            stop=stop_after_attempt(READ_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            retry=retry_if_exception_type(OSError),
            before_sleep=_log_retry,
            reraise=True,
        ):
            with attempt:
                return _compute_daily_ekman(catalog, year)
        raise AssertionError("unreachable: Retrying either returns or raises")

    daily = _attempt()
    daily.to_zarr(cached, mode="w")
    logger.success(f"{year}: cached {cached.name}")
    return daily


def _save(da: xr.DataArray, path) -> None:
    """
    Write one climatology field, without the GRIB baggage and at the source's
    own precision.

    ERA5's stress GRIB attaches a wall of ``GRIB_*`` attributes that ride the
    whole chain through here, and they no longer describe the data: an Ekman
    velocity carrying ``GRIB_units: 'N m**-2'`` next to its true ``units: m/s``
    invites a reader to take the wrong one. float32 because the source is
    int16-packed and decoded — float64 stored twice the bytes to hold precision
    the values never had (296 MB for the day-of-year file).
    """
    out = da.astype("float32")
    out.attrs = {k: v for k, v in da.attrs.items() if not k.startswith("GRIB_")}
    for coord in out.coords.values():
        coord.attrs = {
            k: v for k, v in coord.attrs.items() if not k.startswith("GRIB_")
        }
    out.to_netcdf(path)
    logger.success(f"Saved: {path}")


def main() -> None:
    settings = get_settings()
    clim_dir = settings.CLIMATOLOGY_DIR
    if clim_dir is None:
        raise FileNotFoundError("CLIMATOLOGY_DIR is not configured")
    clim_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = clim_dir / "_ekman_daily_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    catalog = ZarrCatalog(VAR_KEY)
    years = [_daily_ekman_year(catalog, y, cache_dir) for y in BASELINE_YEARS]

    # Concatenate before rolling: a per-year rolling mean would restart at every
    # 1 January and leave the first six days of twenty years wrong.
    ekman = xr.concat(years, dim="time").sortby("time")["ekman_pumping"]

    # min_periods=ROLL_DAYS, not 1: the archive starts on 1998-01-01, so the
    # first six days have no history to average. A partial window there is a
    # 1-to-6-day mean wearing a 7-day mean's name, and it would enter the
    # climatology as if it were one. NaN instead, so those buckets are built
    # from the nineteen years that do have a full window.
    ekman_7d = ekman.rolling(time=ROLL_DAYS, min_periods=ROLL_DAYS).mean()

    # 29 February is excluded from the climatology and borrows 28 February's
    # bucket when the climatology is applied (see calendar_doy). Keeping it here
    # would build one bucket in 365 from five samples against everyone else's
    # twenty.
    is_leap_day = (ekman_7d.time.dt.month == 2) & (ekman_7d.time.dt.day == 29)
    ek_noleap = ekman_7d.where(~is_leap_day, drop=True)

    logger.info("Computing DOY mean climatology")
    doy = calendar_doy(ek_noleap["time"]).rename("dayofyear")
    clim = ek_noleap.groupby(doy).mean("time").compute()
    clim.attrs.update(
        {
            "long_name": "Ekman pumping day-of-year climatology",
            "units": "m/s",
            "baseline": f"{BASELINE_YEARS[0]}-{BASELINE_YEARS[-1]}",
            "description": (
                f"Mean of the {ROLL_DAYS}-day rolling mean of daily Ekman "
                "pumping, per calendar day and grid cell."
            ),
            "doy_convention": (
                "365-day calendar: index 1 is 1 January and index 365 is 31 "
                "December in every year. 29 February is excluded from the mean "
                "and takes index 59 (28 February) when applied."
            ),
        }
    )

    _save(clim, clim_dir / "cds_ekman-doy-mean_80W-10E-0N-70N_1998-2017.nc")

    # Aligned exactly as add_engineered_ekman aligns it, so the threshold is a
    # percentile of the same anomaly the events are later counted against.
    logger.info("Computing monthly 90th percentile of anomalies")
    anom = ek_noleap - clim.sel(dayofyear=calendar_doy(ek_noleap["time"]))
    p90_monthly = (
        anom.rename("ekman_pumping_anom")
        .groupby("time.month")
        .quantile(0.90, dim="time")
        .compute()
    )
    p90_monthly.attrs.update(
        {
            "long_name": "Monthly 90th percentile of the Ekman pumping anomaly",
            "units": "m/s",
            "baseline": f"{BASELINE_YEARS[0]}-{BASELINE_YEARS[-1]}",
            "description": (
                "Upwelling-event threshold: the 90th percentile of the daily "
                f"Ekman anomaly ({ROLL_DAYS}-day rolling mean minus the "
                "day-of-year climatology), per calendar month and grid cell."
            ),
        }
    )

    _save(
        p90_monthly,
        clim_dir / "cds_ekman-monthly-90thquantile_80W-10E-0N-70N_1998-2017.nc",
    )
    logger.info(
        f"Cached per-year ekman left in {cache_dir} — delete it to reclaim ~3 GB."
    )


if __name__ == "__main__":
    main()
