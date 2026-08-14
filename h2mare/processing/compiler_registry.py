"""
Registry mapping var_key → compile processor for h2ds compilation.

Add a new entry to COMPILE_PROCESSORS when a variable needs custom handling
during the Zarr compilation step. Variables not registered here use
``compile_default``, which opens the catalog and interpolates to the base grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd
import xarray as xr

from h2mare.models import step_freq
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import DateRange
from h2mare.utils.spatial import clip_land_data

if TYPE_CHECKING:
    from h2mare.processing.compiler import Compiler

# Each processor receives the Compiler instance (for grid/bbox/config access),
# the variable's ZarrCatalog (None for system variables like bathy and moon),
# and the compilation DateRange.
CompileProcessor = Callable[
    ["Compiler", Optional[ZarrCatalog], DateRange],
    Optional[xr.Dataset],
]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _open_or_warn(
    catalog: ZarrCatalog | None,
    var_key: str,
    date_range: DateRange,
    bbox,
    **kwargs,
) -> xr.Dataset | None:
    """Open a dataset from the catalog, returning None and logging on missing data."""
    from loguru import logger

    # CompileProcessor passes None for system variables (bathy, moon), but those
    # have their own processors and never reach here — a None catalog means a
    # non-system var_key was wired to one, which is a bug rather than missing data.
    if catalog is None:
        raise ValueError(
            f"_open_or_warn requires a catalog, got None for '{var_key}'. "
            f"System variables must not be dispatched to a catalog-reading processor."
        )

    try:
        return catalog.open_dataset(
            start_date=date_range.start,
            end_date=date_range.end,
            bbox=bbox,
            **kwargs,
        )
    except FileNotFoundError:
        logger.warning(
            f"No data for {var_key} during "
            f"{date_range.start.date()}–{date_range.end.date()} — skipping."
        )
        return None


# ---------------------------------------------------------------------------
# Registered processors (one per special-cased variable)
# ---------------------------------------------------------------------------


def _compile_bathy(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    var_cfg = compiler.app_config.variables["bathy"]
    if var_cfg.data_file is None:
        raise ValueError("bathy config entry is missing required 'data_file' field")
    data_path = compiler.remote_store_root / var_cfg.local_folder / var_cfg.data_file
    ds = xr.open_dataset(data_path).sel(
        lon=slice(compiler.bbox.xmin, compiler.bbox.xmax),
        lat=slice(compiler.bbox.ymin, compiler.bbox.ymax),
    )
    return ds.interp_like(compiler.base_grid, method="linear", assume_sorted=True)


def _compile_moon(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset:
    # Lazy import breaks the compiler.py ↔ compiler_registry.py cycle.
    from h2mare.processing.compiler import calculate_moon_phase

    dates = pd.date_range(date_range.start, date_range.end, freq="D")
    lat = float(compiler.base_grid.lat.mean().values)
    lon = float(compiler.base_grid.lon.mean().values)
    moon_phase = calculate_moon_phase(lat, lon, dates)
    da = xr.DataArray(
        np.broadcast_to(
            np.array(moon_phase)[:, None, None],
            (len(dates), len(compiler.base_grid.lat), len(compiler.base_grid.lon)),
        ),
        name="moon_phase",
        dims=["time", "lat", "lon"],
        coords={
            "time": dates,
            "lat": compiler.base_grid.lat,
            "lon": compiler.base_grid.lon,
        },
    )
    return clip_land_data(da.to_dataset())


def _compile_depth_var(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """Generic processor for 3-D variables: selects depth levels from compile_depth_slices."""
    assert catalog is not None
    var_key = catalog.var_key
    depths = compiler.app_config.variables[var_key].compile_depth_slices
    assert depths is not None, f"{var_key} has no compile_depth_slices configured"

    ds = _open_or_warn(catalog, var_key, date_range, compiler.bbox, chunks={"depth": 1})
    if ds is None:
        return None
    ds_interp = ds.sel(depth=depths, method="nearest").interp_like(
        compiler.base_grid, method="linear", assume_sorted=True
    )
    return xr.Dataset(
        {
            f"{var_key}_{target}": ds_interp[var_key].isel(depth=i).drop_vars("depth")
            for i, target in enumerate(depths)
        }
    )


def _atm_accum_from_hourly(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """
    Derive the daily ekman/rain features from an hourly atm-accum-avg store.

    The window is widened by ``_EKMAN_WARMUP_DAYS`` so the rolling features start
    warm, which replaces ``get_previous_dates_da``'s trip to the store: there is
    no stored daily ``ekman_pumping`` to seed from once the store is hourly, and
    reading real hours is strictly better than reading back a previous day's
    output. The warm-up days are trimmed off before the result is returned.
    """
    from h2mare.processing.core.cds import (
        _EKMAN_WARMUP_DAYS,
        add_engineered_ekman,
        compute_curl_and_ekman,
        daily_total_rain,
    )

    warm = DateRange(
        date_range.start - pd.Timedelta(days=_EKMAN_WARMUP_DAYS),
        date_range.end,
    )
    ds = _open_or_warn(catalog, "atm-accum-avg", warm, compiler.var_config.bbox)
    if ds is None:
        return None

    # Both of these already take hourly in and give daily out — they are the
    # same functions the daily convert path calls, just later.
    ds_ekman = compute_curl_and_ekman(ds)
    features = add_engineered_ekman(
        ds_ekman["ekman_pumping"], var_key="atm-accum-avg", seed_from_store=False
    )
    merged = xr.merge(
        [ds_ekman, features, daily_total_rain(ds)], compat="override", join="outer"
    )
    return merged.sel(time=slice(date_range.start, date_range.end))


def _compile_atm_accum_avg(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """
    h2ds stays daily whatever cadence the atm-accum-avg store is kept at.

    A daily store already holds the derived features, so this only regrids. An
    hourly store holds raw stress and precipitation, so the whole ekman chain is
    computed here instead — yielding the same h2ds columns either way.
    """
    # .get, not [...]: step_freq already treats a missing entry as daily, which
    # keeps stand-in configs on the path every existing store actually uses.
    var_config = compiler.app_config.variables.get("atm-accum-avg")
    if step_freq(var_config) == "h":
        ds = _atm_accum_from_hourly(compiler, catalog, date_range)
    else:
        ds = _open_or_warn(
            catalog, "atm-accum-avg", date_range, compiler.var_config.bbox
        )
    if ds is None:
        return None
    # Coords ride along from the climatology alignment (.sel on dayofyear/month)
    # whether the features came off disk or were just computed.
    ds = ds.drop_vars(["dayofyear", "month", "quantile"], errors="ignore")
    return ds.interp_like(compiler.base_grid, method="linear", assume_sorted=True)


def _compile_waves(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """
    h2ds stays daily whatever cadence the waves store is kept at.

    With ``time_step: hourly`` the convert step leaves the resample undone, so it
    happens here — through the same ``daily_waves`` the daily path has always
    used, so both cadences yield identical h2ds columns.
    """
    from h2mare.processing.core.cds import (
        daily_waves,
        direction_to_uv,
        uv_to_direction,
    )

    ds = _open_or_warn(catalog, "waves", date_range, compiler.var_config.bbox)
    if ds is None:
        return None
    if step_freq(compiler.app_config.variables["waves"]) == "h":
        ds = daily_waves(ds)

    # mdts is a direction in degrees, so it cannot be regridded by a weighted
    # average: interpolating between 350° and 10° sweeps the long way round and
    # lands on 180°. Interpolate the unit-vector components instead, then
    # recombine. swh is a magnitude and interpolates directly.
    components = direction_to_uv(ds["mdts"])
    to_interp = ds[["swh"]].assign(u_ts=components["u_ts"], v_ts=components["v_ts"])
    out = to_interp.interp_like(compiler.base_grid, method="linear", assume_sorted=True)
    out["mdts"] = uv_to_direction(out["u_ts"], out["v_ts"])
    out["mdts"].attrs.update(ds["mdts"].attrs)
    return out.drop_vars(["u_ts", "v_ts"])


def _compile_sst(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    from h2mare.processing.compiler import postprocess_sst_fdist

    ds = _open_or_warn(catalog, "sst", date_range, compiler.var_config.bbox)
    if ds is None:
        return None
    ds = postprocess_sst_fdist(ds)
    return ds.interp_like(compiler.base_grid, method="linear", assume_sorted=True)


# ---------------------------------------------------------------------------
# Default processor (open from catalog + interp to base grid)
# ---------------------------------------------------------------------------


def compile_default(
    compiler: Compiler,
    catalog: ZarrCatalog | None,
    date_range: DateRange,
) -> xr.Dataset | None:
    """Fallback processor: open from catalog and interpolate to the base grid."""
    from loguru import logger

    if catalog is None:
        return None

    try:
        ds = catalog.open_dataset(
            start_date=date_range.start,
            end_date=date_range.end,
            bbox=compiler.var_config.bbox,
        )
    except FileNotFoundError:
        logger.warning(
            f"No data during "
            f"{date_range.start.date()}–{date_range.end.date()} — skipping."
        )
        return None
    return ds.interp_like(compiler.base_grid, method="linear", assume_sorted=True)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

COMPILE_PROCESSORS: dict[str, CompileProcessor] = {
    "bathy": _compile_bathy,
    "moon": _compile_moon,
    "o2": _compile_depth_var,
    "thetao": _compile_depth_var,
    "atm-accum-avg": _compile_atm_accum_avg,
    "sst": _compile_sst,
    "waves": _compile_waves,
}
