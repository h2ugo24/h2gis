"""
Pilot: evaluate storing ERA5 as hourly Zarr instead of GRIB + daily Zarr.

Read-only against the store. Writes scratch Zarr to this script's directory.

For one var_key + year it reports:
  1. size   — hourly Zarr (float32 vs int16) against the source GRIB
  2. fidelity — int16 round-trip error vs the original values
  3. reproducibility — daily derived from the int16 hourly Zarr, compared against
     the daily Zarr already in the store (this is what says "safe to drop GRIB")
  4. extraction — hour-level extraction through Extractor.from_dataset

Answers "should this variable go hourly, and at what encoding?" without
touching the pipeline or the store — it reads the raw GRIB directly and writes
only to --out.

Use it before converting a new group. For the open atm-accum-avg question
(ekman_anom_lag14 and n_upwell_events_* did not reproduce), run it once as-is
and once with the int16 comparison removed: if float32 also fails, the cause is
this script's single-year isolation starving the rolling windows, not int16
quantisation.

Usage:
    uv run python scripts/hourly_pilot.py --var-key waves --year 2020
    uv run python scripts/hourly_pilot.py --var-key atm-accum-avg --year 2020 --keep
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

warnings.filterwarnings("ignore")


def size_gb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def hr(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def build_int16_encoding(ds: xr.Dataset, level: int = 9) -> dict:
    """int16 scale/offset matched to each variable's actual range (GRIB-like 16-bit)."""
    enc = {}
    for v in ds.data_vars:
        vmin = float(ds[v].min().compute())
        vmax = float(ds[v].max().compute())
        enc[v] = {
            "dtype": "int16",
            "scale_factor": (vmax - vmin) / 65000.0,
            "add_offset": (vmin + vmax) / 2.0,
            "_FillValue": -32767,
            "compressors": [zarr.codecs.ZstdCodec(level=level)],
        }
    return enc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--var-key", default="waves")
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--keep", action="store_true", help="keep the scratch zarr stores")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "h2mare_hourly_pilot",
        help="where to write the scratch stores (never the real store)",
    )
    args = ap.parse_args()

    from h2mare.config import get_settings
    from h2mare.processing.registry import PROCESSORS
    from h2mare.storage.xarray_helpers import rename_dims

    settings = get_settings()
    cfg = settings.app_config.variables[args.var_key]
    store = Path(settings.STORE_ROOT) / cfg.local_folder

    grib = sorted((store / str(args.year)).glob("*.grib"))
    if not grib:
        raise SystemExit(f"no GRIB found in {store / str(args.year)}")
    grib_bytes = sum(f.stat().st_size for f in grib)

    hr(
        f"{args.var_key} {args.year}  —  {len(grib)} GRIB file(s), "
        f"{grib_bytes / 1e9:.2f} GB"
    )

    # Mirror Netcdf2Zarr._open_dataset exactly. _get_ds_for_month is essential:
    # accumulated ERA5 files carry reference times that spill into the next
    # month, so without the trim combine_by_coords sees a non-monotonic time axis.
    import h2mare.processing.core.cds as _cds

    def _preprocess(d: xr.Dataset) -> xr.Dataset:
        d = rename_dims(d)
        d = _cds.merge_time_step(d)
        return _cds._get_ds_for_month(d)

    ds = xr.open_mfdataset(
        sorted(grib),
        engine="cfgrib",
        combine="by_coords",
        decode_timedelta=True,
        chunks={"time": 168},
        preprocess=_preprocess if cfg.merge_time_step else None,
    )
    ds = rename_dims(ds)
    ds = ds[[v for v in ds.data_vars]]
    # open_mfdataset gives one dask chunk per source file, so the time chunks are
    # ragged (month lengths). Zarr needs uniform chunks, so rechunk explicitly.
    ds = ds.chunk({"time": 168, "lat": -1, "lon": -1})
    n_val = int(np.prod([ds.sizes[d] for d in ("time", "lat", "lon")])) * len(
        ds.data_vars
    )
    print(f"vars={list(ds.data_vars)}  dims={dict(ds.sizes)}")
    print(f"values={n_val / 1e6:.0f}M   GRIB={grib_bytes / n_val:.3f} bytes/value")

    SCRATCH = args.out
    SCRATCH.mkdir(parents=True, exist_ok=True)
    paths = {}

    # ---- 1. size -----------------------------------------------------------
    hr("1. SIZE")
    for label, enc in (
        (
            "float32",
            {
                v: {"compressors": [zarr.codecs.ZstdCodec(level=0)]}
                for v in ds.data_vars
            },
        ),
        ("int16", build_int16_encoding(ds)),
    ):
        p = SCRATCH / f"{args.var_key}_{args.year}_{label}.zarr"
        if p.exists():
            shutil.rmtree(p)
        ds.to_zarr(p, mode="w", encoding=enc, consolidated=False)
        paths[label] = p
        gb = size_gb(p)
        print(
            f"  hourly {label:8s} {gb:7.3f} GB   {gb * 1e9 / n_val:5.3f} B/value   "
            f"{gb / (grib_bytes / 1e9):.3f}x GRIB"
        )
    print(f"  source GRIB      {grib_bytes / 1e9:7.3f} GB")

    # ---- 2. fidelity -------------------------------------------------------
    hr("2. INT16 FIDELITY (vs original GRIB values)")
    i16 = xr.open_zarr(paths["int16"], consolidated=False)
    for v in ds.data_vars:
        err = float(np.abs(ds[v] - i16[v]).max().compute())
        rng = float((ds[v].max() - ds[v].min()).compute())
        print(
            f"  {v:10s} max|err|={err:.6g}   range={rng:.6g}   "
            f"rel={err / rng if rng else 0:.2e}"
        )

    # ---- 3. daily reproducibility -----------------------------------------
    hr("3. DAILY DERIVED FROM INT16 HOURLY  vs  DAILY ZARR IN STORE")
    stored = sorted(store.glob(f"*_{args.year}.zarr"))
    proc = PROCESSORS.get(args.var_key)
    if not stored:
        print(f"  (no stored daily zarr matching *_{args.year}.zarr — skipped)")
    elif proc is None:
        print(f"  (no registered processor for {args.var_key} — skipped)")
    else:
        try:
            derived = proc(i16, cfg, args.var_key)
            old = xr.open_zarr(stored[0], consolidated=False)
            print(f"  store: {stored[0].name}")
            for v in derived.data_vars:
                if v not in old:
                    print(f"  {v:12s} (not in stored daily — derived only)")
                    continue
                a, b = xr.align(derived[v], old[v], join="inner")
                if a.size == 0:
                    print(f"  {v:12s} no overlap")
                    continue
                err = float(np.abs(a - b).max().compute())
                rng = float((b.max() - b.min()).compute())
                flag = "OK" if (rng and err / rng < 1e-3) else "CHECK"
                print(
                    f"  {v:12s} max|diff|={err:.6g}  range={rng:.6g}  "
                    f"rel={err / rng if rng else 0:.2e}  [{flag}]"
                )
            old.close()
        except Exception as e:
            print(f"  processor/compare failed: {type(e).__name__}: {e}")

    # ---- 4. hourly extraction ---------------------------------------------
    hr("4. HOURLY EXTRACTION (Extractor.from_dataset)")
    from h2mare.processing.extractor import Extractor

    def extraction_check() -> None:
        var0 = str(list(i16.data_vars)[0])
        day = pd.Timestamp(f"{args.year}-06-15")
        # Use the store's own timestamps for that day, not a synthetic
        # date_range — a constructed range can miss the stored stamps and
        # silently reindex to NaN.
        tsel = pd.DatetimeIndex(i16.time.values)
        times = tsel[(tsel >= day) & (tsel < day + pd.Timedelta(days=1))]

        # Pick a point with real (non-NaN) data across that day, else the
        # comparison is meaningless over land.
        block = i16[var0].sel(time=times).compute()
        ok = np.isfinite(block.values).all(axis=0)
        if not ok.any():
            print("  no fully-valid point on this day — skipped")
            return
        iy, ix = np.argwhere(ok)[0]
        lat0 = float(i16.lat.values[iy])
        lon0 = float(i16.lon.values[ix])
        df = pd.DataFrame(
            {"id": range(len(times)), "time": times, "lat": lat0, "lon": lon0}
        )
        ex = Extractor(
            df, index_col="id", time_col="time", lon_col="lon", lat_col="lat"
        )
        out = ex.extract_from_dataset(i16, vars=[var0])
        got = out[var0].to_numpy()
        want = block.values[:, iy, ix]
        n_distinct = len(np.unique(got[~np.isnan(got)]))
        print(
            f"  {var0} at ({lat0:.2f},{lon0:.2f}) on {day.date()}, "
            f"{len(times)} hourly points"
        )
        print(f"  distinct extracted values : {n_distinct}/{len(times)}")
        print(f"  max|extracted - direct sel|: {np.nanmax(np.abs(got - want)):.6g}")
        print(
            "  -> hour-level resolution preserved"
            if n_distinct > 1 and np.allclose(got, want, equal_nan=True)
            else "  -> MISMATCH (investigate)"
        )

    try:
        extraction_check()
    except Exception as e:
        print(f"  extraction check failed: {type(e).__name__}: {e}")

    i16.close()
    ds.close()
    if not args.keep:
        for p in paths.values():
            shutil.rmtree(p, ignore_errors=True)
        print(f"\n(scratch stores removed; --keep to retain them in {SCRATCH})")


if __name__ == "__main__":
    main()
