"""End-to-end check of an hourly variable through the REAL pipeline.

Runs the actual Netcdf2Zarr and ZarrCatalog against a staged copy of the raw
files, writing only into --out. The active config.yaml is never touched: the
hourly cadence and int16 encoding are injected as an in-memory AppConfig
override, so this proves the pipeline behaves before any config is changed.

Stage the raw files flat in --raw first (no year folders), mirroring a fresh
download. Copy rather than move: with archive_raw the converter relocates them,
and _cleanup_downloads removes the whole --raw folder at the end of run().

Usage:
    uv run python scripts/hourly_pipeline_check.py --var-key waves --year 2020         --raw /path/to/staged/grib [--compare-daily /path/to/existing_daily.zarr]
"""

import argparse
import calendar
import tempfile
import warnings
from pathlib import Path

import msgspec
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

from h2mare.config import get_settings  # noqa: E402
from h2mare.format_converters.netcdf2zarr import Netcdf2Zarr  # noqa: E402
from h2mare.models import StoreDtype, TimeStep, step_freq  # noqa: E402
from h2mare.processing.core.cds import daily_waves  # noqa: E402
from h2mare.storage.audit import audit_zarr_file  # noqa: E402
from h2mare.storage.zarr_catalog import ZarrCatalog  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--var-key", default="waves")
ap.add_argument("--year", type=int, default=2020)
ap.add_argument("--raw", type=Path, required=True, help="staged raw files (flat)")
ap.add_argument(
    "--out",
    type=Path,
    default=Path(tempfile.gettempdir()) / "h2mare_hourly_check",
    help="scratch store root; never the real store",
)
ap.add_argument(
    "--compare-daily",
    type=Path,
    default=None,
    help="optional existing daily zarr to compare the derived daily against",
)
args = ap.parse_args()

VAR, YEAR, RAW, STORE = args.var_key, args.year, args.raw, args.out / "store"


def hourly_config():
    base = get_settings().app_config
    entry = msgspec.structs.replace(
        base.variables[VAR], time_step=TimeStep.HOURLY, store_dtype=StoreDtype.INT16
    )
    variables = dict(base.variables)
    variables[VAR] = entry
    return msgspec.structs.replace(base, variables=variables)


def hr(t):
    print(f"\n{'=' * 66}\n{t}\n{'=' * 66}")


cfg = hourly_config()
print(
    f"{VAR} time_step = {cfg.variables[VAR].time_step} "
    f"-> step_freq {step_freq(cfg.variables[VAR])!r}"
)

hr("1. CONVERT (real Netcdf2Zarr, scratch store)")
conv = Netcdf2Zarr(VAR, app_config=cfg, store_root=STORE, download_root=RAW)
ok = conv.run(start_date=f"{YEAR}-01-01", end_date=f"{YEAR}-12-31")
print(f"run() -> {ok}")

written = sorted(STORE.rglob("*.zarr"))
print("written stores:", [p.name for p in written])
assert written, "no zarr written"
path = written[0]

ds = xr.open_zarr(path, consolidated=False)
t = pd.DatetimeIndex(ds.time.values)
print(f"  steps={len(t)}  dims={dict(ds.sizes)}")
print(f"  first={t[0]}  last={t[-1]}")
print(
    f"  distinct hours={sorted({x.hour for x in t})[:5]}...  "
    f"unique={t.is_unique}  monotonic={t.is_monotonic_increasing}"
)
expect = 24 * (366 if calendar.isleap(YEAR) else 365)
print(f"  -> {'HOURLY OK' if len(t) == expect else f'EXPECTED {expect}'}")
ds.close()

hr("2. READ BACK THROUGH ZarrCatalog")
cat = ZarrCatalog(VAR, app_config=cfg, store_root=STORE, auto_refresh=True)
got = cat.open_dataset(start_date=f"{YEAR}-06-01", end_date=f"{YEAR}-06-30")
gt = pd.DatetimeIndex(got.time.values)
print(f"  June: steps={len(gt)} (expect 720)  unique={gt.is_unique}")
print(f"  last stamp={gt[-1]} (must be 30th 23:00, not 30th 00:00)")
print(f"  -> {'NO COLLAPSE, NO TRUNCATION' if len(gt) == 720 else 'PROBLEM'}")
got.close()

hr("3. AGGREGATE TO DAILY  vs  AN EXISTING DAILY STORE")
if args.compare_daily is None:
    print("  (no --compare-daily given — skipped)")
else:
    # daily_waves is the waves aggregator; other groups have their own, so this
    # section is waves-specific. For another var_key, swap in its processor.
    src = xr.open_zarr(path, consolidated=False)
    derived = daily_waves(src)
    old = xr.open_zarr(args.compare_daily, consolidated=False)
    print(f"  derived steps={derived.sizes['time']}  stored steps={old.sizes['time']}")
    for v in ("swh", "mdts"):
        a, b = xr.align(derived[v], old[v], join="inner")
        err = float(np.abs(a - b).max().compute())
        rng = float((b.max() - b.min()).compute())
        note = "expected to differ (circular fix)" if v == "mdts" else "must match"
        print(
            f"  {v:5s} max|diff|={err:.6g}  range={rng:.6g}  "
            f"rel={err / rng:.2e}   <- {note}"
        )
    old.close()
    src.close()

hr("4. AUDIT AT HOURLY CADENCE")
gap, slices, err = audit_zarr_file(path, freq="h")
print(f"  error={err}")
print(f"  axis gap={'none' if gap is None else f'{len(gap.missing)} missing steps'}")
print(f"  -> {'CLEAN' if gap is None and err is None else 'SEE ABOVE'}")

hr("5. AUDIT THE SAME STORE ON THE DAILY GRID (the old behaviour)")
gap_d, _, _ = audit_zarr_file(path, freq="D")
print(
    f"  axis gap={'none' if gap_d is None else f'{len(gap_d.missing)} missing'}"
    "   (a daily grid cannot see hourly holes either way)"
)
