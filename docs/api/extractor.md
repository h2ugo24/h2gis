# Extractor

`Extractor` extracts environmental values at point locations (CSV / `DataFrame`) or
spatial geometries (SHP / `GeoDataFrame`). By default it reads from the h2ds Zarr stores
via `ZarrCatalog`, but it can also extract against an arbitrary in-memory `xarray` dataset
that is not yet in the store (see [`extract_from_dataset()`](#extract_from_dataset)).

```python
from h2mare.processing.extractor import Extractor

extractor = Extractor("data/points.csv", time_col="ls_date", index_col="idlance")
df = extractor.run("sst")                       # returns a DataFrame
extractor.run("sst", output_path="out.csv")     # or writes a CSV and returns None
```

The input (points or geometries) is supplied to the **constructor**, not to `run()`.
There are no `start_date` / `end_date` arguments: the extraction window is the set of
dates present in the input, intersected with each variable's store coverage.

---

## Constructor

```python
Extractor(
    file_path,            # path to .csv/.shp, or an in-memory DataFrame/GeoDataFrame
    *,
    time_col=None,        # default "time"
    index_col=None,       # default auto "__row_id__"
    lon_col=None,         # default "lon"  (CSV only)
    lat_col=None,         # default "lat"  (CSV only)
    app_config=None,      # default: settings.app_config
    store_root=None,      # default: STORE_ROOT
    crs=4326,             # EPSG code for geometry extraction (SHP only)
    log_file=None,        # default: LOGS_DIR/extractor.log
)
```

| Parameter | Default | Description |
|---|---|---|
| `file_path` | — | A `.csv` / `.shp` path, or an in-memory `pd.DataFrame` (points) / `gpd.GeoDataFrame` (geometries). The suffix or object type selects point vs geometry mode. |
| `time_col` | `"time"` | Name of the time column in the input. |
| `index_col` | `"__row_id__"` | Name of the column to use as the output index. If omitted, a sequential `__row_id__` index is created and preserved through the pipeline (never written to output). |
| `lon_col` | `"lon"` | Longitude column name (CSV/point input only). |
| `lat_col` | `"lat"` | Latitude column name (CSV/point input only). |
| `app_config` | `settings.app_config` | Override the application configuration (variable registry, depth slices, etc.). |
| `store_root` | `STORE_ROOT` | Root directory of the Zarr stores. |
| `crs` | `4326` | EPSG code that geometries are reprojected to (SHP/geometry input only). |
| `log_file` | `LOGS_DIR/extractor.log` | Extraction log file. The first `Extractor` in the process fixes this; later values are ignored. |

---

## `run()`

```python
extractor.run(
    var_dict=None,       # which var_keys / variables to extract; None = all
    output_path=None,    # None → return DataFrame; path → write CSV, return None
    n_workers=8,         # parallel workers for geometry (SHP) extraction only
)
```

| Parameter | Description |
|---|---|
| `var_dict` | Selects what to extract. A `str` (single `var_key`), a `list[str]` (several `var_keys`), or a `dict[var_key, vars]` to pick specific variables inside a `var_key` (e.g. `{"radiation": ["tisr", "ssrd"]}`). `None` extracts every `var_key` in config (excluding compiled `h2mare`-source outputs). |
| `output_path` | If `None`, `run()` returns the result `DataFrame`. If a path is given, the result is written to **CSV** and `run()` returns `None`. |
| `n_workers` | Number of `ThreadPoolExecutor` workers. **Only used for geometry (SHP) extraction**; point (CSV) extraction is vectorized and ignores it. |

```python
var_dict = {"seapodym": [], "radiation": ["tisr", "ssrd", "slhf"]}
extractor = Extractor("input.csv", time_col="ls_date", index_col="idlance")
results = extractor.run(var_dict, output_path="out.csv", n_workers=12)
```

Extraction is checkpointed per `var_key` (`INTERIM_DIR/extraction_checkpoint.feather`), so
an interrupted run resumes where it stopped.

---

## `extract_from_dataset()`

Extract against an arbitrary in-memory `xarray` object instead of the store — useful for
new data that has not been ingested yet. It runs the same engine as `run()` but bypasses
`ZarrCatalog`, reusing the points/geometries already prepared in the constructor.

```python
extractor.extract_from_dataset(
    ds,                       # xr.Dataset | xr.DataArray with coords lon, lat, [time]
    *,
    vars=None,                # subset of data_vars (xr.Dataset only)
    n_workers=8,              # geometry (SHP) extraction only
    clip_to_coverage=False,   # drop input rows outside the ds extent → NaN
) -> pd.DataFrame
```

| Parameter | Description |
|---|---|
| `ds` | Gridded data with coords named `lon`, `lat`, and optionally `time`. For geometry input the dataset is assumed to be in `crs` — its CRS is overwritten (not reprojected) to match the geometries. |
| `vars` | Subset of variables to extract. Only valid when `ds` is an `xr.Dataset`; passing it with a `DataArray` raises `TypeError`. |
| `n_workers` | Parallel workers for geometry (SHP) extraction only. |
| `clip_to_coverage` | When `True`, input rows whose location (and time, if `ds` has a time coord) fall outside the `ds` extent are dropped and surface as `NaN` in the result. Default `False`, since nearest-neighbour (CSV) and clip-or-NaN (SHP) already handle out-of-extent inputs. |

Only config-free preparation is applied. Config-driven steps that the store path performs
— depth-slice expansion, `rename_lonlat`, and store date/bbox coverage resolution — are
the **caller's** responsibility: prepare `ds` beforehand.

```python
import numpy as np, pandas as pd, xarray as xr

ds = xr.Dataset(
    {"sst": (("time", "lat", "lon"), np.arange(2 * 3 * 3.0).reshape(2, 3, 3))},
    coords={"time": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "lat": [40, 41, 42], "lon": [-10, -9, -8]},
)
pts = pd.DataFrame({"time": ["2021-01-01", "2021-01-02"], "lon": [-10, -8], "lat": [40, 42]})
out = Extractor(pts).extract_from_dataset(ds)   # sst at the two nearest grid/time cells
```

---

## Output format

`run()` produces a single tabular result, indexed by `index_col`:

- **Returned as a `DataFrame`** when `output_path` is `None`.
- **Written as a CSV** when `output_path` is given (and `run()` returns `None`). If the
  file already exists, overlapping columns are dropped from the existing file and the new
  columns are joined in.

Columns are the **input columns carried through**, plus one column per extracted variable:

| Input | Carried-through columns | Extracted columns |
|---|---|---|
| CSV / points | `time`, `lon`, `lat` | one column per variable (e.g. `sst`, `tisr`); depth-sliced variables expand to `var_<depth>` |
| SHP / geometries | `time`, `geometry` | one column per variable; `bathy` additionally yields a `bathy_std` column (mean / std over each geometry) |

---

## Parallelism

Geometry (SHP) extraction runs each geometry through a `ThreadPoolExecutor`, sized by the
`n_workers` argument (default `8`). Point (CSV) extraction is fully vectorized — it uses a
cached KDTree for the nearest grid cell and `searchsorted` for the nearest time — and does
not use `n_workers`.
