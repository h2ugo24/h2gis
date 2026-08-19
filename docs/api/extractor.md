# Extractor

`Extractor` extracts environmental values at point locations (CSV / `DataFrame`) or
spatial geometries (SHP / `GeoDataFrame`). By default it reads the per-variable Zarr stores
via `ZarrCatalog`, falling back to the compiled h2ds for variables stored hourly (see
[Cadence](#cadence)); it can also extract against an arbitrary in-memory `xarray` dataset
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
    index_col,            # REQUIRED — name of a unique key column already in the data
    time_col=None,        # default "time"
    lon_col=None,         # default "lon"  (CSV only)
    lat_col=None,         # default "lat"  (CSV only)
    app_config=None,      # default: settings.app_config
    store_root=None,      # default: STORE_ROOT
    crs=4326,             # EPSG code for geometry extraction (SHP only)
    time_resolution="auto",  # "auto" | "daily" | "native" — see Cadence
    log_file=None,        # default: LOGS_DIR/extractor.log
)
```

| Parameter | Default | Description |
|---|---|---|
| `file_path` | — | A `.csv` / `.shp` path, or an in-memory `pd.DataFrame` (points) / `gpd.GeoDataFrame` (geometries). The suffix or object type selects point vs geometry mode. |
| `index_col` | **required** | Name of the unique key column used to merge results back onto your input. It must already exist in the data and be unique — the Extractor consumes the key, it never creates one. A missing or duplicated key raises `ValueError`. Use [`ensure_row_id`](#establishing-the-merge-key-ensure_row_id) to establish it. |
| `time_col` | `"time"` | Name of the time column in the input. |
| `lon_col` | `"lon"` | Longitude column name (CSV/point input only). |
| `lat_col` | `"lat"` | Latitude column name (CSV/point input only). |
| `app_config` | `settings.app_config` | Override the application configuration (variable registry, depth slices, etc.). |
| `store_root` | `STORE_ROOT` | Root directory of the Zarr stores. |
| `crs` | `4326` | EPSG code that geometries are reprojected to (SHP/geometry input only). |
| `time_resolution` | `"auto"` | How the input's own timestamps are read, which decides what an hourly `var_key` returns. See [Cadence](#cadence). |
| `log_file` | `LOGS_DIR/extractor.log` | Extraction log file. The first `Extractor` in the process fixes this; later values are ignored. |

---

## Cadence

Some variables are stored **hourly** (`time_step: hourly` in `config.yaml` — currently
`atm-instante`, `atm-accum-avg`, `radiation`, `waves`), and the daily reduction plus every
feature derived from it is produced at compile time and written only to h2ds. The
per-variable Zarr is therefore the raw *source*; **h2ds is the daily *product***.

The Extractor picks which one answers, per `var_key`, from what your input asks for:

| store cadence | input cadence | source | you get |
|---|---|---|---|
| daily | any | per-variable Zarr | unchanged — the day's value |
| hourly | date-only | **h2ds** | the daily aggregate and every derived feature |
| hourly | sub-daily | hourly Zarr + h2ds | stored fields at the nearest hour; daily-by-construction features broadcast across the day |

Consequences worth knowing:

- **Extraction of an hourly `var_key` at daily cadence needs a current compile.** If h2ds
  is missing a column the `var_key` publishes, extraction raises and names
  `uv run h2mare compile` rather than returning a thinner frame.
- **Units differ between the two sources.** The hourly store holds the raw source, so ERA5
  `msl` is **Pa** there and **hPa** in h2ds, and `tp` is a per-hour accumulation rather
  than the day's total. Column names stay the same either way, so a sub-daily run logs a
  warning naming the stored units.
- **Some features have no hourly value at all.** `ekman_anom` is a 7-day rolling mean
  against a day-of-year climatology; `wind_max` is a maximum over a day. These are daily by
  construction, so a sub-daily run serves each sample the value for the day it falls in.
- The store the input never reaches is never opened, and h2ds is opened **once** per
  `Extractor` no matter how many hourly `var_keys` a `run()` walks.

### `time_resolution`

`"auto"` (default) infers the input cadence: a time component that **varies** across rows
means you want hours; a date-only input, or one uniformly stamped (`14:00:00` on every
row — someone's export default rather than a real hour), means you want days.

Override it when that inference is wrong for your data:

| value | behaviour |
|---|---|
| `"auto"` | infer, as above |
| `"daily"` | always truncate to the day, even for varying stamps — forces the h2ds route |
| `"native"` | never truncate, so a uniform `14:00` is honoured as a real hour |

```python
# A uniform 14:00 stamp that really does mean 14:00
Extractor(pts, index_col="row_id", time_resolution="native").run("radiation")
```

Daily stores have only one cadence to give, so this argument does not affect them.

---

## Establishing the merge key (`ensure_row_id`)

The Extractor keeps only the columns it needs (`time`/`lon`/`lat` for points,
`time`/`geometry` for geometries) and returns one column per extracted variable. To stitch
those results back onto your *original* dataframe, you need a key that exists on **both**
sides. Because the Extractor never sees your other columns, that key must be established
**before** extraction, on the frame you keep — not invented inside the Extractor.

`ensure_row_id` is the helper for that step:

```python
from h2mare.processing.extractor import ensure_row_id

df = ensure_row_id(df)        # adds a unique "row_id" column (or validates an existing one)
```

| `col` state in `data` | Behaviour |
|---|---|
| present, unique | returned unchanged |
| present, duplicated | raises `ValueError` — a duplicated key collapses rows on merge-back |
| absent | adds a positional `0..n-1` key (on a copy) — safe to merge back only against this exact frame, in this order |

### Merging results back

```python
df = ensure_row_id(df)                              # key now lives on your frame
out = Extractor(df, index_col="row_id").run("sst")  # result indexed by row_id
enriched = df.merge(out, left_on="row_id", right_index=True)
```

Passing a `.csv`/`.shp` **path** instead of a frame is still supported, but with
`index_col` required the file must already contain that key column — there is no
auto-provisioning fallback. For files, read them in, call `ensure_row_id`, and pass the
frame.

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
pts = ensure_row_id(pts)
out = Extractor(pts, index_col="row_id").extract_from_dataset(ds)   # sst at nearest cells
```

---

## Output format

`run()` produces a single tabular result, indexed by `index_col`:

- **Returned as a `DataFrame`** when `output_path` is `None`.
- **Written as a CSV** when `output_path` is given (and `run()` returns `None`). The
  `index_col` key is written as a column so you can merge the result back. If the file
  already exists, overlapping columns are dropped from the existing file and the new
  columns are joined in.

Columns are the **input columns carried through**, plus one column per extracted variable:

| Input | Carried-through columns | Extracted columns |
|---|---|---|
| CSV / points | `time`, `lon`, `lat` | one column per variable (e.g. `sst`, `tisr`); depth-sliced variables expand to `var_<depth>` |
| SHP / geometries | `time`, `geometry` | one column per variable; `bathy` additionally yields a `bathy_std` column (mean / std over each geometry) |

The in-memory `run()` return keeps the SHP `geometry` column as a `GeoDataFrame` with live
shapely geometries (restored from the input on a checkpoint resume, since feather cannot
round-trip them). It is dropped only when writing to CSV, where shapely geometries would
serialize to WKT strings that cannot be read back as geometries.

---

## Parallelism

Geometry (SHP) extraction runs each geometry through a `ThreadPoolExecutor`, sized by the
`n_workers` argument (default `8`). Point (CSV) extraction is fully vectorized — it uses a
cached KDTree for the nearest grid cell and `searchsorted` for the nearest time — and does
not use `n_workers`.
