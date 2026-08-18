# Configuration

H2MARE is configured through two files: `config.yaml` (variable definitions and processing parameters) and `.env` (paths and credentials).

---

## config.yaml

### Variable entries

Each key under `variables:` defines one data stream:

```yaml
variables:
  sst:
    local_folder: CMEMS_SST           # subdirectory under STORE_ROOT
    source_vars: [analysed_sst, ...]  # variable names inside the source file
    dataset_id_rep: <cmems-id>        # reprocessed (multiyear) dataset ID
    dataset_id_nrt: <cmems-id>        # near-real-time dataset ID (optional)
    source: cmems                     # cmems | aviso | cds
    archive_raw: false                # keep raw files in store (true) or delete after convert (false)
    pattern: "(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})"  # filename date pattern
    subset: true                      # CMEMS only: subset() vs get() download API
    bbox: [-80, 0, 10, 70]           # [xmin, ymin, xmax, ymax]
    depth_range: [0.0, 500.0]        # [min_depth, max_depth]

  radiation:
    local_folder: CDS_Radiation
    source: cds
    archive_raw: false
    time_step: hourly                 # keep the source cadence; h2ds stays daily
    store_dtype: int16                # scale/offset packed, ~2/3 the size
    merge_time_step: true             # GRIB time x step grid
    dataset_id_rep: reanalysis-era5-single-levels
```

Both `time_step` and `store_dtype` are properties of the **store**, not of a run:
each takes effect when a Zarr is created and an append inherits it, so changing
either on an existing variable means re-converting it.

| Field | Required | Description |
|---|---|---|
| `local_folder` | yes | Subdirectory under `STORE_ROOT` for this variable's Zarr files |
| `source_vars` | yes | Variable names to extract from source files |
| `dataset_id_rep` | yes | Reprocessed dataset identifier |
| `dataset_id_nrt` | no | Near-real-time dataset identifier. Omit for reanalysis-only products |
| `source` | yes | Provider: `cmems`, `aviso`, or `cds` |
| `archive_raw` | yes | Whether to keep this variable's raw NetCDF/GRIB files by moving them into the store after conversion (`true`), or delete them per-period once converted (`false`). |
| `pattern` | download vars | Regex matched against each raw filename to extract date component(s). Unmatched optional groups are dropped before parsing. Its capture groups must agree with `filename_date_range`: with `true`, up to **2** groups giving `(start, end)` — one group alone is read as a single day, which is why the shipped patterns make the range tail optional (`(\d{4}-\d{2}-\d{2})(?:-(\d{4}-\d{2}-\d{2}))?`); with `false`, the groups are joined with `-` and parsed as a single date (e.g. `(\d{8})` → `20210115`; `(\d{4})(\d{2})(\d{2})` → `2021-01-15`). Omit for derived/system variables (`bathy`, `moon`, `h2ds`) that are never matched against filenames. |
| `subset` | CMEMS only | Chooses the CMEMS download API: `true` (default) uses `copernicusmarine.subset()` (spatial/variable subset honoring `bbox`/`source_vars`); `false` uses `copernicusmarine.get()` to fetch full original files. Ignored for non-CMEMS sources. |
| `merge_time_step` | no | Set to `true` for CDS/ERA5 accumulated or averaged variables whose GRIB files have a 2-D `time × step` coordinate grid instead of a flat `time` axis (e.g. `atm-accum-avg`, `radiation`). Triggers a preprocess step that merges the two dimensions and trims overlapping timestamps at month edges. Default `false`. |
| `filename_date_range` | no | Set to `true` when the `pattern` captures a `(start, end)` date range (e.g. CMEMS/CDS files named `2021-01-01-2021-01-31.nc`). A **one-day** request is named with a single date instead (`2026-07-31.nc`), so make the second group optional and it will be read as that one day — without this a single-day repair download matches nothing and is discarded. Leave `false` (default) when the pattern yields a single date (e.g. AVISO FSLE: `_20210115_`). Controls how `Netcdf2Zarr` expands filenames into daily time steps. |
| `known_gaps` | no | Days the provider never published, so they can never be downloaded, converted or backfilled. Each entry is a date (`2025-06-02`) or a closed interval (`2025-06-02/2025-06-05`). Excluded from the gap checks and from `h2mare audit`, which reports how many were suppressed. Needed because a source shipping one file per day leaves an *axis* hole when it skips one — AVISO has no `fsle` file for 2025-06-02 and its remote listing jumps `20250601` → `20250603` — which is otherwise indistinguishable from data the pipeline lost. Only for gaps confirmed absent at the source; anything else is a defect and belongs fixed, not listed. |
| `time_step` | no | Cadence of this variable's own Zarr: `daily` (default) or `hourly`. An hourly store keeps the source's native axis and moves the daily reduction to compile time, so h2ds stays daily either way. Distinct from `time_resolution`, which chooses the *file* period (one Zarr per year or per month) — a store can be hourly and still written one file per year. The gap checks read this so they compare a store against a calendar at its own resolution; a daily grid cannot see a missing hour, and an hourly grid over a daily store would report 23 phantom gaps a day. Flipping it on an existing store requires re-converting: the store is written at one cadence and the check expects the other, which fails the write verification rather than corrupting anything. |
| `store_dtype` | no | On-disk encoding: `float32` (default, byte-identical to what the pipeline has always written) or `int16`, which stores scale/offset-packed integers at roughly two thirds the size. Safe for ERA5, whose GRIB is already ~16-bit packed, so the packing discards quantisation noise rather than signal. The scale spans each variable's own measured range, so the encoding step makes one pass over the data before the first byte is written — expect a silent minutes-long pause on a large store. Applied only when a store is **created**; appends inherit whatever encoding the store already has, so changing this on an existing store does nothing until it is re-converted. |
| `expect_contiguous_time` | no | Whether this product publishes an unbroken time axis at its own cadence (see `time_step`). `true` (default) lets the convert step reject a Zarr whose axis skips a step inside the range it just wrote. Set `false` only for a source that legitimately publishes an irregular axis. Distinct from `known_gaps`, which suppresses named days on an otherwise contiguous axis, and from `time_step`, which asks how finely the axis is sampled rather than whether it may skip. |
| `raw_include` | no | Regex matched (via `re.search`) against each raw filename; only matching files are converted. Use when a download directory holds files the pipeline must not read — AVISO ships META3.2 eddy trajectories as `long`/`short`/`untracked` variants side by side, and only the long ones belong in the store (the `untracked` files carry no `track` variable at all). Omit (default) to convert every file the date `pattern` matches. |
| `bbox` | no | Bounding box for subset. If omitted, the full available extent is downloaded |
| `depth_range` | no | Depth range for 3D variables (e.g. `o2`) |
| `data_file` | no | Filename of the static source file at the configured output resolution (e.g. 0.25°). Used by compile-only variables such as `bathy` |
| `data_file_hires` | no | Filename of the high-resolution static source file. Used by `bathy` when extracting at full native resolution (e.g. from SHP geometries) |
| `trajectory_format` | no | Set to `true` for trajectory-format datasets (e.g. `eddies`) that require spatial binning before they can be stored as a gridded Zarr. The standard `open_mfdataset` pipeline is bypassed entirely. Default `false`. |
| `rename_lonlat` | no | Set to `true` for variables whose Zarr store uses `lon`/`lat` coordinate names that must be renamed to `x`/`y` before `rioxarray` clip during extraction (e.g. AVISO `fsle`, `eddies`). Default `false`. |
| `extract_depth_slices` | no | Depth levels (metres) to extract when slicing a 3-D variable during `Extractor` runs. Each level becomes a separate output column (e.g. `[0, 100, 500]` → `o2_0`, `o2_100`, `o2_500`). **Applies only to variables with a `depth` dimension** (`thetao`, `o2`); omit for every 2-D variable, including static fields such as `bathy` whose hires file is used for geometry extraction but which carry no depth axis. |
| `compile_depth_slices` | no | Depth levels (metres) to select when compiling a 3-D variable into h2ds. Each level becomes a separate output variable (e.g. `[0, 100, 500, 1000]` → `o2_0`, `o2_100`, `o2_500`, `o2_1000`). Same 3-D-only rule as `extract_depth_slices`; can differ from it. |
| `compiled_vars` | no | Exact variable names as they appear in the compiled h2ds Zarr for this var_key, accounting for any renames or derived variables produced during the Convert step (e.g. `sst` → `[sst, analysis_error, sst_std, sst_fdist]`). Used by `h2mare parquet --add-var` to select only the relevant columns from the h2ds Zarr without the caller needing to know internal variable names. |

### Validation

h2mare warns at load time if `config.yaml` contains top-level keys other than `variables`, `global_attrs`, and `variable_attrs`. Unknown keys are ignored, but the warning helps catch typos like `varibles:` before they cause a silent misconfiguration.

### The `h2ds` key

The special `h2ds` variable defines the output grid for the compile step:

```yaml
  h2ds:
    local_folder: h2ds
    dataset_id_rep: compiled-data-0.25deg-P1D
    source: h2mare
    bbox: [-80, 0, 10, 70]
```

The `bbox` here sets the spatial extent of the compiled dataset.

---

## .env

| Variable | Required | Description |
|---|---|---|
| `STORE_ROOT` | yes | Root path for Zarr output (can be an external drive) |
| `H2MARE_ROOT` | no | Directory containing `config.yaml` and `.env`. Overrides the default auto-detection (walking up from the current working directory). Set it when running `h2mare` from an unrelated directory or when another project imports h2mare. See [Installation](installation.md#where-to-place-these-files). |
| `CMEMS_USERNAME` | CMEMS only | Copernicus Marine account username |
| `CMEMS_PASSWORD` | CMEMS only | Copernicus Marine account password |
| `AVISO_USERNAME` | AVISO only | AVISO account username |
| `AVISO_PASSWORD` | AVISO only | AVISO account password |
| `AVISO_FTP_SERVER` | AVISO only | FTP server hostname |

CDS / ERA5 credentials are handled by the `cdsapi` package and stored in `~/.cdsapirc`.

---

## Adding a new variable

1. Add an entry under `variables:` in `config.yaml` with the correct `source`, `dataset_id_rep`, and `local_folder`.
2. Add `variable_attrs` entries for each output variable name (used to set metadata in compiled Zarr files).
3. If the variable is a CDS/ERA5 accumulated or averaged product (GRIB files with a `time × step` structure), set `merge_time_step: true` in its config entry.
4. If each downloaded file covers a date range encoded in its filename as two groups (e.g. `2021-01-01-2021-01-31.nc`), set `filename_date_range: true` and make the second group optional so a one-day download still parses. Leave it unset for variables whose filenames encode a single date (e.g. AVISO FSLE).
5. If the variable is a trajectory dataset that requires spatial binning (observations indexed by `obs`, not a lat/lon/time grid), set `trajectory_format: true`.
6. If the source is new, implement a downloader class inheriting from `BaseDownloader` and register it in `h2mare/cli/main.py`.
