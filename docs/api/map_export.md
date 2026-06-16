# Map export (display-optimized Zarr)

`export_map_zarr` rewrites a per-period Zarr store into a **map-optimized sibling
store** for interactive visualization, leaving the canonical store untouched.

```python
from h2mare.format_converters.zarr_map_export import export_map_zarr

export_map_zarr()                       # h2ds → h2ds_map (default)
export_map_zarr(var_key="sst")          # any registered var_key
export_map_zarr(source_root="/some/zarr/dir")   # config-free
```

## Why two layouts

The pipeline's Zarr stores are chunked for **extraction** — a long time block per
small spatial tile, so reading a point's full time series touches few chunks
(`chunk_dataset(..., layout="timeseries")`, the default). That layout is the
opposite of what an interactive map wants: drawing one date's full-grid field has
to decompress a whole time block per frame.

The **map layout** (`layout="map"`) keeps the spatial dims contiguous and pins a
small time chunk, so a single full-grid field reads as one small chunk. The two
layouts are mutually exclusive optima, so the map copy lives in a **separate
store** rather than replacing the canonical one.

| | `timeseries` (default) | `map` |
|---|---|---|
| Contiguous (budget-filled) axis | time | space |
| Minimized axis | space (256-cell tiles) | time (`map_time_chunk`) |
| Cheap read | point/region full time series | one date, full grid |
| Used by | `Extractor` | interactive maps / animation |

For h2ds (280 × 360 grid, float32) the map chunk is `time × 280 × 360`:

| `map_time_chunk` | chunk size | chunks/var (365-day file) |
|---|---|---|
| 1 | 0.39 MB | 365 |
| **14 (default)** | **5.38 MB** | **27** |
| 30 | 11.5 MB | 13 |

`map_time_chunk=1` makes any single day one chunk (best for arbitrary-day
scrubbing); larger values pack more days per chunk to cut the chunk-file count
(better for sequential playback or object storage). A single-day `.sel` still
reads one chunk regardless.

!!! note "Budget caps space, not time"
    The 32 MB byte budget caps the **spatial** extent in the map layout (a single
    full-grid timestep is tiled down only if it exceeds the budget — hi-res
    grids). Time is pinned to `map_time_chunk`, not grown to the budget: filling
    time would force a single-day read to decompress dozens of unwanted days.

---

## export_map_zarr

Rechunk each per-period Zarr in a store into a map-optimized sibling store. The
source is resolved as: explicit `source_root` if given, otherwise
`ZarrCatalog(var_key).store_root` (defaults to `"h2ds"`).

```python
export_map_zarr(
    *,
    var_key="h2ds",         # store to export when source_root is None
    source_root=None,       # explicit source dir; overrides var_key
    dest_root=None,         # defaults to <source_root>_map (sibling)
    years=None,             # restrict to these years (None = all)
    map_time_chunk=14,      # time chunk for the map layout
    target_mb=32,           # caps spatial tiles only if one timestep exceeds it
    overwrite=False,        # re-export files already present in dest
)
```

| Parameter | Default | Description |
|---|---|---|
| `var_key` | `"h2ds"` | Variable key whose store to export when `source_root` is `None` |
| `source_root` | `None` | Explicit source store dir. Overrides `var_key` when given (no config needed) |
| `dest_root` | `None` | Destination dir. Defaults to `<source_root>_map`, leaving the source untouched |
| `years` | `None` | Restrict to these years (matched against each filename). `None` = all |
| `map_time_chunk` | `14` | Time chunk for the map layout. `1` = smallest/most random-friendly; larger = fewer chunk files |
| `target_mb` | `32` | Per-chunk byte budget; only caps spatial tiles if one full-grid timestep exceeds it |
| `overwrite` | `False` | Re-export files that already exist in dest. Default skips them |

Returns the list of map store paths written this run. Raises `FileNotFoundError`
if the resolved source has no `.zarr` files (after the optional `years` filter).

### Properties

- **Pure projection** of the source — the canonical store is never modified, so
  the map store can be rebuilt at any time and the two layouts can never diverge.
- **Memory-bounded** — rechunking large-time → small-time is a pure split, so
  dask reads each source block once and streams to disk (no whole-year load).
- **Atomic writes** — each file is written to a hidden temp dir then renamed into
  place, so a crash never leaves a half-written store.

!!! warning "It is a full copy"
    The `_map` store is a same-order-of-magnitude duplicate of the source
    (h2ds is ~139 GB). Treat it as a derived, rebuildable artifact and watch
    disk usage.

### Reading the map store

```python
import xarray as xr

ds = xr.open_dataset(
    f"{STORE_ROOT}/h2ds_map/h2mare_..._2024.zarr",
    engine="zarr",
    consolidated=False,
)
field = ds["sst"].sel(time="2024-07-01")   # one ~5 MB chunk, no year bulk-load
```

---

## chunk_dataset layout argument

`export_map_zarr` is a thin driver over `chunk_dataset`
(`h2mare.storage.xarray_helpers`), which gained the layout selector:

```python
chunk_dataset(ds, layout="timeseries")          # default — extraction
chunk_dataset(ds, layout="map", map_time_chunk=14)   # display
```

`chunk_dataset` logs the resulting layout, chunk shape and size for **every**
dataset that passes through it (convert, compile, and this export), e.g.:

```
chunk_dataset: layout='map' chunk {'time': 14, 'lat': 280, 'lon': 360} ~5.38 MB (float32, e.g. 'sst')
```
