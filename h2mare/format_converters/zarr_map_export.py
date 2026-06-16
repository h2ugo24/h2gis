"""
Export a map-optimized copy of a per-period Zarr store (the "_map" store).

h2mare's canonical Zarr stores are chunked for point/geometry extraction (large
time blocks, small spatial tiles). Interactive map tools instead read one
full-grid field per timestep and animate across dates — there a large time chunk
forces a whole time block to be decompressed per frame. This rewrites a store
into a *sibling* (e.g. ``h2ds`` → ``h2ds_map``) using the ``"map"`` chunk layout
(``{time: map_time_chunk, lat: full, lon: full}``) so a single-day field reads as
one small chunk.

Works on any per-period store (h2ds by default, or any var_key, or an arbitrary
``source_root``). The export is a pure projection of the source: it never writes
back to it and can be rebuilt at any time, so the two layouts can never diverge.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Sequence

import xarray as xr
from loguru import logger

from h2mare.storage.xarray_helpers import chunk_dataset
from h2mare.storage.zarr_catalog import ZarrCatalog


def export_map_zarr(
    *,
    var_key: str = "h2ds",
    source_root: Optional[Path] = None,
    dest_root: Optional[Path] = None,
    years: Optional[Sequence[int]] = None,
    map_time_chunk: int = 14,
    target_mb: int = 32,
    overwrite: bool = False,
) -> list[Path]:
    """
    Rechunk each per-period Zarr in a store into a map-optimized sibling store.

    The source is resolved in priority order:
      1. ``source_root`` if given — used directly, no config needed.
      2. otherwise ``ZarrCatalog(var_key).store_root`` (defaults to ``"h2ds"``).

    Args:
        var_key: Variable key whose store to export when ``source_root`` is None.
            Defaults to ``"h2ds"`` (the compiled store).
        source_root: Explicit source store dir. Overrides ``var_key`` when given.
        dest_root: Destination dir. Defaults to ``<source_root>_map`` (sibling),
            leaving the source untouched.
        years: Restrict to these years (matched against each filename). None = all.
            Best-effort substring match; assumes the year appears in the file name
            (true for the year/yearmonth date formats h2mare writes).
        map_time_chunk: Time chunk for the map layout (passed to chunk_dataset).
            Default 14: a single-day field still reads one chunk, while packing
            14 days per chunk keeps the chunk-file count down. Set 1 for the
            smallest/most-random-friendly chunks.
        target_mb: Per-chunk byte budget; only caps spatial tiles if one
            full-grid timestep exceeds it.
        overwrite: Re-export files that already exist in dest. Default skips them.

    Returns:
        Paths of the map stores written this run.

    Raises:
        FileNotFoundError: If the resolved source dir has no ``.zarr`` files
            (after the optional ``years`` filter).
    """
    if source_root is not None:
        src_dir = Path(source_root)
    else:
        src_dir = ZarrCatalog(var_key).store_root

    dst_dir = (
        Path(dest_root)
        if dest_root is not None
        else src_dir.parent / f"{src_dir.name}_map"
    )
    dst_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(src_dir.glob("*.zarr"))
    if years is not None:
        wanted = {str(y) for y in years}
        src_files = [p for p in src_files if any(y in p.stem for y in wanted)]
    if not src_files:
        raise FileNotFoundError(f"No .zarr files found under {src_dir}")

    logger.info(
        f"Map-export: {len(src_files)} file(s) from {src_dir} -> {dst_dir} "
        f"(map_time_chunk={map_time_chunk})"
    )

    written: list[Path] = []
    for src in src_files:
        dst = dst_dir / src.name
        if dst.exists() and not overwrite:
            logger.info(f"Skip (exists): {dst.name}")
            continue

        logger.info(f"Rechunking {src.name} -> map layout")
        ds = xr.open_zarr(src, consolidated=False)
        try:
            ds = chunk_dataset(
                ds, target_mb=target_mb, layout="map", map_time_chunk=map_time_chunk
            )
            # Drop stale chunk encodings inherited from the source so to_zarr
            # writes the new map grid instead of rejecting it as unsafe
            # (same pattern as storage.py:_append_data).
            for var in ds.variables:
                ds[var].encoding.pop("chunks", None)
                ds[var].encoding.pop("preferred_chunks", None)

            # Atomic-ish swap: write to a hidden temp dir, then rename into place
            # (mirrors parquet_store's .tmp_write_* → rename).
            tmp = dst_dir / f".tmp_{src.name}"
            if tmp.exists():
                shutil.rmtree(tmp)
            # No consolidated metadata: matches the rest of the store, which is
            # always written plain and read with consolidated=False.
            ds.to_zarr(tmp, mode="w")
        finally:
            ds.close()

        if dst.exists():
            shutil.rmtree(dst)
        tmp.replace(dst)
        written.append(dst)
        logger.success(f"Wrote {dst.name}")

    logger.success(f"Map-export complete: {len(written)} file(s) -> {dst_dir}")
    return written
