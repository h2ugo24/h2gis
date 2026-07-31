"""
Bring Zarr filenames in a store back in line with what the pipeline generates,
and quarantine files written on a corrupted grid.

The eddies store accumulated two problems that show up as one symptom — a mix of
``*_80W-9E-0N-70N_*`` and ``*_80W-10E-0N-70N_*`` filenames:

  * Most files carry a stale label. Every eddies file holds the same extent
    (lon -80..10, lat 0..70) on the same uniform 0.1 degree grid, so the ``9E``
    label is simply what an older labelling produced. Renaming is cosmetic.
  * One file is genuinely different: it was written on a grid unioned across the
    whole store (see EDDIESProcessor._grid_from_store), leaving a doubled axis of
    near-duplicate points. Renaming that one would hide a real defect behind a
    standard-looking name, so it is quarantined instead.

Only the geo-extent token of the filename is rewritten, using
``BBox.from_dataset(ds).to_label()`` — the same value the labelling code derives.
Everything else in the name is left alone, deliberately: the Compiler writes
h2ds with a ``name_key`` override (``compiled-data-0.25deg-P1D``) rather than
the var_key, so rebuilding a whole filename from ``build_file_path`` defaults
would silently rename that store to a convention it never used.

What it does, in order:

  1. **Quarantine** any file whose lat/lon axis is degenerate, by moving it into
     ``<store_root>/_quarantine/``. The catalog globs ``*.zarr`` at the top level
     only, so this drops the file from the store without deleting it.
  2. **Rename** every remaining file whose on-disk name differs from its
     canonical name. Skipped when the target already exists.
  3. **Reload** the catalog so its absolute paths and the Parquet sidecar match
     the renamed files, then print the dataset breakdown.

Safety:
  * --dry-run (default) only reads and reports. Nothing is moved or renamed.
  * Quarantine moves, never deletes; restore by moving the directory back.
  * Renames are refused when the destination exists, so no file is overwritten.
  * A quarantined file leaves a gap in coverage. Re-convert it from raw
    afterwards; ``h2mare convert`` infers the window from store coverage, so it
    will target exactly the gap.

Run:
    uv run python scripts/standardize_zarr_filenames.py --var-keys eddies
    uv run python scripts/standardize_zarr_filenames.py --var-keys eddies --apply
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import xarray as xr
from loguru import logger

from h2mare.processing.core.aviso import _is_degenerate_axis
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import BBox

# The geo-extent token as create_label_from_dataset writes it, e.g. 80W-10E-0N-70N.
_GEO_RE = re.compile(
    r"\d+(?:\.\d+)?[WE]-\d+(?:\.\d+)?[WE]-\d+(?:\.\d+)?[NS]-\d+(?:\.\d+)?[NS]"
)

QUARANTINE_DIR = "_quarantine"


def _degenerate_axes(ds: xr.Dataset) -> list[str]:
    """Names of coordinate axes that carry near-duplicate points."""
    return [
        name
        for name in ("lat", "lon")
        if name in ds.coords and _is_degenerate_axis(ds[name].values)
    ]


def plan(catalog: ZarrCatalog) -> tuple[list[Path], list[tuple[Path, Path]], list[str]]:
    """Work out what to quarantine and what to rename, without touching anything.

    Two passes on purpose: a file bound for quarantine has already vacated its
    name by the time renames run, so the collision check must ignore it.
    Deciding both in one pass would refuse the very rename the quarantine frees
    up, and would do so depending on iteration order.
    """
    quarantine: list[Path] = []
    renames: list[tuple[Path, Path]] = []
    skipped: list[str] = []
    canonical_of: dict[Path, Path] = {}

    for path in sorted(catalog.store_root.glob("*.zarr")):
        try:
            with xr.open_zarr(path, consolidated=False) as ds:
                bad = _degenerate_axes(ds)
                geo = BBox.from_dataset(ds).to_label()
        except Exception as e:
            skipped.append(f"{path.name}: unreadable ({e})")
            continue

        if not _GEO_RE.search(path.name):
            skipped.append(f"{path.name}: no geo-extent token in the filename")
            continue

        canonical_of[path] = path.with_name(_GEO_RE.sub(geo, path.name, count=1))

        if bad:
            logger.warning(
                f"{path.name}: degenerate {'/'.join(bad)} axis — quarantining "
                "(written on a unioned grid; needs re-converting from raw)"
            )
            quarantine.append(path)

    vacated = set(quarantine)
    for path, canonical in canonical_of.items():
        if path in vacated or canonical.name == path.name:
            continue
        if canonical.exists() and canonical not in vacated:
            skipped.append(f"{path.name}: target {canonical.name} already exists")
            continue
        renames.append((path, canonical))

    return quarantine, sorted(renames), skipped


def apply(catalog: ZarrCatalog, quarantine, renames) -> None:
    """Move quarantined files aside, then rename the rest."""
    if quarantine:
        dest_dir = catalog.store_root / QUARANTINE_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in quarantine:
            dest = dest_dir / path.name
            if dest.exists():
                logger.error(f"Quarantine target already exists, skipping: {dest}")
                continue
            path.rename(dest)
            logger.success(f"Quarantined {path.name} -> {QUARANTINE_DIR}/")

    for path, canonical in renames:
        if canonical.exists():
            logger.error(f"Target appeared since planning, skipping: {canonical.name}")
            continue
        path.rename(canonical)
        logger.info(f"Renamed {path.name} -> {canonical.name}")


def report(catalog: ZarrCatalog) -> None:
    """Reload the catalog so paths match the files, then show the breakdown."""
    df = catalog.reload()
    cov = catalog.get_time_coverage()
    logger.info(f"Catalog reloaded: {df['path'].nunique()} file(s), coverage {cov}")
    if "dataset" in df.columns:
        for dataset, group in df.groupby("dataset", sort=True):
            logger.info(
                f"  {dataset}: {group['start_date'].min().date()} -> "
                f"{group['end_date'].max().date()}"
            )


def process(var_key: str, do_apply: bool) -> None:
    catalog = ZarrCatalog(var_key, auto_refresh=False)
    if not catalog.store_root.exists():
        logger.warning(f"[{var_key}] store not found: {catalog.store_root}")
        return

    quarantine, renames, skipped = plan(catalog)

    logger.info(
        f"[{var_key}] {len(quarantine)} to quarantine, {len(renames)} to rename, "
        f"{len(skipped)} skipped"
    )
    for path, canonical in renames:
        logger.info(f"  {path.name}  ->  {canonical.name}")
    for note in skipped:
        logger.warning(f"  skipped {note}")

    if not do_apply:
        logger.info(f"[{var_key}] dry run — nothing written. Re-run with --apply.")
        return

    if not quarantine and not renames:
        logger.info(f"[{var_key}] nothing to do")
        return

    apply(catalog, quarantine, renames)
    report(catalog)

    if quarantine:
        logger.warning(
            f"[{var_key}] {len(quarantine)} file(s) quarantined — that period is now "
            "missing from the store. Re-convert it from raw; the converter infers "
            "the window from store coverage and will target exactly the gap."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--var-keys",
        nargs="+",
        required=True,
        help="Variable keys to process. Required — this tool moves and renames files.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the moves and renames. Without this, nothing is written.",
    )
    args = parser.parse_args()

    var_keys: list[str] = args.var_keys
    for var_key in var_keys:
        try:
            process(var_key, args.apply)
        except Exception as e:
            logger.opt(exception=True).error(f"[{var_key}] failed: {e}")


if __name__ == "__main__":
    main()
