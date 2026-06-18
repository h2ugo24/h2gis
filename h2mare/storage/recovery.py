"""
Crash-recovery sweeps for Zarr and Parquet stores.

The write paths in :mod:`h2mare.storage.storage` and
:mod:`h2mare.storage.parquet_store` are atomic against *exceptions* (write to a
sibling temp, then swap, restoring on failure). They are not, on their own,
atomic against a *hard kill* — SIGKILL, power loss, an OS crash — which can
interrupt a swap and leave orphaned temp or backup artifacts behind.

These sweeps reconcile that leftover state before gap detection runs, so a
resumed pipeline neither trusts a partially written store nor loses data
stranded in a backup, and so orphaned temp files cannot pollute the full-store
globs the read layer relies on.

Both functions are idempotent and safe to call on a clean store (they no-op).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, Sequence

from loguru import logger

ZARR_TMP_SUFFIX = ".zarr.tmp"
ZARR_BAK_SUFFIX = ".zarr.bak"
PARQUET_TMP_PREFIX = ".tmp_write_"


def recover_zarr_store(store_root: Path) -> int:
    """
    Reconcile orphaned ``*.zarr.bak`` / ``*.zarr.tmp`` dirs under *store_root*.

    Mirrors the swap in :func:`h2mare.storage.storage._append_data` and the
    new-file write in :func:`~h2mare.storage.storage.write_append_zarr`:

    - ``foo.zarr.bak`` with ``foo.zarr`` **missing** — the swap crashed after the
      backup move; the backup holds the only full copy, so restore it.
    - ``foo.zarr.bak`` with ``foo.zarr`` **present** — the swap completed but the
      backup cleanup did not; drop the stale backup.
    - ``foo.zarr.tmp`` — an interrupted write that was never promoted; discard it.

    Args:
        store_root: Directory holding ``*.zarr`` stores for one variable.

    Returns:
        Number of orphaned entries acted on.
    """
    store_root = Path(store_root)
    if not store_root.exists():
        return 0

    acted = 0

    for bak in store_root.glob("*" + ZARR_BAK_SUFFIX):
        primary = bak.with_name(bak.name[: -len(".bak")])
        if primary.exists():
            logger.warning(f"Removing stale backup {bak.name} (primary present).")
            shutil.rmtree(bak, ignore_errors=True)
        else:
            logger.warning(
                f"Restoring {bak.name} → {primary.name} (interrupted swap recovered)."
            )
            shutil.move(str(bak), str(primary))
        acted += 1

    for tmp in store_root.glob("*" + ZARR_TMP_SUFFIX):
        logger.warning(f"Discarding interrupted Zarr write {tmp.name}.")
        shutil.rmtree(tmp, ignore_errors=True)
        acted += 1

    return acted


def _partition_from_tmp(
    tmp_name: str, partition_by: Sequence[str]
) -> Optional[tuple[str, ...]]:
    """
    Reconstruct a partition tuple from a ``.tmp_write_<v1>_<v2>...`` dir name.

    The temp name joins partition values with ``_`` (see
    ``ParquetStore.atomic_partition_write``), so it can only be split back
    unambiguously when no value itself contains ``_`` and the count matches
    ``partition_by``. Returns ``None`` when reconstruction is ambiguous — the
    caller then discards the temp dir rather than guessing a destination.
    """
    raw = tmp_name[len(PARQUET_TMP_PREFIX) :]
    parts = raw.split("_")
    if len(parts) != len(partition_by):
        return None
    return tuple(parts)


def recover_parquet_store(parquet_root: Path, partition_by: Sequence[str]) -> int:
    """
    Reconcile orphaned ``.tmp_write_<partition>`` dirs under *parquet_root*.

    ``ParquetStore.atomic_partition_write`` writes a partition to
    ``.tmp_write_<vals>``, then ``rmtree``\\ s the final partition dir and
    renames the temp into place. A crash *between* the rmtree and the rename
    leaves the partition's only complete copy in the temp dir; an earlier crash
    leaves a temp dir whose final partition still exists. In both cases the temp
    dir's parquet files would otherwise be swept up by the store-wide
    ``rglob('*.parquet')`` scans. So:

    - final partition **missing** (and temp has data) → promote the temp into place.
    - final partition **present**, or destination unreconstructable → discard the temp.

    Args:
        parquet_root: Root of the Hive-partitioned store.
        partition_by: Ordered partition column names (e.g. ``["year", "month"]``),
            used to rebuild the destination path from the temp dir name.

    Returns:
        Number of orphaned temp dirs acted on.
    """
    parquet_root = Path(parquet_root)
    if not parquet_root.exists():
        return 0

    acted = 0
    for tmp in parquet_root.glob(PARQUET_TMP_PREFIX + "*"):
        if not tmp.is_dir():
            continue

        partition = _partition_from_tmp(tmp.name, partition_by)
        final: Optional[Path] = None
        if partition is not None:
            final = parquet_root
            for col, val in zip(partition_by, partition):
                final = final / f"{col}={val}"

        if final is not None and not final.exists() and any(tmp.rglob("*.parquet")):
            logger.warning(
                f"Promoting interrupted partition write {tmp.name} → "
                f"{final.relative_to(parquet_root)}."
            )
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp), str(final))
        else:
            logger.warning(f"Discarding orphaned partition temp {tmp.name}.")
            shutil.rmtree(tmp, ignore_errors=True)
        acted += 1

    return acted
