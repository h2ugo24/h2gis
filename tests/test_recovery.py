"""
Crash-recovery and write-atomicity tests.

Covers the hard-kill scenarios that the exception-atomic write paths cannot
handle on their own: orphaned ``*.zarr.bak`` / ``*.zarr.tmp`` dirs, stranded
``.tmp_write_*`` parquet partitions, and the new-file Zarr write's temp→promote
guarantee. Each test targets behavior introduced by the recovery sweep, so it
errors/fails on the pre-fix code.
"""

import shutil

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from h2mare.storage.parquet_store import iter_store_parquet_files
from h2mare.storage.recovery import recover_parquet_store, recover_zarr_store
from h2mare.storage.storage import write_append_zarr


def _make_ds(start: str = "2020-01-01", n_days: int = 5, seed: int = 0) -> xr.Dataset:
    times = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    data = rng.uniform(10.0, 30.0, size=(n_days, 3, 3))
    return xr.Dataset(
        {"sst": (["time", "lat", "lon"], data)},
        coords={
            "time": times,
            "lat": [30.0, 35.0, 40.0],
            "lon": [-10.0, -5.0, 0.0],
        },
    )


# ---------------------------------------------------------------------------
# recover_zarr_store
# ---------------------------------------------------------------------------


class TestRecoverZarrStore:
    def test_orphaned_backup_restored_when_primary_missing(self, tmp_path):
        """A *.zarr.bak with no primary (crashed mid-swap) is restored."""
        primary = tmp_path / "sst_2020.zarr"
        bak = tmp_path / "sst_2020.zarr.bak"
        _make_ds().to_zarr(bak)

        acted = recover_zarr_store(tmp_path)

        assert acted == 1
        assert primary.exists()
        assert not bak.exists()
        ds = xr.open_zarr(primary)
        assert len(ds.time) == 5
        ds.close()

    def test_stale_backup_removed_when_primary_present(self, tmp_path):
        """A *.zarr.bak alongside a present primary is dropped, primary untouched."""
        primary = tmp_path / "sst_2020.zarr"
        bak = tmp_path / "sst_2020.zarr.bak"
        _make_ds(n_days=5).to_zarr(primary)
        _make_ds(n_days=3).to_zarr(bak)

        recover_zarr_store(tmp_path)

        assert not bak.exists()
        ds = xr.open_zarr(primary)
        assert len(ds.time) == 5  # primary preserved, not overwritten by backup
        ds.close()

    def test_interrupted_tmp_discarded(self, tmp_path):
        """A *.zarr.tmp from an interrupted write is removed."""
        tmp = tmp_path / "sst_2020.zarr.tmp"
        _make_ds().to_zarr(tmp)

        recover_zarr_store(tmp_path)

        assert not tmp.exists()

    def test_clean_store_is_noop(self, tmp_path):
        primary = tmp_path / "sst_2020.zarr"
        _make_ds().to_zarr(primary)
        assert recover_zarr_store(tmp_path) == 0
        assert primary.exists()

    def test_missing_store_root_is_noop(self, tmp_path):
        assert recover_zarr_store(tmp_path / "does_not_exist") == 0


# ---------------------------------------------------------------------------
# recover_parquet_store / iter_store_parquet_files
# ---------------------------------------------------------------------------


def _write_parquet(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"sst": [1.0, 2.0]}).write_parquet(path)


class TestRecoverParquetStore:
    def test_stranded_partition_promoted_when_final_missing(self, tmp_path):
        """.tmp_write_<part> with its final partition gone is promoted into place."""
        _write_parquet(tmp_path / ".tmp_write_2021_5" / "part-0.parquet")

        acted = recover_parquet_store(tmp_path, ["year", "month"])

        assert acted == 1
        assert (tmp_path / "year=2021" / "month=5" / "part-0.parquet").exists()
        assert not (tmp_path / ".tmp_write_2021_5").exists()

    def test_temp_discarded_when_final_present(self, tmp_path):
        """.tmp_write_<part> is dropped when the committed partition already exists."""
        _write_parquet(tmp_path / "year=2021" / "month=5" / "part-committed.parquet")
        _write_parquet(tmp_path / ".tmp_write_2021_5" / "part-0.parquet")

        recover_parquet_store(tmp_path, ["year", "month"])

        assert not (tmp_path / ".tmp_write_2021_5").exists()
        assert (tmp_path / "year=2021" / "month=5" / "part-committed.parquet").exists()

    def test_unreconstructable_temp_discarded(self, tmp_path):
        """A temp dir whose name can't map to the partition layout is discarded."""
        # Three value-parts but only two partition columns → ambiguous → discard.
        _write_parquet(tmp_path / ".tmp_write_2021_5_extra" / "part-0.parquet")

        recover_parquet_store(tmp_path, ["year", "month"])

        assert not (tmp_path / ".tmp_write_2021_5_extra").exists()

    def test_missing_root_is_noop(self, tmp_path):
        assert recover_parquet_store(tmp_path / "nope", ["year", "month"]) == 0


class TestIterStoreParquetFiles:
    def test_excludes_tmp_write_dirs(self, tmp_path):
        _write_parquet(tmp_path / "year=2020" / "month=1" / "part-0.parquet")
        _write_parquet(tmp_path / ".tmp_write_2020_1" / "part-0.parquet")

        found = iter_store_parquet_files(tmp_path)

        names = {p.parent.name for p in found}
        assert "month=1" in names
        assert not any(".tmp_write" in str(p) for p in found)
        assert len(found) == 1


# ---------------------------------------------------------------------------
# write_append_zarr — new-file write atomicity (Fix A)
# ---------------------------------------------------------------------------


class TestNewWriteAtomicity:
    def test_crash_during_promote_leaves_no_partial_final(self, tmp_path, monkeypatch):
        """
        If the process dies during the temp→final promote, the final *.zarr
        must not exist as a partial store (gap detection would trust it). The
        verified data survives in *.zarr.tmp for recovery to discard.
        """
        path = tmp_path / "sst.zarr"

        real_move = shutil.move

        def boom(src, dst, *a, **k):
            # Fail only the promote of our tmp → final, not the backup restore.
            if str(src).endswith(".zarr.tmp"):
                raise OSError("simulated hard kill during promote")
            return real_move(src, dst, *a, **k)

        monkeypatch.setattr("h2mare.storage.storage.shutil.move", boom)

        with pytest.raises(OSError):
            write_append_zarr("sst", _make_ds(), path)

        assert not path.exists()
        assert path.with_name(path.name + ".tmp").exists()


# ---------------------------------------------------------------------------
# write_append_zarr — orphaned backup restore (Fix B)
# ---------------------------------------------------------------------------


class TestOrphanedBackupRestore:
    def test_append_restores_backup_instead_of_overwriting(self, tmp_path):
        """
        With the primary missing but a *.zarr.bak present (crashed swap), a new
        append must restore the backup and extend it — not take the new-file
        path and drop the store's history.
        """
        path = tmp_path / "sst.zarr"
        bak = path.with_name(path.name + ".bak")
        _make_ds("2020-01-01", 5).to_zarr(bak)  # only the backup survives

        write_append_zarr("sst", _make_ds("2020-01-06", 5), path)

        assert not bak.exists()
        ds = xr.open_zarr(path)
        assert len(ds.time) == 10  # history (5) + increment (5), nothing lost
        ds.close()


# ---------------------------------------------------------------------------
# _resolve_overlap — bbox mismatch raises a clear ValueError (Fix E)
# ---------------------------------------------------------------------------


class TestBboxOverlapError:
    def test_non_overlapping_bbox_raises_valueerror(self, tmp_path):
        path = tmp_path / "sst.zarr"
        _make_ds("2020-01-01", 5).to_zarr(path)

        # Same variable (forces the time-extension/_resolve_overlap path) but a
        # disjoint longitude band → no geographic overlap.
        far = _make_ds("2020-01-06", 5)
        far = far.assign_coords(lon=[50.0, 55.0, 60.0])

        with pytest.raises(ValueError, match="do not overlap"):
            write_append_zarr("sst", far, path)
