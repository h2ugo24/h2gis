"""One-time migration helpers for Zarr provenance metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from h2mare.types import DateLike, DateRange

if TYPE_CHECKING:
    from h2mare.storage.zarr_catalog import ZarrCatalog


def records_for_window(manifest: list[dict], window: DateRange) -> list[dict]:
    """
    Build ``source_datasets`` records for the part of *window* each dataset covered.

    The generic converter path matches a whole raw file to one manifest entry,
    which suits per-period source files. It does not suit eddies: a single
    trajectory file spans years and both the rep and nrt periods, while each
    period is written to its own Zarr. Intersecting the written window with the
    manifest attributes each period correctly.

    Entries that do not overlap *window* are dropped.
    """
    records = []
    for entry in manifest:
        start = max(pd.to_datetime(entry["start"]).normalize(), window.start)
        end = min(pd.to_datetime(entry["end"]).normalize(), window.end)
        if start > end:
            continue
        records.append(
            {
                "dataset_id": entry["dataset_id"],
                "dataset_type": entry["dataset_type"],
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
            }
        )
    return sorted(records, key=lambda r: r["start_date"])


def merge_records(existing: list[dict], new: list[dict]) -> list[dict]:
    """
    Combine provenance records, widening the span of any dataset present in both.

    Periods are appended to incrementally, so a Zarr written across several runs
    accumulates coverage. Replacing the attribute with only the latest run's
    records — as the generic converter path does — would drop the earlier part
    of the same file.
    """
    merged: dict[str, dict] = {}
    for record in [*existing, *new]:
        key = record["dataset_id"]
        if key not in merged:
            merged[key] = dict(record)
            continue
        merged[key]["start_date"] = min(merged[key]["start_date"], record["start_date"])
        merged[key]["end_date"] = max(merged[key]["end_date"], record["end_date"])
    return sorted(merged.values(), key=lambda r: r["start_date"])


def write_provenance_for_window(
    zarr_path: Path, manifest: list[dict], window: DateRange
) -> list[dict]:
    """
    Stamp ``source_datasets`` onto *zarr_path* for the given written *window*.

    Merges with whatever the file already carries, so repeated appends widen the
    recorded coverage instead of overwriting it. Returns the records written
    (empty when the manifest covers none of the window).
    """
    import zarr

    records = records_for_window(manifest, window)
    if not records:
        return []

    root = zarr.open_group(str(zarr_path), mode="r+")
    raw = root.attrs.get("source_datasets")
    existing = json.loads(raw) if raw else []
    combined = merge_records(existing, records)
    root.attrs["source_datasets"] = json.dumps(combined)
    return combined


def backfill_provenance(catalog: "ZarrCatalog", rep_end_date: DateLike) -> int:
    """
    Retroactively write provenance for existing Zarr files that pre-date
    automatic tracking by Netcdf2Zarr.

    For each Zarr file in the catalog's store_root that has no
    ``source_datasets`` attribute:

    * Entire file falls within rep period  -> single rep entry.
    * Entire file falls after rep end date -> single nrt entry
      (only written when dataset_id_nrt is configured).
    * File spans the rep/nrt boundary    -> two entries split at
      rep_end_date / rep_end_date + 1 day.

    Call once after upgrading. The rep end date is obtainable without
    re-downloading data from the downloader for that variable's source —
    ``get_rep_availability().end`` on either CMEMSDownloader or AVISODownloader.
    Use the one matching ``var_config.source``; for AVISO variables the value
    comes from the FTP directory listing, not the CMEMS API.

    Files that already carry ``source_datasets`` are skipped, so this never
    overwrites existing provenance — including provenance that has since gone
    stale because the source republished part of the nrt period as rep. Fixing
    a stale file means re-converting it from raw, not calling this.

    Args:
        catalog: The variable's ZarrCatalog.
        rep_end_date: Last date covered by the reprocessed (rep) dataset.

    Returns:
        Number of zarr files updated.

    Example::

        from h2mare.storage.zarr_catalog import ZarrCatalog
        from h2mare.downloader.cmems_downloader import CMEMSDownloader

        rep_end = CMEMSDownloader("sst").get_rep_availability().end
        n = ZarrCatalog("sst").backfill_provenance(rep_end)
        print(f"Written {n} sidecars")

        # AVISO variables (fsle, eddies) use their own downloader:
        from h2mare.downloader.aviso_downloader import AVISODownloader

        rep_end = AVISODownloader("fsle").get_rep_availability().end
    """
    rep_end = pd.to_datetime(rep_end_date).normalize()
    nrt_start = rep_end + pd.Timedelta(days=1)
    has_nrt = catalog.var_config.dataset_id_nrt is not None

    if not catalog.store_root.exists():
        catalog._log("warning", f"Store root not found: {catalog.store_root}")
        return 0

    import zarr

    written = 0
    for zarr_path in sorted(catalog.store_root.glob("*.zarr")):
        try:
            ds = xr.open_zarr(zarr_path, consolidated=False)
            already_set = ds.attrs.get("source_datasets") is not None
            z_start = pd.to_datetime(ds.time.min().compute().item()).normalize()
            z_end = pd.to_datetime(ds.time.max().compute().item()).normalize()
            ds.close()
        except Exception as e:
            catalog._log("warning", f"Could not read {zarr_path.name}: {e}")
            continue

        if already_set:
            catalog._log(
                "debug",
                f"Provenance already in zarr attrs, skipping: {zarr_path.name}",
            )
            continue

        records = []

        if z_end <= rep_end or not has_nrt:
            records.append(
                {
                    "dataset_id": catalog.var_config.dataset_id_rep,
                    "dataset_type": "rep",
                    "start_date": z_start.strftime("%Y-%m-%d"),
                    "end_date": z_end.strftime("%Y-%m-%d"),
                }
            )
        elif z_start > rep_end:
            records.append(
                {
                    "dataset_id": catalog.var_config.dataset_id_nrt,
                    "dataset_type": "nrt",
                    "start_date": z_start.strftime("%Y-%m-%d"),
                    "end_date": z_end.strftime("%Y-%m-%d"),
                }
            )
        else:
            records.append(
                {
                    "dataset_id": catalog.var_config.dataset_id_rep,
                    "dataset_type": "rep",
                    "start_date": z_start.strftime("%Y-%m-%d"),
                    "end_date": rep_end.strftime("%Y-%m-%d"),
                }
            )
            records.append(
                {
                    "dataset_id": catalog.var_config.dataset_id_nrt,
                    "dataset_type": "nrt",
                    "start_date": nrt_start.strftime("%Y-%m-%d"),
                    "end_date": z_end.strftime("%Y-%m-%d"),
                }
            )

        root = zarr.open_group(str(zarr_path), mode="r+")
        root.attrs["source_datasets"] = json.dumps(records)

        # Remove any legacy sidecar now that provenance lives in zarr attrs
        prov_file = zarr_path.parent / (zarr_path.stem + "_prov.json")
        if prov_file.exists():
            prov_file.unlink()

        catalog._log(
            "info",
            f"Wrote backfilled provenance for {zarr_path.name} ({len(records)} source(s))",
        )
        written += 1

    if written:
        catalog.reload()
        catalog._log(
            "info",
            f"Backfill complete: {written} zarr file(s) updated, catalog reloaded",
        )
    else:
        catalog._log("info", "Backfill complete: no files needed provenance")

    return written
