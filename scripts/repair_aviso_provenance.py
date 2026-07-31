"""
Repair rep/nrt provenance on existing AVISO Zarr stores.

Until the download manifest was added for AVISO, ``Netcdf2Zarr._write_provenance``
silently no-opped for fsle and eddies: no ``source_datasets`` attribute was ever
written, so ``ZarrCatalog`` fell back to a single row labelled ``dataset_id_rep``
and stamped near-real-time data as delayed-time. That is why ``h2mare catalog``
showed delayed-time and near-real-time ranges that overlap, which cannot happen
on the FTP server where the two directories are disjoint.

New conversions now label themselves. This tool repairs what is already on disk.

It reads the live rep/nrt availability from the AVISO FTP server, so it needs the
same credentials as a download (``AVISO_FTP_SERVER`` / ``AVISO_USERNAME`` /
``AVISO_PASSWORD`` in ``.env``).

Two kinds of problem are reported:

  * **missing** — no ``source_datasets`` at all; the catalog is silently calling
    this file delayed-time. ``--apply`` fixes these.
  * **stale** — provenance exists but its rep/nrt boundary disagrees with what
    the FTP server currently reports, e.g. because AVISO has since republished
    part of the near-real-time period as delayed-time. Reported only:
    ``backfill_provenance`` deliberately skips files that already carry
    provenance, and rewriting them is a judgement call about re-downloading.

Safety:
  * --dry-run (default) only reads. Nothing is written, no FTP downloads occur.
  * --apply calls ``ZarrCatalog.backfill_provenance``, which writes the
    ``source_datasets`` attribute on files that lack it and then reloads the
    catalog. Existing provenance is never overwritten.

Run:
    uv run python scripts/repair_aviso_provenance.py                     # dry run, all AVISO vars
    uv run python scripts/repair_aviso_provenance.py --var-keys fsle     # one variable
    uv run python scripts/repair_aviso_provenance.py --apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr
from loguru import logger

from h2mare.config import get_settings
from h2mare.downloader.aviso_downloader import AVISODownloader
from h2mare.storage.zarr_catalog import ZarrCatalog


def _aviso_var_keys() -> list[str]:
    """Every configured variable served from the AVISO FTP server."""
    return [
        key
        for key, entry in get_settings().app_config.variables.items()
        if entry.source == "aviso"
    ]


def _read_provenance(zarr_path: Path) -> Optional[list[dict]]:
    """Return the recorded ``source_datasets`` for *zarr_path*, or None."""
    import zarr

    try:
        raw = zarr.open_group(str(zarr_path), mode="r").attrs.get("source_datasets")
    except Exception as e:
        logger.warning(f"Could not read {zarr_path.name}: {e}")
        return None
    return json.loads(raw) if raw else None


def _file_span(zarr_path: Path) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        ds = xr.open_zarr(zarr_path, consolidated=False)
        t = pd.to_datetime(ds.time.values)
        ds.close()
        return t.min().normalize(), t.max().normalize()
    except Exception as e:
        logger.warning(f"Could not read time axis of {zarr_path.name}: {e}")
        return None


def _classify(
    records: Optional[list[dict]],
    span: tuple[pd.Timestamp, pd.Timestamp],
    rep_end: pd.Timestamp,
) -> tuple[str, str]:
    """Classify one file as ok / missing / stale, with a human-readable reason."""
    start, end = span

    if records is None:
        implied = "nrt" if start > rep_end else "rep"
        return "missing", (
            f"no provenance — catalog reads it as delayed-time; "
            f"FTP says this span is {implied}"
        )

    # What the recorded provenance claims per type.
    claimed_rep_end = max(
        (pd.Timestamp(r["end_date"]) for r in records if r["dataset_type"] == "rep"),
        default=None,
    )
    claimed_nrt_start = min(
        (pd.Timestamp(r["start_date"]) for r in records if r["dataset_type"] == "nrt"),
        default=None,
    )

    # A file entirely inside the rep period should carry no nrt entry, and vice versa.
    if claimed_nrt_start is not None and claimed_nrt_start <= rep_end:
        return "stale", (
            f"claims nrt from {claimed_nrt_start.date()}, but FTP delayed-time "
            f"now runs to {rep_end.date()}"
        )
    if claimed_rep_end is not None and claimed_rep_end > rep_end:
        return "stale", (
            f"claims rep to {claimed_rep_end.date()}, beyond the FTP "
            f"delayed-time end {rep_end.date()}"
        )
    return "ok", ""


def inspect(var_key: str) -> dict:
    """Report provenance health for one variable against the live FTP listing."""
    logger.info(f"[{var_key}] querying AVISO FTP for dataset availability…")
    dl = AVISODownloader(var_key)
    rep_avail = dl.get_rep_availability()
    nrt_avail = dl.get_nrt_availability()

    rep_end = pd.Timestamp(rep_avail.end).normalize()
    logger.info(
        f"[{var_key}] FTP delayed-time: {pd.Timestamp(rep_avail.start).date()} → {rep_end.date()}"
        + (
            f" | near-real-time: {pd.Timestamp(nrt_avail.start).date()} → "
            f"{pd.Timestamp(nrt_avail.end).date()}"
            if nrt_avail
            else " | near-real-time: not configured"
        )
    )

    catalog = ZarrCatalog(var_key, auto_refresh=False)
    buckets: dict[str, list[str]] = {"ok": [], "missing": [], "stale": []}

    for zarr_path in sorted(catalog.store_root.glob("*.zarr")):
        span = _file_span(zarr_path)
        if span is None:
            continue
        status, reason = _classify(_read_provenance(zarr_path), span, rep_end)
        buckets[status].append(zarr_path.name)
        if status != "ok":
            logger.warning(
                f"[{var_key}] {status.upper():7} {zarr_path.name} "
                f"({span[0].date()} → {span[1].date()}): {reason}"
            )

    logger.info(
        f"[{var_key}] {len(buckets['ok'])} ok, {len(buckets['missing'])} missing, "
        f"{len(buckets['stale'])} stale"
    )
    return {"var_key": var_key, "rep_end": rep_end, **buckets}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--var-keys",
        nargs="*",
        help="AVISO variable keys to check (default: every variable with source: aviso).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write provenance for files that have none. Without this, nothing is written.",
    )
    args = parser.parse_args()

    var_keys = args.var_keys or _aviso_var_keys()
    if not var_keys:
        logger.error("No AVISO variables configured.")
        return

    reports = []
    for var_key in var_keys:
        try:
            reports.append(inspect(var_key))
        except Exception as e:
            logger.opt(exception=True).error(f"[{var_key}] inspection failed: {e}")

    if not args.apply:
        total_missing = sum(len(r["missing"]) for r in reports)
        total_stale = sum(len(r["stale"]) for r in reports)
        logger.info(
            f"Dry run — {total_missing} file(s) would be repaired, "
            f"{total_stale} stale file(s) need a decision. Re-run with --apply."
        )
        return

    for report in reports:
        var_key = report["var_key"]
        if not report["missing"]:
            logger.info(f"[{var_key}] nothing to repair")
            continue
        n = ZarrCatalog(var_key).backfill_provenance(report["rep_end"])
        logger.success(f"[{var_key}] wrote provenance for {n} file(s)")

    stale = [(r["var_key"], f) for r in reports for f in r["stale"]]
    if stale:
        logger.warning(
            f"{len(stale)} file(s) carry provenance that disagrees with the FTP "
            "listing and were left untouched: "
            + ", ".join(f"{v}/{f}" for v, f in stale)
        )


if __name__ == "__main__":
    main()
