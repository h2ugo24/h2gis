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


#: What a record says.
#:
#: ``start_date``/``end_date`` are what the file holds from that dataset, and
#: together the records account for every day on its time axis. One dataset for
#: a year means that year's first and last day; a year where rep hands over to
#: nrt splits at the handover. So a file states its coverage the same way
#: whatever sequence of runs built it.
#:
#: They are read off the axis on every write rather than accumulated, which is
#: what keeps them from drifting: a download window is what was *asked* for, and
#: recording that instead is how a store appended to over a hundred runs ended
#: up claiming only the last week of them.
#:
#: ``days`` counts the days actually present in the span — equal to its length
#: unless the source skipped some. ``updated`` is when the record was last
#: rewritten, which is the one fact the axis cannot supply.
_DERIVED = ("start_date", "end_date", "days", "updated")


def _anchor(record: dict) -> str:
    """
    First day attributed to a record's dataset, used to order the handovers.

    ``start_date`` under either layout: the covered one it now means, and the
    requested one it used to. A narrower value than the truth is harmless
    because :func:`annotate_covered` only uses it for ordering — the first
    dataset on the axis is extended back to the file's own start regardless,
    which is what repairs a file whose history was never fully recorded.
    """
    return record["start_date"]


def records_for_window(manifest: list[dict], window: DateRange) -> list[dict]:
    """
    Build ``source_datasets`` records for the part of *window* each dataset covered.

    The generic converter path matches a whole raw file to one manifest entry,
    which suits per-period source files. It does not suit eddies: a single
    trajectory file spans years and both the rep and nrt periods, while each
    period is written to its own Zarr. Intersecting the written window with the
    manifest attributes each period correctly.

    The dates here are provisional: they say where each dataset's contribution
    begins, which is what :func:`annotate_covered` needs to place the handovers
    before replacing them with what the axis actually holds.

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
    return sorted(records, key=_anchor)


def merge_records(existing: list[dict], new: list[dict]) -> list[dict]:
    """
    Combine provenance records, widening the span of any dataset present in both.

    Periods are appended to incrementally, so a Zarr written across several runs
    accumulates coverage. Replacing the attribute with only the latest run's
    records — as the generic converter path used to — would drop the earlier
    part of the same file.

    A dataset seen twice keeps the earliest start and the latest end, which is
    all h2ds needs: it merges the spans its sources already worked out and has
    no time axis of its own to recompute them from. Every per-variable store
    does have one, and :func:`annotate_covered` runs afterwards and replaces
    these dates with what it holds — so merging here settles *which* datasets a
    file involves and roughly where each begins, not what it finally says.

    ``days`` is dropped rather than combined: summing double-counts a re-convert
    of a period already present, and taking a maximum understates a genuine
    append. Same for ``updated``, which belongs to a write, not to a span.
    """
    merged: dict[str, dict] = {}
    for record in [*existing, *new]:
        key = record["dataset_id"]
        span = {
            k: v
            for k, v in record.items()
            if not k.startswith("delivered_") and k not in ("days", "updated")
        }
        if key not in merged:
            merged[key] = span
            continue

        held = merged[key]
        held["start_date"] = min(held["start_date"], span["start_date"])
        held["end_date"] = max(held["end_date"], span["end_date"])
    return sorted(merged.values(), key=_anchor)


def annotate_covered(records: list[dict], stored: pd.DatetimeIndex) -> list[dict]:
    """
    Share the store's time axis out between the datasets that supplied it.

    Every day on the axis is attributed, so the records account for the whole
    file: one dataset for a year gets that year's first and last day, and a year
    where rep hands over to nrt splits at the handover. The records say the same
    thing whether the file was written in one run or a hundred.

    Each dataset holds from where it starts until the next one begins, and the
    earliest is extended back to the first day on the axis. That last part is
    what repairs a file whose history was only partly recorded: a store filled
    week by week ends up with a record naming the most recent week, and reading
    it literally would claim the year holds seven days. Extending it back says
    what is true — that dataset is what the file is made of.

    ``days`` counts what is actually present in the span, so a source that
    skipped days shows fewer than the span is long. ``updated`` stamps the
    write, being the one fact no amount of reading the axis can recover.

    A file with no time axis, or records for datasets it holds nothing from,
    yields nothing rather than dates that describe no data.
    """
    stored = pd.DatetimeIndex(stored).normalize().unique().sort_values()
    if len(stored) == 0 or not records:
        return []

    ordered = sorted(records, key=_anchor)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    out = []

    for i, record in enumerate(ordered):
        # The first dataset owns everything up to the second one, however late
        # its own record happens to start.
        start = stored[0] if i == 0 else max(pd.to_datetime(_anchor(record)), stored[0])
        if i + 1 < len(ordered):
            end = pd.to_datetime(_anchor(ordered[i + 1])) - pd.Timedelta(days=1)
        else:
            end = stored[-1]

        inside = stored[(stored >= start) & (stored <= end)]
        if len(inside) == 0:
            continue

        out.append(
            {
                "dataset_id": record["dataset_id"],
                "dataset_type": record["dataset_type"],
                "start_date": inside.min().strftime("%Y-%m-%d"),
                "end_date": inside.max().strftime("%Y-%m-%d"),
                "days": int(len(inside)),
                "updated": today,
            }
        )
    return out


def write_provenance_for_window(
    zarr_path: Path,
    manifest: list[dict],
    window: DateRange,
    stored: pd.DatetimeIndex | None = None,
) -> list[dict]:
    """
    Stamp ``source_datasets`` onto *zarr_path* for the given written *window*.

    Merges with whatever the file already carries, so repeated appends widen the
    recorded coverage instead of overwriting it. Returns the records written
    (empty when the manifest covers none of the window).

    Args:
        stored: The store's time axis, used to record what was actually
            delivered against each requested span. Read from *zarr_path* when
            omitted.
    """
    import zarr

    records = records_for_window(manifest, window)
    if not records:
        return []

    root = zarr.open_group(str(zarr_path), mode="r+")
    raw = root.attrs.get("source_datasets")
    existing = json.loads(raw) if raw else []
    combined = merge_records(existing, records)
    combined = annotate_covered(
        combined, stored if stored is not None else read_store_dates(zarr_path)
    )
    root.attrs["source_datasets"] = json.dumps(combined)
    return combined


def read_store_dates(zarr_path: Path) -> pd.DatetimeIndex:
    """Time axis of a Zarr store. Coordinate read only — no data is touched."""
    try:
        with xr.open_zarr(zarr_path, consolidated=False) as ds:
            if "time" not in ds.coords:
                return pd.DatetimeIndex([])
            return pd.DatetimeIndex(ds.time.values).normalize().unique().sort_values()
    except Exception:
        return pd.DatetimeIndex([])


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

        # The split above reconstructs which dataset supplied which window; the
        # covered dates then come off the store's own axis, exactly as they do
        # on the convert path, so a backfilled file and a converted one say the
        # same kind of thing.
        records = annotate_covered(records, read_store_dates(zarr_path))
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


# ---------------------------------------------------------------------------
# Compiled (h2ds) provenance
# ---------------------------------------------------------------------------

#: Attribute h2ds carries its per-source provenance under.
#:
#: Deliberately not ``source_datasets``. That one is a flat list, which suits a
#: store fed by a single product; h2ds merges a dozen sources, so its records
#: have to say which variable each belongs to and the value is a mapping. Reusing
#: the name would hand a reader a dict where it expected a list.
COMPILED_PROVENANCE_ATTR = "source_datasets_by_variable"


def read_source_datasets(zarr_path: Path) -> list[dict]:
    """
    Provenance records held by a store, or ``[]`` when it has none.

    Absent, unreadable and malformed all return empty rather than raising:
    provenance is metadata about a compile, and losing it must never be the
    reason a compile fails.
    """
    import zarr

    try:
        root = zarr.open_group(str(zarr_path), mode="r")
        raw = root.attrs.get("source_datasets")
    except Exception:
        return []
    if not isinstance(raw, str) or not raw:
        return []
    try:
        records = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return records if isinstance(records, list) else []


def collect_source_datasets(catalog: "ZarrCatalog", window: DateRange) -> list[dict]:
    """
    Merge the provenance of every store file overlapping *window*.

    A compile window routinely spans several per-period files, and a variable
    can switch from rep to nrt part-way through, so the records are merged
    rather than taken from the first file found.
    """
    records: list[dict] = []
    for path in catalog.get_paths_in_range(window.start, window.end):
        records = merge_records(records, read_source_datasets(Path(path)))
    return records


def write_compiled_provenance(
    zarr_path: Path, by_variable: dict[str, list[dict]]
) -> dict[str, list[dict]]:
    """
    Record on h2ds which dataset delivered which dates, for each source variable.

    Merged into whatever the file already holds, never replacing it: ``run -v
    sst`` recompiles one variable's columns into a file that already carries
    every other variable's records, and replacing would silently drop them —
    the same trap the per-variable path documents in ``write_provenance``.

    Written after the Zarr exists rather than through ``ds.attrs``, so it does
    not depend on how the write path combines attributes, and so a compile is
    never failed by its own bookkeeping.
    """
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r+")
    raw = root.attrs.get(COMPILED_PROVENANCE_ATTR)
    existing = json.loads(raw) if isinstance(raw, str) and raw else {}

    combined = dict(existing)
    for var_key, records in by_variable.items():
        combined[var_key] = merge_records(existing.get(var_key, []), records)

    root.attrs[COMPILED_PROVENANCE_ATTR] = json.dumps(combined, sort_keys=True)
    return combined


#: Root attributes a refresh must carry across rather than overwrite, because
#: they are written by the pipeline itself and have no counterpart in config.
_DERIVED_ROOT_ATTRS = (COMPILED_PROVENANCE_ATTR,)


def refresh_root_attrs(zarr_path: Path, global_attrs: dict) -> dict:
    """
    Make a store's root attributes match config rather than its own history.

    ``xr.concat`` keeps the first dataset's attributes, so appending to an
    existing store preserves whatever globals it was created with and a config
    change never reaches it. h2ds went on advertising a ``products ID`` block
    for a fortnight after that key was deleted from config — still holding two
    dataset ids that had since been corrected — because no compile ever rewrote
    the root.

    Replaces rather than updates, so a key removed from config is removed here
    too; updating alone would have left ``products ID`` in place forever.
    Attributes the pipeline derives rather than reads from config are carried
    across.
    """
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r+")
    preserved = {k: root.attrs[k] for k in _DERIVED_ROOT_ATTRS if k in root.attrs}
    merged = {**global_attrs, **preserved}
    root.attrs.put(merged)
    return merged
