"""
Backfill CF metadata onto Zarr stores that were written before it existed.

``apply_cf_attrs`` only reaches newly written data. An append does not rewrite
``lat``/``lon``, and it never revisits a variable's attributes, so every store
already on disk keeps whatever it was created with — which for the coordinates
is nothing at all, and for the derived variables (``sst``, ``gke``, the ``_std``
layers) is nothing at all either, since xarray drops attributes on arithmetic.

This rewrites **metadata only**. No data array is read or written: variable and
coordinate attributes are small JSON blobs beside the chunks, so a full pass
over the production store is a matter of seconds rather than the hours a
recompile would take.

What it writes is not decided here. Variable attributes come from
``resolve_cf_attrs`` and coordinates from ``_CF_COORD_ATTRS``, both shared with
the write path, so a repaired store and a freshly written one cannot disagree.

It also drops a stray ``valid_time``. The hourly waves path used to leave one
behind — a byte-for-byte duplicate of ``time`` — and the store could not be
reconverted once the raw GRIBs were gone. Only ever removed when it really is
identical to ``time``; a ``valid_time`` that differs is data, and is reported
and left alone.

Usage:
    uv run python scripts/repair_cf_attrs.py waves              # dry run
    uv run python scripts/repair_cf_attrs.py waves --apply
    uv run python scripts/repair_cf_attrs.py --all              # survey everything
    uv run python scripts/repair_cf_attrs.py --all --apply
    uv run python scripts/repair_cf_attrs.py --all --apply --drop-legacy-description
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from h2mare import get_settings
from h2mare.storage.provenance import (
    extent_root_attrs,
    pipeline_root_attrs,
    refresh_root_attrs,
)
from h2mare.storage.xarray_helpers import (
    _CF_COORD_ATTRS,
    _SOURCE_ENCODING_ATTRS,
    _SOURCE_ENCODING_PREFIXES,
    resolve_cf_attrs,
)

#: The duplicate coordinate the hourly waves path used to leave behind.
_STRAY_COORD = "valid_time"


def store_paths(var_key: str) -> list[Path]:
    """
    Every Zarr store for *var_key*, by name.

    Globbed rather than taken from ``ZarrCatalog``, which cannot be constructed
    for a time-less store at all: its scanner reads ``ds.time.min()`` to build
    the coverage index, so ``ZarrCatalog("bathy")`` raises on the ETOPO store
    before returning any paths. Repairing metadata needs no coverage index, and
    bathy needs its coordinates labelled like everything else.
    """
    settings = get_settings()
    config = settings.app_config.variables.get(var_key)
    if config is None:
        return []
    return sorted((settings.STORE_ROOT / config.local_folder).glob("*.zarr"))


def is_compiled(var_key: str) -> bool:
    """
    Whether *var_key* is the compiled store rather than a native one.

    By ``source: h2mare`` rather than by the name ``h2ds``, matching how
    ``Extractor._compiled_var_key`` identifies it, so a deployment that names
    its compiled store differently still routes correctly.
    """
    config = get_settings().app_config.variables.get(var_key)
    return getattr(config, "source", None) == "h2mare"


def _stale_source_attrs(attrs: dict, *, drop_grib: bool) -> list[str]:
    """Attribute names describing the source file rather than this store."""
    prefixes = _SOURCE_ENCODING_PREFIXES if drop_grib else ()
    return [
        name
        for name in attrs
        if (prefixes and name.startswith(prefixes)) or name in _SOURCE_ENCODING_ATTRS
    ]


def _description_is_superseded(
    current: dict, desired: dict, *, any_wording: bool = False
) -> bool:
    """
    Whether a leftover ``description`` can go now that ``comment`` is written.

    Config renamed ``description`` to ``comment`` — the name CF and ACDD define
    — but the repair only ever writes the attributes config *names*, so the old
    key would sit beside its replacement forever holding the same text. On h2ds
    it does exactly that; 2108 variables carried both.

    By default dropped only where it says nothing the store loses: empty, which
    is what CMEMS ships, or word for word what ``comment`` now carries. Anything
    else might be the source's own, and that is not ours to discard.

    ``any_wording`` widens it to every ``description`` config now has a
    ``comment`` for. What that catches is h2mare's own earlier prose, kept
    verbatim from a version of the code that has since been deleted — the native
    ``sst_std`` still says "Derived from a rolling mean (3*3 cells) from sst
    native resolution", against config's fuller sentence. Nothing in h2mare
    writes ``description`` any more, so on these stores nothing else can be
    producing it; it stays opt-in only because that reasoning is about this
    repo's history rather than anything the attribute itself declares.
    """
    if "description" not in current or "comment" not in desired:
        return False
    existing = current["description"]
    if any_wording:
        return True
    return not str(existing).strip() or existing == desired["comment"]


def plan_variable(
    current: dict,
    var_name: str,
    *,
    native_var_key: str | None,
    drop_legacy_description: bool = False,
) -> tuple[dict, list[str]]:
    """(attributes to set, attribute names to delete) for one variable."""
    desired = resolve_cf_attrs(var_name, native_var_key)
    sets = {
        key: value
        for key, value in desired.items()
        if value is not None and current.get(key) != value
    }
    deletes = [
        key for key, value in desired.items() if value is None and key in current
    ]
    deletes += _stale_source_attrs(current, drop_grib=native_var_key is None)
    if _description_is_superseded(
        current, desired, any_wording=drop_legacy_description
    ):
        deletes.append("description")

    # Config describing a variable but naming no units is a positive statement,
    # not an omission: the eddy track ids are ordinal labels and CF lets a label
    # variable carry none. Writing only what config names leaves the old value
    # in place, so the stores still held 'ordinal' and 'unitless' — neither of
    # which udunits parses, and the only unparseable units left anywhere.
    # Guarded on `desired` so a variable config says nothing about keeps its own.
    if desired and "units" not in desired and "units" in current:
        deletes.append("units")

    return sets, sorted(set(deletes))


def _stray_is_a_duplicate(path: Path) -> bool | None:
    """True if valid_time duplicates time, False if it differs, None if absent."""
    with xr.open_zarr(path, consolidated=False) as ds:
        if _STRAY_COORD not in ds.variables:
            return None
        if "time" not in ds.variables:
            return False
        stray, time = ds[_STRAY_COORD].values, ds["time"].values
        return bool(stray.shape == time.shape and np.array_equal(stray, time))


def _clean_coordinates_attr(root: zarr.Group, names: set[str]) -> dict[str, str | None]:
    """
    Rewrite each variable's ``coordinates`` attribute to name only real ones.

    The waves store listed ``number time step meanSea latitude longitude
    valid_time``, of which five never existed in the store and ``time`` is a
    dimension coordinate, which CF says must *not* be listed. Removing
    valid_time without this leaves xarray naming coordinates it cannot find.
    """
    changes: dict[str, str | None] = {}
    for name in names:
        listed = str(root[name].attrs.get("coordinates", "")).split()
        if not listed:
            continue
        # Worth listing only if it exists in the store and is an auxiliary
        # coordinate — one whose sole dimension is not itself. A dimension
        # coordinate belongs in `dimension_names`, not here.
        kept = [
            c
            for c in listed
            if c in root and tuple(root[c].metadata.dimension_names or ()) != (c,)
        ]
        changes[name] = " ".join(kept) if kept else None
    return changes


def repair(var_key: str, *, apply: bool, drop_legacy_description: bool = False) -> int:
    """Report (and optionally write) the metadata repair for one var_key."""
    paths = store_paths(var_key)
    if not paths:
        print(f"[{var_key}] no stores found")
        return 0

    native_var_key = None if is_compiled(var_key) else var_key
    kind = "compiled" if native_var_key is None else "native"
    print(f"[{var_key}] {len(paths)} store(s), {kind}")

    changed = 0
    for path in paths:
        root = zarr.open_group(str(path), mode="r" if not apply else "r+")
        members = {name for name, _ in root.arrays()}
        coords = set(_CF_COORD_ATTRS) & members
        data_vars = members - coords - {_STRAY_COORD}

        notes: list[str] = []
        var_plans: dict[str, tuple[dict, list[str]]] = {}
        for name in sorted(data_vars):
            sets, deletes = plan_variable(
                dict(root[name].attrs),
                name,
                native_var_key=native_var_key,
                drop_legacy_description=drop_legacy_description,
            )
            if sets or deletes:
                var_plans[name] = (sets, deletes)
                detail = ", ".join(
                    [f"+{k}" for k in sorted(sets)] + [f"-{k}" for k in deletes]
                )
                notes.append(f"{name}: {detail}")

        coord_plans: dict[str, dict] = {}
        for name in sorted(coords):
            desired = _CF_COORD_ATTRS[name]
            current = dict(root[name].attrs)
            sets = {k: v for k, v in desired.items() if current.get(k) != v}
            if sets:
                coord_plans[name] = sets
                notes.append(f"{name}: +{', +'.join(sorted(sets))}")

        stray = _stray_is_a_duplicate(path)
        drop_stray = stray is True
        if stray is False:
            notes.append(f"{_STRAY_COORD}: differs from time — LEFT ALONE, inspect it")
        elif drop_stray:
            notes.append(f"{_STRAY_COORD}: duplicate of time — remove")

        if not notes:
            continue

        changed += 1
        if not apply:
            print(f"  WOULD FIX {path.name}")
            for note in notes:
                print(f"      {note}")
            continue

        for name, (sets, deletes) in var_plans.items():
            attrs = root[name].attrs
            for key in deletes:
                del attrs[key]
            if sets:
                attrs.update(sets)
        for name, sets in coord_plans.items():
            root[name].attrs.update(sets)

        if drop_stray:
            # Delete before the cleanup, not after: _clean_coordinates_attr
            # keeps any name it can still resolve in the store, so with the
            # array still present it would keep the very one being removed.
            del root[_STRAY_COORD]
            for name, value in _clean_coordinates_attr(root, data_vars).items():
                attrs = root[name].attrs
                if value is None:
                    attrs.pop("coordinates", None)
                else:
                    attrs["coordinates"] = value

        if native_var_key is None:
            refresh_root_attrs(path, get_settings().global_attrs)
        else:
            root.attrs.update({**pipeline_root_attrs(), **extent_root_attrs(path)})

        # Re-consolidate: these stores carry consolidated metadata, and every
        # edit above went to the arrays' own metadata. Left stale, a reader that
        # trusts the consolidated copy still sees the attributes — and still
        # sees valid_time.
        zarr.consolidate_metadata(root.store)

        print(f"  FIXED {path.name}")
        for note in notes:
            print(f"      {note}")

    if not changed:
        print("  already correct")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("var_keys", nargs="*", help="variable(s) to repair")
    parser.add_argument(
        "--all", action="store_true", help="repair every configured var_key"
    )
    parser.add_argument(
        "--apply", action="store_true", help="write the repair (default is a dry run)"
    )
    parser.add_argument(
        "--drop-legacy-description",
        action="store_true",
        help=(
            "also remove a description whose wording differs from the comment "
            "config now supplies. Nothing in h2mare writes description any "
            "more, so what remains is this repo's own earlier prose"
        ),
    )
    args = parser.parse_args(argv)

    if args.all:
        var_keys = list(get_settings().app_config.variables)
    elif args.var_keys:
        var_keys = args.var_keys
    else:
        parser.error("give at least one var_key, or --all")

    total = 0
    for var_key in var_keys:
        try:
            total += repair(
                var_key,
                apply=args.apply,
                drop_legacy_description=args.drop_legacy_description,
            )
        except Exception as e:  # noqa: BLE001 - one bad store must not stop the survey
            print(f"[{var_key}] could not be repaired: {e}")

    if not args.apply:
        print(f"\nDry run — {total} store(s) would change. Re-run with --apply.")
    else:
        print(f"\nRepaired {total} store(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
