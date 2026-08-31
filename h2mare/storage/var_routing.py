"""
Which store actually holds a given variable name.

A var_key does not always hold everything it publishes. An **hourly** store
keeps the raw source only: its daily reduction and the features derived from it
are computed at compile time and written to the compiled store (h2ds) instead —
``atm-instante`` holds ``u10``/``v10`` and publishes ``wind_mean``. A **3-D**
variable is stored once on a ``depth`` axis and published as one flat column per
level — ``thetao`` holds ``thetao`` and publishes ``thetao_100``. Either way the
name a caller holds — an h2ds column, which is what extraction returned — can be
absent from its own var_key's store.

:func:`catalog_for_var` answers "which catalog holds this name", so a caller no
longer has to keep ``var`` and ``var_key`` consistent by hand. It is the
read-side counterpart of the routing
:func:`~h2mare.processing.extractor.split_vars_by_source` does for extraction,
and follows the same map (``compiled_vars`` in config *is* the var_key → h2ds
column mapping).

It routes on where the name is, not on ``time_step``, which is what lets one
rule cover the hourly and the depth case. The consequence, and the one place it
is deliberately laxer than ``split_vars_by_source``: a daily store genuinely
*missing* something it publishes routes to h2ds rather than raising, because
from here that is indistinguishable from the depth case.
"""

from __future__ import annotations

from typing import Optional

from h2mare.config import AppConfig, get_settings
from h2mare.storage.zarr_catalog import ZarrCatalog
from h2mare.types import DateRange, ReadFrom
from h2mare.validators import validate_var_key

#: ``source`` marking the var_key that holds the compiled dataset, rather than
#: its name — the same marker ``Extractor._normalize_var_dict`` excludes from
#: default runs, so a compiled store named something other than ``h2ds`` is
#: still found.
COMPILED_SOURCE = "h2mare"


def compiled_var_key(app_config: AppConfig) -> str:
    """
    The var_key holding the compiled dataset, found by its ``source``.

    Raises:
        ValueError: if no var_key declares :data:`COMPILED_SOURCE`, or if more
            than one does.
    """
    keys = [
        k
        for k, cfg in app_config.variables.items()
        if getattr(cfg, "source", None) == COMPILED_SOURCE
    ]
    if not keys:
        raise ValueError(
            f"No compiled var_key in config: none has source "
            f"'{COMPILED_SOURCE}'. Reading a variable from the compiled "
            f"store needs one (conventionally 'h2ds')."
        )
    if len(keys) > 1:
        raise ValueError(
            f"Ambiguous compiled var_key: {sorted(keys)} all declare source "
            f"'{COMPILED_SOURCE}'. Exactly one is expected."
        )
    return keys[0]


def catalog_for_var(
    var: str,
    var_key: str,
    *,
    read_from: ReadFrom = "auto",
    app_config: Optional[AppConfig] = None,
) -> ZarrCatalog:
    """
    Open the catalog that holds *var* for *var_key*.

    With ``read_from="auto"`` the native store answers when it holds the name,
    and the compiled store answers when it does not but ``compiled_vars`` says
    the var_key publishes it. ``"native"`` and ``"compiled"`` pin the choice and
    raise if that store cannot serve the name — ``"native"`` is how you ask for
    the raw hourly field behind a daily h2ds column.

    The name is the same on both sides, so only the catalog is returned.

    Args:
        var: Variable name, as it appears in the frame being diagnosed.
        var_key: Storage key the variable belongs to.
        read_from: Which store to read from. Defaults to ``"auto"``.
        app_config: Application configuration. Defaults to ``get_settings()``.

    Returns:
        The :class:`~h2mare.storage.zarr_catalog.ZarrCatalog` to open *var* from.

    Raises:
        ValueError: if *var_key* is not in config, if *var* belongs to neither
            store, or if a pinned ``read_from`` names a store without it.
    """
    app_config = app_config or get_settings().app_config
    var_key = validate_var_key(var_key, app_config)
    var_config = app_config.variables[var_key]
    declared = list(getattr(var_config, "compiled_vars", None) or [])

    def _compiled() -> ZarrCatalog:
        return ZarrCatalog(compiled_var_key(app_config), app_config=app_config)

    # Pinned "compiled" short-circuits: the native store is never opened, so its
    # index is not scanned just to be discarded.
    if read_from == "compiled":
        if var not in declared:
            raise ValueError(
                f"[{var_key}] read_from='compiled' but '{var}' is not one of its "
                f"compiled_vars, so the compiled store holds no column for it. "
                f"'{var_key}' publishes {sorted(declared) or 'nothing'}."
            )
        return _compiled()

    native = ZarrCatalog(var_key, app_config=app_config)
    stored = native.get_variables()
    if var in stored:
        return native

    if read_from == "native":
        raise ValueError(
            f"[{var_key}] read_from='native' but its store holds {sorted(stored)}, "
            f"not '{var}'. Derived and depth-sliced columns only exist in the "
            f"compiled store — drop read_from to be routed there."
        )

    if var in declared:
        return _compiled()

    raise ValueError(
        f"[{var_key}] cannot plot '{var}': not a variable of '{var_key}'. "
        f"Store holds {sorted(stored)}"
        + (f"; config publishes {sorted(declared)}." if declared else ".")
    )


def coverage_for_var(
    var: str,
    var_key: str,
    *,
    read_from: ReadFrom = "auto",
    app_config: Optional[AppConfig] = None,
) -> tuple[ZarrCatalog, DateRange]:
    """
    The catalog *var* is read from, and the dates that catalog really holds it for.

    :func:`catalog_for_var` answers *where*; this answers *how far*, and the
    second question needs the first because the two stores do not move together.
    Store-level coverage is what misleads here: h2ds ends where its
    furthest-ahead source ends, so every variable behind that one reads as
    covered across the stretch where it is only NaN padding — 21 days for an
    ERA5 var_key against a store carried to yesterday by altimetry, and months
    for a slower one. Asked of the routed catalog, per variable, the answer
    narrows to where the values are real (see
    :meth:`~h2mare.storage.zarr_catalog.ZarrCatalog.get_var_coverage`).

    Args:
        var: Variable name, as it appears in the frame being diagnosed.
        var_key: Storage key the variable belongs to.
        read_from: Which store to read from. Defaults to ``"auto"``.
        app_config: Application configuration. Defaults to ``get_settings()``.

    Returns:
        The catalog *var* is read from, paired with its coverage there.

    Raises:
        ValueError: anything :func:`catalog_for_var` raises, or if the routed
            store lists no file holding *var*.
    """
    cat = catalog_for_var(var, var_key, read_from=read_from, app_config=app_config)
    coverage = cat.get_var_coverage(var)
    if coverage is None:
        raise ValueError(
            f"[{var_key}] no coverage for '{var}' in the '{cat.var_key}' store: "
            f"the store lists the name but no file in it carries the variable. "
            f"Its catalog may be stale — refresh it, or compile the dates in."
        )
    return cat, coverage
