"""
Classes representing Data models for spatial and variable configurations.
"""

from enum import Enum
from typing import Optional

import msgspec


class TimeStep(str, Enum):
    """
    Native cadence of a variable's stored Zarr — how far apart its time steps are.

    Distinct from :class:`h2mare.types.FilePeriod`, which is about the storage
    layout rather than the data: a store can be hourly and still be written one
    file per year.
    """

    DAILY = "daily"
    HOURLY = "hourly"


class StoreDtype(str, Enum):
    """
    On-disk encoding for a variable's Zarr.

    ``FLOAT32`` (the default) writes what the pipeline has always written and is
    byte-identical to it. ``INT16`` stores scale/offset-packed integers instead,
    matching the ~16-bit packing ERA5's GRIB already uses — so it discards
    quantisation noise rather than signal, at roughly two thirds the size.

    Opt-in per variable: a store keeps whatever encoding it was created with,
    and appends inherit it.
    """

    FLOAT32 = "float32"
    INT16 = "int16"


def step_freq(var_config) -> str:
    """
    Pandas frequency alias matching a variable's cadence — ``"h"`` or ``"D"``.

    Used by the gap checks so they compare a store against a calendar at its own
    resolution: a daily grid cannot see a missing hour, and an hourly grid built
    for a daily store would report 23 phantom gaps per day.

    Takes any object with a ``time_step`` attribute and defaults to daily, so
    stand-in configs and entries predating the field keep the old behaviour.
    """
    return "h" if getattr(var_config, "time_step", None) is TimeStep.HOURLY else "D"


class KeyVarConfigEntry(msgspec.Struct):
    """Configuration for a single Key variable/dataset."""

    # Subdirectory under STORE_ROOT for this variable's Zarr files.
    local_folder: str
    # Variable names to extract from source files.
    source_vars: str | list[str]
    # Reprocessed (multiyear) dataset identifier.
    dataset_id_rep: str
    # Provider: "cmems", "aviso", or "cds".
    source: str
    # Whether this variable's raw NetCDF/GRIB files are archived into the store
    # (and kept) after conversion, or deleted per-period. Required and explicit:
    # True keeps raw files, False deletes them.
    archive_raw: bool
    # Near-real-time dataset identifier. Omit for reanalysis-only products.
    dataset_id_nrt: Optional[str] = None
    # CMEMS only. Chooses the copernicusmarine download API: True (default)
    # downloads via subset() (spatial/variable subset honoring bbox/source_vars);
    # False downloads full original files via get(). Ignored for non-CMEMS
    # sources (aviso, cds, …), which do not consult this field.
    subset: Optional[bool] = True
    # Set True for CDS/ERA5 accumulated or averaged variables whose GRIB files
    # have a 2-D time×step coordinate grid instead of a flat time axis
    # (e.g. atm-accum-avg, radiation). Triggers a preprocess step that merges
    # the two dimensions and trims overlapping timestamps at month edges.
    merge_time_step: bool = False
    # Regex matched (via re.search) against each raw filename to extract the
    # date component(s); the capture groups must agree with filename_date_range:
    #   - filename_date_range=True  -> exactly 2 groups, each a full parseable
    #     date (start, end); the file expands to that daily range.
    #   - filename_date_range=False -> the groups are joined with "-" and parsed
    #     as one date (e.g. (\d{8}) -> "20210115";
    #     (\d{4})(\d{2})(\d{2}) -> "2021-01-15").
    # None for derived/system variables (bathy, moon, h2ds) that are never
    # matched against downloaded filenames.
    pattern: Optional[str] = None
    # Set True when the filename pattern has two capture groups encoding a
    # (start, end) date range (e.g. CMEMS/CDS: "2021-01-01-2021-01-31.nc").
    # Set False (default) when the pattern yields a single date
    # (e.g. AVISO FSLE: "_20210115_").
    filename_date_range: bool = False
    # Regex matched (via re.search) against each raw filename; only matching
    # files are converted. Use when a download directory holds files the
    # pipeline must not read — AVISO ships META3.2 eddy trajectories as
    # long/short/untracked variants alongside each other, and only the long
    # ones belong in the store (the untracked files have no `track` variable at
    # all). None (default) converts every file the date pattern matches.
    raw_include: Optional[str] = None
    # Bounding box [xmin, ymin, xmax, ymax] for spatial subsetting.
    bbox: Optional[tuple[float, float, float, float]] = None
    # Depth range [min_depth, max_depth] for 3-D variables (e.g. o2, thetao).
    depth_range: Optional[tuple[float, float]] = None
    # Filename of the static source file at the configured output resolution.
    # Used by compile-only variables such as bathy.
    data_file: Optional[str] = None
    # High-resolution static source file. Used by bathy when extracting at
    # full native resolution (e.g. from SHP geometries).
    data_file_hires: Optional[str] = None
    # Set True for trajectory-format datasets (e.g. eddies) that require
    # spatial binning before they can be stored as a gridded Zarr.
    # The standard open_mfdataset pipeline is bypassed entirely.
    trajectory_format: bool = False
    # Set True for variables whose Zarr store uses lon/lat coordinate names that
    # must be renamed to x/y before rioxarray clip (e.g. AVISO fsle, eddies).
    rename_lonlat: bool = False
    # Depth levels (metres) to extract when slicing a 3-D variable during
    # Extractor runs. Each level becomes a separate output column
    # (e.g. [0, 100, 500] → o2_0, o2_100, o2_500). None = no depth slicing.
    extract_depth_slices: Optional[list[int]] = None
    # Depth levels (metres) to select when compiling a 3-D variable into h2ds.
    # Each level becomes a separate output variable (e.g. [0, 100, 500, 1000]
    # → o2_0, o2_100, o2_500, o2_1000). None = no depth slicing in compiler.
    compile_depth_slices: Optional[list[int]] = None
    # Exact variable names as they appear in the compiled h2ds Zarr for this
    # var_key. Used to select only these columns when adding a variable to an
    # existing Parquet store (--add-var). None means not yet declared.
    compiled_vars: Optional[list[str]] = None
    # Cadence of this variable's stored Zarr. DAILY (the default, and what every
    # existing store is) means one step per calendar day. HOURLY keeps the
    # source's sub-daily axis instead of aggregating it away at convert time —
    # for ERA5 that preserves the native resolution the raw GRIB already has.
    #
    # Read paths normalize a DAILY store's stamps to midnight (sources often
    # publish at 12:00); doing that to an HOURLY store would collapse 24 steps
    # onto one timestamp, so the normalization is skipped for it.
    time_step: TimeStep = TimeStep.DAILY
    # On-disk encoding. Default writes exactly what it always has; INT16 packs
    # to scale/offset integers (see StoreDtype). Only consulted when a store is
    # first created — an existing store keeps its own encoding through appends.
    store_dtype: StoreDtype = StoreDtype.FLOAT32
    # Days the provider never published, which therefore cannot be downloaded,
    # converted or backfilled. Each entry is either "YYYY-MM-DD" or a closed
    # interval "YYYY-MM-DD/YYYY-MM-DD".
    #
    # Needed because a source that ships one file per day produces an *axis*
    # hole when it skips one, which is otherwise indistinguishable from data
    # the pipeline lost — AVISO simply has no fsle file for 2025-06-02, and its
    # remote listing jumps 20250601 → 20250603. Without somewhere to record
    # that, the gap checks would report it on every run forever, and a check
    # that cries wolf is one people stop reading.
    #
    # Only for gaps confirmed absent at the source. Anything else is a defect
    # and belongs fixed, not listed.
    known_gaps: Optional[list[str]] = None

    def __post_init__(self):
        if self.bbox is not None:
            lon_min, lat_min, lon_max, lat_max = self.bbox
            if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
                raise ValueError("Longitude must be between -180 and 180")
            if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
                raise ValueError("Latitude must be between -90 and 90")
            if lon_min >= lon_max:
                raise ValueError("lon_min must be less than lon_max")
            if lat_min >= lat_max:
                raise ValueError("lat_min must be less than lat_max")

        if self.depth_range is not None:
            if self.depth_range[0] >= self.depth_range[1]:
                raise ValueError("depth_min must be less than depth_max")

        # Range-mode parsing unpacks exactly two capture groups (start, end);
        # fail fast at config load rather than deep in the convert step with a
        # cryptic "not enough values to unpack".
        if self.filename_date_range:
            if self.pattern is None:
                raise ValueError(
                    "filename_date_range=True requires a `pattern` with 2 capture "
                    "groups (start, end)"
                )
            import re

            ngroups = re.compile(self.pattern).groups
            if ngroups != 2:
                raise ValueError(
                    "filename_date_range=True requires `pattern` to have exactly 2 "
                    f"capture groups (start, end); got {ngroups} in {self.pattern!r}"
                )


class SecretsConfig(msgspec.Struct):
    """External service credentials."""

    aviso_ftp_server: Optional[str] = None
    aviso_username: Optional[str] = None
    aviso_password: Optional[str] = None


# VariablesConfig is now a plain dict — kept as a type alias for compatibility
VariablesConfig = dict[str, KeyVarConfigEntry]

# Variables that are derived/computed inside the pipeline and never downloaded
# from an external source. Excluded from download loops and catalog date-range
# inference; each has its own dedicated processing path in the Compiler.
SYSTEM_VAR_KEYS: frozenset[str] = frozenset({"h2ds", "bathy", "moon"})


class AppConfig(msgspec.Struct):
    """Complete application configuration."""

    variables: VariablesConfig
    secrets: SecretsConfig
