"""
h2mare audit — check the stores for days that silently went missing.

Coverage everywhere else in the pipeline is a frontier: a start and an end. A
year holding January 1st and December 31st reports itself complete no matter
how much is missing in between, which is how AVISO_FSLE 1999 shipped with 128
of 365 days and passed every check.

This reads the time axes and reports days that are absent from the middle of a
store's own span. It reads coordinates only — the full production store takes
about a minute — so there is nothing to cache and no incremental mode to get
wrong. Run it after a pipeline run, and periodically: no in-process check
survives a hard kill, and a hard kill is the likeliest origin of a ragged store.

Exits non-zero when anything is found, so it can gate a scheduled run.

Examples
--------
    # Every configured variable
    uv run h2mare audit --all

    # One variable
    uv run h2mare audit fsle

    # Also look for present-but-unusable slices (reads data — slower)
    uv run h2mare audit fsle --values

    # Parity check on the Parquet store, from footer stats (no data read)
    uv run h2mare audit --parquet

    # Show which days are excluded as known source gaps
    uv run h2mare audit --all --known
"""

from typing import Optional

import pandas as pd
import typer

from h2mare.config import get_settings
from h2mare.types import DateRange

app = typer.Typer()


def _print_var_audit(audit, show_all: bool, show_known: bool = False) -> None:
    """Render one variable's findings. Returns nothing; prints to stdout."""
    known = (
        f"  ({audit.n_known_gaps} known source gap(s) excluded)"
        if audit.n_known_gaps
        else ""
    )

    def _known_lines() -> None:
        """List the suppressed days, so the list can be audited in place."""
        if show_known and audit.n_known_gaps:
            typer.echo("           known source gaps (never published upstream):")
            for block in _blocks(audit.known_gaps):
                typer.echo(f"             {block}")

    if not audit.store_exists:
        # Not a pass: nothing was checked. `moon` is computed at compile time
        # and has no store, but a missing store is also what a misconfigured
        # STORE_ROOT or an unmounted drive looks like, and "[OK] 0 file(s)"
        # would report both as healthy.
        typer.echo(f"  [SKIP] {audit.var_key:<16} no store directory on disk")
        return

    if audit.ok:
        # A variable with suppressed days prints under --known even without
        # --show-ok; otherwise the one thing --known exists to show would be
        # invisible on a clean store, which is the normal case.
        if show_all or (show_known and audit.n_known_gaps):
            typer.echo(f"  [OK]   {audit.var_key:<16} {audit.n_files} file(s){known}")
            _known_lines()
        return

    typer.echo(f"\n  [FAIL] {audit.var_key:<16} {audit.n_files} file(s){known}")
    _known_lines()

    for gap in audit.gaps:
        span = f"{gap.span[0].date()} → {gap.span[1].date()}"
        typer.echo(f"    {gap.path.name}  (span {span})")
        typer.echo(f"      {len(gap.missing)} day(s) absent from the axis:")
        for block in _blocks(gap.missing):
            typer.echo(f"        {block}")

    # One line per day rather than per (variable, day). All of a var_key's
    # columns share dates by config convention, so listing each separately just
    # doubles the output — chl reported 22 findings for 11 days.
    by_day: dict[tuple, list[str]] = {}
    for issue in audit.slices:
        key = (issue.path.name, issue.date, issue.kind, issue.detail)
        by_day.setdefault(key, []).append(issue.variable)

    for (fname, date, kind, detail), variables in sorted(
        by_day.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        cols = ", ".join(sorted(variables))
        typer.echo(f"    {fname}  {date.date()}  [{kind}] {detail}  ({cols})")

    for error in audit.errors:
        typer.echo(f"    ! {error}")


def _blocks(dates, max_blocks: int = 12) -> list[str]:
    """Compress a date index into contiguous 'a → b' lines for display."""
    from h2mare.storage.audit import contiguous_blocks

    runs = contiguous_blocks(dates)
    shown = [
        str(a.date()) if a == b else f"{a.date()} → {b.date()} ({(b - a).days + 1}d)"
        for a, b in runs[:max_blocks]
    ]
    if len(runs) > max_blocks:
        shown.append(f"… and {len(runs) - max_blocks} more block(s)")
    return shown


def audit(
    var_key: Optional[str] = typer.Argument(
        None,
        help="Variable key to audit (e.g. sst, fsle). Omit with --all.",
    ),
    all_vars: bool = typer.Option(
        False, "--all", "-a", is_flag=True, help="Audit every configured variable."
    ),
    check_values: bool = typer.Option(
        False,
        "--values",
        is_flag=True,
        help="Also report present-but-unusable slices (empty or single-valued). "
        "Reads data, so this is much slower than the axis check.",
    ),
    check_parquet: bool = typer.Option(
        False,
        "--parquet",
        is_flag=True,
        help="Check the Parquet store for wholly-null columns, from footer "
        "statistics. Reads no data.",
    ),
    show_ok: bool = typer.Option(
        False, "--show-ok", is_flag=True, help="List variables that passed too."
    ),
    show_known: bool = typer.Option(
        False,
        "--known",
        is_flag=True,
        help="List the days excluded via each variable's known_gaps config "
        "entry, rather than only counting them.",
    ),
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Bound --values to dates on or after this (YYYY-MM-DD). The value "
        "scan is disk-bound over the whole store — chl alone is 97 GB — so "
        "auditing more than one variable without this is an hours-long job.",
    ),
) -> None:
    """Report days missing from the middle of a store's own time span."""
    from h2mare.storage.audit import audit_parquet_nulls, audit_var_key

    settings = get_settings()

    if not var_key and not all_vars and not check_parquet:
        typer.echo("Provide a variable key, or use --all / --parquet.", err=True)
        raise typer.Exit(code=1)

    findings = 0

    keys: list[str] = []
    if all_vars:
        keys = list(settings.app_config.variables.keys())
    elif var_key:
        if var_key not in settings.app_config.variables:
            typer.echo(
                f"Unknown variable key '{var_key}'. "
                f"Available: {', '.join(settings.app_config.variables)}.",
                err=True,
            )
            raise typer.Exit(code=1)
        keys = [var_key]

    value_window = None
    if since:
        try:
            value_window = DateRange(
                start=pd.Timestamp(since), end=pd.Timestamp.now().normalize()
            )
        except Exception:
            typer.echo(f"Could not parse --since {since!r} as a date.", err=True)
            raise typer.Exit(code=1) from None
        if not check_values:
            typer.echo("--since only bounds --values; ignoring it.", err=True)

    n_gaps = n_slices = 0

    if keys:
        checks = "axis + value check" if check_values else "axis check"
        bound = f", values since {value_window.start.date()}" if value_window else ""
        typer.echo(f"\nAuditing {len(keys)} variable(s) — {checks}{bound}")
        for key in keys:
            try:
                result = audit_var_key(
                    key,
                    check_values=check_values,
                    value_window=value_window,
                    progress=check_values,
                )
            except Exception as e:
                typer.echo(f"  [SKIP] {key:<16} {e}")
                continue
            _print_var_audit(result, show_ok, show_known)
            n_gaps += len(result.gaps)
            # Count the days shown, not the (variable, day) pairs behind them.
            # The display collapses a var_key's columns onto one line, so
            # counting issues reported 22 findings above 11 printed lines.
            n_slices += len({(s.path, s.date) for s in result.slices})
            findings += (
                len(result.gaps)
                + len({(s.path, s.date) for s in result.slices})
                + len(result.errors)
            )

    if check_parquet:
        typer.echo("\nAuditing Parquet store — wholly-null columns")
        nulls = audit_parquet_nulls(settings.PARQUET_DIR)
        if nulls:
            for path, column in nulls:
                typer.echo(f"  [FAIL] {column} is entirely null in {path.name}")
            findings += len(nulls)
        else:
            typer.echo("  [OK]   no wholly-null columns")

    if findings:
        # Advice per finding type. The two need opposite responses, and a
        # single hardcoded line told anyone looking at a value finding to
        # "re-run the download" — which cannot fix a day the provider never
        # published, and reads as confident while being exactly wrong.
        typer.echo(f"\n{findings} finding(s).")
        if n_gaps:
            typer.echo(
                "  Days absent from the axis are pipeline defects: re-run the "
                "download and convert for those dates."
            )
        if n_slices:
            typer.echo(
                "  Days present but empty are usually source gaps, and "
                "re-running will not fill them. Confirm against the provider; "
                "if the data was never published, record it in the variable's "
                "known_gaps so this stops being reported."
            )
        raise typer.Exit(code=1)

    typer.echo("\nNo gaps found.")


app.command()(audit)
