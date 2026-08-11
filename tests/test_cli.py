"""Tests for CLI commands — argument validation and error paths."""

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pandas as pd
from typer.testing import CliRunner

from h2mare.cli import _configure, _use_utf8_console
from h2mare.cli.audit import app as audit_app
from h2mare.cli.catalog import _print_catalog
from h2mare.cli.catalog import app as catalog_app
from h2mare.cli.compile import app as compile_app
from h2mare.cli.main import app as main_app
from h2mare.cli.nc2zarr import app as nc2zarr_app
from h2mare.models import AppConfig

_runner = CliRunner()

_MINIMAL_APP_CONFIG = msgspec.convert(
    {
        "variables": {
            "sst": {
                "local_folder": "sst",
                "source_vars": ["analysed_sst"],
                "dataset_id_rep": "cmems-sst",
                "source": "cmems",
                "archive_raw": False,
                "pattern": r".*\.nc",
            }
        },
        "secrets": {},
    },
    AppConfig,
)


def _mock_settings(tmp_path: Path) -> MagicMock:
    m = MagicMock()
    m.LOGS_DIR = tmp_path
    m.STORE_ROOT = tmp_path / "store"
    m.PARQUET_DIR = tmp_path / "parquet"
    m.app_config = _MINIMAL_APP_CONFIG
    return m


# ---------------------------------------------------------------------------
# cli/main.py — run command
# ---------------------------------------------------------------------------


class TestRunCLI:
    def test_only_start_date_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(main_app, ["--start-date", "2021-01-01"])
        assert result.exit_code == 1

    def test_only_end_date_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(main_app, ["--end-date", "2021-12-31"])
        assert result.exit_code == 1

    def test_start_not_before_end_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(
                main_app,
                ["--start-date", "2021-12-31", "--end-date", "2021-01-01"],
            )
        assert result.exit_code == 1

    def test_single_day_range_is_accepted(self, tmp_path):
        # Regression: start == end used to be rejected, making a one-day
        # download impossible even though DateRange allows it.
        with (
            patch(
                "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.main.PipelineManager") as mock_pm,
        ):
            mock_pm.return_value.run.return_value = True
            result = _runner.invoke(
                main_app,
                ["-v", "sst", "--start-date", "2021-06-01", "--end-date", "2021-06-01"],
            )
        assert result.exit_code == 0

    def test_unknown_var_key_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(main_app, ["-v", "nonexistent"])
        assert result.exit_code == 1

    def test_missing_store_root_exits_with_code_1(self, tmp_path):
        ms = _mock_settings(tmp_path)
        ms.STORE_ROOT = None
        with patch("h2mare.cli.main.get_settings", return_value=ms):
            result = _runner.invoke(main_app, ["-v", "sst"])
        assert result.exit_code == 1

    def test_successful_run_exits_with_code_0(self, tmp_path):
        with (
            patch(
                "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.main.PipelineManager") as mock_pm,
        ):
            mock_pm.return_value.run.return_value = True
            result = _runner.invoke(main_app, ["-v", "sst"])
        assert result.exit_code == 0

    def test_failed_pipeline_exits_with_code_1(self, tmp_path):
        with (
            patch(
                "h2mare.cli.main.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.main.PipelineManager") as mock_pm,
        ):
            mock_pm.return_value.run.return_value = False
            result = _runner.invoke(main_app, ["-v", "sst"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# cli/compile.py — compile command
# ---------------------------------------------------------------------------


class TestCompileCLI:
    def test_only_start_date_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(compile_app, ["--start-date", "2021-01-01"])
        assert result.exit_code == 1

    def test_only_end_date_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(compile_app, ["--end-date", "2021-12-31"])
        assert result.exit_code == 1

    def test_start_not_before_end_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(
                compile_app,
                ["--start-date", "2021-12-31", "--end-date", "2021-01-01"],
            )
        assert result.exit_code == 1

    def test_unknown_var_key_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(compile_app, ["-v", "nonexistent"])
        assert result.exit_code == 1

    def test_single_day_range_is_accepted(self, tmp_path):
        # Regression: start == end used to be rejected (see TestRunCLI).
        with (
            patch(
                "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.processing.compiler.Compiler") as mock_compiler,
        ):
            result = _runner.invoke(
                compile_app,
                ["-v", "sst", "--start-date", "2021-06-01", "--end-date", "2021-06-01"],
            )
        assert result.exit_code == 0
        mock_compiler.return_value.run.assert_called_once()

    def test_valid_call_invokes_compiler(self, tmp_path):
        with (
            patch(
                "h2mare.cli.compile.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.processing.compiler.Compiler") as mock_compiler,
        ):
            result = _runner.invoke(
                compile_app,
                ["-v", "sst", "--start-date", "2021-01-01", "--end-date", "2021-12-31"],
            )
        assert result.exit_code == 0
        mock_compiler.return_value.run.assert_called_once()


# ---------------------------------------------------------------------------
# cli/nc2zarr.py — convert command
# ---------------------------------------------------------------------------


class TestConvertCLI:
    def test_unknown_var_key_logs_error_and_exits_0(self, tmp_path):
        with patch(
            "h2mare.cli.nc2zarr.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(nc2zarr_app, ["-v", "nonexistent"])
        assert result.exit_code == 0

    def test_valid_var_key_invokes_netcdf2zarr(self, tmp_path):
        with (
            patch(
                "h2mare.cli.nc2zarr.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.nc2zarr.Netcdf2Zarr") as mock_n2z,
        ):
            result = _runner.invoke(nc2zarr_app, ["-v", "sst"])
        assert result.exit_code == 0
        mock_n2z.return_value.run.assert_called_once()


# ---------------------------------------------------------------------------
# cli/catalog.py — catalog command
# ---------------------------------------------------------------------------


class TestCatalogCLI:
    def test_no_var_key_and_no_all_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(catalog_app, [])
        assert result.exit_code == 1

    def test_unknown_var_key_prints_error_and_continues(self, tmp_path):
        with patch(
            "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(catalog_app, ["nonexistent"])
        assert (
            "Unknown" in result.output
            or "unknown" in result.output
            or result.exit_code == 0
        )

    def test_valid_var_key_calls_print_catalog(self, tmp_path):
        with (
            patch(
                "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.catalog._print_catalog") as mock_print,
        ):
            result = _runner.invoke(catalog_app, ["sst"])
        assert result.exit_code == 0
        mock_print.assert_called_once_with("sst", False)

    def test_all_flag_calls_print_catalog_for_each_var(self, tmp_path):
        with (
            patch(
                "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.cli.catalog._print_catalog") as mock_print,
        ):
            result = _runner.invoke(catalog_app, ["--all"])
        assert result.exit_code == 0
        mock_print.assert_called_once_with("sst", False)


# ---------------------------------------------------------------------------
# cli/__init__.py — console encoding
# ---------------------------------------------------------------------------


class TestConsoleEncoding:
    @staticmethod
    def _cp1252_stream() -> io.TextIOWrapper:
        """Stand-in for the cp1252 stdout Windows hands the CLI."""
        return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")

    def test_catalog_summary_survives_cp1252_console(self, monkeypatch):
        cat = MagicMock()
        cat.df = pd.DataFrame()
        cat.summary.return_value = {
            "num_files": 29,
            "time_coverage": SimpleNamespace(
                start=pd.Timestamp("1998-01-01"), end=pd.Timestamp("2026-08-06")
            ),
            "variables": {"sst"},
            "total_timesteps": 10444,
            "store_root": "D:/GlobalData/CMEMS_SST",
            "catalog_path": "sst_zarr_catalog.parquet",
            "last_scanned": pd.Timestamp("2026-08-10"),
        }
        monkeypatch.setattr(
            "h2mare.storage.zarr_catalog.ZarrCatalog", lambda *a, **k: cat
        )
        monkeypatch.setattr(sys, "stdout", self._cp1252_stream())
        monkeypatch.setattr(sys, "stderr", self._cp1252_stream())

        _use_utf8_console()
        _print_catalog("sst", show_rows=False)
        sys.stdout.flush()

        printed = sys.stdout.buffer.getvalue().decode("utf-8")
        assert "1998-01-01 → 2026-08-06" in printed

    def test_utf8_console_leaves_utf8_streams_alone(self, monkeypatch):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", stream)
        _use_utf8_console()
        assert sys.stdout.errors == "strict"

    def test_every_command_configures_the_console(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(
            "h2mare.cli._use_utf8_console", lambda: calls.append("console")
        )
        monkeypatch.setattr("h2mare.cli.configure_logging", lambda *a, **k: None)
        _configure()
        assert calls == ["console"]


# ---------------------------------------------------------------------------
# cli/audit.py — audit command
# ---------------------------------------------------------------------------


class TestAuditCLI:
    def test_no_target_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(audit_app, [])
        assert result.exit_code == 1

    def test_unknown_var_key_exits_with_code_1(self, tmp_path):
        with patch(
            "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
        ):
            result = _runner.invoke(audit_app, ["nonexistent"])
        assert result.exit_code == 1
        assert "Unknown" in result.output

    def test_clean_store_exits_zero(self, tmp_path):
        with (
            patch(
                "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.storage.audit.audit_var_key") as mock_audit,
        ):
            mock_audit.return_value = SimpleNamespace(
                var_key="sst",
                n_files=3,
                gaps=[],
                slices=[],
                errors=[],
                ok=True,
                n_known_gaps=0,
                known_gaps=pd.DatetimeIndex([]),
                store_exists=True,
            )
            result = _runner.invoke(audit_app, ["sst"])
        assert result.exit_code == 0
        assert "No gaps found" in result.output

    def test_findings_exit_non_zero(self, tmp_path):
        """The command gates a scheduled run, so a gap must fail the process."""
        gap = SimpleNamespace(
            path=Path("cmems_sst_2026.zarr"),
            span=(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-08-06")),
            missing=pd.DatetimeIndex([pd.Timestamp("2026-07-31")]),
        )
        with (
            patch(
                "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.storage.audit.audit_var_key") as mock_audit,
        ):
            mock_audit.return_value = SimpleNamespace(
                var_key="sst",
                n_files=1,
                gaps=[gap],
                slices=[],
                errors=[],
                ok=False,
                n_known_gaps=0,
                known_gaps=pd.DatetimeIndex([]),
                store_exists=True,
            )
            result = _runner.invoke(audit_app, ["sst"])
        assert result.exit_code == 1
        assert "2026-07-31" in result.output


class TestAuditKnownGapsDisplay:
    """A suppression list nothing can print is one that grows unnoticed."""

    def _result(self, *, ok: bool, known: list[str]):
        gap = SimpleNamespace(
            path=Path("aviso_fsle_2025.zarr"),
            span=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
            missing=pd.DatetimeIndex([pd.Timestamp("2025-09-09")]),
        )
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in known])
        return SimpleNamespace(
            var_key="fsle",
            n_files=29,
            gaps=[] if ok else [gap],
            slices=[],
            errors=[],
            ok=ok,
            known_gaps=idx,
            n_known_gaps=len(idx),
            store_exists=True,
        )

    def _run(self, tmp_path, result, args):
        with (
            patch(
                "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.storage.audit.audit_var_key", return_value=result),
        ):
            return _runner.invoke(audit_app, args)

    def test_failing_var_still_reports_the_suppressed_count(self, tmp_path):
        """Regression: the count was built but never interpolated on [FAIL]."""
        out = self._run(
            tmp_path, self._result(ok=False, known=["2025-06-02"]), ["sst"]
        ).output

        assert "[FAIL]" in out
        assert "1 known source gap(s) excluded" in out

    def test_known_flag_lists_the_dates(self, tmp_path):
        out = self._run(
            tmp_path, self._result(ok=True, known=["2025-06-02"]), ["sst", "--known"]
        ).output

        assert "2025-06-02" in out

    def test_known_flag_shows_a_passing_var_without_show_ok(self, tmp_path):
        """Otherwise the one thing --known exists for is invisible when clean."""
        out = self._run(
            tmp_path, self._result(ok=True, known=["2025-06-02"]), ["sst", "--known"]
        ).output

        assert "fsle" in out

    def test_known_flag_is_quiet_for_a_var_without_gaps(self, tmp_path):
        out = self._run(
            tmp_path, self._result(ok=True, known=[]), ["sst", "--known"]
        ).output

        assert "known source gaps" not in out

    def test_dates_are_not_listed_without_the_flag(self, tmp_path):
        out = self._run(
            tmp_path, self._result(ok=True, known=["2025-06-02"]), ["sst", "--show-ok"]
        ).output

        assert "1 known source gap(s) excluded" in out
        assert "2025-06-02" not in out

    def test_an_interval_is_rendered_as_a_range(self, tmp_path):
        out = self._run(
            tmp_path,
            self._result(ok=True, known=["2025-06-02", "2025-06-03", "2025-06-04"]),
            ["sst", "--known"],
        ).output

        assert "2025-06-02 → 2025-06-04" in out


class TestAuditReporting:
    """Advice and labels have to match the check that produced the findings."""

    def _result(self, *, gaps=(), slices=(), store_exists=True):
        return SimpleNamespace(
            var_key="chl",
            n_files=29,
            gaps=list(gaps),
            slices=list(slices),
            errors=[],
            ok=not (gaps or slices),
            known_gaps=pd.DatetimeIndex([]),
            n_known_gaps=0,
            store_exists=store_exists,
        )

    def _gap(self):
        return SimpleNamespace(
            path=Path("a_2025.zarr"),
            span=(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
            missing=pd.DatetimeIndex([pd.Timestamp("2025-06-02")]),
        )

    def _slice(self, variable="chl", date="1999-01-25"):
        return SimpleNamespace(
            path=Path("chl_1999.zarr"),
            variable=variable,
            date=pd.Timestamp(date),
            kind="empty",
            detail="no finite values",
        )

    def _run(self, tmp_path, result, args):
        with (
            patch(
                "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.storage.audit.audit_var_key", return_value=result),
        ):
            return _runner.invoke(audit_app, args)

    def test_value_findings_do_not_advise_re_downloading(self, tmp_path):
        """Re-running cannot fill a day the provider never published."""
        out = self._run(
            tmp_path, self._result(slices=[self._slice()]), ["sst", "--values"]
        ).output

        assert "re-run the download" not in out.lower()
        assert "known_gaps" in out

    def test_axis_findings_do_advise_re_downloading(self, tmp_path):
        out = self._run(tmp_path, self._result(gaps=[self._gap()]), ["sst"]).output

        assert "re-run the" in out.lower()

    def test_both_kinds_get_their_own_advice(self, tmp_path):
        out = self._run(
            tmp_path,
            self._result(gaps=[self._gap()], slices=[self._slice()]),
            ["sst", "--values"],
        ).output

        assert "re-run the" in out.lower()
        assert "known_gaps" in out

    def test_header_names_the_value_check(self, tmp_path):
        out = self._run(
            tmp_path, self._result(), ["sst", "--values", "--show-ok"]
        ).output

        assert "axis + value check" in out

    def test_header_says_axis_only_by_default(self, tmp_path):
        out = self._run(tmp_path, self._result(), ["sst", "--show-ok"]).output

        assert "— axis check" in out

    def test_a_day_is_reported_once_across_variables(self, tmp_path):
        """chl reported 22 findings for 11 days, one per column."""
        out = self._run(
            tmp_path,
            self._result(slices=[self._slice("chl"), self._slice("chl_fdist")]),
            ["sst", "--values"],
        ).output

        assert out.count("1999-01-25") == 1
        assert "chl, chl_fdist" in out

    def test_a_missing_store_is_skipped_not_passed(self, tmp_path):
        out = self._run(
            tmp_path, self._result(store_exists=False), ["sst", "--show-ok"]
        ).output

        assert "[SKIP]" in out
        assert "[OK]" not in out

    def test_since_without_values_is_flagged(self, tmp_path):
        result = self._run(tmp_path, self._result(), ["sst", "--since", "2025-01-01"])

        assert "only bounds --values" in result.output

    def test_unparseable_since_exits_1(self, tmp_path):
        result = self._run(
            tmp_path, self._result(), ["sst", "--values", "--since", "nonsense"]
        )

        assert result.exit_code == 1


class TestCatalogMissingStore:
    """The warning belongs in the output, attributable to the variable.

    resolve_store_path warns "will be created when data is added", which is
    false for a read-only inspector and doubly so for `moon` — computed at
    compile time, it never gets a store at all.
    """

    def test_catalog_does_not_warn_about_a_missing_store(self, tmp_path):
        from h2mare.storage.zarr_catalog import ZarrCatalog

        with (
            patch(
                "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch.object(ZarrCatalog, "__init__", return_value=None) as mock_init,
            patch.object(
                ZarrCatalog,
                "df",
                new_callable=lambda: property(lambda s: pd.DataFrame()),
            ),
            patch.object(ZarrCatalog, "summary", return_value={"num_files": 0}),
        ):
            _runner.invoke(catalog_app, ["sst"])

        assert mock_init.call_args.kwargs.get("warn_if_missing") is False

    def test_missing_store_is_annotated_in_the_output(self, tmp_path):
        from h2mare.storage.zarr_catalog import ZarrCatalog

        gone = tmp_path / "definitely_not_here"
        with (
            patch(
                "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch.object(ZarrCatalog, "__init__", return_value=None),
            patch.object(
                ZarrCatalog,
                "df",
                new_callable=lambda: property(lambda s: pd.DataFrame()),
            ),
            patch.object(
                ZarrCatalog,
                "summary",
                return_value={"num_files": 0, "store_root": str(gone)},
            ),
        ):
            result = _runner.invoke(catalog_app, ["sst"])

        assert "(does not exist)" in result.output

    def test_an_existing_store_is_not_annotated(self, tmp_path):
        from h2mare.storage.zarr_catalog import ZarrCatalog

        with (
            patch(
                "h2mare.cli.catalog.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch.object(ZarrCatalog, "__init__", return_value=None),
            patch.object(
                ZarrCatalog,
                "df",
                new_callable=lambda: property(lambda s: pd.DataFrame()),
            ),
            patch.object(
                ZarrCatalog,
                "summary",
                return_value={"num_files": 0, "store_root": str(tmp_path)},
            ),
        ):
            result = _runner.invoke(catalog_app, ["sst"])

        assert "(does not exist)" not in result.output


class TestAuditFindingCount:
    """The count has to match the lines printed, or it reads as a discrepancy."""

    def _run(self, tmp_path, slices, args):
        result = SimpleNamespace(
            var_key="chl",
            n_files=29,
            gaps=[],
            slices=list(slices),
            errors=[],
            ok=False,
            known_gaps=pd.DatetimeIndex([]),
            n_known_gaps=0,
            store_exists=True,
        )
        with (
            patch(
                "h2mare.cli.audit.get_settings", return_value=_mock_settings(tmp_path)
            ),
            patch("h2mare.storage.audit.audit_var_key", return_value=result),
        ):
            return _runner.invoke(audit_app, args)

    def _slice(self, variable, date="1999-01-25"):
        return SimpleNamespace(
            path=Path("chl_1999.zarr"),
            variable=variable,
            date=pd.Timestamp(date),
            kind="empty",
            detail="no finite values",
        )

    def test_one_day_across_two_columns_counts_once(self, tmp_path):
        out = self._run(
            tmp_path,
            [self._slice("chl"), self._slice("chl_fdist")],
            ["sst", "--values"],
        ).output

        assert "1 finding(s)" in out

    def test_distinct_days_count_separately(self, tmp_path):
        out = self._run(
            tmp_path,
            [
                self._slice("chl", "1999-01-25"),
                self._slice("chl_fdist", "1999-01-25"),
                self._slice("chl", "1999-11-17"),
            ],
            ["sst", "--values"],
        ).output

        assert "2 finding(s)" in out

    def test_the_count_matches_the_lines_printed(self, tmp_path):
        out = self._run(
            tmp_path,
            [self._slice("chl"), self._slice("chl_fdist")],
            ["sst", "--values"],
        ).output

        printed = sum(1 for line in out.splitlines() if "[empty]" in line)
        assert f"{printed} finding(s)" in out
