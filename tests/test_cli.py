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
            )
            result = _runner.invoke(audit_app, ["sst"])
        assert result.exit_code == 1
        assert "2026-07-31" in result.output
