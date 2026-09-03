"""Tests for downloader/cds_downloader.py — CDSDownloader date logic and download_file."""

import json
from unittest.mock import MagicMock, patch

import msgspec
import pandas as pd
import pytest

from h2mare.downloader.cds_downloader import CDSDownloader
from h2mare.models import AppConfig
from h2mare.types import DateRange

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CDS_ENTRY = {
    "local_folder": "atm",
    "source_vars": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    "dataset_id_rep": "reanalysis-era5-single-levels",
    "source": "cds",
    "archive_raw": True,
    "pattern": r".*\.grib",
    "subset": False,
    "bbox": (-20, 25, 15, 55),
}


def _make_config(entry=_CDS_ENTRY) -> AppConfig:
    return msgspec.convert({"variables": {"atm": entry}, "secrets": {}}, AppConfig)


@pytest.fixture
def dl(tmp_path):
    return CDSDownloader(
        "atm",
        app_config=_make_config(),
        store_root=tmp_path / "store",
        download_root=tmp_path / "downloads",
    )


# ---------------------------------------------------------------------------
# _resolve_date_range — explicit inputs
# ---------------------------------------------------------------------------


class TestResolveExplicit:
    def test_explicit_dates_return_date_range(self, dl):
        result = dl._resolve_date_range("2020-01-01", "2020-12-31")
        assert result is not None
        assert pd.Timestamp(result.start) == pd.Timestamp("2020-01-01")
        assert pd.Timestamp(result.end) == pd.Timestamp("2020-12-31")

    def test_start_after_end_returns_none(self, dl):
        result = dl._resolve_date_range("2025-01-01", "2020-01-01")
        assert result is None

    def test_same_start_and_end_returns_date_range(self, dl):
        result = dl._resolve_date_range("2021-06-15", "2021-06-15")
        assert result is not None
        assert pd.Timestamp(result.start) == pd.Timestamp(result.end)


# ---------------------------------------------------------------------------
# _resolve_date_range — inferred from store
# ---------------------------------------------------------------------------


class TestResolveInferred:
    def test_no_store_coverage_raises_value_error(self, dl):
        with patch(
            "h2mare.downloader.cds_downloader.get_store_coverage", return_value=None
        ):
            with pytest.raises(ValueError, match="No existing data"):
                dl._resolve_date_range(None, None)

    def test_start_inferred_from_store_end(self, dl):
        coverage = DateRange("2020-01-01", "2021-12-31")
        with patch(
            "h2mare.downloader.cds_downloader.get_store_coverage", return_value=coverage
        ):
            result = dl._resolve_date_range(None, "2022-06-30")
        assert result is not None
        assert pd.Timestamp(result.start) == pd.Timestamp("2022-01-01")

    def test_end_inferred_returns_past_month_end(self, dl):
        coverage = DateRange("2015-01-01", "2019-12-31")
        with (
            patch(
                "h2mare.downloader.cds_downloader.get_store_coverage",
                return_value=coverage,
            ),
            patch("pandas.Timestamp.now", return_value=pd.Timestamp("2024-06-15")),
        ):
            result = dl._resolve_date_range("2020-01-01", None)
        assert result is not None
        # day=15 >= 10 → MonthEnd(-1) → end of May 2024
        assert pd.Timestamp(result.end) == pd.Timestamp("2024-05-31")


# ---------------------------------------------------------------------------
# run — dry_run
# ---------------------------------------------------------------------------


class TestRun:
    def test_dry_run_returns_false(self, dl):
        with patch.object(
            dl,
            "_resolve_date_range",
            return_value=DateRange("2020-01-01", "2020-03-31"),
        ):
            result = dl.run(dry_run=True)
        assert result is False

    def test_no_tasks_returns_false(self, dl):
        with patch.object(dl, "_resolve_date_range", return_value=None):
            result = dl.run()
        assert result is False


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    def _mock_client(self):
        mock_client = MagicMock()
        mock_client.retrieve.return_value.download.return_value = None
        return mock_client

    def test_full_year_uses_all_12_months(self, dl, tmp_path):
        with patch(
            "h2mare.downloader.cds_downloader.cdsapi.Client",
            return_value=self._mock_client(),
        ) as MockClient:
            dl.download_file(DateRange("2020-01-01", "2020-12-31"), output_dir=tmp_path)
            request = MockClient.return_value.retrieve.call_args[0][1]
        assert len(request["month"]) == 12

    def test_full_year_uses_31_days(self, dl, tmp_path):
        with patch(
            "h2mare.downloader.cds_downloader.cdsapi.Client",
            return_value=self._mock_client(),
        ) as MockClient:
            dl.download_file(DateRange("2020-01-01", "2020-12-31"), output_dir=tmp_path)
            request = MockClient.return_value.retrieve.call_args[0][1]
        assert len(request["day"]) == 31

    def test_partial_month_uses_single_month(self, dl, tmp_path):
        with patch(
            "h2mare.downloader.cds_downloader.cdsapi.Client",
            return_value=self._mock_client(),
        ) as MockClient:
            dl.download_file(DateRange("2020-03-10", "2020-03-20"), output_dir=tmp_path)
            request = MockClient.return_value.retrieve.call_args[0][1]
        assert request["month"] == ["03"]

    def test_partial_month_day_range_is_start_to_end(self, dl, tmp_path):
        with patch(
            "h2mare.downloader.cds_downloader.cdsapi.Client",
            return_value=self._mock_client(),
        ) as MockClient:
            dl.download_file(DateRange("2020-03-10", "2020-03-20"), output_dir=tmp_path)
            request = MockClient.return_value.retrieve.call_args[0][1]
        assert request["day"][0] == "10"
        assert request["day"][-1] == "20"
        assert len(request["day"]) == 11

    def test_dataset_id_passed_to_retrieve(self, dl, tmp_path):
        with patch(
            "h2mare.downloader.cds_downloader.cdsapi.Client",
            return_value=self._mock_client(),
        ) as MockClient:
            dl.download_file(DateRange("2020-06-01", "2020-06-30"), output_dir=tmp_path)
            dataset_id = MockClient.return_value.retrieve.call_args[0][0]
        assert dataset_id == "reanalysis-era5-single-levels"


# ---------------------------------------------------------------------------
# Download manifest
#
# Netcdf2Zarr stamps source_datasets from this file and returns early without
# one, which is why the ERA5 variables carried no provenance at all — not a
# stale record but no record, in any year. CMEMS and AVISO have always written
# one.
# ---------------------------------------------------------------------------


class TestManifest:
    def _run(self, dl, tmp_path, start="2020-01-01", end="2020-03-31"):
        with patch.object(dl, "download_file"):
            dl.run(start_date=start, end_date=end, output_dir=tmp_path)
        return json.loads((tmp_path / "h2mare_manifest.json").read_text())

    def test_a_manifest_is_written(self, dl, tmp_path):
        assert self._run(dl, tmp_path)

    def test_one_record_per_download_window(self, dl, tmp_path):
        records = self._run(dl, tmp_path)
        assert [(r["start"], r["end"]) for r in records] == [
            ("2020-01-01", "2020-01-31"),
            ("2020-02-01", "2020-02-29"),
            ("2020-03-01", "2020-03-31"),
        ]

    def test_every_record_names_the_reanalysis_product(self, dl, tmp_path):
        """ERA5 has no rep/nrt handover — one product covers the lot."""
        records = self._run(dl, tmp_path)
        assert {r["dataset_id"] for r in records} == {"reanalysis-era5-single-levels"}
        assert {r["dataset_type"] for r in records} == {"rep"}

    def test_written_before_the_downloads_run(self, dl, tmp_path):
        """A window that fails still has to say what was asked of it."""
        seen = {}

        def _spy(dt, output_dir=None):
            seen["existed"] = (tmp_path / "h2mare_manifest.json").exists()

        with patch.object(dl, "download_file", side_effect=_spy):
            dl.run(start_date="2020-01-01", end_date="2020-01-31", output_dir=tmp_path)

        assert seen["existed"]
