"""Tests for config.py — Settings class."""

import warnings

import pytest

from h2mare.config import Settings, get_settings

_MINIMAL_CONFIG_YAML = """\
global_attrs:
  title: test dataset

variable_attrs:
  sst:
    long_name: Sea Surface Temperature
    units: K

variables:
  sst:
    local_folder: sst
    source_vars:
      - analysed_sst
    dataset_id_rep: cmems-rep-sst
    source: cmems
    archive_raw: false
    pattern: '.*\\.nc'
    subset: true
    bbox:
      - -80
      - 0
      - 10
      - 70
"""


# ---------------------------------------------------------------------------
# _find_project_root
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    def test_h2mare_root_env_takes_priority(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        s = Settings()
        assert s.BASE_DIR == tmp_path.resolve()

    def test_h2mare_root_sets_project_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        s = Settings()
        assert s._project_mode is True

    def test_base_dir_is_resolved_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        s = Settings()
        assert s.BASE_DIR.is_absolute()


# ---------------------------------------------------------------------------
# directory creation
# ---------------------------------------------------------------------------


class TestDirectoryCreation:
    def test_settings_creates_nothing_under_base_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        Settings()
        assert list(tmp_path.iterdir()) == []

    def test_discovered_root_creates_nothing(self, tmp_path, monkeypatch):
        """A config.yaml in cwd makes tmp_path the project root — still no data/."""
        monkeypatch.delenv("H2MARE_ROOT", raising=False)
        monkeypatch.delenv("STORE_ROOT", raising=False)
        (tmp_path / "config.yaml").write_text(_MINIMAL_CONFIG_YAML)
        monkeypatch.chdir(tmp_path)
        s = Settings()
        assert s.BASE_DIR == tmp_path.resolve()
        assert not (tmp_path / "data").exists()
        assert not (tmp_path / "logs").exists()

    def test_ensure_directories_creates_tree_when_called(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        s = Settings()
        s.ensure_directories()
        for d in (
            s.DOWNLOADS_DIR,
            s.INTERIM_DIR,
            s.ZARR_DIR,
            s.PARQUET_DIR,
            s.METADATA_DIR,
            s.LOGS_DIR,
        ):
            assert d.is_dir()


# ---------------------------------------------------------------------------
# _get_store_dir
# ---------------------------------------------------------------------------


class TestGetStoreDir:
    def test_store_root_env_returned_as_path(self, tmp_path, monkeypatch):
        store = tmp_path / "my_store"
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.setenv("STORE_ROOT", str(store))
        s = Settings()
        assert s.STORE_ROOT == store.resolve()

    def test_missing_store_root_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        s = Settings()
        assert s.STORE_ROOT is None


class TestOverrideStoreRoot:
    """
    Backs --store-path. Applied to settings rather than threaded through each
    step, so the places nothing passes an argument to — the compiler's own
    per-variable catalogs, cds.get_previous_dates_da — follow it too.
    """

    def test_override_replaces_the_env_value(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.setenv("STORE_ROOT", str(tmp_path / "from_env"))
        s = Settings()

        s.override_store_root(tmp_path / "from_flag")

        assert s.STORE_ROOT == (tmp_path / "from_flag").resolve()

    def test_per_variable_paths_follow_the_override(self, tmp_path, monkeypatch):
        """The point of overriding at the source: resolution downstream moves."""
        import msgspec

        from h2mare.models import AppConfig
        from h2mare.utils.paths import resolve_store_path

        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.setenv("STORE_ROOT", str(tmp_path / "from_env"))
        s = Settings()
        entry = {
            "local_folder": "sst",
            "source_vars": ["analysed_sst"],
            "dataset_id_rep": "cmems-rep-sst",
            "source": "cmems",
            "archive_raw": False,
            "pattern": r".*\.nc",
        }
        cfg = msgspec.convert({"variables": {"sst": entry}, "secrets": {}}, AppConfig)
        var_config = cfg.variables["sst"]

        monkeypatch.setattr("h2mare.utils.paths.get_settings", lambda: s)
        s.override_store_root(tmp_path / "from_flag")

        assert resolve_store_path(var_config, warn_if_missing=False) == (
            tmp_path / "from_flag" / "sst"
        )

    def test_climatology_dir_follows_the_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.setenv("STORE_ROOT", str(tmp_path / "from_env"))
        s = Settings()

        s.override_store_root(tmp_path / "from_flag")

        assert s.CLIMATOLOGY_DIR == (tmp_path / "from_flag").resolve() / "Climatology"


# ---------------------------------------------------------------------------
# load_app_config
# ---------------------------------------------------------------------------


class TestLoadAppConfig:
    def _make_settings(self, tmp_path, monkeypatch) -> Settings:
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        (tmp_path / "config.yaml").write_text(_MINIMAL_CONFIG_YAML)
        return Settings()

    def test_loads_configured_variable(self, tmp_path, monkeypatch):
        s = self._make_settings(tmp_path, monkeypatch)
        config = s.load_app_config()
        assert "sst" in config.variables

    def test_raises_when_config_yaml_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        s = Settings()
        with pytest.raises(FileNotFoundError):
            s.load_app_config()

    def test_second_call_returns_same_object(self, tmp_path, monkeypatch):
        s = self._make_settings(tmp_path, monkeypatch)
        config1 = s.load_app_config()
        config2 = s.load_app_config()
        assert config1 is config2

    def test_global_attrs_populated(self, tmp_path, monkeypatch):
        s = self._make_settings(tmp_path, monkeypatch)
        s.load_app_config()
        assert s._global_attrs.get("title") == "test dataset"

    def test_warns_when_aviso_vars_but_missing_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        monkeypatch.delenv("AVISO_FTP_SERVER", raising=False)
        monkeypatch.delenv("AVISO_USERNAME", raising=False)
        monkeypatch.delenv("AVISO_PASSWORD", raising=False)
        aviso_yaml = _MINIMAL_CONFIG_YAML.replace("source: cmems", "source: aviso")
        (tmp_path / "config.yaml").write_text(aviso_yaml)
        s = Settings()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s.load_app_config()
        assert any(issubclass(warning.category, RuntimeWarning) for warning in w)

    def test_warns_when_subset_set_on_non_cmems_var(self, tmp_path, monkeypatch):
        """`subset` only affects CMEMS; setting it elsewhere logs a warning."""
        from loguru import logger

        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        cds_yaml = _MINIMAL_CONFIG_YAML.replace("source: cmems", "source: cds")
        (tmp_path / "config.yaml").write_text(cds_yaml)
        s = Settings()

        messages: list[str] = []
        sink_id = logger.add(messages.append, level="WARNING")
        try:
            s.load_app_config()
        finally:
            logger.remove(sink_id)

        assert any("subset" in m and "non-CMEMS" in m for m in messages)


# ---------------------------------------------------------------------------
# get_var_info
# ---------------------------------------------------------------------------


class TestGetVarInfo:
    def _make_settings(self, tmp_path, monkeypatch) -> Settings:
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        (tmp_path / "config.yaml").write_text(_MINIMAL_CONFIG_YAML)
        return Settings()

    def test_returns_attrs_for_known_var(self, tmp_path, monkeypatch):
        s = self._make_settings(tmp_path, monkeypatch)
        info = s.get_var_info("sst")
        assert info.get("long_name") == "Sea Surface Temperature"
        assert info.get("units") == "K"

    def test_returns_empty_dict_for_unknown_var(self, tmp_path, monkeypatch):
        s = self._make_settings(tmp_path, monkeypatch)
        assert s.get_var_info("nonexistent") == {}


# ---------------------------------------------------------------------------
# get_available_var_keys
# ---------------------------------------------------------------------------


class TestGetAvailableVarKeys:
    def test_returns_list_of_configured_keys(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        (tmp_path / "config.yaml").write_text(_MINIMAL_CONFIG_YAML)
        s = Settings()
        keys = s.get_available_var_keys()
        assert isinstance(keys, list)
        assert "sst" in keys


# ---------------------------------------------------------------------------
# get_settings — cached factory
# ---------------------------------------------------------------------------


class TestGetSettingsFactory:
    def test_returns_settings_instance(self):
        assert isinstance(get_settings(), Settings)

    def test_same_object_returned_on_repeated_calls(self):
        assert get_settings() is get_settings()

    def test_cache_clear_produces_fresh_instance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("H2MARE_ROOT", str(tmp_path))
        monkeypatch.delenv("STORE_ROOT", raising=False)
        get_settings.cache_clear()
        try:
            s = get_settings()
            assert s.BASE_DIR == tmp_path.resolve()
        finally:
            get_settings.cache_clear()  # restore default behaviour for other tests
