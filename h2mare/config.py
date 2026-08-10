"""
Configuration management for h2mare project
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import msgspec
import yaml
from dotenv import load_dotenv
from loguru import logger

from h2mare.models import AppConfig


class Settings:
    """Application settings and paths."""

    def __init__(self):

        # project root: directory containing h2mare's config.yaml
        self.BASE_DIR, self._project_mode = self._find_project_root()

        # Load .env file
        self._load_dotenv()

        # === Data Directories ===
        self.DATA_DIR = self.BASE_DIR / "data"

        # Raw Data (immutable)
        self.RAW_DIR = self.DATA_DIR / "raw"
        self.DOWNLOADS_DIR = self.RAW_DIR / "downloads"

        # Interim data (processing stages)
        self.INTERIM_DIR = self.DATA_DIR / "interim"

        # Processed data (final outputs)
        self.PROCESSED_DIR = self.DATA_DIR / "processed"
        self.ZARR_DIR = self.PROCESSED_DIR / "zarr"
        self.PARQUET_DIR = self.PROCESSED_DIR / "parquet"
        self.METADATA_DIR = self.PROCESSED_DIR / "metadata"

        # Logs
        self.LOGS_DIR = self.BASE_DIR / "logs"

        # External Storage (where all data lives)
        self.STORE_ROOT = self._get_store_dir()

        # No directories are created here: Settings() runs on any import, and
        # BASE_DIR may be a directory that merely contains a config.yaml. Writers
        # mkdir their own parents.

        # Application config (lazy loaded)
        self._app_config: Optional[AppConfig] = None
        self._global_attrs = None
        self._variable_attrs = None

    def _find_project_root(self) -> tuple[Path, bool]:
        """Find the h2mare project root and whether we are in project mode.

        Returns (root_path, is_project_mode). is_project_mode is True only when
        an h2mare config.yaml is found, meaning h2mare is being run as a project
        rather than used as a library dependency inside another project.

        Search order:
        1. H2MARE_ROOT env var (explicit override) → project mode.
        2. Walk up from cwd() looking for config.yaml only.
        3. Walk up from __file__ looking for config.yaml only (editable installs).
        4. Fallback: ~/.h2mare → library mode, no directories created.
        """
        if root_env := os.getenv("H2MARE_ROOT"):
            return Path(root_env).resolve(), True

        for start in (Path.cwd(), Path(__file__).resolve().parent):
            current = start.resolve()
            while current != current.parent:
                if (current / "config.yaml").exists():
                    return current, True
                current = current.parent

        return Path.home() / ".h2mare", False

    def _load_dotenv(self):
        """Load environment variables from .env file."""
        env_file = self.BASE_DIR / ".env"
        if env_file.exists():
            load_dotenv(env_file)

    def _get_store_dir(self) -> Path | None:
        """Get external storage directory from environment."""
        if store_dir := os.getenv("STORE_ROOT"):
            return Path(store_dir).resolve()
        return None

    @property
    def CLIMATOLOGY_DIR(self) -> Path | None:
        if self.STORE_ROOT is None:
            return None
        return self.STORE_ROOT / "Climatology"

    def ensure_directories(self):
        """Scaffold the project directory tree under BASE_DIR. Opt-in, not automatic."""
        dirs = [
            self.DOWNLOADS_DIR,
            self.INTERIM_DIR,
            self.ZARR_DIR,
            self.PARQUET_DIR,
            self.METADATA_DIR,
            self.LOGS_DIR,
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_app_config(self, config_path: Optional[Path] = None) -> AppConfig:
        """
        Load application configuration from YAML.

        Args:
            config_path: Path to config.yaml. If None, uses BASE_DIR/config.yaml

        Returns:
            Validated AppConfig instance
        """
        if self._app_config is not None:
            return self._app_config

        if config_path is None:
            config_path = self.BASE_DIR / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"config.yaml not found at {config_path}\n"
                f"Expected location: {self.BASE_DIR / 'config.yaml'}"
            )

        # Load YAML
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Warn on unrecognised top-level keys — catches typos like "varibles:"
        _KNOWN_KEYS = {"variables", "global_attrs", "variable_attrs"}
        unknown = set(config_dict) - _KNOWN_KEYS
        if unknown:
            logger.warning(
                f"config.yaml contains unrecognised top-level key(s): "
                f"{sorted(unknown)} — these will be ignored. "
                f"Expected keys: {sorted(_KNOWN_KEYS)}."
            )

        # Extract global and variable metadata (not part of AppConfig)
        self._global_attrs = config_dict.get("global_attrs", {})
        self._variable_attrs = config_dict.get("variable_attrs", {})

        # Add secrets from environment
        secrets_dict = {
            "aviso_ftp_server": os.getenv("AVISO_FTP_SERVER"),
            "aviso_username": os.getenv("AVISO_USERNAME"),
            "aviso_password": os.getenv("AVISO_PASSWORD"),
        }
        config_dict["secrets"] = secrets_dict

        # Warn early if AVISO variables are configured but credentials are absent
        aviso_vars = [
            k
            for k, v in config_dict.get("variables", {}).items()
            if isinstance(v, dict) and v.get("source") == "aviso"
        ]
        if aviso_vars:
            missing = [k for k, v in secrets_dict.items() if v is None]
            if missing:
                import warnings

                warnings.warn(
                    f"AVISO variables {aviso_vars} configured but env secrets missing: {missing}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # `subset` only affects CMEMS downloads (subset() vs get() API choice);
        # it is ignored for every other source. Flag it when set elsewhere so a
        # misplaced `subset:` is not silently no-op'd.
        misplaced_subset = [
            k
            for k, v in config_dict.get("variables", {}).items()
            if isinstance(v, dict) and v.get("source") != "cmems" and "subset" in v
        ]
        if misplaced_subset:
            logger.warning(
                f"`subset` is set on non-CMEMS variable(s) {sorted(misplaced_subset)} "
                "but only applies to CMEMS downloads — it will be ignored there."
            )

        self._app_config = msgspec.convert(config_dict, AppConfig)
        return self._app_config

    def get_available_var_keys(self) -> list[str]:
        """
        Get list of available variable keys from config.
        """
        if self._app_config is None:
            self._app_config = self.load_app_config()
        return list(self._app_config.variables.keys())

    def get_var_info(self, var_name: str) -> dict:
        """Get variable attributes from yaml. Returns {} if var_name is not in config."""
        if self._variable_attrs is None:
            self.load_app_config()
        return (self._variable_attrs or {}).get(var_name, {})

    @property
    def global_attrs(self) -> dict:
        if self._global_attrs is None:
            self.load_app_config()
        return self._global_attrs or {}

    @property
    def variable_attrs(self) -> dict:
        if self._variable_attrs is None:
            self.load_app_config()
        return self._variable_attrs or {}

    @property
    def app_config(self) -> AppConfig:
        """Lazy-loaded application config."""
        if self._app_config is None:
            self._app_config = self.load_app_config()
        return self._app_config


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application-wide Settings instance (cached after first call).

    Tests can reset the cache with ``get_settings.cache_clear()`` before
    monkeypatching environment variables to obtain a fresh instance.
    """
    return Settings()
