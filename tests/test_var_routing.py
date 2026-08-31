"""Which store answers for a variable name — h2mare.storage.var_routing."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from h2mare.models import TimeStep
from h2mare.storage import var_routing as routing_module
from h2mare.storage.var_routing import (
    catalog_for_var,
    compiled_var_key,
    coverage_for_var,
)
from h2mare.types import DateRange

# ---------------------------------------------------------------------------
# Config fixtures — the three shapes that route differently
# ---------------------------------------------------------------------------


def _hourly_config() -> SimpleNamespace:
    """atm-instante-shaped: stores the raw source, publishes the daily reduction."""
    return SimpleNamespace(
        compiled_vars=["msl", "u10", "v10", "tcc", "wind_mean", "wind_max"],
        time_step=TimeStep.HOURLY,
        source="cds",
        local_folder="CDS_Atm_instante",
        store_root=None,
    )


def _depth_config() -> SimpleNamespace:
    """thetao-shaped: stores one variable on a depth axis, publishes the levels."""
    return SimpleNamespace(
        compiled_vars=["thetao_0", "thetao_100", "thetao_500"],
        time_step=TimeStep.DAILY,
        source="cmems",
        local_folder="CMEMS_Thetao",
        store_root=None,
    )


def _h2ds_config() -> SimpleNamespace:
    """The compiled store, found by ``source: h2mare`` rather than by its name."""
    return SimpleNamespace(
        compiled_vars=[],
        time_step=TimeStep.DAILY,
        source="h2mare",
        local_folder="h2ds",
        store_root=None,
    )


@pytest.fixture
def opened(monkeypatch):
    """Stub ZarrCatalog; returns the list of var_keys opened, in order.

    ``get_variables`` answers from ``stores``, so a var_key absent from it
    stands for a store holding nothing.
    """
    stores = {
        "atm-instante": {"msl", "u10", "v10", "tcc"},
        "thetao": {"thetao"},
        "h2ds": {"wind_mean", "wind_max", "thetao_100"},
    }
    # h2ds runs to 08-21 as a store; wind_mean, coming from an ERA5 var_key,
    # stops three weeks earlier and is NaN padding after that.
    coverages = {
        ("atm-instante", "u10"): DateRange("2026-01-01", "2026-07-31"),
        ("h2ds", "wind_mean"): DateRange("2026-01-01", "2026-07-31"),
    }
    seen: list[str] = []

    def _factory(var_key, **_kw):
        seen.append(var_key)
        catalog = MagicMock()
        catalog.var_key = var_key
        catalog.get_variables.return_value = stores.get(var_key, set())
        catalog.get_var_coverage.side_effect = lambda v, _k=var_key: coverages.get(
            (_k, v)
        )
        return catalog

    monkeypatch.setattr(routing_module, "ZarrCatalog", _factory)
    return seen


def _config(**over) -> SimpleNamespace:
    variables = {
        "atm-instante": _hourly_config(),
        "thetao": _depth_config(),
        "h2ds": _h2ds_config(),
    }
    variables.update(over)
    return SimpleNamespace(variables=variables)


def _route(var, var_key, **kwargs):
    return catalog_for_var(var, var_key, app_config=_config(), **kwargs)


# ---------------------------------------------------------------------------


class TestAutoRouting:
    """
    An hourly store holds the raw source only, and a 3-D store holds one
    variable on a depth axis. Either way the name a caller holds — an h2ds
    column, which is what extraction returned — can be absent from its own
    var_key's store, and must route to the compiled one instead of raising.
    """

    def test_derived_daily_name_routes_to_the_compiled_store(self, opened):
        cat = _route("wind_mean", "atm-instante")

        assert cat.var_key == "h2ds"
        assert opened == ["atm-instante", "h2ds"]

    def test_raw_name_stays_on_the_hourly_store(self, opened):
        """`u10` is in the store; nothing is gained by going to h2ds for it."""
        cat = _route("u10", "atm-instante")

        assert cat.var_key == "atm-instante"
        assert opened == ["atm-instante"]

    def test_depth_sliced_name_routes_to_the_compiled_store(self, opened):
        """Not an hourly case: the store holds `thetao`, h2ds holds the levels."""
        cat = _route("thetao_100", "thetao")

        assert cat.var_key == "h2ds"

    def test_name_in_neither_store_names_both_sides(self, opened):
        with pytest.raises(ValueError) as err:
            _route("not_a_var", "atm-instante")

        msg = str(err.value)
        assert "not a variable of 'atm-instante'" in msg
        assert "u10" in msg  # what the store holds
        assert "wind_mean" in msg  # what config publishes

    def test_unknown_var_key_is_rejected_before_any_store_is_opened(self, opened):
        with pytest.raises(ValueError, match="not found in config"):
            _route("wind_mean", "nope")

        assert opened == []


class TestPinnedReadFrom:
    """`read_from` is how you overrule the routing in either direction."""

    def test_native_pins_the_raw_store(self, opened):
        cat = _route("u10", "atm-instante", read_from="native")

        assert cat.var_key == "atm-instante"

    def test_native_refuses_a_name_its_store_does_not_hold(self, opened):
        with pytest.raises(ValueError) as err:
            _route("wind_mean", "atm-instante", read_from="native")

        assert "drop read_from" in str(err.value)

    def test_compiled_pins_h2ds_without_opening_the_native_store(self, opened):
        cat = _route("msl", "atm-instante", read_from="compiled")

        assert cat.var_key == "h2ds"
        assert opened == ["h2ds"]

    def test_compiled_refuses_a_name_the_var_key_does_not_publish(self, opened):
        with pytest.raises(ValueError) as err:
            _route("not_a_var", "atm-instante", read_from="compiled")

        assert "not one of its compiled_vars" in str(err.value)


class TestCoverageForVar:
    """
    Routing says *which* store; this says *how far* that store has the variable.
    Both are needed because a compiled store ends where its furthest-ahead
    source ends, and pads everything slower with NaN out to that date.
    """

    def test_coverage_comes_from_the_routed_store_per_variable(self, opened):
        cat, coverage = coverage_for_var(
            "wind_mean", "atm-instante", app_config=_config()
        )

        assert cat.var_key == "h2ds"
        # h2ds's own end is later; wind_mean's is where ERA5 stopped.
        assert coverage.end == pd.Timestamp("2026-07-31")

    def test_native_variable_is_asked_of_its_own_store(self, opened):
        cat, coverage = coverage_for_var("u10", "atm-instante", app_config=_config())

        assert cat.var_key == "atm-instante"
        assert coverage.start == pd.Timestamp("2026-01-01")

    def test_a_store_with_no_file_for_the_name_says_so(self, opened):
        """Routable (config publishes it) but nothing on disk carries it yet."""
        with pytest.raises(ValueError, match="no coverage for 'wind_max'"):
            coverage_for_var("wind_max", "atm-instante", app_config=_config())


class TestCompiledVarKey:
    def test_found_by_source_not_by_name(self):
        cfg = SimpleNamespace(
            variables={"atm-instante": _hourly_config(), "my_compiled": _h2ds_config()}
        )
        assert compiled_var_key(cfg) == "my_compiled"

    def test_config_without_one_says_so(self):
        cfg = SimpleNamespace(variables={"atm-instante": _hourly_config()})
        with pytest.raises(ValueError, match="No compiled var_key"):
            compiled_var_key(cfg)

    def test_two_candidates_are_ambiguous(self):
        cfg = SimpleNamespace(variables={"a": _h2ds_config(), "b": _h2ds_config()})
        with pytest.raises(ValueError, match="Ambiguous compiled var_key"):
            compiled_var_key(cfg)
