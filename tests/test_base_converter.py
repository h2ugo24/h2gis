"""Unit tests for BaseConverter — the format_converters ABC.

The class is deliberately minimal (it only enforces ``run()``), which is exactly
why it needs a test: nothing else fails if a converter stops inheriting from it,
or if the abstract method quietly loses its decorator and lets an incomplete
subclass instantiate.
"""

from __future__ import annotations

import inspect

import pytest

from h2mare.format_converters.base import BaseConverter
from h2mare.format_converters.netcdf2zarr import Netcdf2Zarr
from h2mare.format_converters.zarr2parquet import Zarr2Parquet


class TestRunContract:
    def test_subclass_without_run_cannot_be_instantiated(self):
        class Incomplete(BaseConverter):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_implementing_run_instantiates(self):
        class Complete(BaseConverter):
            def run(self) -> bool:
                return True

        assert Complete().run() is True

    def test_base_class_itself_is_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            BaseConverter()  # type: ignore[abstract]


class TestConcreteConvertersRegister:
    """The ABC only buys anything if the real converters actually inherit it."""

    @pytest.mark.parametrize("converter", [Netcdf2Zarr, Zarr2Parquet])
    def test_converter_subclasses_base(self, converter):
        assert issubclass(converter, BaseConverter)

    @pytest.mark.parametrize("converter", [Netcdf2Zarr, Zarr2Parquet])
    def test_converter_overrides_run(self, converter):
        """Inheriting without overriding would leave the abstract stub in place
        and make the class impossible to construct."""
        assert converter.run is not BaseConverter.run
        assert not getattr(converter.run, "__isabstractmethod__", False)

    @pytest.mark.parametrize("converter", [Netcdf2Zarr, Zarr2Parquet])
    def test_run_returns_a_bool_per_the_contract(self, converter):
        """PipelineManager branches on the return value, so an implementation
        annotated as returning something else is a real mismatch."""
        annotation = inspect.signature(converter.run).return_annotation
        assert annotation in (bool, "bool")
