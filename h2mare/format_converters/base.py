"""Abstract base class for format converters."""

from __future__ import annotations

from abc import ABC, abstractmethod


# Intentionally minimal — converters have different enough constructors that
# shared __init__ setup would be contrived. The base only enforces the run() contract.
class BaseConverter(ABC):
    """
    Contract shared by the format converters: each one exposes ``run()``.

    Deliberately minimal — see the note above. It exists so callers can hold a
    converter without knowing which one it is, not to share implementation.
    """

    @abstractmethod
    def run(self) -> bool:
        """Perform the conversion. Returns True on success."""
        ...
