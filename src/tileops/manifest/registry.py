"""The op class a manifest entry names (docs/design/manifest.md § Top-Level Fields)."""

from __future__ import annotations

import importlib

__all__ = ["op_class"]


def op_class(name: str, entry: dict) -> type:
    """The class of op *name*: `tileops.<family>.<name>`.

    Raises:
        ImportError: The module does not import.
        AttributeError: The module has no attribute *name*.
    """
    return getattr(importlib.import_module(f"tileops.{entry['family']}"), name)
