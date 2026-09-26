"""The op class a manifest entry names (docs/design/manifest.md § Top-Level Fields)."""

from __future__ import annotations

import importlib

from .signature import is_legacy

__all__ = ["op_class"]


def op_class(name: str, entry: dict) -> type:
    """The class of op *name*: `tileops.<family>.<name>`, or where a legacy entry's `source.op` is.

    Raises:
        ImportError: The module does not import.
        AttributeError: The module has no attribute *name*.
    """
    if is_legacy(entry):
        module = entry["source"]["op"].removesuffix(".py").replace("/", ".")
    else:
        module = f"tileops.{entry['family']}"
    return getattr(importlib.import_module(module), name)
