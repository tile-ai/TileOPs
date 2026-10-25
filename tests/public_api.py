"""The public surface of TileOPs, as data: every public name and the signature a caller binds.

`surface()` walks the public modules and returns one string per public name, sorted, so a
change to any name, signature, default or torch operator name shows as a line in a diff.
`python -m tests.public_api` rewrites the snapshot `test_public_api_snapshot.py` compares to.
"""

import importlib
import inspect
import json
import re
from pathlib import Path

import tileops

SNAPSHOT = Path(__file__).with_name("public_api_snapshot.json")
# Public modules besides the op families, which `tileops._FAMILIES` lists.
MODULES = ("tileops.authoring", "tileops.backend", "tileops.manifest", "tileops.trace")


def _signature(obj: object) -> str:
    try:
        text = str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(?)"
    # A default without a repr of its own prints its address, which differs every run.
    return re.sub(r" at 0x[0-9a-f]+>", ">", text)


def _members(cls: type) -> list[str]:
    """Each public member a class defines or inherits from TileOPs, with its signature."""
    out = []
    for name in sorted(dir(cls)):
        if name.startswith("_"):
            continue
        owner = next(k for k in cls.__mro__ if name in vars(k))
        if not owner.__module__.startswith("tileops"):
            continue
        value = vars(owner)[name]
        if isinstance(value, property):
            out.append(f"{name}: property")
        elif isinstance(value, (staticmethod, classmethod)):
            out.append(f"{name}{_signature(value.__func__)}")
        elif callable(value):
            out.append(f"{name}{_signature(value)}")
        else:
            out.append(f"{name}: attribute")
    return out


def _describe(module: str, name: str, obj: object) -> list[str]:
    path = f"{module}.{name}"
    if inspect.isclass(obj):
        lines = [f"{path}: class{_signature(obj.__init__)}"]
        if module == "tileops.authoring" or not issubclass(obj, tileops.Op):
            lines += [f"{path}.{member}" for member in _members(obj)]
        else:
            lines.append(f"{path}.forward{_signature(obj.forward)}")
            lines += [f"{path} torch operator: {op}" for op in obj.compile_op_names]
        return lines
    if callable(obj):
        return [f"{path}{_signature(obj)}"]
    return [f"{path}: {type(obj).__name__}"]


def surface() -> list[str]:
    """The public surface, one line per name or member."""
    lines = [f"tileops.{name}" for name in tileops.__all__]
    for module in (*(f"tileops.{f}" for f in tileops._FAMILIES), *MODULES):
        mod = importlib.import_module(module)
        for name in mod.__all__:
            lines += _describe(module, name, getattr(mod, name))
    return sorted(lines)


if __name__ == "__main__":
    SNAPSHOT.write_text(json.dumps(surface(), indent=0) + "\n")
    print(f"wrote {SNAPSHOT}")
