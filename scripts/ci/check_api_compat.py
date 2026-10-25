#!/usr/bin/env python3
"""Fail when the public API breaks compatibility with the last release.

Imports the public modules of two source trees, the latest ``vX.Y.Z`` tag and the working tree,
each in its own interpreter, and compares what they expose. A change breaks compatibility
when code written against the release stops working or binds differently: a removed name,
member, property setter or torch operator name, a positional slot that binds another
parameter, a keyword no longer accepted, a changed default, or a new required parameter.
Additions pass. Return types and behavior are not compared; review covers them. Without a
release tag the check passes.

    python scripts/ci/check_api_compat.py [--base <git ref>]
"""

import argparse
import dataclasses
import importlib
import inspect
import json
import re
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Public modules besides the op families, which `tileops._FAMILIES` lists.
MODULES = ("tileops.authoring", "tileops.backend", "tileops.manifest", "tileops.trace")
_POSITIONAL = ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")
_VARIADIC = ("VAR_POSITIONAL", "VAR_KEYWORD")


def _parameters(obj: object) -> "list[list] | None":
    """``[name, kind, default]`` per parameter, the default as its repr or None when absent."""
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    out = []
    for p in signature.parameters.values():
        if p.name == "self":
            continue
        default = None
        if p.default is not inspect.Parameter.empty:
            # A default without a repr of its own prints its address, which differs every run.
            default = re.sub(r" at 0x[0-9a-f]+>", ">", repr(p.default))
        out.append([p.name, p.kind.name, default])
    return out


def _entry(kind: str, obj: object = None) -> dict:
    entry = {"kind": kind}
    if obj is not None:
        entry["params"] = _parameters(obj)
    return entry


def _class_members(path: str, cls: type, skip: tuple = ()) -> dict:
    """Each public member a class defines or inherits from TileOPs, except those of *skip*,
    and the fields of a dataclass or named tuple, which an instance holds."""
    fields = (
        [f.name for f in dataclasses.fields(cls)]
        if dataclasses.is_dataclass(cls)
        else list(getattr(cls, "_fields", ()))
    )
    out = {f"{path}.{name}": _entry("attribute") for name in fields if not name.startswith("_")}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        owner = next(k for k in cls.__mro__ if name in vars(k))
        if not owner.__module__.startswith("tileops") or owner in skip or name in fields:
            continue
        value = vars(owner)[name]
        if isinstance(value, property):
            out[f"{path}.{name}"] = _entry(
                "property" if value.fset is None else "settable property"
            )
        elif isinstance(value, (staticmethod, classmethod)):
            out[f"{path}.{name}"] = _entry("method", value.__func__)
        elif callable(value):
            out[f"{path}.{name}"] = _entry("method", value)
        else:
            out[f"{path}.{name}"] = _entry("attribute")
    return out


def describe() -> dict:
    """The public surface of the `tileops` on `sys.path`, keyed by dotted path."""
    import tileops

    out = {f"tileops.{name}": _entry("name") for name in tileops.__all__}
    for module in (*(f"tileops.{f}" for f in tileops._FAMILIES), *MODULES):
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError as error:
            if error.name != module:
                raise
            continue  # a module added after the release being described
        for name in mod.__all__:
            path, obj = f"{module}.{name}", getattr(mod, name)
            if not inspect.isclass(obj):
                if callable(obj) and not inspect.isroutine(obj):
                    # An instance such as `tileops.trace.trace`: callers use its members.
                    out[path] = _entry("instance")
                    out.update(_class_members(path, type(obj)))
                else:
                    out[path] = _entry("function", obj) if callable(obj) else _entry("value")
                continue
            # The class itself: its signature is what construction binds, `__new__` included.
            out[path] = _entry("class", obj)
            if module != "tileops.authoring" and issubclass(obj, tileops.Op):
                # A concrete op: callers bind its constructor and forward, a compiled graph
                # names its torch operators, and `tileops.authoring.Op` covers what it inherits.
                out[f"{path}.forward"] = _entry("method", obj.forward)
                out.update(_class_members(path, obj, skip=tileops.Op.__mro__))
                for op in obj.compile_op_names:
                    out[f"{path} torch operator {op}"] = _entry("operator")
            else:
                out.update(_class_members(path, obj))
    return out


def _parameter_breaks(old: list, new: list) -> list[str]:
    """What makes a call written against *old* fail, or bind differently, against *new*.

    A call passes positional arguments by slot and the rest by keyword. It keeps working when
    every slot binds the same parameter, every keyword is still accepted, nothing optional
    becomes required, and no default changes. A positional-only or variadic parameter is
    matched by position or kind, since a call never names it.
    """
    out = []
    old_slots = [p for p in old if p[1] in _POSITIONAL]
    new_slots = [p for p in new if p[1] in _POSITIONAL]
    for i, (name, kind, _default) in enumerate(old_slots):
        if i >= len(new_slots):
            out.append(f"positional parameter `{name}` (slot {i}) removed")
            continue
        now = new_slots[i]
        if kind == "POSITIONAL_OR_KEYWORD" and (now[0], now[1]) != (name, kind):
            out.append(f"slot {i} was `{name}`, is now `{now[0]}` ({now[1]})")
    by_name = {p[0]: p for p in new}
    for name, kind, default in old:
        if kind in _VARIADIC:
            if not any(p[1] == kind for p in new):
                out.append(f"`{'*' if kind == 'VAR_POSITIONAL' else '**'}{name}` removed")
            continue
        if kind == "KEYWORD_ONLY":
            now = by_name.get(name)
            if now is None or now[1] not in ("KEYWORD_ONLY", "POSITIONAL_OR_KEYWORD"):
                out.append(f"keyword parameter `{name}` removed")
                continue
        else:
            index = [p[0] for p in old_slots].index(name)
            if index >= len(new_slots):
                continue
            now = new_slots[index]
        if default is not None and now[2] != default:
            out.append(f"parameter `{name}` default changed from {default} to {now[2]}")
    # A parameter a call may omit today must stay omittable.
    required_slots = {i for i, p in enumerate(old_slots) if p[2] is None}
    # Every call passes a required parameter it can name, by position or by keyword.
    required_names = {
        p[0] for p in old if p[1] != "POSITIONAL_ONLY" and p[1] not in _VARIADIC and p[2] is None
    }
    for i, (name, _kind, default) in enumerate(new_slots):
        if default is None and i not in required_slots and name not in required_names:
            out.append(f"parameter `{name}` (slot {i}) is required, and calls may omit it")
    for name, kind, default in new:
        if kind == "KEYWORD_ONLY" and default is None and name not in required_names:
            out.append(f"keyword parameter `{name}` is required, and calls may omit it")
    return out


def breaks(old: dict, new: dict) -> list[str]:
    """What in *new* breaks code written against *old*."""
    out = []
    for path, before in sorted(old.items()):
        after = new.get(path)
        if after is None:
            out.append(f"{path}: removed")
        elif after["kind"] != before["kind"]:
            out.append(f"{path}: was a {before['kind']}, is now a {after['kind']}")
        elif before.get("params") is not None and after.get("params") is not None:
            out += [f"{path}: {b}" for b in _parameter_breaks(before["params"], after["params"])]
    return out


def _latest_tag() -> "str | None":
    """The highest final release tag, found by version rather than ancestry, which a shallow
    clone lacks."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "tag", "--list", "v*", "--sort=-v:refname"],
        capture_output=True,
        text=True,
        check=True,
    )
    final = [t for t in result.stdout.split() if re.fullmatch(r"v\d+\.\d+\.\d+", t)]
    return final[0] if final else None


def _surface_of(src: Path) -> dict:
    """`describe()` run on the package under *src*, in a fresh interpreter."""
    code = (
        f"import sys; sys.path.insert(0, {str(src)!r}); "
        f"sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        "import json, tileops, check_api_compat; "
        # An installed tileops must not stand in for the tree being described.
        f"assert tileops.__file__.startswith({str(src)!r}), tileops.__file__; "
        "print(json.dumps(check_api_compat.describe()))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"importing the package under {src} failed:\n{result.stderr}")
    return json.loads(result.stdout.splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base", help="git ref of the release to compare with (default: the latest vX.Y.Z tag)"
    )
    args = parser.parse_args()
    base = args.base or _latest_tag()
    if base is None:
        print("no release tag yet; nothing to compare")
        return 0
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", base, "src"], capture_output=True, check=True
    ).stdout
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=BytesIO(archive)) as tar:
            tar.extractall(tmp, filter="data")
        old = _surface_of(Path(tmp) / "src")
    new = _surface_of(REPO_ROOT / "src")
    found = breaks(old, new)
    for line in found:
        print(line)
    print(f"{len(found)} incompatible change(s) against {base}")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
