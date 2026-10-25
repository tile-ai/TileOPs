#!/usr/bin/env python3
"""Fail when the public API breaks compatibility with the last release.

Imports the public modules of two source trees, the latest ``vX.Y.Z`` tag and the working tree,
each in its own interpreter, and compares what they expose. Removing a public name, a class
member or a torch operator name breaks compatibility, and so does changing a parameter a
caller may pass: its name, kind or default, or the order of positional parameters. Adding a
name, or a parameter that has a default, does not. Without a release tag the check passes.

    python scripts/ci/check_api_compat.py [--base <git ref>]
"""

import argparse
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


def _class_members(path: str, cls: type) -> dict:
    """Each public member a class defines or inherits from TileOPs."""
    out = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        owner = next(k for k in cls.__mro__ if name in vars(k))
        if not owner.__module__.startswith("tileops"):
            continue
        value = vars(owner)[name]
        if isinstance(value, property):
            out[f"{path}.{name}"] = _entry("property")
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
        except ModuleNotFoundError:
            continue  # a module added after the release being described
        for name in mod.__all__:
            path, obj = f"{module}.{name}", getattr(mod, name)
            if not inspect.isclass(obj):
                out[path] = _entry("function", obj) if callable(obj) else _entry("value")
                continue
            out[path] = _entry("class", obj.__init__)
            if module != "tileops.authoring" and issubclass(obj, tileops.Op):
                # A concrete op: callers bind its constructor and forward, and a compiled graph
                # names its torch operators.
                out[f"{path}.forward"] = _entry("method", obj.forward)
                for op in obj.compile_op_names:
                    out[f"{path} torch operator {op}"] = _entry("operator")
            else:
                out.update(_class_members(path, obj))
    return out


def _parameter_breaks(old: list, new: list) -> list[str]:
    out = []
    by_name = {p[0]: p for p in new}
    for name, kind, default in old:
        now = by_name.get(name)
        if now is None:
            out.append(f"parameter `{name}` removed")
        elif now[1] != kind:
            out.append(f"parameter `{name}` changed from {kind} to {now[1]}")
        elif now[2] != default:
            out.append(f"parameter `{name}` default changed from {default} to {now[2]}")
    old_positional = [p[0] for p in old if p[1] in _POSITIONAL]
    new_positional = [p[0] for p in new if p[1] in _POSITIONAL and p[0] in old_positional]
    if new_positional != [n for n in old_positional if n in by_name]:
        out.append(f"positional order changed from {old_positional} to {new_positional}")
    known = {p[0] for p in old}
    for name, kind, default in new:
        if name not in known and default is None and kind not in _VARIADIC:
            out.append(f"new parameter `{name}` has no default")
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
        "import json, check_api_compat; print(json.dumps(check_api_compat.describe()))"
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
