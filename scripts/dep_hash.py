#!/usr/bin/env python3
"""Print a SHA-256 hash of the dependency-related fields in pyproject.toml.

Used by CI workflows to compute a stable venv cache key that only changes
when actual dependencies change, not when linting/style config is modified.
"""

import hashlib
import json
import pathlib

import tomllib

d = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
optional_deps = d.get("project", {}).get("optional-dependencies", {})
optional_deps_filtered = {k: v for k, v in optional_deps.items() if k != "bench"}
deps = {
    "dependencies": d.get("project", {}).get("dependencies", []),
    "optional": optional_deps_filtered,
    "requires-python": d.get("project", {}).get("requires-python", ""),
    "build-requires": d.get("build-system", {}).get("requires", []),
}
print(hashlib.sha256(json.dumps(deps, sort_keys=True).encode()).hexdigest())
