#!/usr/bin/env python3
"""Flag manifest entry and class names that break the naming rules.

M1  An abbreviation in an entry name is spelled as ``ABBREVIATIONS`` fixes.
M2  An entry name holds no ``AGENTIVE`` noun, unless ``AGENTIVE_EXCEPTIONS`` lists it.
M3  Among entries sharing one ``ref_api``, the plain ``<Stem>FwdOp`` / ``<Stem>BwdOp`` is
    flagged, since it does not say which variant it is, unless ``DEFAULTED_SWITCH`` lists
    the API; the plain name is then the default side.
M4  M1 over every class under ``src/tileops/`` and ``workloads/``.

Reads the manifest through ``tileops.manifest.load_manifest``.
"""

import ast
import re
from collections import defaultdict
from pathlib import Path

from common import REPO_ROOT, relative, report, use_source_tree

use_source_tree()

from tileops.manifest import load_manifest, manifest_files  # noqa: E402

# M1: the one spelling, and a pattern matching every other one.
ABBREVIATIONS = {
    "FP8": r"Fp8",
    "INT8": r"Int8",
    "INT4": r"Int4",
    "MoE": r"Moe",
    "TopK": r"Topk",
    "MLP": r"Mlp",
    "FFT": r"Fft",
    "RMS": r"Rms(?=Norm)",
    "NSA": r"Nsa",
    "MLA": r"Mla",
    "GQA": r"Gqa",
    "SSD": r"Ssd",
    "MHA": r"Mha",
    "DSA": r"Dsa",
    "GLA": r"Gla",
    "GDN": r"Gdn",
    "KDA": r"Kda",
    "RoPE": r"Rope",
    "YaRN": r"Yarn",
    "WS": r"Ws(?=[A-Z0-9]|$)",
    "MMA": r"Mma",
    "WGMMA": r"Wgmma",
    "TMA": r"Tma",
}

# M2: a noun naming an actor rather than the step.
AGENTIVE = ("Producer", "Selector", "Indexer", "Builder", "Manager", "Handler")
# Each entry is a review decision, with its reason.
AGENTIVE_EXCEPTIONS = {
    "FP8LightningIndexerFwdOp": "DeepSeek-V3.2 calls this component the lightning indexer",
}

# M3: a `ref_api` whose torch switch has a default, so the plain name is the default side.
DEFAULTED_SWITCH = {
    "torch.nn.functional.max_pool1d",
    "torch.nn.functional.max_pool2d",
    "torch.nn.functional.max_pool3d",
    "torch.nn.functional.adaptive_max_pool2d",
}


_ENTRY_KEY = re.compile(r"([A-Za-z_]\w*)\s*:")


def entry_locations() -> dict[str, str]:
    """The ``path:line`` each manifest entry's key sits on, by entry name."""
    out = {}
    for spec in manifest_files():
        rel = relative(Path(str(spec)))
        for number, line in enumerate(spec.read_text(encoding="utf-8").splitlines(), start=1):
            key = _ENTRY_KEY.match(line)
            if key:
                out.setdefault(key.group(1), f"{rel}:{number}")
    return out


def findings(ops: dict, locations: dict[str, str] | None = None) -> list[str]:
    """M1-M3 over the manifest entries *ops*, each finding at its entry's *locations* key."""
    locations = locations or {}

    def at(name: str) -> str:
        return locations.get(name, "manifest")

    out = []
    for name in sorted(ops):
        for right, wrong in ABBREVIATIONS.items():
            if re.search(wrong, name):
                out.append(f"{at(name)}: M1 {name}: spell this abbreviation {right}")
        for word in AGENTIVE:
            if word in name and name not in AGENTIVE_EXCEPTIONS:
                out.append(f"{at(name)}: M2 {name}: {word!r} names an actor, not the step")

    shared = defaultdict(list)
    for name, entry in ops.items():
        if entry.get("ref_api"):
            shared[entry["ref_api"]].append(name)
    for ref, names in sorted(shared.items()):
        if len(names) < 2 or ref in DEFAULTED_SWITCH:
            continue
        stem = "".join(p.capitalize() for p in ref.split(".")[-1].split("_"))
        plain = [n for n in names if n.startswith(stem) and n[len(stem) :] in ("FwdOp", "BwdOp")]
        for name in sorted(plain):
            out.append(
                f"{at(name)}: M3 {name}: {ref} has no default side, so this name has to say "
                f"which of {sorted(names)} it is"
            )
    return out


CLASS_ROOTS = ("src/tileops", "workloads")


def class_findings() -> list[str]:
    """M4 over every class under CLASS_ROOTS, by the spellings M1 already fixes."""
    out = []
    for root in CLASS_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for right, wrong in ABBREVIATIONS.items():
                    if re.search(wrong, node.name):
                        out.append(
                            f"{relative(path)}:{node.lineno}: M4 {node.name}: spell this "
                            f"abbreviation {right}"
                        )
    return out


def main() -> int:
    return report(findings(load_manifest(), entry_locations()) + class_findings())


if __name__ == "__main__":
    raise SystemExit(main())
