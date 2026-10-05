#!/usr/bin/env python3
"""Lint manifest op names against the rules that hold over every entry.

Three rules, each stated so it decides every entry on its own:

M1  One spelling per abbreviation. The stem of an entry that names a `ref_api` comes from
    that API (`rms_norm` -> `RmsNorm`), and this rule then fixes how its abbreviations are
    written, so M1 never competes for the stem.
M2  No agentive noun. A name says what the step computes, not who computes it.
    `AGENTIVE_EXCEPTIONS` is an explicit allowlist: what this lint decides is membership
    in it, nothing more. Putting an entry on it is a review decision, recorded there with
    its reason, not something the rule derives.
M3  Entries sharing one `ref_api` each carry the word that tells them apart, unless the
    torch API's own switch has a default: the plain name is then the default side.
M4  M1 again, over the class definitions a caller reads. M1 reaches manifest entries only,
    and a spelling that holds there drifts in the layers underneath unless something checks
    them too. M4 checks one thing, how an abbreviation is written, and leaves every other
    naming question to review.

A name no rule reaches is left alone. Most entries are in that position: a little over
half name no `ref_api` at all, and for those neither an upstream name nor a sibling
constrains the choice.

Usage: ``op_naming_lint.py``. Exits 1 on a finding.
"""

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "src/tileops/manifest/spec"

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
# Allowed by review, with the reason. Not derived from anything; adding a line is a
# decision, and the reason is what a later reader weighs it against.
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


def entries() -> dict:
    ops = {}
    for path in sorted(SPEC.glob("*.yaml")):
        if path.name == "types.yaml":
            continue
        ops.update(yaml.safe_load(path.read_text()) or {})
    return ops


def findings(ops: dict) -> list[str]:
    out = []
    for name in sorted(ops):
        for right, wrong in ABBREVIATIONS.items():
            if re.search(wrong, name):
                out.append(f"M1 {name}: spell this abbreviation {right}")
        for word in AGENTIVE:
            if word in name and name not in AGENTIVE_EXCEPTIONS:
                out.append(f"M2 {name}: {word!r} names an actor, not the step")

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
                f"M3 {name}: {ref} has no default side, so this name has to say which "
                f"of {sorted(names)} it is"
            )
    return out


CLASS_ROOTS = ("src/tileops", "workloads")


def class_findings() -> list[str]:
    """M4 over every class under CLASS_ROOTS, by the spellings M1 already fixes.

    Every class, not a suffix-filtered subset: a base class and a bare `FusedMoE` carry the
    same abbreviations as the kernels beside them.
    """
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
                        rel = path.relative_to(REPO_ROOT)
                        out.append(
                            f"M4 {rel}:{node.lineno} {node.name}: spell this abbreviation {right}"
                        )
    return out


def main() -> int:
    found = findings(entries()) + class_findings()
    for line in found:
        print(line, file=sys.stderr)
    if found:
        print(f"\n{len(found)} naming finding(s)", file=sys.stderr)
        return 1
    print("op naming: no findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
