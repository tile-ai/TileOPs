"""Generate a before/after FLOP/byte table for the per-element FLOP
convention rollout (``docs/design/roofline.md`` §1.3).

Pure formula evaluation — no GPU, no kernel JIT. The "after" column
loads the affected manifest entries from this checkout and evaluates
each entry's ``roofline.flops`` / ``roofline.bytes`` expression on the
representative workload shape. The "before" column hard-codes the
coefficients that lived on the same entries on ``upstream/testbed``
immediately before the convention commit (see ``git log`` /
``git diff upstream/testbed`` on this branch).

Usage:
    python scripts/perf/flop_convention_delta.py \
        --out docs/perf/flop_convention_delta.csv

Reproducible from a clean checkout: the script only reads YAML and
evaluates simple Python expressions; it does not import ``tileops.ops``
and so does not trigger TileLang JIT compilation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "tileops" / "manifest"


def _product(seq) -> int:
    out = 1
    for v in seq:
        out *= int(v)
    return out


def _load_op(family_file: str, op_name: str) -> dict:
    with (MANIFEST_DIR / family_file).open() as f:
        data = yaml.safe_load(f)
    return data[op_name]


class _Shape:
    """Minimal stand-in for a tensor exposing ``.shape`` and ``.ndim``."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)


def _eval_inline(roofline: dict, tensor_shapes: Mapping[str, tuple[int, ...]],
                 elem_bytes: int) -> tuple[int, int]:
    """Evaluate ``vars`` then ``flops`` / ``bytes`` in a sealed namespace."""
    ns: dict = {
        "product": _product,
        "len": len,
        "min": min,
        "max": max,
        "int": int,
        "elem_bytes": elem_bytes,
    }
    for name, shape in tensor_shapes.items():
        ns[name] = _Shape(shape)
    for vname, vexpr in (roofline.get("vars") or {}).items():
        ns[vname] = eval(vexpr, {"__builtins__": {}}, ns)  # noqa: S307
    flops = int(eval(roofline["flops"], {"__builtins__": {}}, ns))  # noqa: S307
    nbytes = int(eval(roofline["bytes"], {"__builtins__": {}}, ns))  # noqa: S307
    return flops, nbytes


# --- "before" coefficients captured from ``upstream/testbed`` ---
# These are the FLOP coefficients on the named entries before the
# convention commit. Bytes formulas were not changed by the convention,
# so the byte column is computed once from the current expression.
_BEFORE_FLOPS_COEFF: dict[str, int] = {
    "ReluFwdOp": 2,        # was "2 * N"
    "HardtanhFwdOp": 4,    # was "4 * N"
    "ClampFwdOp_Nmult": 2,  # was "2 * n_total" inside clamp_fwd_roofline
}


@dataclass(frozen=True)
class Row:
    family: str
    op_name: str
    label: str
    shape: tuple[int, ...]
    dtype_name: str
    elem_bytes: int
    flops_before: int
    flops_after: int
    bytes_before: int
    bytes_after: int


def _row_activation() -> Row:
    """ReLU on the Llama-3.1-8B prefill hidden state."""
    op = _load_op("elementwise_unary_activation.yaml", "ReluFwdOp")
    shape = (2048, 4096)
    n = _product(shape)
    flops_after, bytes_after = _eval_inline(
        op["roofline"], {"input": shape}, elem_bytes=2,
    )
    return Row(
        family="activation",
        op_name="ReluFwdOp",
        label="hidden-state-prefill",
        shape=shape,
        dtype_name="float16",
        elem_bytes=2,
        flops_before=_BEFORE_FLOPS_COEFF["ReluFwdOp"] * n,
        flops_after=flops_after,
        bytes_before=bytes_after,  # bytes formula unchanged
        bytes_after=bytes_after,
    )


def _row_clamp_scalar() -> Row:
    """Hardtanh (scalar 2-sided clamp) on the same shape."""
    op = _load_op("elementwise_unary_activation.yaml", "HardtanhFwdOp")
    shape = (2048, 4096)
    n = _product(shape)
    flops_after, bytes_after = _eval_inline(
        op["roofline"], {"input": shape}, elem_bytes=2,
    )
    return Row(
        family="clamp",
        op_name="HardtanhFwdOp",
        label="hidden-state-prefill",
        shape=shape,
        dtype_name="float16",
        elem_bytes=2,
        flops_before=_BEFORE_FLOPS_COEFF["HardtanhFwdOp"] * n,
        flops_after=flops_after,
        bytes_before=bytes_after,
        bytes_after=bytes_after,
    )


def _row_clamp_tensor() -> Row:
    """Tensor-bound 2-sided clamp (func mode in formulas.py)."""
    shape = (4096, 4096)
    n = _product(shape)
    elem_bytes = 2
    # After convention: flops = N_total, bytes = 4 * N_total * elem_bytes.
    flops_after = n
    bytes_after = 4 * n * elem_bytes
    return Row(
        family="min-max",
        op_name="ClampFwdOp",
        label="elementwise-16M",
        shape=shape,
        dtype_name="float16",
        elem_bytes=elem_bytes,
        flops_before=_BEFORE_FLOPS_COEFF["ClampFwdOp_Nmult"] * n,
        flops_after=flops_after,
        bytes_before=bytes_after,
        bytes_after=bytes_after,
    )


def collect() -> list[Row]:
    return [_row_activation(), _row_clamp_scalar(), _row_clamp_tensor()]


def write_csv(rows: list[Row], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "family", "op", "label", "shape", "dtype",
            "flops_before", "flops_after", "flops_delta",
            "bytes_before", "bytes_after", "bytes_delta",
        ])
        for r in rows:
            w.writerow([
                r.family, r.op_name, r.label,
                "x".join(map(str, r.shape)), r.dtype_name,
                r.flops_before, r.flops_after,
                r.flops_after - r.flops_before,
                r.bytes_before, r.bytes_after,
                r.bytes_after - r.bytes_before,
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "perf" / "flop_convention_delta.csv",
    )
    args = parser.parse_args()
    rows = collect()
    write_csv(rows, args.out)
    for r in rows:
        print(
            f"{r.family:<10} {r.op_name:<16} {r.label:<22} "
            f"flops {r.flops_before:>12} -> {r.flops_after:<12} "
            f"bytes {r.bytes_before:>12} -> {r.bytes_after:<12}"
        )


if __name__ == "__main__":
    main()
