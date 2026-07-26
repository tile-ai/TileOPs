"""Benchmarks for the RoPE op family.

Workload shapes, dtypes, layouts, and roofline formulas are loaded from the
ops manifest (``tileops/manifest/position_encoding.yaml``); nothing about a
workload is hard-coded here.

Each op gets its own ``test_*_bench`` function so the manifest validator's
per-op AST check (``scripts/validate_manifest.py`` → ``check_l4_benchmark``)
can tie ``load_workloads("<OpName>")`` / ``ManifestBenchmark("<OpName>", ...)``
calls one-to-one to the manifest entry.

Baselines are bench-local PyTorch rotations applied to pre-computed cos/sin
tables. The tables are built outside the timed window, so the timed baseline
measures only the rotation itself.
"""

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops.rope import (
    RopeLlama31Op,
    RopeLongRopeOp,
    RopeNeoxOp,
    RopeNeoxPositionIdsOp,
    RopeNonNeoxOp,
    RopeYarnOp,
)

# Bench-local: manifest workload entries carry no ``base``; the ops and the
# baseline both use the manifest signature default (``base: 10000.0``).
_BASE = 10000.0


class _RopeWorkload:
    """Minimal :class:`ShapeDtypeWorkload` for the RoPE family.

    Holds ``shape`` and ``dtype`` so :class:`ManifestBenchmark` can call
    ``op.eval_roofline()`` after ``forward()`` has bound the dynamic vars.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


def _mark(idx: int):
    """First manifest workload of an op is the smoke case; the rest are full."""
    return pytest.mark.smoke if idx == 0 else pytest.mark.full


def _layout_params(workloads: list[dict]) -> list:
    """Build ``(shape, dtype, layout)`` params for the 1d/2d RoPE variants."""
    params = []
    for idx, w in enumerate(workloads):
        layout = w["layout"]
        if layout == "1d":
            shape = (w["seq_len"], w["head_dim"])
        else:
            shape = (w["batch"], w["seq_len"], w["num_heads"], w["head_dim"])
        for dtype_name in w["dtypes"]:
            params.append(pytest.param(
                shape, getattr(torch, dtype_name), layout,
                id=f"{w['label']}-{dtype_name}",
                marks=_mark(idx),
            ))
    return params


def _position_ids_params(workloads: list[dict]) -> list:
    """Build ``(shape, dtype, max_position)`` params for the THD variant."""
    params = []
    for idx, w in enumerate(workloads):
        shape = (w["num_tokens"], w["num_heads"], w["head_dim"])
        for dtype_name in w["dtypes"]:
            params.append(pytest.param(
                shape, getattr(torch, dtype_name), w["max_position"],
                id=f"{w['label']}-{dtype_name}",
                marks=_mark(idx),
            ))
    return params


# Bench-local PyTorch baselines


def _rope_tables(
    seq_len: int, head_dim: int, dtype: torch.dtype, *, interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute full-width cos/sin tables of shape ``(seq_len, head_dim)``.

    ``interleaved`` selects the adjacent-pair (RoFormer) table layout; the
    default is the half-split (GPT-NeoX) layout. Frequency *values* are
    variant-specific, but the timed rotation cost depends only on the table
    geometry, which every variant shares.
    """
    half = head_dim // 2
    freqs = 1.0 / (
        _BASE ** (torch.arange(0, half, device="cuda", dtype=torch.float32) / half)
    )
    angles = torch.outer(
        torch.arange(seq_len, device="cuda", dtype=torch.float32), freqs,
    )
    cos, sin = torch.cos(angles), torch.sin(angles)
    if interleaved:
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
    else:
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
    return cos.to(dtype), sin.to(dtype)


def _apply_neox(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """Half-split (GPT-NeoX) rotation."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _apply_non_neox(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """Adjacent-pair (RoFormer) rotation."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rotated * sin


def _profile_rope(
    op,
    bm: ManifestBenchmark,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    layout: str,
    *,
    interleaved: bool = False,
) -> None:
    """Profile op and the bench-local torch rotation on the same input.

    ``ManifestBenchmark`` is constructed at each per-op test's call site with
    the literal op-name constant so the validator AST check can match it; this
    helper only handles input generation and the profile + record pairs.
    """
    x = torch.randn(shape, device="cuda", dtype=dtype)
    params = {"shape": shape, "dtype": dtype, "layout": layout}

    result = bm.profile(op, x)
    BenchmarkReport.record(op, params, result, tag="tileops")

    seq_len = shape[0] if layout == "1d" else shape[1]
    head_dim = shape[-1]
    cos, sin = _rope_tables(seq_len, head_dim, dtype, interleaved=interleaved)
    if layout != "1d":
        cos = cos.view(1, seq_len, 1, head_dim)
        sin = sin.view(1, seq_len, 1, head_dim)
    apply_fn = _apply_non_neox if interleaved else _apply_neox

    def baseline_fn(t: torch.Tensor) -> torch.Tensor:
        return apply_fn(t, cos, sin)

    result_bl = bm.profile(baseline_fn, x)
    BenchmarkReport.record(op, params, result_bl, tag="torch-ref")


# Per-op tests — one block per manifest entry.

_NEOX_OP = "RopeNeoxOp"


@pytest.mark.parametrize(
    "shape, dtype, layout", _layout_params(load_workloads(_NEOX_OP)),
)
def test_rope_neox_bench(
    shape: tuple[int, ...], dtype: torch.dtype, layout: str,
) -> None:
    op = RopeNeoxOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(_NEOX_OP, op, _RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout)


_NON_NEOX_OP = "RopeNonNeoxOp"


@pytest.mark.parametrize(
    "shape, dtype, layout", _layout_params(load_workloads(_NON_NEOX_OP)),
)
def test_rope_non_neox_bench(
    shape: tuple[int, ...], dtype: torch.dtype, layout: str,
) -> None:
    op = RopeNonNeoxOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(_NON_NEOX_OP, op, _RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout, interleaved=True)


_LLAMA31_OP = "RopeLlama31Op"


@pytest.mark.parametrize(
    "shape, dtype, layout", _layout_params(load_workloads(_LLAMA31_OP)),
)
def test_rope_llama31_bench(
    shape: tuple[int, ...], dtype: torch.dtype, layout: str,
) -> None:
    op = RopeLlama31Op(layout=layout, base=_BASE)
    bm = ManifestBenchmark(_LLAMA31_OP, op, _RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout)


_YARN_OP = "RopeYarnOp"


@pytest.mark.parametrize(
    "shape, dtype, layout", _layout_params(load_workloads(_YARN_OP)),
)
def test_rope_yarn_bench(
    shape: tuple[int, ...], dtype: torch.dtype, layout: str,
) -> None:
    op = RopeYarnOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(_YARN_OP, op, _RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout)


_LONGROPE_OP = "RopeLongRopeOp"


@pytest.mark.parametrize(
    "shape, dtype, layout", _layout_params(load_workloads(_LONGROPE_OP)),
)
def test_rope_longrope_bench(
    shape: tuple[int, ...], dtype: torch.dtype, layout: str,
) -> None:
    op = RopeLongRopeOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(_LONGROPE_OP, op, _RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout)


_POSITION_IDS_OP = "RopeNeoxPositionIdsOp"


@pytest.mark.parametrize(
    "shape, dtype, max_position",
    _position_ids_params(load_workloads(_POSITION_IDS_OP)),
)
def test_rope_neox_position_ids_bench(
    shape: tuple[int, int, int], dtype: torch.dtype, max_position: int,
) -> None:
    num_tokens, _, head_dim = shape
    x = torch.randn(shape, device="cuda", dtype=dtype)
    position_ids = torch.arange(
        num_tokens, device="cuda", dtype=torch.int32,
    ) % max_position

    op = RopeNeoxPositionIdsOp(max_position=max_position, base=_BASE)
    bm = ManifestBenchmark(_POSITION_IDS_OP, op, _RopeWorkload(shape, dtype))
    params = {"shape": shape, "dtype": dtype, "max_position": max_position}

    result = bm.profile(op, x, position_ids)
    BenchmarkReport.record(op, params, result, tag="tileops")

    cos, sin = _rope_tables(max_position, head_dim, dtype, interleaved=False)

    def baseline_fn(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        idx = pos.long()
        return _apply_neox(t, cos[idx].unsqueeze(1), sin[idx].unsqueeze(1))

    result_bl = bm.profile(baseline_fn, x, position_ids)
    BenchmarkReport.record(op, params, result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
