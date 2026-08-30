"""Benchmarks for the RoPE op family.

Workload shapes, dtypes, layouts, and roofline formulas are loaded from the
ops manifest (``src/tileops/manifest/position_encoding.yaml``); nothing about a
workload is hard-coded here.

One ``test_*_bench`` per op, so every op this file is declared the benchmark
of records a row of its own.

Baselines build their cos/sin tables outside the timed window, so only the
rotation itself is measured.
"""

import pytest
import torch

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads
from tileops.ops.rope import (
    RopeLlama31FwdOp,
    RopeLongRopeFwdOp,
    RopeNeoxFwdOp,
    RopeNeoxPositionIdsFwdOp,
    RopeNonNeoxFwdOp,
    RopeYarnFwdOp,
)

# Bench-local: manifest workload entries carry no ``base``; the ops and the
# baseline both use the manifest signature default (``base: 10000.0``).
_BASE = 10000.0


class RopeWorkload:
    """Minimal shape/dtype descriptor for the RoPE family.

    Holds ``shape`` and ``dtype`` so :class:`ManifestBenchmark` can call
    ``op.eval_roofline()`` after ``forward()`` has bound the dynamic vars.
    """

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


def _layout_args(w: dict, dtype: torch.dtype) -> tuple:
    """``(shape, dtype, layout)`` for the 1d/2d RoPE variants."""
    if w["layout"] == "1d":
        shape = (w["seq_len"], w["head_dim"])
    else:
        shape = (w["batch"], w["seq_len"], w["num_heads"], w["head_dim"])
    return (shape, dtype, w["layout"])


def _position_ids_args(w: dict, dtype: torch.dtype) -> tuple:
    return ((w["num_tokens"], w["num_heads"], w["head_dim"]), dtype, w["max_position"])


# Bench-local PyTorch baselines


def _rope_tables(seq_len: int, head_dim: int, dtype: torch.dtype):
    """Half-split cos/sin tables, shape ``(seq_len, head_dim)``.

    Frequency values are variant-specific, but the timed rotation cost depends
    only on table geometry, which every RoPE variant shares — so one baseline
    serves all of them.
    """
    half = head_dim // 2
    freqs = 1.0 / (_BASE ** (torch.arange(0, half, device="cuda", dtype=torch.float32) / half))
    angles = torch.outer(
        torch.arange(seq_len, device="cuda", dtype=torch.float32),
        freqs,
    )
    return (
        torch.cat([torch.cos(angles)] * 2, dim=-1).to(dtype),
        torch.cat([torch.sin(angles)] * 2, dim=-1).to(dtype),
    )


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


def _vllm_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    """Return vllm's rotary_embedding and the arguments it rotates.

    It rewrites its query in place and takes it flattened to
    ``[num_tokens, num_heads * head_dim]``, so it gets its own copy; its cache is
    the half-width cos and sin concatenated, not the doubled tables the reference
    indexes; and ``is_neox=True`` is the same half-split rotation.
    """
    fn = vllm_op("rotary_embedding")
    num_tokens = x.shape[0]
    half = head_dim // 2
    cache = torch.cat([cos[:, :half], sin[:, :half]], dim=-1).contiguous()
    positions = position_ids.long()
    query = x.reshape(num_tokens, -1).clone()

    def baseline_fn(positions_i, query_i):
        fn(positions_i, query_i, None, head_dim, cache, True)
        return query_i

    return baseline_fn, (positions, query)


def _profile_rope(
    op, bm: ManifestBenchmark, shape: tuple[int, ...], dtype: torch.dtype, layout: str
) -> None:
    """Profile op and the torch rotation baseline on the same input."""
    x = torch.randn(shape, device="cuda", dtype=dtype)
    params = {"shape": shape, "dtype": dtype, "layout": layout}

    seq_len = shape[0] if layout == "1d" else shape[1]
    cos, sin = _rope_tables(seq_len, shape[-1], dtype)
    if layout != "1d":
        cos, sin = (t.view(1, seq_len, 1, shape[-1]) for t in (cos, sin))

    def baseline_fn(t):
        return _rotate(t, cos, sin)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        x,
        params=params,
    )


# Per-op tests — one block per manifest entry.

_NEOX_OP = "RopeNeoxFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, layout",
    workload_params(load_workloads(_NEOX_OP), _layout_args, smoke_first=True),
)
def test_rope_neox_bench(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    layout: str,
) -> None:
    op = RopeNeoxFwdOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(op, RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout)


_NON_NEOX_OP = "RopeNonNeoxFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, layout",
    workload_params(load_workloads(_NON_NEOX_OP), _layout_args, smoke_first=True),
)
def test_rope_non_neox_bench(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    layout: str,
) -> None:
    op = RopeNonNeoxFwdOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(op, RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout)


_LLAMA31_OP = "RopeLlama31FwdOp"


@pytest.mark.parametrize(
    "shape, dtype, layout",
    workload_params(load_workloads(_LLAMA31_OP), _layout_args, smoke_first=True),
)
def test_rope_llama31_bench(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    layout: str,
) -> None:
    op = RopeLlama31FwdOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(op, RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout)


_YARN_OP = "RopeYarnFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, layout",
    workload_params(load_workloads(_YARN_OP), _layout_args, smoke_first=True),
)
def test_rope_yarn_bench(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    layout: str,
) -> None:
    op = RopeYarnFwdOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(op, RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout)


_LONGROPE_OP = "RopeLongRopeFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, layout",
    workload_params(load_workloads(_LONGROPE_OP), _layout_args, smoke_first=True),
)
def test_rope_longrope_bench(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    layout: str,
) -> None:
    op = RopeLongRopeFwdOp(layout=layout, base=_BASE)
    bm = ManifestBenchmark(op, RopeWorkload(shape, dtype))
    _profile_rope(op, bm, shape, dtype, layout)


_POSITION_IDS_OP = "RopeNeoxPositionIdsFwdOp"


@pytest.mark.parametrize(
    "shape, dtype, max_position",
    workload_params(load_workloads(_POSITION_IDS_OP), _position_ids_args, smoke_first=True),
)
def test_rope_neox_position_ids_bench(
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    max_position: int,
) -> None:
    num_tokens, _, head_dim = shape
    x = torch.randn(shape, device="cuda", dtype=dtype)
    position_ids = (
        torch.arange(
            num_tokens,
            device="cuda",
            dtype=torch.int32,
        )
        % max_position
    )

    op = RopeNeoxPositionIdsFwdOp(max_position=max_position, base=_BASE)
    bm = ManifestBenchmark(op, RopeWorkload(shape, dtype))
    params = {"shape": shape, "dtype": dtype, "max_position": max_position}

    cos, sin = _rope_tables(max_position, head_dim, dtype)

    def baseline_fn(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        idx = pos.long()
        return _rotate(t, cos[idx].unsqueeze(1), sin[idx].unsqueeze(1))

    # vllm rotates in fp32 and rounds once, the reference in the storage dtype, so they
    # agree to one rounding step: over the manifest's rows on an H200, max |Δ| 0.0039
    # in fp16 and 0.0156 in bf16.
    check_fn, check_args = _vllm_rope(x, position_ids, head_dim, cos, sin)
    torch.testing.assert_close(
        check_fn(*check_args).view(x.shape),
        baseline_fn(x, position_ids),
        rtol=1e-2,
        atol=2e-2,
    )
    vllm_fn, vllm_args = _vllm_rope(x, position_ids, head_dim, cos, sin)

    bm.compare(
        {
            "tileops": op,
            VLLM_TAG: (vllm_fn, vllm_args),
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        x,
        position_ids,
        params=params,
    )
