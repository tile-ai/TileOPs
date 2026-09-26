"""Benchmarks for the RoPE op family.

Every case is a manifest call of the op (``src/tileops/manifest/position_encoding.yaml``),
the op is built from the call's parameters and the inputs are the call's tensors.

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
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.rope import (
    RopeLlama31FwdOp,
    RopeLongRopeFwdOp,
    RopeNeoxFwdOp,
    RopeNeoxPositionIdsFwdOp,
    RopeNonNeoxFwdOp,
    RopeYarnFwdOp,
)
from workloads.workload_base import CallWorkload

# The torch baseline's frequency base; the rows run every op at its default base.
_BASE = 10000.0


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


def _bench_rope(op_cls, call) -> None:
    """Profile the op on one manifest call against the torch rotation baseline."""
    workload = CallWorkload(call)
    tensors = call.materialize("cuda")
    op = op_cls(**call.arguments(tensors))
    bm = ManifestBenchmark(op, workload)
    x = tensors["x"]
    layout = call.params["layout"]
    seq_len = x.shape[0] if layout == "1d" else x.shape[1]
    cos, sin = _rope_tables(seq_len, x.shape[-1], x.dtype)
    if layout != "1d":
        cos, sin = (t.view(1, seq_len, 1, x.shape[-1]) for t in (cos, sin))

    def baseline_fn(t):
        return _rotate(t, cos, sin)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        x,
    )


@pytest.mark.parametrize("call", manifest_calls(RopeNeoxFwdOp))
def test_rope_neox_bench(call) -> None:
    _bench_rope(RopeNeoxFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(RopeNonNeoxFwdOp))
def test_rope_non_neox_bench(call) -> None:
    _bench_rope(RopeNonNeoxFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(RopeLlama31FwdOp))
def test_rope_llama31_bench(call) -> None:
    _bench_rope(RopeLlama31FwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(RopeYarnFwdOp))
def test_rope_yarn_bench(call) -> None:
    _bench_rope(RopeYarnFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(RopeLongRopeFwdOp))
def test_rope_longrope_bench(call) -> None:
    _bench_rope(RopeLongRopeFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(RopeNeoxPositionIdsFwdOp))
def test_rope_neox_position_ids_bench(call) -> None:
    workload = CallWorkload(call)
    tensors = call.materialize("cuda")
    x, position_ids = tensors["x"], tensors["position_ids"]
    head_dim = x.shape[-1]
    op = RopeNeoxPositionIdsFwdOp(**call.arguments(tensors))
    bm = ManifestBenchmark(op, workload)

    cos, sin = _rope_tables(call.params["max_position"], head_dim, x.dtype)

    def baseline_fn(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        idx = pos.long()
        return _rotate(t, cos[idx].unsqueeze(1), sin[idx].unsqueeze(1))

    # vllm rotates in fp32 and rounds once, the reference in the storage dtype, so they
    # agree to one rounding step of the storage dtype, which is what the default
    # tolerances allow.
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
    )
