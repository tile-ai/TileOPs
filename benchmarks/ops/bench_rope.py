"""Benchmarks for the RoPE op family.

Every case is a manifest call of the op (``src/tileops/manifest/spec/rope.yaml``),
the op is built from the call's parameters and the inputs are the call's tensors.

One ``test_*_bench`` per op, so every op this file is declared the benchmark
of records a row of its own.

Baselines build their cos/sin tables outside the timed window, so only the
rotation itself is measured.
"""

import pytest
import torch

from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    flashinfer_op,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.rope import (
    LongRoPEFwdOp,
    RoPEFwdOp,
    RoPELlama31FwdOp,
    RoPENeoxPositionIdsFwdOp,
    YaRNFwdOp,
)
from workloads.rope import (
    RoPECall,
    llama31_frequency_tables,
    longrope_frequency_tables,
    rope_frequency_tables,
    yarn_frequency_tables,
)


def _rope_tables(seq_len: int, head_dim: int, dtype: torch.dtype, *, base: float = 10000.0):
    cos, sin = rope_frequency_tables(head_dim, seq_len, base=base, dtype=dtype)
    return torch.cat([cos, cos], dim=-1), torch.cat([sin, sin], dim=-1)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return (x.float() * cos.float() + torch.cat((-x2, x1), dim=-1).float() * sin.float()).to(
        x.dtype
    )


def _rotate_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each adjacent pair, reading the first half of the doubled tables."""
    half = x.shape[-1] // 2
    cos_pairs, sin_pairs = cos[..., :half], sin[..., :half]
    pairs = x.unflatten(-1, (half, 2))
    even, odd = pairs[..., 0].float(), pairs[..., 1].float()
    cos_pairs, sin_pairs = cos_pairs.float(), sin_pairs.float()
    rotated = torch.stack(
        (even * cos_pairs - odd * sin_pairs, odd * cos_pairs + even * sin_pairs), dim=-1
    )
    return rotated.flatten(-2).to(x.dtype)


def _vllm_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    """Return vllm's rotary_embedding, the arguments it rotates, and their refill.

    It rewrites its query in place and takes it flattened to
    ``[num_tokens, num_heads * head_dim]``, so it gets its own copy, refilled before each
    call; its cache is
    the half-width cos and sin concatenated, not the doubled tables the reference
    indexes; and ``is_neox=True`` is the same half-split rotation.
    """
    fn = vllm_op("rotary_embedding")
    num_tokens = x.shape[0]
    half = head_dim // 2
    cache = torch.cat([cos[:, :half], sin[:, :half]], dim=-1).contiguous()
    positions = position_ids.long()
    query = x.reshape(num_tokens, -1).clone()

    def refill():
        query.copy_(x.reshape_as(query))

    def baseline_fn(positions_i, query_i):
        fn(positions_i, query_i, None, head_dim, cache, True)
        return query_i

    return baseline_fn, (positions, query), refill


def _bench_rope(op_cls, call) -> None:
    """Check and time the rotation using this variant's independent frequency tables."""
    workload = RoPECall(call)
    tensors = workload.tensors
    op = op_cls(**call.arguments(tensors))
    bm = ManifestBenchmark(op, workload)
    x = tensors["x"]
    input_layout = call.params["input_layout"]
    seq_len = x.shape[0] if input_layout == "1d" else x.shape[1]
    table_fn = {
        RoPEFwdOp: rope_frequency_tables,
        RoPELlama31FwdOp: llama31_frequency_tables,
        YaRNFwdOp: yarn_frequency_tables,
        LongRoPEFwdOp: longrope_frequency_tables,
    }[op_cls]
    parameters = {k: v for k, v in call.params.items() if k not in ("input_layout", "rope_layout")}
    if "rescale_factors" in tensors:
        parameters["rescale_factors"] = tensors["rescale_factors"]
    cos, sin = table_fn(x.shape[-1], seq_len, dtype=x.dtype, device=x.device, **parameters)
    cos, sin = (torch.cat([table, table], dim=-1) for table in (cos, sin))
    if input_layout != "1d":
        cos, sin = (t.view(1, seq_len, 1, x.shape[-1]) for t in (cos, sin))
    rotate = _rotate if call.params.get("rope_layout", "neox") == "neox" else _rotate_interleaved

    def baseline_fn(t):
        return rotate(t, cos, sin)

    apply_rope = flashinfer_op("rope.apply_rope_with_cos_sin_cache")
    dim = x.shape[-1]
    cache = torch.cat(
        (cos.reshape(seq_len, dim)[:, : dim // 2], sin.reshape(seq_len, dim)[:, : dim // 2]), -1
    ).float()
    positions = torch.arange(seq_len, dtype=torch.int32, device=x.device)
    if input_layout != "1d":
        positions = positions.repeat(x.shape[0])
    scratch = torch.empty((positions.numel(), dim), dtype=x.dtype, device=x.device)
    neox = call.params.get("rope_layout", "neox") == "neox"

    def flashinfer_fn(x):
        rotated, _ = apply_rope(
            positions, x.reshape(positions.numel(), -1), scratch, dim, cache, neox
        )
        return rotated.reshape_as(x)

    functors = {
        "tileops": op,
        "torch-ref": baseline_fn,
        TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
    }
    if x.dtype in (torch.float16, torch.bfloat16):
        functors[FLASHINFER_TAG] = flashinfer_fn
    bm.compare(functors, x)


@pytest.mark.parametrize("call", manifest_calls(RoPEFwdOp))
def test_rope_bench(call) -> None:
    _bench_rope(RoPEFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(RoPELlama31FwdOp))
def test_rope_llama31_bench(call) -> None:
    _bench_rope(RoPELlama31FwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(YaRNFwdOp))
def test_yarn_bench(call) -> None:
    _bench_rope(YaRNFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(LongRoPEFwdOp))
def test_rope_longrope_bench(call) -> None:
    _bench_rope(LongRoPEFwdOp, call)


@pytest.mark.parametrize("call", manifest_calls(RoPENeoxPositionIdsFwdOp))
def test_rope_neox_position_ids_bench(call) -> None:
    workload = RoPECall(call)
    tensors = workload.tensors
    x, position_ids = (tensors["x"], tensors["position_ids"])
    head_dim = x.shape[-1]
    op = RoPENeoxPositionIdsFwdOp(**call.arguments(tensors))
    bm = ManifestBenchmark(op, workload)
    cos, sin = _rope_tables(
        call.params["max_position"], head_dim, x.dtype, base=call.params["base"]
    )

    def baseline_fn(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        idx = pos.long()
        return _rotate(t, cos[idx].unsqueeze(1), sin[idx].unsqueeze(1))

    vllm_fn, vllm_args, vllm_refill = _vllm_rope(x, position_ids, head_dim, cos, sin)
    vllm_shaped = (lambda *a, _f=vllm_fn: _f(*a).view(x.shape), vllm_args, vllm_refill)
    bm.compare(
        {
            "tileops": op,
            VLLM_TAG: vllm_shaped,
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        },
        x,
        position_ids,
    )
