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

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    flashinfer_op,
    vllm_op,
)
from tileops.ops.rope import (
    LongRoPEFwdOp,
    RoPEFwdOp,
    RoPELlama31FwdOp,
    RoPENeoxPositionIdsFwdOp,
    YaRNFwdOp,
)
from workloads.rope import (
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
) -> bench.Implementation:
    """vllm's rotary_embedding on a private query it rewrites, restored before every round.

    vllm takes the query as ``[num_tokens, num_heads * head_dim]`` and a cache of the
    half-width cos and sin concatenated; ``is_neox=True`` is the half-split rotation.
    """
    fn = vllm_op("rotary_embedding")
    num_tokens = x.shape[0]
    half = head_dim // 2
    cache = torch.cat([cos[:, :half], sin[:, :half]], dim=-1).contiguous()
    positions = position_ids.long()
    query = x.reshape(num_tokens, -1).clone()

    def reset() -> None:
        query.copy_(x.reshape_as(query))

    def run(positions_i, query_i):
        fn(positions_i, query_i, None, head_dim, cache, True)
        return query_i.view(x.shape)

    return bench.Implementation(run=run, args=(positions, query), reset=reset)


def _bench_rope(op_cls, case: bench.Case) -> None:
    """Check and time the rotation using this variant's independent frequency tables."""
    params = case.arguments
    op = op_cls(**params)
    (x,) = case.inputs
    input_layout = params["input_layout"]
    seq_len = x.shape[0] if input_layout == "1d" else x.shape[1]
    table_fn = {
        RoPEFwdOp: rope_frequency_tables,
        RoPELlama31FwdOp: llama31_frequency_tables,
        YaRNFwdOp: yarn_frequency_tables,
        LongRoPEFwdOp: longrope_frequency_tables,
    }[op_cls]
    parameters = {k: v for k, v in params.items() if k not in ("input_layout", "rope_layout")}
    cos, sin = table_fn(x.shape[-1], seq_len, dtype=x.dtype, device=x.device, **parameters)
    cos, sin = (torch.cat([table, table], dim=-1) for table in (cos, sin))
    if input_layout != "1d":
        cos, sin = (t.view(1, seq_len, 1, x.shape[-1]) for t in (cos, sin))
    rotate = _rotate if params.get("rope_layout", "neox") == "neox" else _rotate_interleaved

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
    neox = params.get("rope_layout", "neox") == "neox"

    def flashinfer_fn(x):
        rotated, _ = apply_rope(
            positions, x.reshape(positions.numel(), -1), scratch, dim, cache, neox
        )
        return rotated.reshape_as(x)

    implementations = {
        "torch-ref": baseline_fn,
        TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
    }
    if x.dtype in (torch.float16, torch.bfloat16):
        implementations[FLASHINFER_TAG] = flashinfer_fn
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(RoPEFwdOp), ids=lambda case: case.id)
def test_rope_bench(case) -> None:
    _bench_rope(RoPEFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(RoPELlama31FwdOp), ids=lambda case: case.id)
def test_rope_llama31_bench(case) -> None:
    _bench_rope(RoPELlama31FwdOp, case)


@pytest.mark.parametrize("case", bench.cases(YaRNFwdOp), ids=lambda case: case.id)
def test_yarn_bench(case) -> None:
    _bench_rope(YaRNFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LongRoPEFwdOp), ids=lambda case: case.id)
def test_rope_longrope_bench(case) -> None:
    _bench_rope(LongRoPEFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(RoPENeoxPositionIdsFwdOp), ids=lambda case: case.id)
def test_rope_neox_position_ids_bench(case) -> None:
    x, position_ids = case.inputs
    head_dim = x.shape[-1]
    op = RoPENeoxPositionIdsFwdOp(**case.arguments)
    cos, sin = _rope_tables(
        case.params["max_position"], head_dim, x.dtype, base=case.params["base"]
    )

    def baseline_fn(t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        idx = pos.long()
        return _rotate(t, cos[idx].unsqueeze(1), sin[idx].unsqueeze(1))

    bench.Runner(op, case).compare(
        {
            VLLM_TAG: _vllm_rope(x, position_ids, head_dim, cos, sin),
            "torch-ref": baseline_fn,
            TORCH_COMPILE_TAG: compiled_reference(baseline_fn),
        }
    )
