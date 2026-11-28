"""Benchmarks for the RoPE op family.

Baselines build their cos/sin caches outside the timed window, so only the
rotation itself is measured.
"""

import dataclasses

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    baseline_module,
    compiled_reference,
    private_inputs,
    vllm_op,
)
from tileops._ops.rope import (
    LongRoPEFwdOp,
    RoPECosSinFwdOp,
    RoPEFwdOp,
    RoPELlama31FwdOp,
    YaRNFwdOp,
)
from workloads.rope import rope_reference, rope_tables


def _vllm_rope(case: bench.Case, cache: torch.Tensor) -> bench.Implementation:
    """vllm's rotary_embedding on a private query and key it rewrites, restored every round.

    The cache is ``cos ‖ sin`` in the query's dtype, indexed by the effective rows.
    """
    fn = vllm_op("rotary_embedding")
    positions, query, key, *rest = (*case.inputs, None)[:4]
    index = (positions if not rest or rest[0] is None else positions + rest[0]).long()
    params = case.arguments

    def run(index_i, q, k):
        fn(index_i, q, k, params["head_size"], cache, params["is_neox_style"])
        return q, k

    return private_inputs(run, (index, query, key), 1, 2)


def _paired(case: bench.Case, run, writes: frozenset[str]) -> bench.Implementation:
    """*run*, which writes the inputs *writes*, compared with the call only where the call
    writes the same inputs, as the manifest marks them; otherwise timed but not compared."""
    impl = run if isinstance(run, bench.Implementation) else bench.Implementation(run=run)
    if writes == case._call.written:
        return impl
    reason = (
        f"writes {', '.join(sorted(writes)) or 'no input'}; "
        f"this call writes {', '.join(sorted(case._call.written)) or 'no input'}"
    )
    return dataclasses.replace(impl, noncomparable_reason=reason)


def _bench_rope(op_cls, case: bench.Case) -> None:
    """Check and time the rotation against vLLM's kernel, torch and inductor."""
    op = op_cls(**case.arguments)
    params = case.arguments
    cos, sin = rope_tables(op_cls.__name__, params, case.inputs[1])
    cache = torch.cat((cos, sin), dim=-1)

    def reference(*inputs):
        return rope_reference(params, cos, sin, *inputs)

    # vLLM rewrites query and key in place; the torch implementations return new tensors.
    rewritten = frozenset(
        n for n, t in zip(("query", "key"), case.inputs[1:3], strict=True) if t is not None
    )
    implementations = {
        "torch-ref": _paired(case, reference, frozenset()),
        TORCH_COMPILE_TAG: _paired(case, compiled_reference(reference), frozenset()),
        VLLM_TAG: _paired(case, _vllm_rope(case, cache), rewritten),
    }
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(RoPEFwdOp))
def test_rope_bench(case) -> None:
    _bench_rope(RoPEFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(RoPELlama31FwdOp))
def test_rope_llama31_bench(case) -> None:
    _bench_rope(RoPELlama31FwdOp, case)


@pytest.mark.parametrize("case", bench.cases(YaRNFwdOp))
def test_yarn_bench(case) -> None:
    _bench_rope(YaRNFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(LongRoPEFwdOp))
def test_rope_longrope_bench(case) -> None:
    _bench_rope(LongRoPEFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(RoPECosSinFwdOp))
def test_rope_cos_sin_bench(case) -> None:
    """Check and time the rotation against flash-attn's ``apply_rotary_emb``, the op's
    reference API, torch and inductor; each rewrites a private ``x`` where the call is in
    place."""
    op = RoPECosSinFwdOp(**case.arguments)
    params = case.arguments
    apply_rotary_emb = baseline_module("flash_attn.layers.rotary", "flash-attn").apply_rotary_emb

    def flash_attention_fn(x, cos, sin, cu_seqlens, seqlen_offsets):
        return apply_rotary_emb(
            x,
            cos,
            sin,
            interleaved=params["interleaved"],
            inplace=params["inplace"],
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=params["max_seqlen"],
        )

    implementations = {
        "flash-attn": flash_attention_fn,
        "torch-ref": case.torch,
        TORCH_COMPILE_TAG: compiled_reference(case.torch),
    }
    if params["inplace"]:
        implementations = {
            name: private_inputs(fn, case.inputs, 0) for name, fn in implementations.items()
        }
    bench.Runner(op, case).compare(implementations)
