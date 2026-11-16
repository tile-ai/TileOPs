"""Benchmarks for the RoPE op family.

Every case is a manifest call of the op (``src/tileops/manifest/spec/rope.yaml``),
the op is built from the call's parameters and the inputs are the call's tensors.

One ``test_*_bench`` per op, so every op this file is declared the benchmark
of records a row of its own.

Baselines build their cos/sin caches outside the timed window, so only the
rotation itself is measured.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from tileops._ops.rope import LongRoPEFwdOp, RoPEFwdOp, RoPELlama31FwdOp, YaRNFwdOp
from workloads.rope import rope_reference, rope_tables


def _vllm_rope(case: bench.Case, cache: torch.Tensor) -> bench.Implementation:
    """vllm's rotary_embedding on a private query and key it rewrites, restored every round.

    The cache is ``cos ‖ sin`` in the query's dtype, indexed by the effective rows.
    """
    fn = vllm_op("rotary_embedding")
    positions, query, key, *rest = (*case.inputs, None)[:4]
    index = (positions if not rest or rest[0] is None else positions + rest[0]).long()
    params = case.arguments
    query_i = query.clone()
    key_i = None if key is None else key.clone()

    def reset() -> None:
        query_i.copy_(query)
        if key_i is not None:
            key_i.copy_(key)

    def run(index_i, q, k):
        fn(index_i, q, k, params["head_size"], cache, params["is_neox_style"])
        return q, k

    return bench.Implementation(run=run, args=(index, query_i, key_i), reset=reset)


def _bench_rope(op_cls, case: bench.Case) -> None:
    """Check and time the rotation against vLLM's kernel, torch and inductor."""
    op = op_cls(**case.arguments)
    params = case.arguments
    cos, sin = rope_tables(op_cls.__name__, params, case.inputs[1])
    cache = torch.cat((cos, sin), dim=-1)

    def reference(*inputs):
        return rope_reference(params, cos, sin, *inputs)

    implementations = {
        "torch-ref": reference,
        TORCH_COMPILE_TAG: compiled_reference(reference),
        VLLM_TAG: _vllm_rope(case, cache),
    }
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
