"""Benchmark for the gated pooling compressor op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and byte counts come
from the op's ``eval_roofline()`` via :class:`benchmarks.api.Runner`.

The row is timed against the workload's batched torch program. No library call returns this
op's outputs and rings. SGLang's DeepSeek-V4 ``compress_forward`` pools with the same
arithmetic, but its state is pages interleaving kv with the score before ape, and it
returns only the closing rows. vLLM's DeepSeek-V4 compressor keeps every token in a paged
state, and its kernels fuse the following RMSNorm, RoPE, quantization and cache write, or,
for ratio 128, write a float32 row for every token. The torch program reads the closing
tokens' count on the host, so inductor has no single graph of it.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import private_inputs
from tileops.attention import GatedPoolCompressFwdOp


@pytest.mark.parametrize("case", bench.cases(GatedPoolCompressFwdOp), ids=lambda case: case.id)
def test_gated_pool_compress_bench(case) -> None:
    op = GatedPoolCompressFwdOp(**case.arguments)
    # The torch program writes the rings it is given, as the op does: on private copies.
    bench.Runner(op, case).compare({"torch": private_inputs(case.torch, case.inputs, 3, 4)})
