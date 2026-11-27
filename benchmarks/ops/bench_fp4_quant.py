"""Benchmark for the FP4 e2m1 quantization op.

The TileKernels column times ``quant.per_token_cast_and_cast_back`` in place and otherwise
``quant.per_token_cast``, its UE8M0 scales stored as packed bytes, both at TileKernels' default
e2m1 amax floor, ``6 * 2 ** -126``, which is the op's.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    TILEKERNELS_TAG,
    TORCH_COMPILE_TAG,
    baseline_module,
    compiled_reference,
    private_inputs,
)
from tileops.quantization import FP4QuantPerGroupFwdOp


@pytest.mark.parametrize("case", bench.cases(FP4QuantPerGroupFwdOp))
def test_fp4_quant_bench(case) -> None:
    op = FP4QuantPerGroupFwdOp(**case.arguments)
    quant = baseline_module("tile_kernels.quant", TILEKERNELS_TAG)
    block_size = case.arguments["block_size"]

    def tilekernels_fn(x):
        # TileKernels takes two-dimensional x; the leading axes are folded into rows.
        rows = x.view(-1, x.shape[-1])
        if case.arguments["inplace"]:
            quant.per_token_cast_and_cast_back(rows, "e2m1", block_size, round_sf=True)
            return None, None, x
        # Codes two to an int8 byte, and exponent bytes four to an int32: the e8m0 scales.
        q, scale = quant.per_token_cast(
            rows, "e2m1", block_size, round_sf=True, use_packed_ue8m0=True
        )
        leading = x.shape[:-1]
        return (
            q.view(torch.uint8).view(*leading, -1),
            scale.view(torch.float8_e8m0fnu).view(*leading, -1),
            None,
        )

    implementations = {
        TILEKERNELS_TAG: tilekernels_fn,
        "torch-ref": case.reference,
        TORCH_COMPILE_TAG: compiled_reference(case.reference),
    }
    if case.arguments["inplace"]:
        implementations = {
            name: private_inputs(fn, case.inputs, 0) for name, fn in implementations.items()
        }
    bench.Runner(op, case).compare(implementations)
