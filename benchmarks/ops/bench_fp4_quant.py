"""Benchmark for the FP4 e2m1 quantization op.

Workload shapes come from the ops manifest; roofline FLOP and byte counts come from the op's
``eval_roofline()``. The comparison is with the torch reference and ``torch.compile`` of it.
vLLM's FP4 quantizers make NVFP4, e4m3 scales per 16. FlashInfer's ``mxfp4_quantize`` has
the e2m1 and per-32 UE8M0 format but rounds the scale by the OCP MX rule, and on SM90 its
kernel writes no codes. DeepSeek's TileKernels ``per_token_cast`` matches bit for bit but
needs a newer TileLang than the one TileOPs pins.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference, private_inputs
from tileops.quantization import FP4QuantPerGroupFwdOp


@pytest.mark.parametrize("case", bench.cases(FP4QuantPerGroupFwdOp), ids=lambda case: case.id)
def test_fp4_quant_bench(case) -> None:
    op = FP4QuantPerGroupFwdOp(**case.arguments)
    implementations = {
        "torch-ref": case.reference,
        TORCH_COMPILE_TAG: compiled_reference(case.reference),
    }
    if case.arguments["inplace"]:
        # Each fake-quantizes x in place, so each rewrites a private copy, restored per round.
        implementations = {
            name: private_inputs(fn, case.inputs, 0) for name, fn in implementations.items()
        }
    bench.Runner(op, case).compare(implementations)
