"""Compare grouped INT4 quantization with DeepSpeed's fused CUDA kernel."""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import DEEPSPEED_TAG, deepspeed_op
from tileops.quantization import INT4QuantPerGroupFwdOp


@pytest.mark.parametrize("case", bench.cases(INT4QuantPerGroupFwdOp), ids=lambda case: case.id)
def test_int4_quant_per_group_bench(case) -> None:
    workload = case.workload
    op = INT4QuantPerGroupFwdOp(**case.arguments)
    op.autotune()
    quantize = deepspeed_op("quantize")
    asymmetric = deepspeed_op("Asymmetric")

    def deepspeed(w):
        return quantize(w, w.numel() // workload.group_size, 4, asymmetric)

    bench.Runner(op, case).compare(
        {
            DEEPSPEED_TAG: bench.Implementation(
                run=deepspeed,
                noncomparable_reason="vendor uses fast-math division for group scales; rounding can change packed INT4 codes",
            )
        }
    )
