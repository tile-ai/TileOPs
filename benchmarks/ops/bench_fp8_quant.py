"""Benchmark for the FP8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    private_inputs,
    vllm_op,
)
from tileops.quantization import FP8QuantPerTokenGroupFwdOp


@pytest.mark.parametrize("case", bench.cases(FP8QuantPerTokenGroupFwdOp), ids=lambda case: case.id)
def test_fp8_quant_bench(case) -> None:
    op = FP8QuantPerTokenGroupFwdOp(**case.arguments)
    op.request_tune()

    quantize = vllm_op(
        "per_token_group_quant_fp8", "model_executor.layers.quantization.utils.fp8_utils"
    )

    arguments = case.arguments

    def vllm_fn(x, out_q=None):
        # vLLM's CUDA kernel takes two-dimensional x; the leading axes are folded into rows.
        rows = x.reshape(-1, x.shape[-1])
        q, scale = quantize(
            rows,
            arguments["group_size"],
            eps=arguments["eps"],
            dtype=arguments["dtype"],
            column_major_scales=arguments["column_major_scales"],
            tma_aligned_scales=arguments["tma_aligned_scales"],
            out_q=None if out_q is None else out_q.view(rows.shape),
            # The op's None is float32 scales; vLLM would resolve None from DeepGEMM.
            use_ue8m0=bool(arguments["use_ue8m0"]),
        )
        return q.view(x.shape), scale.view(*x.shape[:-1], -1)

    bench.Runner(op, case).compare(
        {
            VLLM_TAG: private_inputs(vllm_fn, case.inputs, *range(1, len(case.inputs))),
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )
