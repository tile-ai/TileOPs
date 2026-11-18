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
        scale = scale.view(*x.shape[:-1], -1).to(arguments["scale_dtype"])
        return q.view(x.shape), scale, None

    implementations = {}
    if arguments["inplace"]:
        print(f"  [skip] {VLLM_TAG}: per_token_group_quant_fp8 has no in-place fake quantization")
    else:
        implementations[VLLM_TAG] = private_inputs(
            vllm_fn, case.inputs, *range(1, len(case.inputs))
        )
    implementations["torch-ref"] = case.reference
    implementations[TORCH_COMPILE_TAG] = compiled_reference(case.reference)
    if arguments["inplace"]:
        # Each fake-quantizes x in place, so each rewrites a private copy, restored per round.
        implementations = {
            name: private_inputs(fn, case.inputs, 0) for name, fn in implementations.items()
        }
    bench.Runner(op, case).compare(implementations)
