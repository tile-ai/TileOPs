"""Benchmark for the FP8 quantization op.

The TileKernels column times ``quant.per_token_cast_and_cast_back`` in place and otherwise
``quant.per_token_cast``, an e8m0 row's UE8M0 scales stored as packed bytes, both with the row's
eps as the amax floor.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    TILEKERNELS_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    baseline_module,
    compiled_reference,
    private_inputs,
    vllm_op,
)
from tileops.quantization import FP8QuantPerTokenGroupFwdOp


@pytest.mark.parametrize("case", bench.cases(FP8QuantPerTokenGroupFwdOp))
def test_fp8_quant_bench(case) -> None:
    op = FP8QuantPerTokenGroupFwdOp(**case.arguments)
    op.request_tune()

    quantize = vllm_op(
        "per_token_group_quant_fp8", "model_executor.layers.quantization.utils.fp8_utils"
    )
    tile_kernels = baseline_module("tile_kernels", TILEKERNELS_TAG)

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

    def tilekernels_fn(x, out_q=None):
        # TileKernels takes two-dimensional x too.
        rows = x.view(-1, x.shape[-1])
        if arguments["inplace"]:
            tile_kernels.quant.per_token_cast_and_cast_back(
                rows, "e4m3", arguments["group_size"], round_sf=True
            )
            return None, None, x
        e8m0 = arguments["scale_dtype"] == torch.float8_e8m0fnu
        q, scale = tile_kernels.quant.per_token_cast(
            rows,
            "e4m3",
            arguments["group_size"],
            round_sf=bool(arguments["use_ue8m0"]),
            # Exponent bytes, four to an int32: the e8m0 scales, without a cast.
            use_packed_ue8m0=e8m0,
            out=None if out_q is None else (out_q.view(rows.shape), None),
        )
        if e8m0:
            scale = scale.view(torch.float8_e8m0fnu)
        return q.view(x.shape), scale.view(*x.shape[:-1], -1), None

    implementations = {}
    if arguments["inplace"]:
        print(f"  [skip] {VLLM_TAG}: per_token_group_quant_fp8 has no in-place fake quantization")
    else:
        implementations[VLLM_TAG] = private_inputs(
            vllm_fn, case.inputs, *range(1, len(case.inputs))
        )
    if case.inputs[0].dtype == torch.float16:
        print(f"  [skip] {TILEKERNELS_TAG}: per_token_cast takes bfloat16 or float32, not float16")
    elif arguments["inplace"]:
        implementations[TILEKERNELS_TAG] = tilekernels_fn
    else:
        implementations[TILEKERNELS_TAG] = private_inputs(
            tilekernels_fn, case.inputs, *range(1, len(case.inputs))
        )
    implementations["torch-ref"] = case.reference
    implementations[TORCH_COMPILE_TAG] = compiled_reference(case.reference)
    if arguments["inplace"]:
        # Each fake-quantizes x in place, so each rewrites a private copy, restored per round.
        implementations = {
            name: private_inputs(fn, case.inputs, 0) for name, fn in implementations.items()
        }
    # TileKernels floors each group maximum at one process-wide value, 1e-4 unless set; this
    # case sets the row's eps and puts the previous value back.
    floor = tile_kernels.get_amax_clamp_for_quant("e4m3", False)
    tile_kernels.set_amax_clamp_for_quant("e4m3", False, arguments["eps"])
    try:
        bench.Runner(op, case).compare(implementations)
    finally:
        tile_kernels.set_amax_clamp_for_quant("e4m3", False, floor)
