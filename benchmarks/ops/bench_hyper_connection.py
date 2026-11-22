"""Benchmarks for the manifold hyper-connection ops, one case per manifest call.

Each pre row times vLLM's ``mhc_pre_tilelang`` and the reference in torch eager and through
inductor. vLLM projects with the ``tf32_hc_prenorm_gemm`` of the DeepGEMM its wheel vendors,
then runs its fused pre kernel. The TF32 projection leaves vLLM's row without a ratio.

Each post row times vLLM's ``mhc_post_tilelang``, the TileLang kernel of DeepSeek's TileKernels
``mhc_post_fwd``. It takes float32 mixes only, as vLLM's pre step emits them, so a row's mixes
are cast before timing.

Each head row times vLLM's ``hc_head_fused_kernel_tilelang``, the head it serves DeepSeek-V4
with. TileKernels has no fused head: its path runs DeepGEMM's TF32 GEMM and three kernels.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, baseline_module, compiled_reference
from tileops.sequence_modeling import (
    HyperConnectionHeadFwdOp,
    HyperConnectionPostFwdOp,
    HyperConnectionPreFwdOp,
)


def _vllm_mhc_pre(params: dict):
    """vLLM's ``mhc_pre_tilelang``: DeepGEMM's TF32 projection, then its fused pre kernel."""
    wrappers = baseline_module("vllm.model_executor.kernels.mhc.tilelang", VLLM_TAG)

    def run(residual, fn, hc_scale, hc_base):
        return wrappers.mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            params["rms_eps"],
            params["hc_pre_eps"],
            params["hc_sinkhorn_eps"],
            params["hc_post_mult_value"],
            params["sinkhorn_repeat"],
        )

    return run


@pytest.mark.parametrize("case", bench.cases(HyperConnectionPreFwdOp), ids=lambda case: case.id)
def test_hyper_connection_pre_bench(case) -> None:
    op = HyperConnectionPreFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {
            VLLM_TAG: bench.Implementation(
                run=_vllm_mhc_pre(case.arguments),
                noncomparable_reason="projects the streams in TF32",
            ),
            "torch-ref": case.torch,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
        }
    )


@pytest.mark.parametrize("case", bench.cases(HyperConnectionPostFwdOp), ids=lambda case: case.id)
def test_hyper_connection_post_bench(case) -> None:
    op = HyperConnectionPostFwdOp(**case.arguments)
    wrappers = baseline_module("vllm.model_executor.kernels.mhc.tilelang", VLLM_TAG)
    x, residual, post_layer_mix, comb_res_mix = case.inputs
    n, c = residual.shape[-2:]

    def vllm_fn(*args):
        return wrappers.mhc_post_tilelang(*args).view(residual.shape)

    vllm = bench.Implementation(
        run=vllm_fn,
        args=(
            x.reshape(-1, c),
            residual.reshape(-1, n, c),
            post_layer_mix.reshape(-1, n, 1).float(),
            comb_res_mix.reshape(-1, n, n).float(),
        ),
    )
    bench.Runner(op, case).compare(
        {
            VLLM_TAG: vllm,
            "torch-ref": case.torch,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
        }
    )


@pytest.mark.parametrize("case", bench.cases(HyperConnectionHeadFwdOp), ids=lambda case: case.id)
def test_hyper_connection_head_bench(case) -> None:
    op = HyperConnectionHeadFwdOp(**case.arguments)
    wrappers = baseline_module("vllm.model_executor.kernels.mhc.tilelang", VLLM_TAG)
    arguments = case.arguments

    def vllm_fn(hidden_states, hc_fn, hc_scale, hc_base):
        # vLLM's kernel takes one token axis.
        n, c = hidden_states.shape[-2:]
        output = wrappers.hc_head_fused_kernel_tilelang(
            hidden_states.reshape(-1, n, c),
            hc_fn,
            hc_scale,
            hc_base,
            arguments["rms_norm_eps"],
            arguments["hc_eps"],
        )
        return output.view(*hidden_states.shape[:-2], c)

    bench.Runner(op, case).compare(
        {
            VLLM_TAG: vllm_fn,
            "torch-ref": case.torch,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
        }
    )
