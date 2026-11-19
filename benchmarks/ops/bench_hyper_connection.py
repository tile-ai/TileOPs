"""Benchmarks for the manifold hyper-connection ops, one case per manifest call.

Each pre row times vLLM's TileLang ``mhc_pre`` kernels and the reference in torch eager and
through inductor. vLLM's ``mhc_pre_tilelang`` projects with DeepGEMM's
``tf32_hc_prenorm_gemm`` where DeepGEMM is installed; the installed DeepGEMM has none, so the
row runs the TileLang projection vLLM falls back to, then its fused pre kernel. That
projection squares each bfloat16 stream element in bfloat16, so vLLM's row has no ratio.

Each post row times vLLM's ``mhc_post_tilelang``, the TileLang kernel of DeepSeek's TileKernels
``mhc_post_fwd``. It takes float32 mixes only, as vLLM's pre step emits them, so a row's mixes
are cast before timing. TileKernels itself needs a newer TileLang than the one TileOPs pins.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, baseline_module, compiled_reference
from tileops.sequence_modeling import HyperConnectionPostFwdOp, HyperConnectionPreFwdOp


def _vllm_mhc_pre(residual: torch.Tensor, params: dict):
    """vLLM's TileLang projection and fused pre kernel, into outputs allocated outside timing."""
    wrappers = baseline_module("vllm.model_executor.kernels.mhc.tilelang", VLLM_TAG)
    kernels = baseline_module("vllm.model_executor.kernels.mhc.tilelang_kernels", VLLM_TAG)
    *lead, n, c = residual.shape
    tokens = residual.numel() // (n * c)
    q = n * n + 2 * n
    f32 = {"dtype": torch.float32, "device": residual.device}
    mixes = torch.empty(1, tokens, q, **f32)
    sqrsum = torch.empty(1, tokens, **f32)
    post_mix = torch.empty(tokens, n, **f32)
    comb_mix = torch.empty(tokens, n * n, **f32)
    layer_input = torch.empty(tokens, c, dtype=residual.dtype, device=residual.device)

    def run(residual, fn, hc_scale, hc_base):
        streams = residual.view(tokens, n, c)
        wrappers._tilelang_hc_prenorm_gemm(streams.view(tokens, n * c), fn, mixes, sqrsum, c, n)
        kernels.mhc_pre_big_fuse_tilelang(
            mixes,
            sqrsum,
            hc_scale,
            hc_base,
            streams,
            post_mix,
            comb_mix,
            layer_input,
            c,
            params["rms_eps"],
            params["hc_pre_eps"],
            params["hc_sinkhorn_eps"],
            params["hc_post_mult_value"],
            params["sinkhorn_repeat"],
            1,
            n,
        )
        return (
            post_mix.view(*lead, n, 1),
            comb_mix.view(*lead, n, n),
            layer_input.view(*lead, c),
        )

    return run


@pytest.mark.parametrize("case", bench.cases(HyperConnectionPreFwdOp), ids=lambda case: case.id)
def test_hyper_connection_pre_bench(case) -> None:
    op = HyperConnectionPreFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {
            VLLM_TAG: bench.Implementation(
                run=_vllm_mhc_pre(case.inputs[0], case.arguments),
                noncomparable_reason="squares the streams in bfloat16 before summing them",
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
