"""Benchmarks for the manifold hyper-connection ops, one case per manifest call.

Each pre row times vLLM's ``mhc_pre_tilelang``, DeepSeek's TileKernels and the reference in
torch eager and through inductor. vLLM projects with the ``tf32_hc_prenorm_gemm`` of the
DeepGEMM its wheel vendors, then runs its fused pre kernel; TileKernels runs the same
projection, then its ``mhc_pre_big_fuse_fwd``. The TF32 projection leaves both rows without a
ratio.

Each post row times TileKernels' ``mhc_post_fwd`` and vLLM's port of it, ``mhc_post_tilelang``.
Both take float32 mixes only, as their pre steps emit them, so a row's mixes are cast before
timing.

Each head row times vLLM's ``hc_head_fused_kernel_tilelang``, the head it serves DeepSeek-V4
with. TileKernels has no fused head: its row runs DeepGEMM's TF32 GEMM, then
``mhc_reduce_partials_and_rmsnorm_fwd``, ``mhc_head_compute_mix_fwd`` and
``mhc_pre_apply_mix_fwd``. Its output is bfloat16, so the TF32 GEMM stays within the case's
tolerance and the row carries a ratio, as vLLM's head row does. Both TileKernels projections
take vLLM's ``compute_num_split`` as their split count.
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
    deepgemm_op,
)
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


def _tf32_projection():
    """DeepGEMM's split-K ``tf32_hc_prenorm_gemm`` of the flattened streams: each split's
    products and squared sums."""
    gemm = deepgemm_op("tf32_hc_prenorm_gemm")
    kernels = baseline_module("vllm.model_executor.kernels.mhc.tilelang_kernels", VLLM_TAG)

    def project(x, fn):
        tokens, k = x.shape
        # DeepGEMM's blocks are 64 tokens by 64 of K.
        splits = kernels.compute_num_split(64, k, -(-tokens // 64))
        products = x.new_empty(splits, tokens, fn.shape[0], dtype=torch.float32)
        sqrsum = x.new_empty(splits, tokens, dtype=torch.float32)
        gemm(x, fn, products, sqrsum, splits)
        return products, sqrsum

    return project


def _tilekernels_mhc_pre(params: dict):
    """TileKernels' pre path: DeepGEMM's TF32 projection, then ``mhc_pre_big_fuse_fwd``."""
    mhc = baseline_module("tile_kernels.mhc", TILEKERNELS_TAG)
    project = _tf32_projection()

    def run(residual, fn, hc_scale, hc_base):
        *s, n, c = residual.shape
        streams = residual.reshape(-1, n, c)
        products, sqrsum = project(streams.flatten(1), fn)
        tokens = streams.shape[0]
        post_mix = residual.new_empty(tokens, n, dtype=torch.float32)
        comb_mix = residual.new_empty(tokens, n * n, dtype=torch.float32)
        layer_input = residual.new_empty(tokens, c)
        mhc.mhc_pre_big_fuse_fwd(
            products,
            sqrsum,
            hc_scale,
            hc_base,
            streams,
            post_mix,
            comb_mix,
            layer_input,
            params["rms_eps"],
            params["hc_pre_eps"],
            params["hc_sinkhorn_eps"],
            params["hc_post_mult_value"],
            params["sinkhorn_repeat"],
            n_splits=products.shape[0],
        )
        return post_mix.view(*s, n, 1), comb_mix.view(*s, n, n), layer_input.view(*s, c)

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
            TILEKERNELS_TAG: bench.Implementation(
                run=_tilekernels_mhc_pre(case.arguments),
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
    mhc = baseline_module("tile_kernels.mhc", TILEKERNELS_TAG)
    x, residual, post_layer_mix, comb_res_mix = case.inputs
    n, c = residual.shape[-2:]

    def vllm_fn(*args):
        return wrappers.mhc_post_tilelang(*args).view(residual.shape)

    def tilekernels_fn(*args):
        # mhc_post_fwd takes a sequence axis ahead of the tokens.
        return mhc.mhc_post_fwd(*(arg.unsqueeze(0) for arg in args)).view(residual.shape)

    args = (
        x.reshape(-1, c),
        residual.reshape(-1, n, c),
        post_layer_mix.reshape(-1, n, 1).float(),
        comb_res_mix.reshape(-1, n, n).float(),
    )
    bench.Runner(op, case).compare(
        {
            VLLM_TAG: bench.Implementation(run=vllm_fn, args=args),
            TILEKERNELS_TAG: bench.Implementation(run=tilekernels_fn, args=args),
            "torch-ref": case.torch,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
        }
    )


@pytest.mark.parametrize("case", bench.cases(HyperConnectionHeadFwdOp), ids=lambda case: case.id)
def test_hyper_connection_head_bench(case) -> None:
    op = HyperConnectionHeadFwdOp(**case.arguments)
    wrappers = baseline_module("vllm.model_executor.kernels.mhc.tilelang", VLLM_TAG)
    mhc = baseline_module("tile_kernels.mhc", TILEKERNELS_TAG)
    project = _tf32_projection()
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

    def tilekernels_fn(hidden_states, padded_fn, hc_scale, hc_base):
        n, c = hidden_states.shape[-2:]
        streams = hidden_states.reshape(-1, n, c)
        products, sqrsum = project(streams.flatten(1), padded_fn)
        splits, tokens, width = products.shape
        mixes = products.new_empty(tokens, width)
        mhc.mhc_reduce_partials_and_rmsnorm_fwd(
            products.view(splits, tokens, 1, width),
            sqrsum.view(splits, tokens, 1),
            products.new_empty(tokens, 1, width),
            sqrsum.new_empty(tokens, 1),
            mixes,
            n * c,
            arguments["rms_norm_eps"],
            splits,
        )
        pre = mixes.new_empty(tokens, n)
        # The mix kernel takes packed mixes, so the n real ones are copied out.
        mhc.mhc_head_compute_mix_fwd(
            mixes[:, :n].contiguous(), hc_scale, hc_base, pre, arguments["hc_eps"]
        )
        output = hidden_states.new_empty(tokens, c)
        mhc.mhc_pre_apply_mix_fwd(streams, pre, output)
        return output.view(*hidden_states.shape[:-2], c)

    hidden_states, hc_fn, hc_scale, hc_base = case.inputs
    # DeepGEMM projects to a multiple of 8 outputs, so hc_fn gains zero rows, outside timing.
    padded_fn = torch.nn.functional.pad(hc_fn, (0, 0, 0, -hc_fn.shape[0] % 8))
    bench.Runner(op, case).compare(
        {
            VLLM_TAG: vllm_fn,
            TILEKERNELS_TAG: bench.Implementation(
                run=tilekernels_fn,
                args=(hidden_states, padded_fn, hc_scale, hc_base),
            ),
            "torch-ref": case.torch,
            TORCH_COMPILE_TAG: compiled_reference(case.torch),
        }
    )
