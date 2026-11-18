"""Benchmark the MXFP4 M-grouped GEMM against vLLM's MXFP4 MoE GEMMs, torch and the FP8 grouped
GEMM.

- ``vllm``: ``vllm._custom_ops.moe_wna16_marlin_gemm`` on ``float4_e2m1f`` weights, the MXFP4 MoE
  GEMM vLLM selects for DeepSeek-V4 on SM90. ``vllm-triton``: ``triton_kernels.matmul_ogs`` as
  vLLM bundles it, its MXFP4 GEMM for GPT-OSS on SM90. Both take the bf16 rows vLLM's serving
  path holds, built once outside the timed call as ``a * a_scale`` (exact under e8m0 scales: an
  fp8 value times a power of two fits bf16). Both read e8m0 weight scales only, so they have no
  tag on the float32-scale rows.
- ``torch`` runs the reference; ``torch-compile`` compiles it on contiguous rows; masked rows
  have no ``torch-compile`` tag.
- ``moe-grouped-gemm-fp8``: ``MoEGroupedGEMMFP8FwdOp`` on the same rows, metadata and output,
  with FP8 weights under 128 x 128 float32 scales; another weight format, so not comparable.

FlashInfer has no single grouped MXFP4 GEMM on SM90 (``cutlass_fused_moe`` runs the whole MoE;
its grouped FP4 GEMMs are SM100+), nor has DeepGEMM any FP4 entry point.
DeepSeek-V4-Flash's own ``fp4_gemm`` is model code, not an installed library, so it has no tag.

Weight repacking, routing tables and per-expert row bounds are prepared once, outside the timed
call, as at model load or by a caller holding the routing on the host. Every comparator is
checked on each expert's valid rows only, which is all the op defines.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from tileops._ops.moe import MoEGroupedGEMMFP8FwdOp, MoEGroupedGEMMMXFP4FwdOp

_dequant_rows = None


def _bf16_rows(a: torch.Tensor, a_scale: torch.Tensor) -> torch.Tensor:
    """``a * a_scale`` in bf16, one compiled kernel."""
    global _dequant_rows
    if _dequant_rows is None:

        def dequant(a, a_scale):
            scale = a_scale.to(torch.bfloat16).repeat_interleave(128, dim=-1)
            return a.to(torch.bfloat16) * scale

        _dequant_rows = torch.compile(dequant, dynamic=False)
    return _dequant_rows(a, a_scale)


def _marlin(case) -> bench.Implementation:
    """vLLM's Marlin MoE GEMM, with ``fused_marlin_moe``'s ``block_size_m`` rule."""
    marlin_utils = "model_executor.layers.quantization.utils.marlin_utils"
    moe_align_block_size = vllm_op(
        "moe_align_block_size", "model_executor.layers.fused_moe.moe_align_block_size"
    )
    marlin_make_workspace_new = vllm_op("marlin_make_workspace_new", marlin_utils)
    marlin_permute_scales = vllm_op("marlin_permute_scales", marlin_utils)
    _repack_marlin_experts = vllm_op("_repack_marlin_experts", f"{marlin_utils}_fp4")
    mxfp4_marlin_process_scales = vllm_op("mxfp4_marlin_process_scales", f"{marlin_utils}_fp4")
    scalar_types = vllm_op("scalar_types", "scalar_type")
    gemm = vllm_op("moe_wna16_marlin_gemm")
    a, a_scale, b, b_scale, metadata = case.inputs
    num_experts, n, packed_k = b.shape
    k = 2 * packed_k
    weights = _repack_marlin_experts(
        b, n, k, torch.empty(0, dtype=torch.int, device=b.device), False
    )
    scales = torch.stack(
        [
            mxfp4_marlin_process_scales(
                marlin_permute_scales(
                    s=b_scale[e].to(torch.bfloat16).T, size_k=k, size_n=n, group_size=32
                ),
                input_dtype=None,
            )
            for e in range(num_experts)
        ]
    )
    # Each materialized row's expert, num_experts where no expert owns it.
    if case.arguments["layout"].kind == "masked":
        max_m = a.shape[1]
        owned = torch.arange(max_m, device=a.device)[None, :] < metadata[:, None].clamp(0, max_m)
        experts = torch.arange(num_experts, device=a.device)[:, None]
        ids = torch.where(owned, experts, num_experts).flatten().to(torch.int32)
    else:
        ids = torch.where(metadata < num_experts, metadata, num_experts).to(torch.int32)
    rows = ids.numel()
    valid = int((ids < num_experts).sum())
    for block_size_m in (8, 16, 32, 48, 64):
        if valid / num_experts / block_size_m < 0.9:
            break
    sorted_ids, expert_ids, padded = moe_align_block_size(
        ids.view(rows, 1), block_size_m, num_experts, ignore_invalid_experts=True
    )
    workspace = marlin_make_workspace_new(a.device, 4)
    route_weights = torch.ones(rows, 1, dtype=torch.float32, device=a.device)
    out = torch.empty(rows, n, dtype=torch.bfloat16, device=a.device)

    rows_bf16 = _bf16_rows(a.view(rows, k), a_scale.view(rows, -1))

    def run():
        gemm(
            rows_bf16,
            out,
            weights,
            None,
            scales,
            None,
            None,
            None,
            None,
            None,
            workspace,
            sorted_ids,
            expert_ids,
            padded,
            route_weights,
            moe_block_size=block_size_m,
            top_k=1,
            mul_topk_weights=False,
            b_q_type=scalar_types.float4_e2m1f,
            size_m=rows,
            size_n=n,
            size_k=k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=True,
            is_zp_float=False,
        )
        return out.view(*a.shape[:-1], n)

    return bench.Implementation(run=run, args=())


def _matmul_ogs(case) -> bench.Implementation:
    """triton_kernels ``matmul_ogs``: ragged over contiguous runs (an aligned run's padding rows
    go with its expert), batched over masked slabs, which computes every ``max_m`` row."""
    # vLLM serves its bundled triton_kernels under that name once it has checked for them.
    assert vllm_op("has_triton_kernels", "utils.import_utils")()
    _swizzle_mxfp4 = vllm_op(
        "_swizzle_mxfp4", "model_executor.layers.quantization.utils.mxfp4_utils"
    )
    from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig, matmul_ogs
    from triton_kernels.routing import RoutingData, compute_expt_data

    a, a_scale, b, b_scale, metadata = case.inputs
    num_experts = b.shape[0]
    weights, flex, scales = _swizzle_mxfp4(b, b_scale.view(torch.uint8), num_warps=8)
    precision = PrecisionConfig(weight_scale=scales, flex_ctx=FlexCtx(rhs_data=flex))
    rows_bf16 = _bf16_rows(a, a_scale)
    if case.arguments["layout"].kind == "masked":
        return bench.Implementation(
            run=lambda: matmul_ogs(rows_bf16, weights, None, precision_config=precision),
            args=(),
        )
    runs = [0] * num_experts
    owner = 0
    for g in metadata.tolist():
        owner = g if g < num_experts else owner
        runs[owner] += 1
    counts = torch.tensor(runs, dtype=torch.int32, device=a.device)
    rows = a.shape[0]
    routing = RoutingData(
        gate_scal=torch.ones(rows, dtype=torch.bfloat16, device=a.device),
        expt_hist=counts,
        n_expts_tot=num_experts,
        n_expts_act=1,
        expt_data=compute_expt_data(counts, num_experts, rows),
    )
    return bench.Implementation(
        run=lambda: matmul_ogs(rows_bf16, weights, None, routing, precision_config=precision),
        args=(),
    )


def _fp8_grouped_gemm(case) -> bench.Implementation:
    """``MoEGroupedGEMMFP8FwdOp`` on the case's rows, scales and metadata, with FP8 weights of
    the same shape under 128 x 128 float32 scales."""
    a, a_scale, b, _, metadata = case.inputs
    num_experts, n, packed_k = b.shape
    k = 2 * packed_k
    generator = torch.Generator(device=a.device).manual_seed(0)
    weights = (torch.randn(num_experts, n, k, device=a.device, generator=generator) * 0.25).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(num_experts, n // 128, k // 128, device=a.device, generator=generator) + 0.5
    op = MoEGroupedGEMMFP8FwdOp(case.arguments["layout"])
    rows_scale = a_scale.float().contiguous()
    out = torch.empty(*a.shape[:-1], n, dtype=torch.bfloat16, device=a.device)
    return bench.Implementation(
        run=lambda: op(a, rows_scale, weights, scales, metadata, out=out),
        args=(),
        noncomparable_reason="FP8 weights under 128 x 128 float32 scales, not MXFP4",
    )


@pytest.mark.parametrize("case", bench.cases(MoEGroupedGEMMMXFP4FwdOp), ids=lambda case: case.id)
def test_moe_grouped_gemm_mxfp4_bench(case) -> None:
    op = MoEGroupedGEMMMXFP4FwdOp(**case.arguments)
    implementations = {"torch": case.torch}
    if case.arguments["layout"].kind == "contiguous":
        implementations[TORCH_COMPILE_TAG] = compiled_reference(case.reference)
    if case.inputs[1].dtype == torch.float8_e8m0fnu:
        implementations[VLLM_TAG] = _marlin(case)
        implementations["vllm-triton"] = _matmul_ogs(case)
    implementations["moe-grouped-gemm-fp8"] = _fp8_grouped_gemm(case)
    bench.Runner(op, case).compare(implementations)
