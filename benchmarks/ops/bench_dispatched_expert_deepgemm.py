"""Compute-only BF16 Expert MLP comparison with DeepGEMM ordinary MoE."""

import argparse

import deep_gemm
import torch
import torch.nn.functional as F

from benchmarks.ops._moe_bench_utils import (
    check_output,
    cuda_time_ms,
    effective_tflops,
    expert_mlp_reference,
    host_time_ms,
    make_expert_sizes,
)
from tileops.ops.moe import DispatchedExpertMLPFwdOp

DTYPE = torch.bfloat16
WARMUP, ITERS = 10, 50
MAX_RANGE_REL_TOL = 0.02
RELATIVE_L2_TOL = 0.01
MODELS = {
    "synthetic": (128, 8, 2048, 1024, [128, 1024]),
    "glm5": (256, 8, 6144, 2048, [1, 32, 256, 512, 1024, 2048]),
    "deepseek-v3": (256, 8, 7168, 2048, [1, 32, 256, 512, 1024, 2048]),
    "qwen3-235b": (128, 8, 7168, 2048, [1, 16, 128, 256, 512, 1024]),
    "qwen35-397b": (512, 10, 4096, 1024, [1, 52, 410, 820, 1639, 3277]),
}


def _run(
    case_name: str,
    num_pairs: int,
    num_experts: int,
    hidden_size: int,
    ffn_size: int,
    distribution: str,
) -> str:
    E, H, F_DIM = num_experts, hidden_size, ffn_size
    sizes = make_expert_sizes(num_pairs, E, distribution)
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(max(1, num_pairs // E))
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    aligned_sizes = [(size + alignment - 1) // alignment * alignment for size in sizes]
    physical_rows = sum(aligned_sizes)

    tight = torch.randn(num_pairs, H, device="cuda", dtype=DTYPE)
    w_gate_up = torch.randn(E, 2 * F_DIM, H, device="cuda", dtype=DTYPE) * 0.02
    w_down = torch.randn(E, H, F_DIM, device="cuda", dtype=DTYPE) * 0.02
    true_sizes = torch.tensor(sizes, device="cuda", dtype=torch.int32)
    true_offsets = torch.tensor(
        [sum(sizes[:expert]) for expert in range(E)],
        device="cuda",
        dtype=torch.int32,
    )

    aligned = torch.empty(physical_rows, H, device="cuda", dtype=DTYPE)
    psum = torch.empty(E, device="cuda", dtype=torch.int32)

    def pack_aligned():
        aligned.zero_()
        tight_start = physical_start = 0
        for expert, (size, aligned_size) in enumerate(zip(sizes, aligned_sizes, strict=True)):
            aligned[physical_start : physical_start + size].copy_(
                tight[tight_start : tight_start + size]
            )
            psum[expert] = physical_start + size
            tight_start += size
            physical_start += aligned_size

    pack_aligned()

    gate_up = torch.empty(physical_rows, 2 * F_DIM, device="cuda", dtype=DTYPE)
    activated = torch.empty(physical_rows, F_DIM, device="cuda", dtype=DTYPE)
    output = torch.empty(physical_rows, H, device="cuda", dtype=DTYPE)

    tileops_unfused = DispatchedExpertMLPFwdOp(
        num_pairs, E, H, F_DIM, DTYPE, use_fused_activation=False
    )
    tileops_fused = DispatchedExpertMLPFwdOp(
        num_pairs, E, H, F_DIM, DTYPE, use_fused_activation=True
    )
    if not tileops_fused.use_fused_activation:
        raise RuntimeError(f"{case_name}: requested TileOps fused activation is not eligible")

    def run_tileops_unfused():
        return tileops_unfused(tight, w_gate_up, w_down, true_sizes, true_offsets)

    def run_tileops_fused():
        return tileops_fused(tight, w_gate_up, w_down, true_sizes, true_offsets)

    def run_gemm1_into(destination):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            aligned,
            w_gate_up,
            destination,
            psum,
            use_psum_layout=True,
            expected_m_for_psum_layout=max(1, num_pairs // E),
        )

    def run_activation_into(gate_up_input, destination):
        torch.mul(
            F.silu(gate_up_input[:, :F_DIM]),
            gate_up_input[:, F_DIM:],
            out=destination,
        )

    def run_gemm2_into(activation_input, destination):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            activation_input,
            w_down,
            destination,
            psum,
            use_psum_layout=True,
            expected_m_for_psum_layout=max(1, num_pairs // E),
        )

    def run_gemm1():
        run_gemm1_into(gate_up)

    def run_activation():
        run_activation_into(gate_up, activated)

    def run_gemm2():
        run_gemm2_into(activated, output)

    def run_deepgemm_pipeline():
        run_gemm1()
        run_activation()
        run_gemm2()

    def run_deepgemm_allocating_pipeline():
        local_gate_up = torch.empty_like(gate_up)
        local_activated = torch.empty_like(activated)
        local_output = torch.empty_like(output)
        run_gemm1_into(local_gate_up)
        run_activation_into(local_gate_up, local_activated)
        run_gemm2_into(local_activated, local_output)
        return local_output

    # Validate every reported backend against an independent FP32 reference.
    reference = expert_mlp_reference(tight, w_gate_up, w_down, sizes)
    tileops_unfused_output = run_tileops_unfused()
    tileops_fused_output = run_tileops_fused()
    run_deepgemm_pipeline()
    deepgemm_valid = torch.empty_like(tileops_unfused_output)
    tight_start = physical_start = 0
    for size, aligned_size in zip(sizes, aligned_sizes, strict=True):
        deepgemm_valid[tight_start : tight_start + size].copy_(
            output[physical_start : physical_start + size]
        )
        tight_start += size
        physical_start += aligned_size
    torch.cuda.synchronize()
    check_kwargs = {
        "max_range_relative_tolerance": MAX_RANGE_REL_TOL,
        "relative_l2_tolerance": RELATIVE_L2_TOL,
    }
    tileops_unfused_error = check_output(
        "TileOps unfused",
        tileops_unfused_output,
        reference,
        **check_kwargs,
    )
    tileops_fused_error = check_output(
        "TileOps fused",
        tileops_fused_output,
        reference,
        **check_kwargs,
    )
    deepgemm_error = check_output(
        "DeepGEMM",
        deepgemm_valid,
        reference,
        **check_kwargs,
    )

    pack_ms = host_time_ms(pack_aligned, warmup=2, iters=10)
    tileops_unfused_ms = cuda_time_ms(run_tileops_unfused, warmup=WARMUP, iters=ITERS)
    tileops_fused_ms = cuda_time_ms(run_tileops_fused, warmup=WARMUP, iters=ITERS)
    gemm1_ms = cuda_time_ms(run_gemm1, warmup=WARMUP, iters=ITERS)
    activation_ms = cuda_time_ms(run_activation, warmup=WARMUP, iters=ITERS)
    gemm2_ms = cuda_time_ms(run_gemm2, warmup=WARMUP, iters=ITERS)
    deepgemm_preallocated_ms = cuda_time_ms(run_deepgemm_pipeline, warmup=WARMUP, iters=ITERS)
    deepgemm_allocating_ms = cuda_time_ms(
        run_deepgemm_allocating_pipeline, warmup=WARMUP, iters=ITERS
    )

    logical_flops = 6 * num_pairs * H * F_DIM
    physical_flops = 6 * physical_rows * H * F_DIM
    empty_experts = sum(size == 0 for size in sizes)
    sizes_tensor = torch.tensor(sizes, dtype=torch.float32)
    return (
        f"{case_name},{distribution},{num_pairs / E:.4f},"
        f"{min(sizes)},{max(sizes)},{sizes_tensor.std(unbiased=False).item():.4f},"
        f"{num_pairs},{physical_rows},{empty_experts},"
        f"{physical_rows / num_pairs - 1:.4f},"
        f"{pack_ms:.4f},"
        f"{tileops_unfused_ms:.4f},{tileops_fused_ms:.4f},"
        f"{gemm1_ms:.4f},{activation_ms:.4f},{gemm2_ms:.4f},"
        f"{deepgemm_preallocated_ms:.4f},{deepgemm_allocating_ms:.4f},"
        f"{effective_tflops(logical_flops, tileops_unfused_ms):.2f},"
        f"{effective_tflops(logical_flops, tileops_fused_ms):.2f},"
        f"{effective_tflops(logical_flops, deepgemm_preallocated_ms):.2f},"
        f"{effective_tflops(logical_flops, deepgemm_allocating_ms):.2f},"
        f"{effective_tflops(physical_flops, deepgemm_preallocated_ms):.2f},"
        f"{tileops_unfused_error[0]:.6f},"
        f"{tileops_unfused_error[1]:.6f},"
        f"{tileops_unfused_error[2]:.6f},"
        f"{tileops_unfused_error[3]:.6f},"
        f"{tileops_fused_error[0]:.6f},"
        f"{tileops_fused_error[1]:.6f},"
        f"{tileops_fused_error[2]:.6f},"
        f"{tileops_fused_error[3]:.6f},"
        f"{deepgemm_error[0]:.6f},{deepgemm_error[1]:.6f},"
        f"{deepgemm_error[2]:.6f},{deepgemm_error[3]:.6f},"
        f"{num_pairs}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=tuple(MODELS),
        default="synthetic",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        help="Token counts; defaults depend on --model.",
    )
    parser.add_argument(
        "--m",
        type=int,
        nargs="+",
        help="Exact logical rows per expert; takes precedence over --tokens.",
    )
    parser.add_argument(
        "--distribution",
        choices=("uniform", "longtail", "hotspot", "router-like"),
        nargs="+",
        default=("uniform",),
    )
    return parser.parse_args()


def main() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    torch.set_grad_enabled(False)
    args = _parse_args()
    num_experts, top_k, hidden_size, ffn_size, default_tokens = MODELS[args.model]
    print(
        f"GPU={torch.cuda.get_device_name()} dtype={DTYPE} "
        f"model={args.model} E={num_experts} K={top_k} "
        f"H={hidden_size} F={ffn_size}"
    )
    print(
        "case,distribution,mean_M,min_M,max_M,std_M,"
        "logical_rows,physical_rows,empty_experts,padding_ratio,pack_ms,"
        "tileops_unfused_ms,tileops_fused_ms,"
        "deepgemm_gemm1_ms,activation_ms,deepgemm_gemm2_ms,"
        "deepgemm_preallocated_ms,deepgemm_allocating_ms,"
        "tileops_unfused_effective_TFLOPS,"
        "tileops_fused_effective_TFLOPS,"
        "deepgemm_preallocated_effective_TFLOPS,"
        "deepgemm_allocating_effective_TFLOPS,"
        "deepgemm_physical_TFLOPS,"
        "tileops_unfused_max_abs,tileops_unfused_rmse,"
        "tileops_unfused_relative_l2,tileops_unfused_max_range_relative,"
        "tileops_fused_max_abs,tileops_fused_rmse,"
        "tileops_fused_relative_l2,tileops_fused_max_range_relative,"
        "deepgemm_max_abs,deepgemm_rmse,deepgemm_relative_l2,"
        "deepgemm_max_range_relative,"
        "valid_rows_checked"
    )
    token_counts = args.tokens or default_tokens
    cases = (
        [(f"M={rows}", rows * num_experts) for rows in args.m]
        if args.m
        else [(f"T={tokens}", tokens * top_k) for tokens in token_counts]
    )
    for distribution in args.distribution:
        for case_name, num_pairs in cases:
            print(
                _run(
                    case_name,
                    num_pairs,
                    num_experts,
                    hidden_size,
                    ffn_size,
                    distribution,
                ),
                flush=True,
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
