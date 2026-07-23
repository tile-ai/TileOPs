"""Compute-only BF16 Expert MLP comparison with DeepGEMM ordinary MoE."""

import argparse
import time

import deep_gemm
import torch
import torch.nn.functional as F

from tileops.ops.moe import DispatchedExpertMLPFwdOp

DTYPE = torch.bfloat16
WARMUP, ITERS = 10, 50
MAX_RANGE_REL_TOL = 0.02
RELATIVE_L2_TOL = 0.01


def _time_ms(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / ITERS


def _host_time_ms(fn, *, warmup: int = 2, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - begin) * 1e3 / iters


def _make_sizes(
    num_pairs: int,
    num_experts: int,
    distribution: str,
) -> list[int]:
    if distribution == "uniform":
        return [
            num_pairs // num_experts + (expert < num_pairs % num_experts)
            for expert in range(num_experts)
        ]

    generator = torch.Generator(device="cpu").manual_seed(42)
    if distribution == "longtail":
        probabilities = 1 / torch.arange(
            1, num_experts + 1, dtype=torch.float64
        ).pow(1.2)
    elif distribution == "hotspot":
        hot_experts = max(1, num_experts // 8)
        probabilities = torch.full((num_experts,), 0.2 / num_experts)
        probabilities[:hot_experts] += 0.8 / hot_experts
    elif distribution == "router-like":
        # Deterministic synthetic categorical routing distribution.
        probabilities = torch.softmax(
            torch.randn(num_experts, generator=generator) * 1.5, dim=0
        )
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    assignments = torch.multinomial(
        probabilities,
        num_samples=num_pairs,
        replacement=True,
        generator=generator,
    )
    return torch.bincount(assignments, minlength=num_experts).tolist()


def _fp32_reference(
    hidden: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    sizes: list[int],
) -> torch.Tensor:
    """Independent per-expert FP32 reference in tight row order."""
    output = torch.empty(
        hidden.shape[0],
        hidden.shape[1],
        device=hidden.device,
        dtype=torch.float32,
    )
    ffn_size = w_down.shape[-1]
    start = 0
    for expert, size in enumerate(sizes):
        if size == 0:
            continue
        rows = hidden[start : start + size].float()
        gate_up = rows @ w_gate_up[expert].float().t()
        activated = F.silu(gate_up[:, :ffn_size]) * gate_up[:, ffn_size:]
        output[start : start + size] = (
            activated @ w_down[expert].float().t()
        )
        start += size
    return output


def _check_output(
    name: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float, float, float]:
    actual_fp32 = actual.float()
    if not torch.isfinite(actual_fp32).all():
        raise AssertionError(f"{name}: output contains NaN or Inf")
    error = actual_fp32 - reference
    max_abs = error.abs().max().item()
    rmse = error.square().mean().sqrt().item()
    relative_l2 = (
        torch.linalg.vector_norm(error)
        / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    ).item()
    max_range_relative = (
        max_abs / reference.abs().max().clamp_min(1e-12).item()
    )
    if (
        max_range_relative > MAX_RANGE_REL_TOL
        or relative_l2 > RELATIVE_L2_TOL
    ):
        raise AssertionError(
            f"{name}: max_abs={max_abs}, rmse={rmse}, "
            f"relative_l2={relative_l2}, "
            f"max_range_relative={max_range_relative}"
        )
    return max_abs, rmse, relative_l2, max_range_relative


def _run(
    case_name: str,
    num_pairs: int,
    num_experts: int,
    hidden_size: int,
    ffn_size: int,
    distribution: str,
) -> None:
    E, H, F_DIM = num_experts, hidden_size, ffn_size
    sizes = _make_sizes(num_pairs, E, distribution)
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(
        max(1, num_pairs // E)
    )
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    aligned_sizes = [
        (size + alignment - 1) // alignment * alignment for size in sizes
    ]
    physical_rows = sum(aligned_sizes)

    tight = torch.randn(num_pairs, H, device="cuda", dtype=DTYPE)
    w_gate_up = (
        torch.randn(E, 2 * F_DIM, H, device="cuda", dtype=DTYPE) * 0.02
    )
    w_down = (
        torch.randn(E, H, F_DIM, device="cuda", dtype=DTYPE) * 0.02
    )
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
        for expert, (size, aligned_size) in enumerate(
            zip(sizes, aligned_sizes, strict=True)
        ):
            aligned[physical_start : physical_start + size].copy_(
                tight[tight_start : tight_start + size]
            )
            psum[expert] = physical_start + size
            tight_start += size
            physical_start += aligned_size

    pack_aligned()

    gate_up = torch.empty(
        physical_rows, 2 * F_DIM, device="cuda", dtype=DTYPE
    )
    activated = torch.empty(
        physical_rows, F_DIM, device="cuda", dtype=DTYPE
    )
    output = torch.empty(physical_rows, H, device="cuda", dtype=DTYPE)

    tileops_unfused = DispatchedExpertMLPFwdOp(
        num_pairs, E, H, F_DIM, DTYPE, use_fused_activation=False
    )
    tileops_fused = DispatchedExpertMLPFwdOp(
        num_pairs, E, H, F_DIM, DTYPE, use_fused_activation=True
    )
    if not tileops_fused.use_fused_activation:
        raise RuntimeError(
            f"{case_name}: requested TileOps fused activation is not eligible"
        )

    def run_tileops_unfused():
        return tileops_unfused(
            tight, w_gate_up, w_down, true_sizes, true_offsets
        )

    def run_tileops_fused():
        return tileops_fused(
            tight, w_gate_up, w_down, true_sizes, true_offsets
        )

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
    reference = _fp32_reference(tight, w_gate_up, w_down, sizes)
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
    tileops_unfused_error = _check_output(
        "TileOps unfused", tileops_unfused_output, reference
    )
    tileops_fused_error = _check_output(
        "TileOps fused", tileops_fused_output, reference
    )
    deepgemm_error = _check_output("DeepGEMM", deepgemm_valid, reference)

    pack_ms = _host_time_ms(pack_aligned)
    tileops_unfused_ms = _time_ms(run_tileops_unfused)
    tileops_fused_ms = _time_ms(run_tileops_fused)
    gemm1_ms = _time_ms(run_gemm1)
    activation_ms = _time_ms(run_activation)
    gemm2_ms = _time_ms(run_gemm2)
    deepgemm_preallocated_ms = _time_ms(run_deepgemm_pipeline)
    deepgemm_allocating_ms = _time_ms(run_deepgemm_allocating_pipeline)

    logical_flops = 6 * num_pairs * H * F_DIM
    physical_flops = 6 * physical_rows * H * F_DIM
    empty_experts = sum(size == 0 for size in sizes)
    sizes_tensor = torch.tensor(sizes, dtype=torch.float32)
    print(
        f"{case_name},{distribution},{num_pairs / E:.4f},"
        f"{min(sizes)},{max(sizes)},{sizes_tensor.std(unbiased=False).item():.4f},"
        f"{num_pairs},{physical_rows},{empty_experts},"
        f"{physical_rows / num_pairs - 1:.4f},"
        f"{pack_ms:.4f},"
        f"{tileops_unfused_ms:.4f},{tileops_fused_ms:.4f},"
        f"{gemm1_ms:.4f},{activation_ms:.4f},{gemm2_ms:.4f},"
        f"{deepgemm_preallocated_ms:.4f},{deepgemm_allocating_ms:.4f},"
        f"{logical_flops / (tileops_unfused_ms / 1e3) / 1e12:.2f},"
        f"{logical_flops / (tileops_fused_ms / 1e3) / 1e12:.2f},"
        f"{logical_flops / (deepgemm_preallocated_ms / 1e3) / 1e12:.2f},"
        f"{logical_flops / (deepgemm_allocating_ms / 1e3) / 1e12:.2f},"
        f"{physical_flops / (deepgemm_preallocated_ms / 1e3) / 1e12:.2f},"
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
        f"{num_pairs}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=(
            "synthetic",
            "glm5",
            "deepseek-v3",
            "qwen3-235b",
            "qwen35-397b",
        ),
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
    if args.model == "glm5":
        num_experts, top_k, hidden_size, ffn_size = 256, 8, 6144, 2048
        token_counts = args.tokens or [1, 32, 256, 512, 1024, 2048]
    elif args.model == "deepseek-v3":
        num_experts, top_k, hidden_size, ffn_size = 256, 8, 7168, 2048
        token_counts = args.tokens or [1, 32, 256, 512, 1024, 2048]
    elif args.model == "qwen3-235b":
        num_experts, top_k, hidden_size, ffn_size = 128, 8, 7168, 2048
        token_counts = args.tokens or [1, 16, 128, 256, 512, 1024]
    elif args.model == "qwen35-397b":
        num_experts, top_k, hidden_size, ffn_size = 512, 10, 4096, 1024
        token_counts = args.tokens or [1, 52, 410, 820, 1639, 3277]
    else:
        num_experts, top_k, hidden_size, ffn_size = 128, 8, 2048, 1024
        token_counts = args.tokens or [128, 1024]
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
    cases = (
        [(f"M={m}", m * num_experts) for m in args.m]
        if args.m
        else [(f"T={tokens}", tokens * top_k) for tokens in token_counts]
    )
    for distribution in args.distribution:
        for case_name, num_pairs in cases:
            _run(
                case_name,
                num_pairs,
                num_experts,
                hidden_size,
                ffn_size,
                distribution,
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
