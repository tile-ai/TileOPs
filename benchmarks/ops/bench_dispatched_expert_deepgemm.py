"""Compute-only BF16 Expert MLP comparison with DeepGEMM ordinary MoE."""

import argparse
import time

import torch
import torch.nn.functional as F

import deep_gemm

from tileops.ops.moe import DispatchedExpertMLPFwdOp

DTYPE = torch.bfloat16
WARMUP, ITERS = 10, 50


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
    elif distribution == "router":
        # Deterministic router-like trace: correlated expert logits produce
        # a reproducible non-uniform assignment with naturally empty experts.
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
            zip(sizes, aligned_sizes)
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

    def run_tileops_unfused():
        return tileops_unfused(
            tight, w_gate_up, w_down, true_sizes, true_offsets
        )

    def run_tileops_fused():
        return tileops_fused(
            tight, w_gate_up, w_down, true_sizes, true_offsets
        )

    def run_gemm1():
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            aligned,
            w_gate_up,
            gate_up,
            psum,
            use_psum_layout=True,
            expected_m_for_psum_layout=max(1, num_pairs // E),
        )

    def run_activation():
        torch.mul(
            F.silu(gate_up[:, :F_DIM]),
            gate_up[:, F_DIM:],
            out=activated,
        )

    def run_gemm2():
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
            activated,
            w_down,
            output,
            psum,
            use_psum_layout=True,
            expected_m_for_psum_layout=max(1, num_pairs // E),
        )

    def run_deepgemm_pipeline():
        run_gemm1()
        run_activation()
        run_gemm2()

    # Correctness compares only valid logical rows; aligned padding is omitted.
    tileops_reference = run_tileops_unfused()
    run_deepgemm_pipeline()
    deepgemm_valid = torch.empty_like(tileops_reference)
    tight_start = physical_start = 0
    for size, aligned_size in zip(sizes, aligned_sizes):
        deepgemm_valid[tight_start : tight_start + size].copy_(
            output[physical_start : physical_start + size]
        )
        tight_start += size
        physical_start += aligned_size
    torch.cuda.synchronize()
    abs_diff = (tileops_reference.float() - deepgemm_valid.float()).abs()
    max_diff = abs_diff.max().item()
    relative_diff = max_diff / max(
        tileops_reference.float().abs().max().item(), 1e-12
    )
    if not torch.allclose(
        tileops_reference.float(),
        deepgemm_valid.float(),
        atol=8e-2,
        rtol=8e-2,
    ):
        raise AssertionError(
            f"{case_name}/{distribution}: TileOps and DeepGEMM disagree; "
            f"max_diff={max_diff}, relative_diff={relative_diff}"
        )

    pack_ms = _host_time_ms(pack_aligned)
    tileops_unfused_ms = _time_ms(run_tileops_unfused)
    tileops_fused_ms = _time_ms(run_tileops_fused)
    gemm1_ms = _time_ms(run_gemm1)
    activation_ms = _time_ms(run_activation)
    gemm2_ms = _time_ms(run_gemm2)
    deepgemm_total_ms = _time_ms(run_deepgemm_pipeline)

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
        f"{deepgemm_total_ms:.4f},"
        f"{logical_flops / (tileops_fused_ms / 1e3) / 1e12:.2f},"
        f"{logical_flops / (deepgemm_total_ms / 1e3) / 1e12:.2f},"
        f"{physical_flops / (deepgemm_total_ms / 1e3) / 1e12:.2f},"
        f"{max_diff:.6f},{relative_diff:.6f},{num_pairs}",
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
        choices=("uniform", "longtail", "hotspot", "router"),
        nargs="+",
        default=("uniform",),
    )
    return parser.parse_args()


def main() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(42)
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
        "deepgemm_total_ms,tileops_effective_TFLOPS,"
        "deepgemm_effective_TFLOPS,deepgemm_physical_TFLOPS,"
        "max_diff,relative_diff,valid_rows_checked"
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
