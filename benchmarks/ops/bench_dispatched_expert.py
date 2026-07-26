"""H100 microbenchmark for the communication-independent expert MLP."""

import torch

from tileops.ops.moe import (
    DispatchedExpertMLPFwdOp,
    ExpertBatch,
    FusedMoEExpertsNopadPersistent3WGFwdOp,
)
from tileops.ops.moe.routed_expert import MoePermuteNopadFwdOp

DTYPE = torch.bfloat16
E, TOP_K, H, F = 128, 8, 2048, 1024
WARMUP, ITERS = 20, 100


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


def _run_shape(num_tokens: int) -> tuple[float, ...]:
    num_pairs = num_tokens * TOP_K
    hidden = torch.randn(num_tokens, H, device="cuda", dtype=DTYPE)
    w_gate_up = (
        torch.randn(E, 2 * F, H, device="cuda", dtype=DTYPE) * 0.02
    )
    w_down = torch.randn(E, H, F, device="cuda", dtype=DTYPE) * 0.02
    # Select unique experts per token and materialize the exact expert-major
    # batch used by the full path. Pure/full timings therefore differ only by
    # routing work, not by expert-size distribution.
    topk_ids = torch.rand(
        num_tokens, E, device="cuda"
    ).topk(TOP_K, dim=-1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn(num_tokens, TOP_K, device="cuda"), dim=-1
    )
    permute = MoePermuteNopadFwdOp(
        total_tokens=num_tokens,
        top_k=TOP_K,
        num_experts=E,
        hidden_size=H,
        dtype=DTYPE,
    )
    expert_input, true_offsets, true_sizes, _, _ = permute(
        hidden, topk_ids
    )
    output_unfused = torch.empty(
        num_tokens, H, device="cuda", dtype=DTYPE
    )
    output_fused = torch.empty_like(output_unfused)
    workspace = torch.empty(0, device="cuda", dtype=DTYPE)

    pure_unfused = DispatchedExpertMLPFwdOp(
        num_pairs, E, H, F, DTYPE, use_fused_activation=False
    )
    pure_fused = DispatchedExpertMLPFwdOp(
        num_pairs, E, H, F, DTYPE, use_fused_activation=True
    )
    full_unfused = FusedMoEExpertsNopadPersistent3WGFwdOp(
        num_tokens, E, TOP_K, H, F, dtype=DTYPE, use_fused_activation=False
    )
    full_fused = FusedMoEExpertsNopadPersistent3WGFwdOp(
        num_tokens, E, TOP_K, H, F, dtype=DTYPE, use_fused_activation=True
    )

    def run_pure_unfused():
        return pure_unfused(
            expert_input, w_gate_up, w_down, true_sizes, true_offsets
        )

    def run_pure_fused():
        return pure_fused(
            expert_input, w_gate_up, w_down, true_sizes, true_offsets
        )

    def run_full_unfused():
        return full_unfused.forward(
            output_unfused,
            hidden,
            w_gate_up,
            w_down,
            topk_weights,
            topk_ids,
            None,
            workspace,
            workspace,
            E,
        )

    def run_full_fused():
        return full_fused.forward(
            output_fused,
            hidden,
            w_gate_up,
            w_down,
            topk_weights,
            topk_ids,
            None,
            workspace,
            workspace,
            E,
        )

    timings = tuple(
        _time_ms(fn)
        for fn in (
            run_pure_unfused,
            run_pure_fused,
            run_full_unfused,
            run_full_fused,
        )
    )
    logical_flops = 6 * num_pairs * H * F
    tflops = tuple(
        logical_flops / (timing / 1e3) / 1e12
        for timing in timings
    )
    return (*timings, *tflops)


def _run_capacity_sweep(capacity: int = 16_384) -> None:
    hidden = torch.randn(capacity, H, device="cuda", dtype=DTYPE)
    w_gate_up = (
        torch.randn(E, 2 * F, H, device="cuda", dtype=DTYPE) * 0.02
    )
    w_down = torch.randn(E, H, F, device="cuda", dtype=DTYPE) * 0.02
    unfused = DispatchedExpertMLPFwdOp(
        capacity, E, H, F, DTYPE, use_fused_activation=False
    )
    fused = DispatchedExpertMLPFwdOp(
        capacity, E, H, F, DTYPE, use_fused_activation=True
    )
    print(
        "capacity,valid_rows,utilization,"
        "capacity_unfused_ms,capacity_fused_ms,"
        "unfused_effective_TFLOPS,fused_effective_TFLOPS"
    )
    for valid_rows in (0, capacity // 64, capacity // 8, capacity):
        sizes = [
            valid_rows // E + (expert < valid_rows % E)
            for expert in range(E)
        ]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        batch = ExpertBatch(
            hidden=hidden,
            expert_offsets=torch.tensor(
                offsets, device="cuda", dtype=torch.int32
            ),
        )
        unfused_ms = _time_ms(
            lambda batch=batch: unfused.forward_batch(
                batch, w_gate_up, w_down
            )
        )
        fused_ms = _time_ms(
            lambda batch=batch: fused.forward_batch(
                batch, w_gate_up, w_down
            )
        )
        logical_flops = 6 * valid_rows * H * F
        print(
            f"{capacity},{valid_rows},{valid_rows / capacity:.6f},"
            f"{unfused_ms:.4f},{fused_ms:.4f},"
            f"{logical_flops / (unfused_ms / 1e3) / 1e12:.2f},"
            f"{logical_flops / (fused_ms / 1e3) / 1e12:.2f}",
            flush=True,
        )


def main() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    print(
        f"GPU={torch.cuda.get_device_name()} dtype={DTYPE} "
        f"E={E} K={TOP_K} H={H} F={F}"
    )
    print(
        "T,M,pure_unfused_ms,pure_fused_ms,"
        "full_unfused_ms,full_fused_ms,"
        "pure_unfused_effective_TFLOPS,pure_fused_effective_TFLOPS,"
        "full_unfused_effective_TFLOPS,full_fused_effective_TFLOPS"
    )
    for num_tokens in (128, 1024):
        values = _run_shape(num_tokens)
        formatted = ",".join(f"{value:.4f}" for value in values)
        print(f"{num_tokens},{num_tokens * TOP_K},{formatted}", flush=True)
        torch.cuda.empty_cache()
    _run_capacity_sweep()


if __name__ == "__main__":
    main()
