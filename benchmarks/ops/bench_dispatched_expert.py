"""H100 microbenchmark for the communication-independent expert MLP."""

import torch

from tileops.ops.moe import (
    DispatchedExpertMLPFwdOp,
    FusedMoEExpertsNopadPersistent3WGFwdOp,
)

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
    expert_input = torch.randn(num_pairs, H, device="cuda", dtype=DTYPE)
    w_gate_up = (
        torch.randn(E, 2 * F, H, device="cuda", dtype=DTYPE) * 0.02
    )
    w_down = torch.randn(E, H, F, device="cuda", dtype=DTYPE) * 0.02
    true_sizes = torch.full(
        (E,), num_pairs // E, device="cuda", dtype=torch.int32
    )
    true_sizes[: num_pairs % E] += 1
    true_offsets = torch.empty(E, device="cuda", dtype=torch.int32)
    true_offsets[0] = 0
    true_offsets[1:] = torch.cumsum(true_sizes[:-1], dim=0)
    topk_ids = torch.randint(
        0, E, (num_tokens, TOP_K), device="cuda", dtype=torch.int32
    )
    topk_weights = torch.softmax(
        torch.randn(num_tokens, TOP_K, device="cuda"), dim=-1
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
    pure_tflops = logical_flops / (timings[1] / 1e3) / 1e12
    full_tflops = logical_flops / (timings[3] / 1e3) / 1e12
    return (*timings, pure_tflops, full_tflops)


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
        "pure_fused_effective_TFLOPS,full_fused_effective_TFLOPS"
    )
    for num_tokens in (128, 1024):
        values = _run_shape(num_tokens)
        formatted = ",".join(f"{value:.4f}" for value in values)
        print(f"{num_tokens},{num_tokens * TOP_K},{formatted}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
