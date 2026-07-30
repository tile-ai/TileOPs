"""Dynamic-capacity benchmark for the dispatched expert MLP."""

import torch

from benchmarks.ops._moe_bench_utils import cuda_time_ms, effective_tflops
from tileops.ops.moe import DispatchedExpertMLPFwdOp, ExpertBatch

DTYPE = torch.bfloat16
E, H, F = 128, 2048, 1024
WARMUP, ITERS = 20, 100


def _run_capacity_sweep(capacity: int = 16_384) -> None:
    hidden = torch.randn(capacity, H, device="cuda", dtype=DTYPE)
    w_gate_up = torch.randn(E, 2 * F, H, device="cuda", dtype=DTYPE) * 0.02
    w_down = torch.randn(E, H, F, device="cuda", dtype=DTYPE) * 0.02
    unfused = DispatchedExpertMLPFwdOp(capacity, E, H, F, DTYPE, use_fused_activation=False)
    fused = DispatchedExpertMLPFwdOp(capacity, E, H, F, DTYPE, use_fused_activation=True)
    print(
        "capacity,valid_rows,utilization,"
        "capacity_unfused_ms,capacity_fused_ms,"
        "unfused_effective_TFLOPS,fused_effective_TFLOPS"
    )
    for valid_rows in (0, capacity // 64, capacity // 8, capacity):
        sizes = [valid_rows // E + (expert < valid_rows % E) for expert in range(E)]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        batch = ExpertBatch(
            hidden=hidden,
            expert_offsets=torch.tensor(offsets, device="cuda", dtype=torch.int32),
        )
        unfused_ms = cuda_time_ms(
            lambda batch=batch: unfused.forward_batch(batch, w_gate_up, w_down),
            warmup=WARMUP,
            iters=ITERS,
        )
        fused_ms = cuda_time_ms(
            lambda batch=batch: fused.forward_batch(batch, w_gate_up, w_down),
            warmup=WARMUP,
            iters=ITERS,
        )
        logical_flops = 6 * valid_rows * H * F
        print(
            f"{capacity},{valid_rows},{valid_rows / capacity:.6f},"
            f"{unfused_ms:.4f},{fused_ms:.4f},"
            f"{effective_tflops(logical_flops, unfused_ms):.2f},"
            f"{effective_tflops(logical_flops, fused_ms):.2f}",
            flush=True,
        )


def main() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    torch.set_grad_enabled(False)
    print(f"GPU={torch.cuda.get_device_name()} dtype={DTYPE} E={E} H={H} F={F}")
    _run_capacity_sweep()


if __name__ == "__main__":
    main()
