"""Per-kernel breakdown profiling: vLLM CUTLASS vs TileOPs padded vs TileOPs nopad.

Decomposes each pipeline into individual stages and measures each separately
so we can identify where time is spent and where the next improvement lies.

Stages measured:
  padded : FusedTopK | Permute | GEMM_gate_up | SiluAndMul | GEMM_down | Unpermute
  nopad  : FusedTopK | Permute | Sched | GEMM_gate_up | SiluAndMul | Sched | GEMM_down | Unpermute
  vLLM   : torch.profiler top-CUDA-kernel breakdown

Usage:
    conda run -n tileops python benchmarks/ops/bench_moe_kernel_breakdown.py
"""

import torch

# ── Optional vLLM ───────────────────────────────────────────────────────────
try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as _vllm_fused_experts
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from tileops.kernels.moe.moe_grouped_gemm_nopad import (
    _SCHED_THREADS,
    _moe_grouped_gemm_kernel,
    _tile_scheduler_kernel,
)
from tileops.ops.elementwise import SiluAndMulOp
from tileops.ops.grouped_gemm import GroupedGemmOp
from tileops.ops.moe import FusedTopKOp, MoePermuteNopadOp, MoePermutePaddedOp, MoeUnpermuteOp

# ── Config ───────────────────────────────────────────────────────────────────
CONFIGS = [
    # (T,    E,   K, H,    F,    scoring,    renorm)
    (512,  128,  8, 2048, 1024, "softmax", False),
    (2048, 128,  8, 2048, 1024, "softmax", False),
    (4096, 128,  8, 2048, 1024, "softmax", False),
    (512,  256,  8, 2048, 1024, "softmax", True),
    (2048, 256,  8, 2048, 1024, "softmax", True),
    (4096, 256,  8, 2048, 1024, "softmax", True),
]
DTYPE = torch.bfloat16
_BLOCK_M = 64   # GroupedGemmOp NT default
WARMUP = 50
ITERS  = 200


# ── Timing utility ───────────────────────────────────────────────────────────

def bench(fn, *args, warmup=WARMUP, iters=ITERS) -> float:
    """Return median single-call latency in ms using CUDA events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def print_breakdown(title: str, stages: list[tuple[str, float]]) -> None:
    total = sum(t for _, t in stages)
    print(f"\n  {'Stage':<26}  {'ms':>7}  {'%':>6}")
    print(f"  {'-'*26}  {'-'*7}  {'-'*6}")
    for name, t in stages:
        print(f"  {name:<26}  {t:7.3f}  {t/total*100:5.1f}%")
    print(f"  {'TOTAL':<26}  {total:7.3f}  100.0%")
    print("  (end-to-end via full op: see main benchmark)")


# ── Input generation ─────────────────────────────────────────────────────────

def gen_inputs(T, E, K, H, F):
    torch.manual_seed(42)
    dev = "cuda"
    hidden  = torch.randn(T, H, dtype=DTYPE, device=dev)
    gating  = torch.randn(T, E, dtype=DTYPE, device=dev)
    w_gu    = torch.randn(E, F * 2, H, dtype=DTYPE, device=dev) * 0.02
    w_down  = torch.randn(E, H, F, dtype=DTYPE, device=dev) * 0.02
    return hidden, gating, w_gu, w_down


# ── Stage decompositions ─────────────────────────────────────────────────────

def profile_padded(T, E, K, H, F, scoring_func, renormalize):
    numel = T * K
    padded = numel + E * (_BLOCK_M - 1)
    hidden, gating, w_gu, w_down = gen_inputs(T, E, K, H, F)

    # Build ops
    topk_op    = FusedTopKOp(T, E, K, scoring_func, renormalize)
    permute_op = MoePermutePaddedOp(T, K, E, H, DTYPE, block_m=_BLOCK_M)
    gemm_gu    = GroupedGemmOp(padded, E, F * 2, H, DTYPE)
    silu_op    = SiluAndMulOp(M=padded, N=F, dtype=DTYPE)
    gemm_dn    = GroupedGemmOp(padded, E, H, F, DTYPE)
    unp_op     = MoeUnpermuteOp(T, K, H, DTYPE, padded_batch_sum=padded)

    # Warm-up full pass to compile all kernels
    tw, tids = topk_op(gating)
    ph, po, ps, _, fi = permute_op(hidden, tids)
    gu = gemm_gu(ph, w_gu, ps, po, po)
    ac = silu_op(gu)
    mm = gemm_dn(ac, w_down, ps, po, po)
    unp_op(mm, fi, tw)
    torch.cuda.synchronize()

    # Pre-compute routing (shared)
    tw, tids = topk_op(gating)
    ph, po, ps, _, fi = permute_op(hidden, tids)
    gu = gemm_gu(ph, w_gu, ps, po, po)
    ac = silu_op(gu)
    mm = gemm_dn(ac, w_down, ps, po, po)

    t_topk    = bench(topk_op, gating)
    t_permute = bench(permute_op, hidden, tids)
    t_gu      = bench(gemm_gu, ph, w_gu, ps, po, po)
    t_silu    = bench(silu_op, gu)
    t_dn      = bench(gemm_dn, ac, w_down, ps, po, po)
    t_unp     = bench(unp_op, mm, fi, tw)

    return [
        ("FusedTopK",     t_topk),
        ("Permute(padded)", t_permute),
        ("GEMM gate+up",  t_gu),
        ("SiluAndMul",    t_silu),
        ("GEMM down",     t_dn),
        ("Unpermute",     t_unp),
    ]


def profile_nopad(T, E, K, H, F, scoring_func, renormalize):
    numel = T * K
    hidden, gating, w_gu, w_down = gen_inputs(T, E, K, H, F)

    # Build ops
    topk_op    = FusedTopKOp(T, E, K, scoring_func, renormalize)
    permute_op = MoePermuteNopadOp(T, K, E, H, DTYPE)
    unp_op     = MoeUnpermuteOp(T, K, H, DTYPE, padded_batch_sum=numel)
    silu_op    = SiluAndMulOp(M=numel, N=F, dtype=DTYPE)

    # Build tile scheduler + GEMM kernels directly (nopad internal)
    block_m, block_n, block_k, num_stages, threads = 64, 256, 64, 2, 128
    max_tiles = numel // block_m + E
    sched_gu_fn = _tile_scheduler_kernel(E, max_tiles, block_m)(_SCHED_THREADS)
    sched_dn_fn = _tile_scheduler_kernel(E, max_tiles, block_m)(_SCHED_THREADS)

    # Warm-up: full pass (also compiles scheduler)
    tw, tids = topk_op(gating)
    ph, to, ts, _, fi = permute_op(hidden, tids)
    tid_gu, tro_gu, tot_gu_t = sched_gu_fn(ts)
    torch.cuda.synchronize()
    total_tiles_gu = int(tot_gu_t.item())
    total_tiles_dn = total_tiles_gu  # same routing

    # Compile GEMM kernels with exact total_tiles (dynamic grid, zero dead CTAs)
    gemm_gu_fn = _moe_grouped_gemm_kernel(numel, total_tiles_gu, E, F * 2, H, "bfloat16")(
                     block_m, block_n, block_k, num_stages, threads)
    gemm_dn_fn = _moe_grouped_gemm_kernel(numel, total_tiles_dn, E, H, F, "bfloat16")(
                     block_m, block_n, block_k, num_stages, threads)

    gu = gemm_gu_fn(ph, w_gu, tid_gu, tro_gu, to, ts)
    ac = silu_op(gu)
    tid_dn, tro_dn, _ = sched_dn_fn(ts)
    mm = gemm_dn_fn(ac, w_down, tid_dn, tro_dn, to, ts)
    unp_op(mm, fi, tw)
    torch.cuda.synchronize()

    # Pre-compute intermediates for individual stage timing
    tw, tids = topk_op(gating)
    ph, to, ts, _, fi = permute_op(hidden, tids)
    tid_gu, tro_gu, _ = sched_gu_fn(ts)
    gu = gemm_gu_fn(ph, w_gu, tid_gu, tro_gu, to, ts)
    ac = silu_op(gu)
    tid_dn, tro_dn, _ = sched_dn_fn(ts)
    mm = gemm_dn_fn(ac, w_down, tid_dn, tro_dn, to, ts)

    t_topk    = bench(topk_op, gating)
    t_permute = bench(permute_op, hidden, tids)
    t_sched   = bench(sched_gu_fn, ts)   # same for gate+up and down
    t_gu      = bench(gemm_gu_fn, ph, w_gu, tid_gu, tro_gu, to, ts)
    t_silu    = bench(silu_op, gu)
    t_dn      = bench(gemm_dn_fn, ac, w_down, tid_dn, tro_dn, to, ts)
    t_unp     = bench(unp_op, mm, fi, tw)

    return [
        ("FusedTopK",      t_topk),
        ("Permute(nopad)", t_permute),
        ("TileSched(×2)",  t_sched * 2),
        ("GEMM gate+up",   t_gu),
        ("SiluAndMul",     t_silu),
        ("GEMM down",      t_dn),
        ("Unpermute",      t_unp),
    ]


def profile_vllm(T, E, K, H, F, scoring_func, renormalize, iters=5):
    """Run vLLM fused_experts under torch.profiler; return sorted CUDA kernel table."""
    hidden, gating, w_gu, w_down = gen_inputs(T, E, K, H, F)
    topk_op = FusedTopKOp(T, E, K, scoring_func, renormalize)
    tw, tids = topk_op(gating)
    # vLLM expects int64 topk_ids
    tids_i64 = tids.to(torch.int64)

    def fn():
        return _vllm_fused_experts(hidden, w_gu, w_down, tw, tids_i64)

    for _ in range(20):
        fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()

    events = prof.key_averages()
    cuda_ev = [(e.key.split("/")[-1][:48], e.device_time_total / iters / 1e3)
               for e in events if e.device_time_total > 0]
    cuda_ev.sort(key=lambda x: x[1], reverse=True)
    total = sum(t for _, t in cuda_ev)
    return cuda_ev, total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    assert torch.cuda.is_available()

    for (T, E, K, H, F, scoring, renorm) in CONFIGS:
        title = f"T={T}, E={E}, K={K}, scoring={scoring}"
        print(f"\n{'='*65}")
        print(f"  {title}")
        print(f"{'='*65}")

        # ── TileOPs padded ────────────────────────────────────────────────
        print("\n[TileOPs PADDED]")
        stages_pad = profile_padded(T, E, K, H, F, scoring, renorm)
        print_breakdown(title, stages_pad)

        # ── TileOPs nopad ─────────────────────────────────────────────────
        print("\n[TileOPs NOPAD]")
        stages_nop = profile_nopad(T, E, K, H, F, scoring, renorm)
        print_breakdown(title, stages_nop)

        # ── vLLM (softmax only) ───────────────────────────────────────────
        if _VLLM_AVAILABLE and scoring == "softmax":
            print("\n[vLLM CUTLASS top CUDA kernels]")
            kv, total = profile_vllm(T, E, K, H, F, scoring, renorm)
            print(f"\n  {'Kernel':<48}  {'ms':>7}  {'%':>6}")
            print(f"  {'-'*48}  {'-'*7}  {'-'*6}")
            for name, t in kv[:15]:
                print(f"  {name:<48}  {t:7.3f}  {t/total*100:5.1f}%")
            print(f"  {'TOTAL (top-15)':<48}  {sum(t for _,t in kv[:15]):7.3f}")
            print(f"  {'TOTAL (all kernels)':<48}  {total:7.3f}")


if __name__ == "__main__":
    main()
