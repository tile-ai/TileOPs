"""NCU profiling target for SSDStatePassingFwdKernel.

Runs all benchmark shapes once (with pre-warm) so that ncu captures
one kernel invocation per shape in a single .ncu-rep file.

Usage
-----
# Full set, both TileOPs and Triton (default):
ncu --set full --target-processes all \\
    -o ncu_reports/state_passing_all.ncu-rep \\
    python scripts/ncu_ssd_state_passing.py

# TileOPs only, skip Triton baseline:
ncu --set full --target-processes all \\
    -o ncu_reports/state_passing_tileops.ncu-rep \\
    python scripts/ncu_ssd_state_passing.py --no-triton

# Single shape (use exact id string):
ncu --set full --target-processes all \\
    -o ncu_reports/state_passing_serving-130m-4k.ncu-rep \\
    python scripts/ncu_ssd_state_passing.py --shape serving-130m-4k

# Override tile config (skips autotune):
python scripts/ncu_ssd_state_passing.py --no-tune \\
    --block-d 128 --threads 128 --no-vectorize
"""

import argparse
import os

import torch

os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")

# ---------------------------------------------------------------------------
# Benchmark shapes — matches bench_mamba.py SHAPES_SSD_STATE_PASSING
# Schema: (id, batch, num_chunks, n_heads, d_state, dtype)
# ---------------------------------------------------------------------------
SHAPES = [
    # ── 130M (n_heads=24) ──
    ("latency-130m-4k",  1,  16, 24, 128, torch.float16),
    ("serving-130m-4k",  8,  16, 24, 128, torch.float16),
    ("longctx-130m-32k", 4, 128, 24, 128, torch.float16),
    # ── 370M (n_heads=32) ──
    ("latency-370m-4k",  1,  16, 32, 128, torch.float16),
    ("serving-370m-4k",  8,  16, 32, 128, torch.float16),
    ("longctx-370m-32k", 4, 128, 32, 128, torch.float16),
    ("throughput-370m-2k", 32, 8, 32, 128, torch.float16),
    # ── 780M (n_heads=48) ──
    ("latency-780m-4k",  1,  16, 48, 128, torch.float16),
    ("serving-780m-4k",  8,  16, 48, 128, torch.float16),
    ("longctx-780m-32k", 4, 128, 48, 128, torch.float16),
    # ── 1.3B (n_heads=64) ──
    ("latency-1p3b-4k",  1,  16, 64, 128, torch.float16),
    ("serving-1p3b-4k",  8,  16, 64, 128, torch.float16),
    ("longctx-1p3b-32k", 2, 128, 64, 128, torch.float16),
    # ── 2.7B (n_heads=80) ──
    ("latency-2p7b-4k",  1,  16, 80, 128, torch.float16),
    ("serving-2p7b-4k",  4,  16, 80, 128, torch.float16),
    ("longctx-2p7b-32k", 2, 128, 80, 128, torch.float16),
]


def make_inputs(batch, num_chunks, n_heads, d_state, dtype, device="cuda"):
    states = torch.randn(batch, num_chunks, n_heads, d_state, dtype=dtype, device=device)
    dA = torch.randn(batch, n_heads, num_chunks, dtype=torch.float32, device=device)
    init = torch.randn(batch, n_heads, d_state, dtype=torch.float32, device=device)
    return states, dA, init


def run_shape(shape_id, batch, num_chunks, n_heads, d_state, dtype, args):
    from tileops.ops.ssd_state_passing import SSDStatePassingFwdOp

    config = None
    if args.block_d is not None:
        config = {
            "block_d": args.block_d,
            "threads": args.threads or (args.block_d // 2 if args.vectorize else 128),
            "vectorize": args.vectorize,
        }

    tune = not args.no_tune and config is None
    op = SSDStatePassingFwdOp(
        batch, num_chunks, n_heads, d_state, dtype=dtype, tune=tune,
    )
    if config is not None:
        op.kernel.config = config

    states, dA, init = make_inputs(batch, num_chunks, n_heads, d_state, dtype)

    # Pre-warm: ensure JIT compilation and autotuner cache are settled.
    for _ in range(3):
        op(states, dA, init)
    torch.cuda.synchronize()

    # Single profiled invocation — ncu captures this.
    op(states, dA, init)
    torch.cuda.synchronize()

    print(f"[TileOPs] {shape_id}: config={op.kernel.config}")


def run_triton_shape(shape_id, batch, num_chunks, n_heads, d_state, dtype):
    try:
        from mamba_ssm.ops.triton.ssd_state_passing import (
            _state_passing_fwd as _mamba_state_passing_fwd,
        )
    except ImportError:
        print(f"[Triton] {shape_id}: mamba_ssm not installed, skipping")
        return

    states, dA, init = make_inputs(batch, num_chunks, n_heads, d_state, dtype)
    states_f32 = states.float()

    for _ in range(3):
        _mamba_state_passing_fwd(states_f32, dA, init, has_initial_states=True)
    torch.cuda.synchronize()

    _mamba_state_passing_fwd(states_f32, dA, init, has_initial_states=True)
    torch.cuda.synchronize()

    print(f"[Triton] {shape_id}: done")


def main():
    parser = argparse.ArgumentParser(description="NCU profiling target for SSDStatePassingFwdKernel")
    parser.add_argument("--shape", default=None, help="Run a single shape by id (e.g. serving-130m-4k)")
    parser.add_argument("--no-triton", action="store_true", help="Skip Triton baseline")
    parser.add_argument("--no-tune", action="store_true", help="Skip autotune, use explicit config")
    parser.add_argument("--block-d", type=int, default=None, help="Explicit block_d override")
    parser.add_argument("--threads", type=int, default=None, help="Explicit threads override")
    parser.add_argument("--vectorize", action="store_true", default=False,
                        help="Use vectorized kernel (threads = block_d // 2)")
    parser.add_argument("--no-vectorize", dest="vectorize", action="store_false",
                        help="Use non-vectorized kernel (default)")
    args = parser.parse_args()

    shapes = SHAPES
    if args.shape is not None:
        shapes = [s for s in SHAPES if s[0] == args.shape]
        if not shapes:
            raise ValueError(f"Unknown shape id: {args.shape!r}. Valid ids: {[s[0] for s in SHAPES]}")

    for shape_id, batch, num_chunks, n_heads, d_state, dtype in shapes:
        print(f"\n=== {shape_id} (B={batch}, C={num_chunks}, H={n_heads}, D={d_state}) ===")
        run_shape(shape_id, batch, num_chunks, n_heads, d_state, dtype, args)
        if not args.no_triton:
            run_triton_shape(shape_id, batch, num_chunks, n_heads, d_state, dtype)


if __name__ == "__main__":
    main()
