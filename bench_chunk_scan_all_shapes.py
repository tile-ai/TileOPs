"""
Nsight Compute profiling harness for SSDChunkScanFwdKernel.

Workflow
--------
1. Construct one kernel object per shape (JIT/autotune excluded from timing).
2. Run correctness check against the PyTorch reference (outside timed region).
3. Warm-up the kernel (outside NCU measurement window).
4. Execute one steady-state call per shape so NCU collects metrics.
5. Emit a markdown summary table alongside the .ncu-rep produced by the caller.

Usage (intended to be called under ncu):
    ncu --set full \\
        --target-processes all \\
        --force-overwrite \\
        -o merged_ssd_chunk_scan_all_shapes \\
        python bench_chunk_scan_all_shapes.py --profile-all-shapes --no-compile-in-loop

The flags --profile-all-shapes and --no-compile-in-loop are accepted but not
required; the script behaves the same with or without them so it can also be
run standalone for a quick sanity-check.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Ensure repo is importable when invoked from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from tileops.kernels.mamba.ssd_chunk_scan import SSDChunkScanFwdKernel

# ---------------------------------------------------------------------------
# Benchmark shapes
# (batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, label)
#
# Model-to-shape mapping (Mamba-2 defaults):
#   n_heads = d_model / 32,  head_dim = 64,  d_state = 128,  chunk_len = 256
#   num_chunks = seq_len // chunk_len  (chunk_len=256: 2k->8, 4k->16, 32k->128)
#   n_groups = 1 (Mamba-2 standard)
#
#   130M -> n_heads=24   370M -> n_heads=32   780M -> n_heads=48
#   1.3B -> n_heads=64   2.7B -> n_heads=80
# ---------------------------------------------------------------------------
_SHAPES = [
    # ── unit-scale (smoke) ──
    (1,  2,   64,  4,  64,  32, 1, torch.float16,  "unit-b1-c2-L64-h4-p64-n32-fp16"),
    (2,  4,   64,  8,  64,  64, 2, torch.float16,  "unit-b2-c4-L64-h8-p64-n64-fp16"),
    (1,  2,  128,  4, 128,  32, 1, torch.bfloat16, "unit-b1-c2-L128-h4-p128-n32-bf16"),
    (2,  2,   64,  4,  64,  32, 2, torch.bfloat16, "unit-b2-c2-L64-h4-p64-n32-bf16"),
    # ── 130M (n_heads=24) ──
    (1,  16, 256, 24, 64, 128, 1, torch.float16, "latency-130m-4k"),
    (8,  16, 256, 24, 64, 128, 1, torch.float16, "serving-130m-4k"),
    (4, 128, 256, 24, 64, 128, 1, torch.float16, "longctx-130m-32k"),
    # ── 370M (n_heads=32) ──
    (1,  16, 256, 32, 64, 128, 1, torch.float16, "latency-370m-4k"),
    (8,  16, 256, 32, 64, 128, 1, torch.float16, "serving-370m-4k"),
    (4, 128, 256, 32, 64, 128, 1, torch.float16, "longctx-370m-32k"),
    (32,  8, 256, 32, 64, 128, 1, torch.float16, "throughput-370m-2k"),
    # ── 780M (n_heads=48) ──
    (1,  16, 256, 48, 64, 128, 1, torch.float16, "latency-780m-4k"),
    (8,  16, 256, 48, 64, 128, 1, torch.float16, "serving-780m-4k"),
    (4, 128, 256, 48, 64, 128, 1, torch.float16, "longctx-780m-32k"),
    (16,  8, 256, 48, 64, 128, 1, torch.float16, "throughput-780m-2k"),
    # ── 1.3B (n_heads=64) ──
    (1,  16, 256, 64, 64, 128, 1, torch.float16, "latency-1p3b-4k"),
    (8,  16, 256, 64, 64, 128, 1, torch.float16, "serving-1p3b-4k"),
    (2, 128, 256, 64, 64, 128, 1, torch.float16, "longctx-1p3b-32k"),
    (8,   8, 256, 64, 64, 128, 1, torch.float16, "throughput-1p3b-2k"),
    # ── 2.7B (n_heads=80) ──
    (1,  16, 256, 80, 64, 128, 1, torch.float16, "latency-2p7b-4k"),
    (4,  16, 256, 80, 64, 128, 1, torch.float16, "serving-2p7b-4k"),
    (2, 128, 256, 80, 64, 128, 1, torch.float16, "longctx-2p7b-32k"),
    (4,   8, 256, 80, 64, 128, 1, torch.float16, "throughput-2p7b-2k"),
]

_N_WARMUP = 10


# ---------------------------------------------------------------------------
# PyTorch reference (correctness gate; not inside timed / NCU region)
# ---------------------------------------------------------------------------

def _ref_forward(x, cb, dA_cumsum, C_mat, prev_states, dt, n_groups):
    """Official-aligned PyTorch reference for chunk scan."""
    b, S, h, p = x.shape
    _, _, c, L = dA_cumsum.shape
    n = C_mat.shape[-1]
    heads_per_group = h // n_groups

    x_chunked = x.float().reshape(b, c, L, h, p)
    C_chunked = C_mat.float().reshape(b, c, L, n_groups, n)
    C_heads = C_chunked[
        :, :, :, torch.arange(h, device=x.device) // heads_per_group, :
    ]

    dA = dA_cumsum.float().permute(0, 2, 3, 1)  # [B, C, L, H]
    y_off = torch.einsum("bclhn,bchpn->bclhp", C_heads, prev_states.float())
    y_off = y_off * torch.exp(dA).unsqueeze(-1)

    cb_heads = cb.float()[
        :, :, torch.arange(h, device=x.device) // heads_per_group, :, :
    ]

    dA_l = dA_cumsum.float().unsqueeze(-1)   # [B, H, C, L, 1]
    dA_s = dA_cumsum.float().unsqueeze(-2)   # [B, H, C, 1, L]
    decay = torch.exp(dA_l - dA_s)
    mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), 0.0)
    decay = decay.permute(0, 2, 1, 3, 4)     # [B, C, H, L, L]

    dt_s = dt.float().permute(0, 2, 1, 3).unsqueeze(-2)  # [B, C, H, 1, L]
    lcb = cb_heads * decay * dt_s             # [B, C, H, L, L]
    y_diag = torch.einsum("bchls,bcshp->bclhp", lcb, x_chunked)

    return (y_off + y_diag).reshape(b, S, h, p)


def _gen_inputs(batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype):
    b, c, L, h, p, n, g = batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups
    S = c * L
    x           = torch.randn(b, S, h, p,    dtype=dtype,         device="cuda") * 0.1
    cb          = torch.randn(b, c, g, L, L, dtype=dtype,         device="cuda") * 0.1
    # Keep cumsum values in a numerically safe range: each step ~U(0, 0.1),
    # so the cumsum over L=256 stays within ~[0, 25.6], well within exp() range.
    dA_cumsum   = -(torch.rand(b, h, c, L, dtype=torch.float32, device="cuda") * 0.1).cumsum(-1)
    C_mat       = torch.randn(b, S, g, n,    dtype=dtype,         device="cuda") * 0.1
    prev_states = torch.randn(b, c, h, p, n, dtype=torch.float32, device="cuda") * 0.1
    dt          = torch.rand( b, h, c, L,    dtype=dtype,         device="cuda") * 0.1 + 0.01
    return x, cb, dA_cumsum, C_mat, prev_states, dt


def _check_correctness(kernel, inputs, n_groups, label):
    x, cb, dA_cumsum, C_mat, prev_states, dt = inputs
    with torch.no_grad():
        out = kernel.forward(x, cb, dA_cumsum, C_mat, prev_states, dt)
        ref = _ref_forward(x, cb, dA_cumsum, C_mat, prev_states, dt, n_groups)
    atol = 1e-2
    rtol = 1e-2
    if not torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol):
        max_err = (out.float() - ref.float()).abs().max().item()
        raise AssertionError(
            f"[{label}] Correctness FAILED: max_abs_err={max_err:.4e} (tol={atol})"
        )
    print(f"  [OK] {label} correctness passed (max_err="
          f"{(out.float()-ref.float()).abs().max().item():.2e})")


# ---------------------------------------------------------------------------
# FLOP / memory helpers for the summary table
# ---------------------------------------------------------------------------

def _compute_flops(batch, num_chunks, chunk_len, n_heads, d_head, d_state):
    b, c, L, h, p, n = batch, num_chunks, chunk_len, n_heads, d_head, d_state
    history = b * c * L * h * 2 * n * p
    diag    = b * c * h * (L * (L + 1) // 2) * 2 * p
    return history + diag


def _compute_memory_bytes(batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype):
    b, c, L, h, p, n, g = (
        batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups,
    )
    S = c * L
    elem = torch.tensor([], dtype=dtype).element_size()
    reads = (
        b * S * h * p
        + b * c * g * L * L
        + b * S * g * n
        + b * c * h * p * n
        + b * h * c * L
    ) * elem
    reads += b * h * c * L * 4   # dA_cumsum float32
    writes = b * S * h * p * 4   # out float32
    return reads + writes


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _emit_summary(rows, out_path):
    header = (
        "| label | latency_ms | tflops | bw_tb_s | "
        "batch | chunks | chunk_len | n_heads | d_head | d_state | dtype |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [
        "# SSD Chunk Scan Forward — Benchmark Summary",
        "",
        header,
        sep,
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['latency_ms']:.4f} | "
            f"{r.get('tflops', float('nan')):.2f} | "
            f"{r.get('bw_tb_s', float('nan')):.2f} | "
            f"{r['batch']} | {r['num_chunks']} | {r['chunk_len']} | "
            f"{r['n_heads']} | {r['d_head']} | {r['d_state']} | "
            f"{r['dtype_str']} |"
        )
    lines.append("")
    text = "\n".join(lines)
    Path(out_path).write_text(text)
    print(f"\nSummary table written to {out_path}")
    print(text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SSD chunk scan NCU profiling harness")
    parser.add_argument("--profile-all-shapes", action="store_true",
                        help="Profile all shapes (default behaviour; flag accepted for NCU CLI)")
    parser.add_argument("--no-compile-in-loop", action="store_true",
                        help="Accepted for NCU CLI compatibility; JIT is always outside the loop")
    parser.add_argument("--summary-out", default="ssd_chunk_scan_summary.md",
                        help="Path for the markdown summary table")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Profiling {len(_SHAPES)} shapes\n")

    summary_rows = []

    for (batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, label) in _SHAPES:
        dtype_str = "float16" if dtype == torch.float16 else "bfloat16"
        print(f"[{label}]")

        # ------------------------------------------------------------------
        # Step 1: construct kernel (JIT/autotune outside timed region)
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        kernel = SSDChunkScanFwdKernel(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
        )
        # Materialise the JIT-compiled kernel object by calling it once
        inputs = _gen_inputs(batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype)
        with torch.no_grad():
            _ = kernel.forward(*inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"  JIT build: {(t1-t0)*1e3:.0f} ms")

        # ------------------------------------------------------------------
        # Step 2: correctness check (outside timed / NCU region)
        # ------------------------------------------------------------------
        _check_correctness(kernel, inputs, n_groups, label)

        # ------------------------------------------------------------------
        # Step 3: warm-up (outside NCU measured region — NCU replays the
        #         kernel independently, but warming up removes one-shot JIT
        #         stalls and ensures the GPU is in steady state)
        # ------------------------------------------------------------------
        for _ in range(_N_WARMUP):
            with torch.no_grad():
                _ = kernel.forward(*inputs)
        torch.cuda.synchronize()

        # ------------------------------------------------------------------
        # Step 4: steady-state call — NCU will intercept this launch
        # ------------------------------------------------------------------
        with torch.no_grad():
            kernel.forward(*inputs)
        torch.cuda.synchronize()

        # ------------------------------------------------------------------
        # Step 5: lightweight CUDA-event timing for the summary table
        #         (independent of NCU; gives a rough latency number)
        # ------------------------------------------------------------------
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        _N_TIME  = 20
        latencies = []
        for _ in range(_N_TIME):
            start_ev.record()
            with torch.no_grad():
                _ = kernel.forward(*inputs)
            end_ev.record()
            torch.cuda.synchronize()
            latencies.append(start_ev.elapsed_time(end_ev))
        latency_ms = sorted(latencies)[len(latencies) // 2]

        flops    = _compute_flops(batch, num_chunks, chunk_len, n_heads, d_head, d_state)
        mem_b    = _compute_memory_bytes(
            batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype,
        )
        tflops   = flops / latency_ms * 1e-9
        bw_tb_s  = mem_b / latency_ms * 1e-9

        print(f"  latency={latency_ms:.4f} ms  tflops={tflops:.2f}  bw={bw_tb_s:.2f} TB/s")

        summary_rows.append({
            "label":      label,
            "latency_ms": latency_ms,
            "tflops":     tflops,
            "bw_tb_s":    bw_tb_s,
            "batch":      batch,
            "num_chunks": num_chunks,
            "chunk_len":  chunk_len,
            "n_heads":    n_heads,
            "d_head":     d_head,
            "d_state":    d_state,
            "n_groups":   n_groups,
            "dtype_str":  dtype_str,
        })

    _emit_summary(summary_rows, args.summary_out)


if __name__ == "__main__":
    main()
