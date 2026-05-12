"""
All-shape autotune script for SSDChunkScanFwdKernel.

For each production shape:
  1. Construct kernel with tune=True (runs full autotune over expanded config space).
  2. Check correctness of the winning config against PyTorch reference.
  3. Time the winning config in steady state.
  4. Record shape, winning config, and latency.

Output: autotune_results.json + autotune_results.md
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from tileops.kernels.mamba.ssd_chunk_scan import SSDChunkScanFwdKernel

# ---------------------------------------------------------------------------
# Production shapes only (unit-scale shapes excluded — autotune overhead not
# worth it for smoke shapes that are never on the critical path).
# (batch, num_chunks, chunk_len, n_heads, d_head, d_state, n_groups, dtype, label)
# ---------------------------------------------------------------------------
_SHAPES = [
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
_N_TIME   = 30


def _gen_inputs(b, c, L, h, p, n, g, dtype):
    S = c * L
    x           = torch.randn(b, S, h, p,    dtype=dtype,         device="cuda") * 0.1
    cb          = torch.randn(b, c, g, L, L, dtype=dtype,         device="cuda") * 0.1
    dA_cumsum   = -(torch.rand(b, h, c, L,   dtype=torch.float32, device="cuda") * 0.1).cumsum(-1)
    C_mat       = torch.randn(b, S, g, n,    dtype=dtype,         device="cuda") * 0.1
    prev_states = torch.randn(b, c, h, p, n, dtype=torch.float32, device="cuda") * 0.1
    dt          = torch.rand( b, h, c, L,    dtype=dtype,         device="cuda") * 0.1 + 0.01
    return x, cb, dA_cumsum, C_mat, prev_states, dt


def _ref(x, cb, dA_cumsum, C_mat, prev_states, dt, n_groups):
    b, S, h, p = x.shape
    _, _, c, L = dA_cumsum.shape
    n = C_mat.shape[-1]
    hpg = h // n_groups
    x_c = x.float().reshape(b, c, L, h, p)
    C_c = C_mat.float().reshape(b, c, L, n_groups, n)
    C_h = C_c[:, :, :, torch.arange(h, device=x.device) // hpg, :]
    dA  = dA_cumsum.float().permute(0, 2, 3, 1)
    y_off = torch.einsum("bclhn,bchpn->bclhp", C_h, prev_states.float())
    y_off = y_off * torch.exp(dA).unsqueeze(-1)
    cb_h  = cb.float()[:, :, torch.arange(h, device=x.device) // hpg, :, :]
    dA_l  = dA_cumsum.float().unsqueeze(-1)
    dA_s  = dA_cumsum.float().unsqueeze(-2)
    decay = torch.exp(dA_l - dA_s)
    mask  = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
    decay = decay.masked_fill(~mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), 0.0)
    decay = decay.permute(0, 2, 1, 3, 4)
    dt_s  = dt.float().permute(0, 2, 1, 3).unsqueeze(-2)
    lcb   = cb_h * decay * dt_s
    y_diag = torch.einsum("bchls,bcshp->bclhp", lcb, x_c)
    return (y_off + y_diag).reshape(b, S, h, p)


def _time_kernel(kern, inputs, n_warmup, n_time):
    for _ in range(n_warmup):
        with torch.no_grad():
            kern.forward(*inputs)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n_time):
        s.record()
        with torch.no_grad():
            kern.forward(*inputs)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sorted(times)[n_time // 2]


def main():
    torch.cuda.set_device(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Autotuning {len(_SHAPES)} shapes with expanded config space\n")

    results = []

    for (b, c, L, h, p, n, g, dtype, label) in _SHAPES:
        print(f"[{label}]")
        inputs = _gen_inputs(b, c, L, h, p, n, g, dtype)

        # --- autotune (JIT + search outside timed region) ---
        t0 = time.perf_counter()
        kern = SSDChunkScanFwdKernel(b, c, L, h, p, n, g, dtype, tune=True)
        t1 = time.perf_counter()
        print(f"  autotune: {(t1-t0):.1f}s  best_config={kern.config}")

        # --- correctness ---
        with torch.no_grad():
            out = kern.forward(*inputs)
            ref = _ref(*inputs, g)
        err = (out.float() - ref.float()).abs().max().item()
        ok  = err < 0.02
        print(f"  correctness: {'OK' if ok else 'FAIL'}  max_err={err:.2e}")
        if not ok:
            print(f"  WARNING: correctness failed for {label}, skipping timing")
            results.append({"label": label, "config": kern.config,
                            "correctness": False, "latency_ms": None})
            continue

        # --- steady-state timing ---
        lat = _time_kernel(kern, inputs, _N_WARMUP, _N_TIME)
        print(f"  latency: {lat:.4f} ms")

        results.append({
            "label":       label,
            "batch":       b,
            "num_chunks":  c,
            "chunk_len":   L,
            "n_heads":     h,
            "d_head":      p,
            "d_state":     n,
            "n_groups":    g,
            "dtype":       "float16" if dtype == torch.float16 else "bfloat16",
            "config":      kern.config,
            "correctness": True,
            "latency_ms":  lat,
        })

    # --- save JSON ---
    json_path = "autotune_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # --- save markdown ---
    md_path = "autotune_results.md"
    lines = [
        "# SSD Chunk Scan — Autotune Results",
        "",
        f"GPU: {torch.cuda.get_device_name(0)}",
        "",
        "| label | latency_ms | block_l | block_p | block_n | block_s | threads |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r["latency_ms"] is None:
            lines.append(f"| {r['label']} | FAIL | — | — | — | — | — |")
            continue
        cfg = r["config"]
        lines.append(
            f"| {r['label']} | {r['latency_ms']:.4f} "
            f"| {cfg['block_l']} | {cfg['block_p']} "
            f"| {cfg['block_n']} | {cfg['block_s']} | {cfg['threads']} |"
        )
    lines.append("")
    Path(md_path).write_text("\n".join(lines))
    print(f"Markdown saved to {md_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
