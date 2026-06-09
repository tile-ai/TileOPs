"""Grouped GEMM benchmark: TileOPs 3-WG persistent kernel vs external baselines.

Compares the pure (no activation) ``GroupedGemmPersistent3WGKernel`` against the
strongest publicly available grouped-GEMM implementations on Hopper, for the
**prefill** regime in bf16:

  * ``torch``    -- ``torch._grouped_mm`` (PyTorch 2.10+, CUTLASS-backed).
  * ``triton``   -- the official Triton ``08-grouped-gemm`` tutorial kernel,
                    ported here and switched fp16 -> bf16. Both the pointer-array
                    kernel and the Hopper TMA variant are run.
  * ``deepgemm`` -- DeepGEMM ``m_grouped_bf16_gemm_nt_contiguous`` (contiguous
                    variable-M grouped GEMM, NT layout, bf16).

A cuBLAS reference (DeepGEMM ``cublaslt_gemm_nt`` looped per expert) is a planned
addition; it is omitted for now because that op intermittently raises
"doesn't have storage" in this environment (a DeepGEMM workspace-allocation bug),
which makes a several-hundred-expert loop unreliable.

Canonical weight layout is ``B = [E, N, K]`` (NT, i.e. ``C = A @ B[e]^T`` per
expert) -- matches our kernel and DeepGEMM directly; torch and the non-TMA
Triton kernel get a transposed ``[E, K, N]`` copy.

Token->expert routing is **uniform** (every expert receives ``M = numel // E``
rows, which is exactly how the model table's M column is derived:
``M = tokens * top_k / num_experts``). Skewed/random distributions are a planned
follow-up.

Cases are real MoE prefill shapes (GLM-5 744B, Llama-4-17B-128E, qwen3.5 397B);
see ``CASES`` below for the per-model parameters.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

# Repeated multi-GB output allocations fragment the caching allocator;
# expandable segments keep the largest cases (M=8192, ~70-130 GB of live
# operands) from OOMing on reserved-but-unallocated memory, and let the
# single-process teardown reclaim cleanly between cases. Must be set before the
# first CUDA init, i.e. before torch is imported.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pytest  # noqa: E402
import torch  # noqa: E402

# Triton is a hard dependency of this benchmark's baseline kernels; skip the
# whole module (rather than error at collection) if it is absent.
triton = pytest.importorskip("triton")  # noqa: E402
tl = pytest.importorskip("triton.language")  # noqa: E402

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport  # noqa: E402
from tileops.kernels.grouped_gemm import GroupedGemmPersistent3WGKernel  # noqa: E402

try:
    import deep_gemm  # noqa: E402

    _HAS_DEEP_GEMM = True
except Exception:  # pragma: no cover - environment dependent
    deep_gemm = None
    _HAS_DEEP_GEMM = False

_DTYPE = torch.bfloat16

# Fixed report group name: all CASES record under this single op so the nightly
# report groups them as one op with N configs. Frozen — changing it resets that
# op's 14-day perf history.
_REPORT_NAME = "grouped_gemm_3wg_baselines"


def _num_sms() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count


def _supports_tma() -> bool:
    return torch.cuda.get_device_capability()[0] >= 9


# ─────────────────────────────────────────────────────────────────────────────
# Triton 08-grouped-gemm tutorial, ported (fp16 -> bf16).
# Upstream: triton-lang/triton python/tutorials/08-grouped-gemm.py
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "NUM_SM": 84}),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "NUM_SM": 128}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "NUM_SM": 84}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "NUM_SM": 128}),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "NUM_SM": 132}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "NUM_SM": 132}),
    ],
    key=["group_size"],
)
@triton.jit
def _grouped_matmul_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,  # [group_size, 3] = <M, N, K>
    g_lds,             # [group_size, 3] = <lda, ldb, ldc>
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for _kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb
            c = accumulator.to(tl.bfloat16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]
            tl.store(c_ptrs, c)
            tile_idx += NUM_SM
        last_problem_end = last_problem_end + num_tiles


_TMA_CONFIGS = [
    triton.Config({"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk}, num_stages=s, num_warps=w)
    for bm in [128]
    for bn in [128, 256]
    for bk in [64, 128]
    for s in [3, 4]
    for w in [4, 8]
]


@triton.autotune(_TMA_CONFIGS, key=["group_size"])
@triton.jit
def _grouped_matmul_tma_kernel(
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    group_gemm_sizes,  # [group_size, 3] = <M, N, K>
    g_lds,             # [group_size, 3] = <lda, ldb, ldc>
    group_size,
    NUM_SM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    dtype = tl.bfloat16
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        if tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(dtype))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(dtype))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(dtype))

            a_desc = tl.make_tensor_descriptor(
                a_ptr, shape=[gm, gk], strides=[lda, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
            b_desc = tl.make_tensor_descriptor(
                b_ptr, shape=[gn, gk], strides=[ldb, 1], block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
            c_desc = tl.make_tensor_descriptor(
                c_ptr, shape=[gm, gn], strides=[ldc, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

            while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
                k = gk
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles
                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N
                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                    accumulator += tl.dot(a, b.T)
                offs_cm = tile_m_idx * BLOCK_SIZE_M
                offs_cn = tile_n_idx * BLOCK_SIZE_N
                c_desc.store([offs_cm, offs_cn], accumulator.to(dtype))
                tile_idx += NUM_SM
        last_problem_end = last_problem_end + num_tiles


def _build_triton_ptrs(group_a, group_b, group_c):
    """Pack per-group tensors into the device pointer/size/ld arrays the kernels expect."""
    dev = group_a[0].device
    a_addrs, b_addrs, c_addrs, g_sizes, g_lds = [], [], [], [], []
    for a, b, c in zip(group_a, group_b, group_c, strict=True):
        m, _ = a.shape
        n = c.shape[1]
        k = a.shape[1]
        a_addrs.append(a.data_ptr())
        b_addrs.append(b.data_ptr())
        c_addrs.append(c.data_ptr())
        g_sizes += [m, n, k]
        g_lds += [a.stride(0), b.stride(0), c.stride(0)]
    return (
        torch.tensor(a_addrs, device=dev),
        torch.tensor(b_addrs, device=dev),
        torch.tensor(c_addrs, device=dev),
        torch.tensor(g_sizes, dtype=torch.int32, device=dev),
        torch.tensor(g_lds, dtype=torch.int32, device=dev),
    )


def _set_triton_allocator():
    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)


# ─────────────────────────────────────────────────────────────────────────────
# Input generation (uniform routing).
# ─────────────────────────────────────────────────────────────────────────────
def gen_inputs(numel, E, N, K):
    """Uniform routing: every expert gets ``numel // E`` rows (a multiple of 128).

    Returns the shared bf16 tensors plus the per-expert metadata each baseline needs.
    """
    assert numel % E == 0, f"numel={numel} not divisible by E={E}"
    per = numel // E
    assert per % 128 == 0, f"per-expert M={per} must be a multiple of 128 (DeepGEMM contiguous)"
    torch.manual_seed(42)
    dev = "cuda"

    A = torch.randn(numel, K, dtype=_DTYPE, device=dev) * 0.02
    B = torch.randn(E, N, K, dtype=_DTYPE, device=dev) * 0.02  # NT: C = A @ B[e]^T

    sizes = torch.full((E,), per, dtype=torch.int32, device=dev)
    offsets = torch.zeros(E, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    m_indices = torch.arange(E, device=dev, dtype=torch.int32).repeat_interleave(per)
    offs_cumsum = torch.cumsum(sizes, dim=0).to(torch.int32)  # torch._grouped_mm: no leading 0
    return A, B, sizes, offsets, m_indices, offs_cumsum, per


def _warmup_gpu(seconds=2.0):
    """Ramp the SM clock to its sustained state before any measurement.

    The benchmark process starts from an idle GPU at base clock, and the
    framework's short per-call warmup does not fully ramp it, which otherwise
    depresses the first case's numbers (cold-start outlier). A couple of seconds
    of dense matmul pins the clock for every case (idempotent on warm ones). Uses
    fp32 — this environment's bf16 ``torch.matmul`` (cublasGemmEx) raises
    CUBLAS_STATUS_INVALID_VALUE, so the measured baselines all avoid it and so
    must the warmup.
    """
    a = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
    b = torch.randn(4096, 4096, dtype=torch.float32, device="cuda")
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(16):
            a = a @ b
        torch.cuda.synchronize()


def _deepgemm_launch(A, B, D, m_indices, tries=8):
    """One DeepGEMM grouped-GEMM call, retrying its intermittent CPU-side
    workspace bug (``RuntimeError: ... doesn't have storage``).

    The error is raised before the kernel launches, so retrying the single call
    costs only host time and never pollutes the measured kernel time. This must
    be per-call, not wrapping the whole measurement loop: the benchmark's timing
    loop makes many calls per measurement, so a ~1-2% per-call failure rate would
    make a whole-measurement retry fail almost every time, whereas a per-call
    retry lets every iteration complete."""
    for _ in range(tries - 1):
        try:
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(A, B, D, m_indices)
            return
        except RuntimeError:
            pass
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(A, B, D, m_indices)


# ─────────────────────────────────────────────────────────────────────────────
# Model cases (prefill, bf16). M = tokens * top_k / num_experts (uniform).
# N/K are the single-expert GEMM dims. up/gate GEMMs fuse gate+up -> N = 2*inter.
# NOTE: qwen3.5 'down' N is carried verbatim from the source table (2048); it
# does not equal hidden (4096) like the other models' down GEMMs -- flagged but
# left as-given pending confirmation.
# ─────────────────────────────────────────────────────────────────────────────
CASES = [
    # label,                         tokens,  E,   top_k, hidden, moe_inter, M,    N,     K
    ("GLM-5-744B  up    T=32768",    32768,   256, 8,     6144,   2048,      1024, 4096,  6144),
    ("GLM-5-744B  up    T=65536",    65536,   256, 8,     6144,   2048,      2048, 4096,  6144),
    ("GLM-5-744B  up    T=131072",   131072,  256, 8,     6144,   2048,      4096, 4096,  6144),
    ("GLM-5-744B  up    T=262144",   262144,  256, 8,     6144,   2048,      8192, 4096,  6144),
    ("Llama4-128E up    T=131072",   131072,  128, 1,     5120,   8192,      1024, 16384, 5120),
    ("qwen3.5-397B up   T~52429",    52429,   512, 10,    4096,   1024,      1024, 2048,  4096),
    ("GLM-5-744B  down  T=32768",    32768,   256, 8,     6144,   2048,      1024, 6144,  2048),
    ("GLM-5-744B  down  T=65536",    65536,   256, 8,     6144,   2048,      2048, 6144,  2048),
    ("GLM-5-744B  down  T=131072",   131072,  256, 8,     6144,   2048,      4096, 6144,  2048),
    ("GLM-5-744B  down  T=262144",   262144,  256, 8,     6144,   2048,      8192, 6144,  2048),
    ("Llama4-128E down  T=131072",   131072,  128, 1,     5120,   8192,      1024, 5120,  8192),
    ("qwen3.5-397B down  T~52429",   52429,   512, 10,    4096,   1024,      1024, 2048,  1024),
]
