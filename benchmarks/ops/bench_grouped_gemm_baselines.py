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
import sys
import time

# Repeated multi-GB output allocations inside do_bench fragment the caching
# allocator; expandable segments keep the largest cases (M=8192, ~70 GB of live
# operands) from OOMing on reserved-but-unallocated memory. Must be set before
# the first CUDA init, i.e. before torch is imported.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
from tilelang.profiler import do_bench  # noqa: E402

from tileops.kernels.grouped_gemm import GroupedGemmPersistent3WGKernel  # noqa: E402

try:
    import deep_gemm  # noqa: E402

    _HAS_DEEP_GEMM = True
except Exception:  # pragma: no cover - environment dependent
    deep_gemm = None
    _HAS_DEEP_GEMM = False

_DTYPE = torch.bfloat16
_WARMUP = 25
_REP = 100

# DeepGEMM's large-M workspace bug ("doesn't have storage") is deterministic
# WITHIN a process (per-call retries cannot recover it) but flips between process
# launches, because DeepGEMM's first-call JIT autotune is nondeterministic and
# some picked configs hit the bug. So the driver relaunches a case subprocess up
# to _DG_MAX_ATTEMPTS times when DeepGEMM drops out; _run_one signals this via
# _DG_RETRY_EXIT.
_DG_RETRY_EXIT = 17
_DG_MAX_ATTEMPTS = 4


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


def tflops(ms, numel, N, K):
    return 2.0 * numel * N * K / ms * 1e-9


def _warmup_gpu(seconds=2.0):
    """Ramp the SM clock to its sustained state before any measurement.

    Each case runs in a fresh subprocess; the FIRST one starts from an idle GPU
    at base clock, and do_bench's 25-iter warmup is too short to fully ramp it,
    which otherwise depresses that case's numbers (cold-start outlier). A couple
    seconds of dense matmul pins the clock for every case (idempotent on warm
    ones). Uses fp32 — this environment's bf16 ``torch.matmul`` (cublasGemmEx)
    raises CUBLAS_STATUS_INVALID_VALUE, so the measured baselines all avoid it
    and so must the warmup.
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
    be per-call, not per-do_bench: a do_bench makes ~125 calls, so a ~1-2%
    per-call failure rate makes whole-do_bench retries fail almost every time,
    whereas a per-call retry lets every iteration complete."""
    for _ in range(tries - 1):
        try:
            deep_gemm.m_grouped_bf16_gemm_nt_contiguous(A, B, D, m_indices)
            return
        except RuntimeError:
            pass
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(A, B, D, m_indices)


# ─────────────────────────────────────────────────────────────────────────────
# Per-case driver.
# ─────────────────────────────────────────────────────────────────────────────
def run_case(label, tokens, E, top_k, hidden, moe_inter, M, N, K):
    numel = M * E  # == tokens * top_k (uniform routing)
    flops = 2.0 * numel * N * K
    sm = _num_sms()

    print(f"\n{'═' * 92}")
    print(f"  {label}")
    print(f"  grouped GEMM: M_total={numel}, N={N}, K={K}, experts={E}, M/expert={M}, bf16")
    print(f"  (MoE prefill: tokens={tokens}, top_k={top_k}, hidden={hidden}, "
          f"moe_inter={moe_inter}; sm_count={sm})")
    print(f"  FLOPs = {flops / 1e12:.3f} TFLOP")
    print(f"{'═' * 92}")

    _warmup_gpu()
    A, B, sizes, offsets, m_indices, offs_cumsum, per = gen_inputs(numel, E, N, K)
    B_KN = B.transpose(1, 2).contiguous()  # [E, K, N] for torch & non-TMA triton

    results = []  # (name, ms)

    # Fairness contract: every timed lambda operates on a PRE-ALLOCATED output
    # and pre-conditioned inputs, so do_bench measures the GEMM kernel alone —
    # not per-call buffer allocation. DeepGEMM (pre-allocated `D`) and both
    # Triton kernels (pre-allocated `group_c`) already do this; tileops-3wg now
    # matches via `out=`. The lone exception is torch._grouped_mm, whose public
    # API has no out-param and allocates its result internally — it stays taxed,
    # but it is only a reference baseline and does not enter the 3wg-vs-deepgemm
    # comparison. Uniform routing makes every per-expert M a multiple of 128, so
    # the 3WG kernel takes its aligned fast path (no A padding), exactly like
    # DeepGEMM's contiguous-layout alignment requirement.

    # ---- ours: 3-WG persistent ----
    k3 = GroupedGemmPersistent3WGKernel(numel=numel, num_experts=E, N=N, K=K, dtype=_DTYPE, sm_count=sm)
    C_3wg = torch.empty(numel, N, dtype=_DTYPE, device="cuda")
    # Tensors are bound as lambda defaults (not free closure vars) so the later
    # ``del`` of the large operands does not make pyflakes flag them as F821.
    ms = do_bench(lambda A=A, B=B, C=C_3wg: k3(A, B, sizes, offsets, out=C),
                  warmup=_WARMUP, rep=_REP)
    results.append(("tileops-3wg", ms))
    del C_3wg
    torch.cuda.empty_cache()

    # ---- torch._grouped_mm (CUTLASS) ----
    ms = do_bench(lambda A=A, B_KN=B_KN: torch._grouped_mm(A, B_KN, offs_cumsum),
                  warmup=_WARMUP, rep=_REP)
    results.append(("torch", ms))
    torch.cuda.empty_cache()

    # ---- deepgemm: contiguous grouped GEMM ----
    if _HAS_DEEP_GEMM:
        D = torch.empty(numel, N, dtype=_DTYPE, device="cuda")
        try:
            ms = do_bench(
                lambda A=A, B=B, D=D: _deepgemm_launch(A, B, D, m_indices),
                warmup=_WARMUP, rep=_REP)
            results.append(("deepgemm", ms))
        except Exception as exc:  # pragma: no cover
            print(f"  [skip] deepgemm grouped: {exc}")
        del D
        torch.cuda.empty_cache()

    # ---- triton 08-grouped-gemm (ported, bf16) ----
    group_a = [A[e * per:(e + 1) * per] for e in range(E)]
    group_c = [torch.empty(per, N, dtype=_DTYPE, device="cuda") for _ in range(E)]
    # non-TMA wants B[e] as [K, N]
    group_b_kn = [B_KN[e] for e in range(E)]
    a_ptrs, b_ptrs, c_ptrs, g_sizes, g_lds = _build_triton_ptrs(group_a, group_b_kn, group_c)
    grid = lambda meta: (meta["NUM_SM"],)  # noqa: E731

    def _triton():
        _grouped_matmul_kernel[grid](a_ptrs, b_ptrs, c_ptrs, g_sizes, g_lds, E)

    try:
        ms = do_bench(_triton, warmup=_WARMUP, rep=_REP)
        results.append(("triton", ms))
    except Exception as exc:  # pragma: no cover
        print(f"  [skip] triton: {exc}")
    del group_c, group_b_kn, B_KN
    torch.cuda.empty_cache()

    # ---- triton + TMA (Hopper, NT layout = our B[e]) ----
    if _supports_tma():
        group_c_tma = [torch.empty(per, N, dtype=_DTYPE, device="cuda") for _ in range(E)]
        group_b_nk = [B[e] for e in range(E)]  # [N, K]
        a_p, b_p, c_p, gs, gl = _build_triton_ptrs(group_a, group_b_nk, group_c_tma)
        _set_triton_allocator()

        def _triton_tma():
            _grouped_matmul_tma_kernel[grid](a_p, b_p, c_p, gs, gl, E, NUM_SM=sm)

        try:
            ms = do_bench(_triton_tma, warmup=_WARMUP, rep=_REP)
            results.append(("triton-tma", ms))
        except Exception as exc:  # pragma: no cover
            print(f"  [skip] triton-tma: {exc}")
        del group_c_tma
        torch.cuda.empty_cache()

    # ---- report ----
    # Absolute TFLOPS are bounded by the sustained SM clock, which on this H200 is
    # power-capped (~700 W -> ~1.4 GHz under full bf16 tensor load, vs 1.98 GHz max),
    # so the meaningful signal is the relative gap, not the raw number.
    ours = next((ms for n, ms in results if n == "tileops-3wg"), None)
    print(f"\n  {'impl':>14}  {'ms':>9}  {'TFLOPS':>9}  {'vs ours':>9}")
    print(f"  {'-' * 14}  {'-' * 9}  {'-' * 9}  {'-' * 9}")
    for name, ms in results:
        spd = f"{ms / ours:.2f}x" if ours else "-"
        print(f"  {name:>14}  {ms:>9.3f}  {tflops(ms, numel, N, K):>9.1f}  {spd:>9}")

    del A, B, group_a
    torch.cuda.empty_cache()
    return results


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


def _run_one(idx):
    """Run a single case in-process (used by the per-case subprocess).

    Exits with ``_DG_RETRY_EXIT`` when DeepGEMM is installed but dropped out of
    this case (its process-deterministic large-M workspace bug), so the driver
    relaunches a fresh process and re-rolls DeepGEMM's JIT autotune."""
    label, tokens, E, top_k, hidden, moe_inter, M, N, K = CASES[idx]
    results = run_case(label, tokens, E, top_k, hidden, moe_inter, M, N, K)
    if _HAS_DEEP_GEMM and not any(name == "deepgemm" for name, _ in results):
        sys.exit(_DG_RETRY_EXIT)


def main():
    """Drive all cases, each in its own subprocess.

    The largest cases (M=8192) hold ~70-130 GB of live operands; running them in
    the same process leaks across cases (do_bench / autotune retain references)
    and OOMs. A fresh process per case lets the OS reclaim everything, at the cost
    of recompiling kernels each time.
    """
    import subprocess

    only = None
    if "--case" in sys.argv:
        only = int(sys.argv[sys.argv.index("--case") + 1])

    if only is not None:
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  deep_gemm: "
              f"{'yes' if _HAS_DEEP_GEMM else 'NO (deepgemm skipped)'}")
        _run_one(only)
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}  |  deep_gemm: "
          f"{'yes' if _HAS_DEEP_GEMM else 'NO (deepgemm skipped)'}  |  "
          f"{len(CASES)} cases, one subprocess each")
    for idx in range(len(CASES)):
        for attempt in range(1, _DG_MAX_ATTEMPTS + 1):
            rc = subprocess.run([sys.executable, "-u", __file__, "--case", str(idx)],
                                env=os.environ).returncode
            if rc != _DG_RETRY_EXIT:
                break
            if attempt < _DG_MAX_ATTEMPTS:
                print(f"  [case {idx}: DeepGEMM hit its workspace bug; relaunching "
                      f"(attempt {attempt + 1}/{_DG_MAX_ATTEMPTS})]")
        if rc == _DG_RETRY_EXIT:
            print(f"  [case {idx}: DeepGEMM still skipped after {_DG_MAX_ATTEMPTS} "
                  f"launches — reporting without it]")
        elif rc != 0:
            print(f"  [case {idx} exited rc={rc}]")


if __name__ == "__main__":
    main()
