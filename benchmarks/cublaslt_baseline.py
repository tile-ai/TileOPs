"""cuBLASLt *searched-best* GEMM baseline for benchmarks.

``torch.matmul`` dispatches to cuBLAS/cuBLASLt's **default** heuristic (the
top-1 algorithm). For some shapes — small-M / tall-skinny / awkward-N — that
default is measurably slower than the best algorithm cuBLASLt *can* run (e.g.
it under-uses split-K). Benchmarking only against the default therefore risks
crediting a TileOPs kernel with a "win" that a cuBLASLt user with algorithm
selection would not concede.

This module provides that stronger baseline. It searches three sources and keeps
the fastest: cuBLASLt's heuristic top picks (``cublasLtMatmulAlgoGetHeuristic``),
a full enumeration of every algorithm id (``cublasLtMatmulAlgoGetIds``) crossed
with its capability grid — tile, stages, split-K, reduction scheme, CTA swizzle,
custom option — validated by ``cublasLtMatmulAlgoCheck``, and the plain
``torch.matmul`` path (so the result is never slower than the default). The
*selection* happens once at construction; the returned callable runs a single
op so the benchmark harness times it with the **same CUPTI protocol** as every
other entry (no cudaEvent / launch-overhead skew between baselines).

Talks to ``libcublasLt`` through ``ctypes`` (no build step) and reads the device
pointers of the caller's torch tensors directly, so inputs are byte-identical to
the ``tileops`` and ``torch-cublas`` entries. Falls back to ``None`` when
cuBLASLt is unavailable, so callers keep the plain ``torch-cublas`` baseline.

Only ``trans_a == False`` (the manifest GEMM layouts: NN and NT) is supported;
other layouts return ``None``.
"""

import ctypes
from typing import Callable, Optional

import torch

__all__ = ["CublasLtBestGemm", "make_cublaslt_best"]

# ── cuBLAS / CUDA enum constants (library_types.h, cublas_api.h, cublasLt.h) ──
_CUDA_R_16F = 2
_CUDA_R_32F = 0
_CUDA_R_16BF = 14
_CUBLAS_COMPUTE_32F = 68  # bf16/fp16 inputs, fp32 accumulate
_CUBLAS_OP_N = 0
_CUBLAS_OP_T = 1
_DESC_TRANSA = 3  # cublasLtMatmulDescAttributes_t
_DESC_TRANSB = 4
_PREF_MAX_WORKSPACE = 1  # cublasLtMatmulPreferenceAttributes_t (SEARCH_MODE=0)
_STATUS_SUCCESS = 0

# cublasLtMatmulAlgoCapAttributes_t  /  ...ConfigAttributes_t  /  reduction schemes
_CAP_SPLITK = 0
_CAP_REDUCTION_MASK = 1
_CAP_SWIZZLE = 2
_CAP_TILE_IDS = 6
_CAP_CUSTOM_MAX = 7
_CAP_STAGES_IDS = 13
_CFG_TILE_ID = 1
_CFG_SPLITK_NUM = 2
_CFG_REDUCTION = 3
_CFG_SWIZZLE = 4
_CFG_CUSTOM = 5
_CFG_STAGES_ID = 6
_RED_NONE = 0
_RED_SCHEME_BITS = (1, 2, 4)  # inplace / compute_type / output_type (cublasLtReductionScheme_t)
_SPLITK_CANDIDATES = (2, 3, 4, 5, 6, 8, 12, 16, 32)

_TORCH_TO_CUDA_DT = {
    torch.float16: _CUDA_R_16F,
    torch.bfloat16: _CUDA_R_16BF,
}


class _MatmulAlgo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]  # cublasLtMatmulAlgo_t


class _HeuristicResult(ctypes.Structure):
    _fields_ = [
        ("algo", _MatmulAlgo),
        ("workspaceSize", ctypes.c_size_t),
        ("state", ctypes.c_int),  # cublasStatus_t
        ("wavesCount", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


_lib: Optional[ctypes.CDLL] = None


def _sonames() -> tuple[str, ...]:
    """Library names to try, most specific first.

    The soname carries the CUDA major version, so it moves with the toolkit —
    a pinned one silently costs this baseline on any build with a different
    version, and silence is the failure mode a benchmark can least afford. Take
    the version from the torch that is going to run the comparison, so the
    library matches the runtime rather than whatever this file was written
    against; the unversioned name is a last resort, since the CUDA wheels ship
    only the versioned one.
    """
    major = (torch.version.cuda or "").split(".")[0]
    return ((f"libcublasLt.so.{major}",) if major.isdigit() else ()) + ("libcublasLt.so",)


def _load() -> Optional[ctypes.CDLL]:
    """Load libcublasLt once; return None if unavailable."""
    global _lib
    if _lib is not None:
        return _lib
    for name in _sonames():
        try:
            _lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if _lib is None:
        return None
    p = ctypes.c_void_p
    sz = ctypes.c_size_t
    i = ctypes.c_int
    # Only argtypes that matter for correctness are pinned; handles are void*.
    _lib.cublasLtCreate.argtypes = [ctypes.POINTER(p)]
    _lib.cublasLtMatmulDescCreate.argtypes = [ctypes.POINTER(p), i, i]
    _lib.cublasLtMatmulDescSetAttribute.argtypes = [p, i, p, sz]
    _lib.cublasLtMatrixLayoutCreate.argtypes = [
        ctypes.POINTER(p),
        i,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int64,
    ]
    _lib.cublasLtMatmulPreferenceCreate.argtypes = [ctypes.POINTER(p)]
    _lib.cublasLtMatmulPreferenceSetAttribute.argtypes = [p, i, p, sz]
    _lib.cublasLtMatmulAlgoGetHeuristic.argtypes = [
        p,
        p,
        p,
        p,
        p,
        p,
        p,
        i,
        ctypes.POINTER(_HeuristicResult),
        ctypes.POINTER(i),
    ]
    _lib.cublasLtMatmul.argtypes = [
        p,
        p,
        p,
        p,
        p,
        p,
        p,
        p,
        p,
        p,
        p,
        p,
        ctypes.POINTER(_MatmulAlgo),
        p,
        sz,
        p,
    ]
    _lib.cublasLtMatmulAlgoGetIds.argtypes = [
        p,
        i,
        i,
        i,
        i,
        i,
        i,
        i,
        ctypes.POINTER(i),
        ctypes.POINTER(i),
    ]
    _lib.cublasLtMatmulAlgoInit.argtypes = [p, i, i, i, i, i, i, i, ctypes.POINTER(_MatmulAlgo)]
    _lib.cublasLtMatmulAlgoCapGetAttribute.argtypes = [
        ctypes.POINTER(_MatmulAlgo),
        i,
        p,
        sz,
        ctypes.POINTER(sz),
    ]
    _lib.cublasLtMatmulAlgoConfigSetAttribute.argtypes = [ctypes.POINTER(_MatmulAlgo), i, p, sz]
    _lib.cublasLtMatmulAlgoCheck.argtypes = [
        p,
        p,
        p,
        p,
        p,
        p,
        ctypes.POINTER(_MatmulAlgo),
        ctypes.POINTER(_HeuristicResult),
    ]
    return _lib


def _ck(status: int, where: str) -> None:
    if status != _STATUS_SUCCESS:
        raise RuntimeError(f"cuBLASLt {where} failed with status {status}")


class CublasLtBestGemm:
    """Callable bound to the fastest cuBLAS algorithm found for one shape.

    Computes ``C = op(A) @ op(B)`` (row-major, bf16/fp16 in, fp32 accumulate)
    with ``trans_a == False`` and ``trans_b`` selecting NN or NT — matching the
    manifest GEMM workloads. Construction searches the heuristic picks, the full
    algorithm-id × capability enumeration, and the ``torch.matmul`` default, then
    keeps the fastest; ``__call__`` runs exactly one op with it.

    Args:
        m: Rows of ``op(A)`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        dtype: ``torch.float16`` or ``torch.bfloat16``.
        trans_b: ``True`` for NT (``A @ Bᵀ``, ``B`` stored ``[n, k]``), ``False``
            for NN (``A @ B``, ``B`` stored ``[k, n]``).
        workspace_mb: cuBLASLt workspace ceiling (enables split-K algorithms).
        n_candidates: Heuristic candidate algorithms to request (before the full
            enumeration is added to the pool).

    Raises:
        RuntimeError: cuBLASLt unavailable, dtype unsupported, or no candidate
            algorithm ran.
    """

    def __init__(
        self,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        trans_b: bool,
        workspace_mb: int = 256,
        n_candidates: int = 32,
    ) -> None:
        lib = _load()
        if lib is None:
            raise RuntimeError("libcublasLt not found")
        if dtype not in _TORCH_TO_CUDA_DT:
            raise RuntimeError(f"unsupported dtype {dtype}")
        self._lib = lib
        self.m, self.n, self.k = m, n, k
        self.dtype, self.trans_b = dtype, trans_b
        io_dt = _TORCH_TO_CUDA_DT[dtype]
        vp = ctypes.c_void_p

        self._ws_bytes = workspace_mb * 1024 * 1024
        self._ws = torch.empty(self._ws_bytes, dtype=torch.uint8, device="cuda")
        self._c = torch.empty((m, n), dtype=dtype, device="cuda")

        # Row-major C[m,n] = A[m,k] @ op(B) is computed column-major as
        # Cᵀ[n,m] = op(B)ᵀ · Aᵀ. With A_lt := B and B_lt := A (trans_a is
        # always False here so op(A)ᵀ = Aᵀ = the [k,m] col-major view of A):
        #   NT (trans_b): B is [n,k] row-major = [k,n] col-major (ld=k), opA=T.
        #   NN (else):    B is [k,n] row-major = [n,k] col-major (ld=n), opA=N.
        handle = vp()
        _ck(lib.cublasLtCreate(ctypes.byref(handle)), "Create")
        self._handle = handle
        desc = vp()
        _ck(
            lib.cublasLtMatmulDescCreate(ctypes.byref(desc), _CUBLAS_COMPUTE_32F, _CUDA_R_32F),
            "MatmulDescCreate",
        )
        self._desc = desc
        op_a = _CUBLAS_OP_T if trans_b else _CUBLAS_OP_N
        op_b = _CUBLAS_OP_N
        self._set_attr(desc, _DESC_TRANSA, ctypes.c_int(op_a))
        self._set_attr(desc, _DESC_TRANSB, ctypes.c_int(op_b))

        # Layouts of the *stored* column-major matrices.
        a_rows, a_cols, a_ld = (k, n, k) if trans_b else (n, k, n)  # A_lt := B
        self._io_dt = io_dt
        self._La = self._layout(io_dt, a_rows, a_cols, a_ld)
        self._Lb = self._layout(io_dt, k, m, k)  # B_lt := A
        self._Lc = self._layout(io_dt, n, m, n)  # C_lt (=[m,n] row-major)
        self._alpha = ctypes.c_float(1.0)
        self._beta = ctypes.c_float(0.0)

        # Candidate pool = (1) cuBLASLt's heuristic top picks, plus (2) a full
        # enumeration of every algorithm id × its capability grid (tile, split-K,
        # reduction, swizzle, stages, custom option) validated by AlgoCheck. The
        # heuristic alone returned as few as 8 candidates for some shapes; the
        # enumeration is what makes this a genuine "best effort" search.
        pref = vp()
        _ck(lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "PreferenceCreate")
        self._set_attr(pref, _PREF_MAX_WORKSPACE, ctypes.c_size_t(self._ws_bytes), pref=True)
        results = (_HeuristicResult * n_candidates)()
        returned = ctypes.c_int(0)
        _ck(
            lib.cublasLtMatmulAlgoGetHeuristic(
                handle,
                desc,
                self._La,
                self._Lb,
                self._Lc,
                self._Lc,
                pref,
                n_candidates,
                results,
                ctypes.byref(returned),
            ),
            "AlgoGetHeuristic",
        )
        candidates = [
            results[i].algo for i in range(returned.value) if results[i].state == _STATUS_SUCCESS
        ]
        candidates += self._enumerate_candidates()
        if not candidates:
            raise RuntimeError("cuBLASLt found no runnable algorithm")
        self.n_searched = len(candidates)
        self._best_algo = self._pick_best(candidates)

    # ── construction helpers ──
    def _set_attr(self, obj, attr, cval, pref: bool = False) -> None:
        fn = (
            self._lib.cublasLtMatmulPreferenceSetAttribute
            if pref
            else self._lib.cublasLtMatmulDescSetAttribute
        )
        _ck(fn(obj, attr, ctypes.byref(cval), ctypes.sizeof(cval)), "SetAttribute")

    def _layout(self, dt, rows, cols, ld):
        lay = ctypes.c_void_p()
        _ck(
            self._lib.cublasLtMatrixLayoutCreate(ctypes.byref(lay), dt, rows, cols, ld),
            "MatrixLayoutCreate",
        )
        return lay

    def _matmul(self, a: torch.Tensor, b: torch.Tensor, algo: _MatmulAlgo) -> int:
        # A_lt := B, B_lt := A (see the __init__ layout note).
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        return self._lib.cublasLtMatmul(
            self._handle,
            self._desc,
            ctypes.byref(self._alpha),
            ctypes.c_void_p(b.data_ptr()),
            self._La,
            ctypes.c_void_p(a.data_ptr()),
            self._Lb,
            ctypes.byref(self._beta),
            ctypes.c_void_p(self._c.data_ptr()),
            self._Lc,
            ctypes.c_void_p(self._c.data_ptr()),
            self._Lc,
            ctypes.byref(algo),
            ctypes.c_void_p(self._ws.data_ptr()),
            self._ws_bytes,
            stream,
        )

    # ── full algorithm enumeration (best-effort search) ──
    def _cap_array(self, algo: _MatmulAlgo, cap: int, maxn: int = 64) -> list:
        buf = (ctypes.c_uint32 * maxn)()
        written = ctypes.c_size_t(0)
        st = self._lib.cublasLtMatmulAlgoCapGetAttribute(
            ctypes.byref(algo), cap, buf, ctypes.sizeof(buf), ctypes.byref(written)
        )
        if st != _STATUS_SUCCESS:
            return []
        return [buf[i] for i in range(written.value // 4)]

    def _cap_scalar(self, algo: _MatmulAlgo, cap: int, default: int = 0) -> int:
        val = ctypes.c_uint32(0)
        written = ctypes.c_size_t(0)
        st = self._lib.cublasLtMatmulAlgoCapGetAttribute(
            ctypes.byref(algo), cap, ctypes.byref(val), 4, ctypes.byref(written)
        )
        return val.value if st == _STATUS_SUCCESS else default

    def _cfg(self, algo: _MatmulAlgo, attr: int, value: int) -> None:
        v = ctypes.c_uint32(value)
        self._lib.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(algo), attr, ctypes.byref(v), 4)

    def _check_ok(self, algo: _MatmulAlgo) -> bool:
        """True if the configured algo is valid and fits the workspace budget."""
        res = _HeuristicResult()
        st = self._lib.cublasLtMatmulAlgoCheck(
            self._handle,
            self._desc,
            self._La,
            self._Lb,
            self._Lc,
            self._Lc,
            ctypes.byref(algo),
            ctypes.byref(res),
        )
        return (
            st == _STATUS_SUCCESS
            and res.state == _STATUS_SUCCESS
            and res.workspaceSize <= self._ws_bytes
        )

    def _enumerate_candidates(self, max_ids: int = 64, max_total: int = 256) -> list:
        """Enumerate every algorithm id × its capability grid (tile, stages,
        split-K, reduction scheme, CTA swizzle, custom option), keeping each
        combination that ``cublasLtMatmulAlgoCheck`` validates and fits the
        workspace. This is the exhaustive counterpart to the heuristic top-N;
        together they form the strongest cuBLAS baseline we can construct."""
        lib = self._lib
        ids = (ctypes.c_int * max_ids)()
        nret = ctypes.c_int(0)
        dt = self._io_dt
        if (
            lib.cublasLtMatmulAlgoGetIds(
                self._handle,
                _CUBLAS_COMPUTE_32F,
                _CUDA_R_32F,
                dt,
                dt,
                dt,
                dt,
                max_ids,
                ids,
                ctypes.byref(nret),
            )
            != _STATUS_SUCCESS
        ):
            return []
        out: list = []
        for i in range(nret.value):
            algo = _MatmulAlgo()
            if (
                lib.cublasLtMatmulAlgoInit(
                    self._handle,
                    _CUBLAS_COMPUTE_32F,
                    _CUDA_R_32F,
                    dt,
                    dt,
                    dt,
                    dt,
                    ids[i],
                    ctypes.byref(algo),
                )
                != _STATUS_SUCCESS
            ):
                continue
            tiles = self._cap_array(algo, _CAP_TILE_IDS) or [0]
            stages = self._cap_array(algo, _CAP_STAGES_IDS) or [0]
            splitk_ok = self._cap_scalar(algo, _CAP_SPLITK)
            red_mask = self._cap_scalar(algo, _CAP_REDUCTION_MASK)
            swizzles = [0, 1] if self._cap_scalar(algo, _CAP_SWIZZLE) >= 1 else [0]
            customs = list(range(min(self._cap_scalar(algo, _CAP_CUSTOM_MAX) + 1, 2)))
            # split-K plans: always the plain single-K pass; add the sweep (with
            # each reduction scheme the algo's mask allows) when it is supported.
            plans = [(1, _RED_NONE)]
            if splitk_ok:
                plans += [
                    (sk, bit)
                    for sk in _SPLITK_CANDIDATES
                    for bit in _RED_SCHEME_BITS
                    if red_mask & bit
                ]
            for tile in tiles:
                for stage in stages:
                    for swz in swizzles:
                        for cust in customs:
                            for sk, red in plans:
                                self._cfg(algo, _CFG_TILE_ID, tile)
                                self._cfg(algo, _CFG_STAGES_ID, stage)
                                self._cfg(algo, _CFG_SWIZZLE, swz)
                                self._cfg(algo, _CFG_CUSTOM, cust)
                                self._cfg(algo, _CFG_SPLITK_NUM, sk)
                                self._cfg(algo, _CFG_REDUCTION, red)
                                if self._check_ok(algo):
                                    out.append(_MatmulAlgo.from_buffer_copy(algo))
                                    if len(out) >= max_total:
                                        return out
        return out

    def _pick_best(self, algos) -> _MatmulAlgo:
        """Select the fastest runnable candidate for the reported CUPTI timing.

        Ranking uses the **minimum** cudaEvent time over many L2-flushed
        iterations, not the mean: the fastest observed run is the one least
        perturbed by clock drift or a neighbour's preemption, so min is robust to
        the transient slowdowns that made an earlier mean-based pass mis-rank
        candidates (and even pick one slower than cuBLASLt's own top heuristic).
        Since the heuristic list already contains that top pick, min-ranking
        never selects a candidate slower than the default. The chosen algorithm's
        latency is re-measured later by the shared CUPTI harness."""
        a = torch.randn(self.m, self.k, dtype=self.dtype, device="cuda")
        b = torch.randn(
            (self.n, self.k) if self.trans_b else (self.k, self.n), dtype=self.dtype, device="cuda"
        )
        flush = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")

        def min_ms(run, iters):
            for _ in range(5):
                flush.zero_()
                run()
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(iters):
                flush.zero_()
                ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                ev0.record()
                run()
                ev1.record()
                ev1.synchronize()
                best = min(best, ev0.elapsed_time(ev1))
            return best

        def algo_ms(al, it):
            return min_ms(lambda: self._matmul(a, b, al), it)

        torch_run = (lambda: torch.matmul(a, b.T)) if self.trans_b else (lambda: torch.matmul(a, b))

        runnable = [al for al in algos if self._matmul(a, b, al) == _STATUS_SUCCESS]
        torch.cuda.synchronize()
        if not runnable:
            raise RuntimeError("no runnable cuBLASLt algorithm among candidates")
        # First pass: cheap min-time screen over the whole pool to find finalists.
        screened = sorted(runnable, key=lambda al: algo_ms(al, 12))[:6]
        # Second pass: more iterations on the finalists to pick a stable winner.
        best_algo = min(screened, key=lambda al: algo_ms(al, 80))
        # A "best cuBLAS" baseline must never be slower than the default
        # torch.matmul path (which may take a route our search misses), so
        # include it as a candidate and defer to it when it wins.
        self._use_torch = min_ms(torch_run, 80) < algo_ms(best_algo, 80)
        return best_algo

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Dispatch to whichever won selection: the default torch.matmul path or
        # the searched cuBLASLt algorithm. Either way this is the fastest cuBLAS
        # we found for the shape, timed by the caller's CUPTI harness.
        if self._use_torch:
            return torch.matmul(a, b.T if self.trans_b else b)
        _ck(self._matmul(a, b, self._best_algo), "Matmul")
        return self._c


def make_cublaslt_best(
    m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool
) -> Optional[Callable]:
    """Build a cuBLASLt searched-best GEMM callable, or None if unavailable.

    Returns ``None`` (so the caller keeps the plain ``torch-cublas`` baseline)
    when cuBLASLt is missing, the dtype is unsupported, ``trans_a`` is True, or
    algorithm selection fails.
    """
    if trans_a:
        return None
    try:
        return CublasLtBestGemm(m, n, k, dtype, trans_b)
    except RuntimeError:
        return None


if __name__ == "__main__":
    # Self-test: correctness vs torch + best latency, NN and NT.
    from benchmarks.benchmark_base import bench_kernel

    for label, m, n, k, tb in [
        ("NT decode-gate-up", 128, 2112, 7168, True),
        ("NT prefill-attn", 4096, 4096, 7168, True),
        ("NN square-1k", 1024, 1024, 1024, False),
    ]:
        for dt in (torch.bfloat16,):
            a = torch.randn(m, k, dtype=dt, device="cuda")
            b = (
                torch.randn(n, k, dtype=dt, device="cuda")
                if tb
                else torch.randn(k, n, dtype=dt, device="cuda")
            )
            fn = make_cublaslt_best(m, n, k, dt, False, tb)
            if fn is None:
                print(f"{label}: cuBLASLt unavailable")
                continue
            c = fn(a, b)
            ref = a.float() @ (b.float().T if tb else b.float())
            err = (c.float() - ref).abs().max().item()
            errcu = ((a @ (b.T if tb else b)).float() - ref).abs().max().item()
            flops = 2 * m * n * k
            tf_best = flops / bench_kernel(lambda x, y, f=fn: f(x, y), args=(a, b)) * 1e-9 / 1e3
            tf_def = (
                flops
                / bench_kernel(lambda x, y, t=tb: torch.matmul(x, y.T if t else y), args=(a, b))
                * 1e-9
                / 1e3
            )
            ok = err <= max(errcu * 3, 0.5)
            print(
                f"{label} {m}x{n}x{k} {str(dt)[6:]}: correct={ok} (err={err:.3f}/cu={errcu:.3f}) "
                f"cuBLASLt_best={tf_best * 1e3:.0f}T torch_default={tf_def * 1e3:.0f}T "
                f"gain={tf_best / tf_def:.3f}x",
                flush=True,
            )
