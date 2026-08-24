"""cuBLASLt *searched-best* GEMM baseline for benchmarks.

``torch.matmul`` dispatches to cuBLAS/cuBLASLt's **default** heuristic (the
top-1 algorithm). For some shapes — small-M / tall-skinny / awkward-N — that
default is measurably slower than the best algorithm cuBLASLt *can* run (e.g.
it under-uses split-K). Benchmarking only against the default therefore risks
crediting a TileOPs kernel with a "win" that a cuBLASLt user with algorithm
selection would not concede.

This module provides that stronger baseline. It ranks cuBLASLt's heuristic picks
for the shape (``cublasLtMatmulAlgoGetHeuristic``) against ``torch.matmul`` and
keeps the fastest, through ``timing.bench_kernel`` -- the instrument that will
report the row, so nothing can win selection on a timer the report does not use.
Selection runs once at construction; the returned callable runs a single op.

Measured on H200 over the 16 ``GemmFwdOp`` workload rows: the search is worth
1.09x on ``wide-n-24576``, 1.06x on ``ds-v3-prefill-down`` and 1.03-1.04x on
three low-``m`` rows, with no row below 0.996x. Those are the rows where cuBLAS's
default heuristic is weakest, and where this library's kernel is furthest behind
-- which is why the baseline has to be here. Enumerating every algorithm id
crossed with its capability grid (256 further candidates) was measured too and
never won a row the heuristic list had not already won, so this searches the
heuristic list alone.

``ctypes`` because the algorithm API has no Python binding here -- not because
the library is missing. ``libcublasLt`` comes in with torch (its ``nvidia-cublas``
dependency) and is what the ``torch-cublas`` entry already runs on; what no
installed package exposes is cuBLASLt's algorithm enumeration, torch offering
only the backend choice (``torch.backends.cuda.preferred_blas_library``). So this
opens the same library torch loaded, under the soname of the CUDA major that
torch was built for, and reads the device pointers of the caller's tensors
directly -- which keeps the inputs byte-identical to the ``tileops`` and
``torch-cublas`` entries. Falls back to ``None`` when the library cannot be
opened, so callers keep ``torch-cublas``.

Only ``trans_a == False`` (the manifest GEMM layouts: NN and NT) is supported;
other layouts return ``None``.
"""

import contextlib
import ctypes
import statistics
from typing import Callable, Optional

import torch

from benchmarks.timing import bench_kernel

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
    for destroy in (
        "cublasLtDestroy",
        "cublasLtMatmulDescDestroy",
        "cublasLtMatrixLayoutDestroy",
        "cublasLtMatmulPreferenceDestroy",
    ):
        getattr(_lib, destroy).argtypes = [p]
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
    return _lib


def _ck(status: int, where: str) -> None:
    if status != _STATUS_SUCCESS:
        raise RuntimeError(f"cuBLASLt {where} failed with status {status}")


class CublasLtBestGemm:
    """Callable bound to the fastest cuBLAS algorithm found for one shape.

    Computes ``C = op(A) @ op(B)`` (row-major, bf16/fp16 in, fp32 accumulate)
    with ``trans_a == False`` and ``trans_b`` selecting NN or NT — matching the
    manifest GEMM workloads. Construction times cuBLASLt's heuristic picks against
    the ``torch.matmul`` default and keeps the fastest; ``__call__`` runs exactly
    one op with it.

    Args:
        m: Rows of ``op(A)`` / ``C``.
        n: Columns of ``op(B)`` / ``C``.
        k: Contraction dim.
        dtype: ``torch.float16`` or ``torch.bfloat16``.
        trans_b: ``True`` for NT (``A @ Bᵀ``, ``B`` stored ``[n, k]``), ``False``
            for NN (``A @ B``, ``B`` stored ``[k, n]``).
        workspace_mb: cuBLASLt workspace ceiling (enables split-K algorithms).
        n_candidates: Heuristic candidate algorithms to request; cuBLASLt returns
            what it has, which on the workload shapes here is around eight.

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

        # Candidate pool: cuBLASLt's heuristic picks for this shape.
        pref = vp()
        _ck(lib.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "PreferenceCreate")
        self._pref = pref
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

    def _pick_best(self, algos) -> _MatmulAlgo:
        """Return the fastest runnable candidate, or defer to ``torch.matmul``.

        Ranked through ``timing.bench_kernel`` -- the instrument that will report
        the row -- so a candidate cannot win selection on a timer the report does
        not use. ``torch.matmul`` is ranked as one of the candidates: a "best
        cuBLAS" entry that reads slower than the plain one would be worse than no
        entry at all.
        """
        a = torch.randn(self.m, self.k, dtype=self.dtype, device="cuda")
        b = torch.randn(
            (self.n, self.k) if self.trans_b else (self.k, self.n), dtype=self.dtype, device="cuda"
        )

        def busy_ms(run) -> float:
            return statistics.median(s.device_busy_ms for s in bench_kernel(run))

        runnable = [al for al in algos if self._matmul(a, b, al) == _STATUS_SUCCESS]
        torch.cuda.synchronize()
        if not runnable:
            raise RuntimeError("cuBLASLt returned no runnable algorithm")

        best_algo = min(runnable, key=lambda al: busy_ms(lambda: self._matmul(a, b, al)))
        torch_run = (lambda: torch.matmul(a, b.T)) if self.trans_b else (lambda: torch.matmul(a, b))
        self._use_torch = busy_ms(torch_run) < busy_ms(lambda: self._matmul(a, b, best_algo))
        return best_algo

    def __del__(self) -> None:
        """Release the cuBLASLt objects this instance created.

        One benchmark process builds one of these per workload row, so without
        this the handles, the matmul descriptor and the three layouts accumulate
        for the length of the run.
        """
        lib = getattr(self, "_lib", None)
        if lib is None:
            return
        for handle, destroy in (
            (getattr(self, "_La", None), "cublasLtMatrixLayoutDestroy"),
            (getattr(self, "_Lb", None), "cublasLtMatrixLayoutDestroy"),
            (getattr(self, "_Lc", None), "cublasLtMatrixLayoutDestroy"),
            (getattr(self, "_desc", None), "cublasLtMatmulDescDestroy"),
            (getattr(self, "_pref", None), "cublasLtMatmulPreferenceDestroy"),
            (getattr(self, "_handle", None), "cublasLtDestroy"),
        ):
            if handle is not None:
                # Teardown can run while the interpreter is tearing down too;
                # a failure here has nobody left to report to.
                with contextlib.suppress(Exception):
                    getattr(lib, destroy)(handle)

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
