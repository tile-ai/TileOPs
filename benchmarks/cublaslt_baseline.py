"""cuBLASLt *searched-best* GEMM baseline for benchmarks.

``torch.matmul`` dispatches to cuBLAS/cuBLASLt's **default** heuristic (the
top-1 algorithm). For some shapes — small-M / tall-skinny / awkward-N — that
default is measurably slower than the best algorithm cuBLASLt *can* run (e.g.
it under-uses split-K). Benchmarking only against the default therefore risks
crediting a TileOPs kernel with a "win" that a cuBLASLt user with algorithm
selection would not concede.

This module provides that stronger baseline. It ranks cuBLASLt's heuristic picks
for the shape against ``torch.matmul`` and keeps the fastest, through
``timing.bench_kernel`` -- the instrument that will report the row, so nothing
can win selection on a timer the report does not use. Selection runs once at
construction; the returned callable runs a single op.

Measured on H200 over the ``GemmFwdOp`` workload rows: the search is worth about
1.10x on ``wide-n-24576``, 1.04-1.06x on ``ds-v3-prefill-down`` and 1.02x on the
low-``m`` rows. Those are the rows where cuBLAS's default heuristic is weakest,
and where this library's kernel is furthest behind -- which is why the baseline
has to be here. Where the search does not win, selection defers to
``torch.matmul`` and the two entries run the same call; they still read about a
percent apart, which is this instrument's run-to-run spread and also the size of
the smallest wins above. Enumerating every algorithm id crossed with its
capability grid (256 further candidates) was measured too and never won a row the
heuristic list had not already won, so this searches the heuristic list alone.

The algorithm API comes from ``nvmath-python``: torch exposes only the backend
choice (``torch.backends.cuda.preferred_blas_library``), not algorithm
enumeration. nvmath maps the ``libcublasLt`` that torch already brought in (its
``nvidia-cublas`` dependency), which is the same library the ``torch-cublas``
entry runs on, and takes the caller's tensors directly -- so the inputs stay
byte-identical to the ``tileops`` and ``torch-cublas`` entries.

Only ``trans_a == False`` (the manifest GEMM layouts: NN and NT) is supported;
other layouts return ``None``.
"""

import contextlib
import statistics
from typing import Callable, Optional

import torch

from benchmarks.timing import bench_kernel

__all__ = ["CublasLtBestGemm", "make_cublaslt_best"]

#: dtypes this baseline serves; fp32 accumulation for both.
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

#: cuBLASLt workspace ceiling. Sized to admit the split-K algorithms, which are
#: what the search wins on; a smaller ceiling drops them from the candidate list
#: before ranking ever sees them.
_WORKSPACE_BYTES = 256 * 1024 * 1024

#: Heuristic candidates to request. cuBLASLt returns what it has, which on the
#: workload shapes here is around eight.
_N_CANDIDATES = 8


def _busy_ms(run: Callable[[], object]) -> float:
    """Median device-busy time of *run*, on the instrument that reports the row."""
    return statistics.median(s.device_busy_ms for s in bench_kernel(run))


class CublasLtBestGemm:
    """Callable bound to the fastest cuBLAS algorithm found for one shape.

    Computes ``C = A @ op(B)`` (row-major, bf16/fp16 in, fp32 accumulate) with
    ``trans_b`` selecting NN or NT — matching the manifest GEMM workloads.
    Construction times cuBLASLt's heuristic picks against the ``torch.matmul``
    default and keeps the fastest; ``__call__`` runs exactly one op with it.

    Args:
        a: Left operand, ``[m, k]``.
        b: Right operand, ``[n, k]`` under NT or ``[k, n]`` under NN.
        trans_b: ``True`` for NT (``A @ Bᵀ``), ``False`` for NN (``A @ B``).
        workspace_bytes: cuBLASLt workspace ceiling.
        n_candidates: Heuristic candidate algorithms to request.

    Raises:
        RuntimeError: dtype unsupported, or the plan produced no algorithm.
    """

    def __init__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        trans_b: bool,
        workspace_bytes: int = _WORKSPACE_BYTES,
        n_candidates: int = _N_CANDIDATES,
    ) -> None:
        from nvmath.linalg.advanced import (
            Matmul,
            MatmulComputeType,
            MatmulOptions,
            MatmulPlanPreferences,
        )

        if a.dtype not in _SUPPORTED_DTYPES:
            raise RuntimeError(f"unsupported dtype {a.dtype}")
        self.trans_b = trans_b
        self._a = a
        self._b = b
        rhs = b.t() if trans_b else b

        # COMPUTE_32F is stated rather than left to the default: fp32 accumulation
        # is what the TileOPs kernel this baseline is timed against does, and a
        # baseline accumulating narrower would be a different computation.
        self._mm = Matmul(
            a,
            rhs,
            options=MatmulOptions(
                compute_type=MatmulComputeType.COMPUTE_32F,
                memory_limit=workspace_bytes,
            ),
        )
        algorithms = self._mm.plan(preferences=MatmulPlanPreferences(limit=n_candidates))
        if not algorithms:
            self._mm.free()
            raise RuntimeError("cuBLASLt returned no runnable algorithm")
        self.n_searched = len(algorithms)

        self._algorithm = min(
            algorithms, key=lambda al: _busy_ms(lambda: self._mm.execute(algorithm=al))
        )
        # torch.matmul is ranked as one of the candidates: a "best cuBLAS" entry
        # that reads slower than the plain one would be worse than no entry at
        # all. Measured necessary -- without it two of the four rows checked on
        # 2026-08-26 read 0.996x instead of 1.000x.
        self._use_torch = _busy_ms(lambda: torch.matmul(a, rhs)) < _busy_ms(
            lambda: self._mm.execute(algorithm=self._algorithm)
        )

    def free(self) -> None:
        """Release the cuBLASLt plan and its workspace.

        One benchmark process builds one of these per workload row, so without
        this the plans and their 256 MB workspaces accumulate for the run.
        """
        mm = getattr(self, "_mm", None)
        if mm is not None:
            mm.free()
            self._mm = None

    def __del__(self) -> None:
        # Teardown can run while the interpreter is tearing down too; a failure
        # here has nobody left to report to.
        with contextlib.suppress(Exception):
            self.free()

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        rhs = b.t() if self.trans_b else b
        if a is not self._a or b is not self._b:
            # The plan is bound to the operands construction saw. Rebinding is a
            # host-side pointer swap outside the timed kernel, so a caller that
            # hands over different tensors still gets the right answer.
            self._mm.reset_operands(a=a, b=rhs)
            self._a, self._b = a, b
        if self._use_torch:
            return torch.matmul(a, rhs)
        return self._mm.execute(algorithm=self._algorithm)


def make_cublaslt_best(
    a: torch.Tensor, b: torch.Tensor, *, trans_a: bool, trans_b: bool
) -> Optional[Callable]:
    """Build a cuBLASLt searched-best GEMM callable, or None for a row it cannot take.

    Returns ``None`` (so the caller keeps the plain ``torch-cublas`` baseline)
    when ``trans_a`` is True, the dtype is unsupported, or the plan produced no
    algorithm. A missing ``nvmath`` is **not** caught here: it is a declared
    runner dependency, so its absence is a degraded image and has to fail the
    row rather than quietly drop the tag.
    """
    if trans_a:
        return None
    try:
        return CublasLtBestGemm(a, b, trans_b)
    except RuntimeError:
        return None
