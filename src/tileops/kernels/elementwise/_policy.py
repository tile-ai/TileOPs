"""What an elementwise kernel is built with: its launch config, output dtype and strategy."""

import warnings
from dataclasses import dataclass

import torch

from tileops.kernels.kernel_base import Kernel

from ._broadcast import row_tile_leaves_tail
from ._dtype import BOOL_STORAGE_DTYPE, _torch_dtype_nbytes

_AUTOTUNE_THREADS = (128, 256, 512)
_DEFAULT_THREADS = 128
_DIRECT_THREADS = 256
_BYTES_PER_THREAD = 16
_MIN_NUM_PER_THREAD = 4
_BOOL_OUTPUT_MAX_NPT = 4
_MAX_THREADS = 1024
_TARGET_BLOCKS = 256


def default_launch_config(
    *,
    strategy: str,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    n_total: int | None,
    stores_bool: bool = True,
    bytes_per_thread: int = _BYTES_PER_THREAD,
    min_num_per_thread: int = _MIN_NUM_PER_THREAD,
    row_broadcast_inner: int | None = None,
    default_threads: int | None = None,
) -> dict:
    """Return the default launch config for one elementwise specialization.

    *bytes_per_thread* is how much each thread carries and *min_num_per_thread*
    how few elements the shrink below leaves it; see
    ``_ElementwiseKernel.BYTES_PER_THREAD`` and ``MIN_NUM_PER_THREAD``.
    *row_broadcast_inner* is the row extent a broadcast block walks, or
    ``None``; see ``_tail_dominated``. *default_threads* replaces the
    strategy's thread count; see ``_ElementwiseKernel.DEFAULT_THREADS``.
    """
    # A direct block covers ``threads`` elements where a vectorized one covers
    # ``threads * num_per_thread``: the elements per block, not the thread count,
    # are what has to stay wide enough to keep the memory pipe busy.
    threads = default_threads or (_DIRECT_THREADS if strategy == "direct" else _DEFAULT_THREADS)

    elem_bytes = _torch_dtype_nbytes(input_dtype)
    npt = max(_MIN_NUM_PER_THREAD, bytes_per_thread // elem_bytes)

    if output_dtype == torch.bool and stores_bool:
        capped = min(npt, _BOOL_OUTPUT_MAX_NPT)
        if strategy != "direct":  # a direct block spans `threads` whatever npt says
            threads = min(_MAX_THREADS, threads * npt // capped)
        npt = capped
    elif _torch_dtype_nbytes(output_dtype) < elem_bytes and not _tail_dominated(
        row_broadcast_inner, n_total, threads, npt, output_dtype == torch.bool
    ):
        # A narrower result leaves the store short of a vector, so widen to cover it.
        npt *= 2

    while (
        n_total is not None
        and strategy != "direct"
        and npt > min_num_per_thread
        and n_total < threads * npt * _TARGET_BLOCKS
    ):
        npt //= 2
    return {"strategy": strategy, "threads": threads, "num_per_thread": npt}


def _tail_dominated(
    inner: int | None, n_total: int | None, threads: int, npt: int, staged: bool
) -> bool:
    """Whether doubling the block width pushes more columns onto the guarded tail path.

    A row-broadcast block covers columns of one row, and the remainder the width
    does not cover runs the slower per-lane path. A width that leaves no
    remainder there has nothing to push onto it.
    """
    if inner is None or n_total is None:
        return False
    if not row_tile_leaves_tail(inner, n_total // inner, threads, npt * 2, staged):
        return False
    return inner % (threads * npt * 2) > inner % (threads * npt)


def elementwise_autotune_configs(
    dtype: torch.dtype,
    strategy: str | None = None,
    bytes_per_thread: int = _BYTES_PER_THREAD,
    min_num_per_thread: int = _MIN_NUM_PER_THREAD,
) -> list[dict]:
    """Return the launch configs to time for one elementwise specialization.

    The swept elements-per-thread brackets the default the same *bytes_per_thread*
    produces, so a kernel can always land back on its shipped config, and reaches
    *min_num_per_thread* where a kernel lowers it.
    """
    # A direct body takes no num_per_thread: the key would name no parameter to bind,
    # and the sweep would time one kernel three times over.
    if strategy == "direct":
        return [{"threads": t} for t in _AUTOTUNE_THREADS]
    default = max(_MIN_NUM_PER_THREAD, bytes_per_thread // _torch_dtype_nbytes(dtype))
    npts = tuple(
        sorted({min_num_per_thread, max(min_num_per_thread, default // 2), default, default * 2})
    )
    return [{"threads": t, "num_per_thread": n} for t in _AUTOTUNE_THREADS for n in npts]


@dataclass(frozen=True)
class ElementwiseOutputPlan:
    logical_dtype: torch.dtype
    kernel_output_dtype: str
    bool_via_int8: bool = False


def elementwise_output_plan(
    input_dtype: torch.dtype,
    declared_output_dtype: torch.dtype | None = None,
    *,
    strategy: str | None = None,
    bool_storage: bool = False,
) -> ElementwiseOutputPlan:
    """Return the dtype the kernel writes, and the dtype the caller sees."""
    logical_dtype = declared_output_dtype or input_dtype
    # Every strategy but `direct` can store the result through an int8 buffer, and
    # wants to: a bool store lowers to one byte per lane, where int8 vectorises.
    bool_via_int8 = bool_storage and declared_output_dtype == torch.bool and strategy != "direct"
    kernel_output_dtype = (
        BOOL_STORAGE_DTYPE if bool_via_int8 else Kernel.dtype_to_str(logical_dtype)
    )
    return ElementwiseOutputPlan(logical_dtype, kernel_output_dtype, bool_via_int8)


def _bool_output_needs_scalar(
    input_dtype: torch.dtype,
    declared_output_dtype: torch.dtype | None,
) -> bool:
    return declared_output_dtype == torch.bool and input_dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
    )


def _validate_strategy(requested: str | None, strategies: list[str]) -> None:
    if requested is not None and requested not in strategies:
        raise ValueError(f"Unknown strategy '{requested}', expected one of {strategies}")


def _warn_direct_override(
    requested: str | None, kernel_name: str, dtype: torch.dtype | None = None
) -> None:
    if requested is None or requested == "direct":
        return
    dtype_msg = "dtype=torch.bool" if dtype is None else f"dtype={dtype} with torch.bool output"
    warnings.warn(
        f"{kernel_name}: {dtype_msg} requires strategy='direct' "
        f"(TileLang cannot lower vectorised boolx<N>); "
        f"overriding requested strategy={requested!r}.",
        RuntimeWarning,
        stacklevel=3,
    )


def choose_unary_strategy(
    *,
    requested: str | None,
    strategies: list[str],
    default_strategy: str,
    input_dtype: torch.dtype,
    declared_output_dtype: torch.dtype | None,
) -> str:
    _validate_strategy(requested, strategies)
    if input_dtype == torch.bool:
        _warn_direct_override(requested, "UnaryKernel")
        return "direct"
    if _bool_output_needs_scalar(input_dtype, declared_output_dtype):
        _warn_direct_override(requested, "UnaryKernel", input_dtype)
        return "direct"
    return requested or default_strategy


def choose_binary_strategy(
    *,
    requested: str | None,
    strategies: list[str],
    default_strategy: str,
    input_dtype: torch.dtype,
    declared_output_dtype: torch.dtype | None,
    same_shape: bool,
) -> str:
    _validate_strategy(requested, strategies)
    if input_dtype == torch.bool:
        _warn_direct_override(requested, "BinaryKernel")
        return "direct"
    if _bool_output_needs_scalar(input_dtype, declared_output_dtype):
        _warn_direct_override(requested, "BinaryKernel", input_dtype)
        return "direct"
    if requested == "register_copy" and not same_shape:
        return "explicit_parallel"
    return requested or ("register_copy" if same_shape else default_strategy)
