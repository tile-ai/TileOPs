"""What an elementwise kernel is built with: its launch config, output dtype and strategy."""

from dataclasses import dataclass

import torch

from tileops._kernels.elementwise._broadcast import row_tile_leaves_tail, takes_flat_grid
from tileops._kernels.elementwise._dtype import BOOL_STORAGE_DTYPE
from tileops._kernels.kernel_base import Kernel

# One 16-byte vector, and the fewest elements a thread carries; the default config and the
# autotune sweep both derive elements per thread from them.
_BYTES_PER_THREAD = 16
_MIN_NUM_PER_THREAD = 4
# Fewest blocks a launch should spread its elements over.
TARGET_BLOCKS = 256


def default_launch_config(
    *,
    strategy: str,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    n_total: int,
    bytes_per_thread: int,
    min_num_per_thread: int,
    resident_threads: int,
    stores_bool: bool = True,
    row_broadcast_inner: int | None = None,
    register_copy_num_per_thread: int = _MIN_NUM_PER_THREAD,
    one_wave_in_l2: bool = False,
) -> dict:
    """Return the default launch config for one elementwise specialization.

    *bytes_per_thread*, *min_num_per_thread* and *register_copy_num_per_thread* are the
    kernel's ``_ElementwiseKernel`` attributes of those names. *row_broadcast_inner* is
    the row extent a broadcast block walks, or ``None``. *resident_threads* is how many
    threads the device holds at once. *one_wave_in_l2* enables wider blocks for
    inexpensive unary bodies held in L2.
    """

    def _tail_dominated(inner: int | None, threads: int, npt: int, staged: bool) -> bool:
        """Whether doubling the block width pushes more row-broadcast columns onto the
        per-lane tail path."""
        if inner is None:
            return False
        if not row_tile_leaves_tail(inner, n_total // inner, threads, npt * 2, staged):
            return False
        return inner % (threads * npt * 2) > inner % (threads * npt)

    # A direct block covers ``threads`` elements where a vectorized one covers
    # ``threads * num_per_thread``: the elements per block, not the thread count,
    # are what has to stay wide enough to keep the memory pipe busy.
    direct_threads = 256
    vectorized_threads = 128
    threads = direct_threads if strategy == "direct" else vectorized_threads

    elem_bytes = input_dtype.itemsize
    npt = max(_MIN_NUM_PER_THREAD, bytes_per_thread // elem_bytes)
    # A bool result is staged through fragments (``BinaryKernel.stage_broadcast``).
    staged = output_dtype == torch.bool

    if output_dtype == torch.bool and stores_bool:
        bool_output_max_npt = 4  # elements per thread a bool store is capped at
        capped = min(npt, bool_output_max_npt)
        if strategy != "direct":  # a direct block spans `threads` whatever npt says
            max_threads = 1024  # CUDA's cap on threads per block
            threads = min(max_threads, threads * npt // capped)
        npt = capped
    else:
        if output_dtype.itemsize < elem_bytes and not _tail_dominated(
            row_broadcast_inner, threads, npt, staged
        ):
            # A narrower result leaves the store short of a vector, so widen to cover it.
            npt *= 2
        if strategy == "register_copy":
            npt = max(npt, _register_copy_floor(register_copy_num_per_thread, elem_bytes))

    # From this many waves an 8-byte operand streams at least as fast with one 16-byte
    # vector a thread on twice the blocks; below it the second vector's bytes in flight
    # win. A narrower result keeps its elements so its store stays a whole vector.
    one_vector_min_waves = 8
    if (
        elem_bytes == 8
        and output_dtype.itemsize == elem_bytes
        and strategy != "direct"
        and -(-n_total // (threads * npt)) >= one_vector_min_waves * (resident_threads // threads)
    ):
        npt //= 2

    while (
        strategy != "direct"
        and npt > min_num_per_thread
        and n_total < threads * npt * TARGET_BLOCKS
    ):
        npt //= 2

    # Block dispatch, not DRAM bandwidth, paces a narrower result's flat row-broadcast grid
    # (``takes_flat_grid``), so it runs 512-thread blocks, three or four resident an SM.
    flat_threads = 512
    if (
        strategy != "direct"
        and row_broadcast_inner is not None
        and output_dtype.itemsize < elem_bytes
        and n_total >= flat_threads * npt * TARGET_BLOCKS
        and all(
            takes_flat_grid(row_broadcast_inner, n_total // row_broadcast_inner, t, npt, staged)
            for t in (threads, flat_threads)
        )
    ):
        threads = flat_threads

    # A launch held in L2 keeps its stores there, so loads in flight set its time, not DRAM
    # bandwidth: it runs one wave, each thread holding its share of the resident threads.
    # Whole blocks preserve vectorized loads; ragged sizes retain the default.
    if one_wave_in_l2 and strategy == "register_copy":
        # 256-thread blocks, not 512: a partial last wave of wide blocks leaves more SMs idle.
        # A small grid keeps them only while it spans four blocks; below that, 128-thread
        # blocks are faster.
        wide_threads, fewest_blocks = 256, 4
        share = -(-n_total // resident_threads)
        # At most four 16-byte vectors a thread: within one wave, more bytes in flight a
        # thread beat more threads. A larger share runs more waves.
        most = 4 * _BYTES_PER_THREAD // elem_bytes
        wave_npt = min(most, max(npt, 1 << (share - 1).bit_length()))
        shrunk = n_total < threads * npt * TARGET_BLOCKS
        widen = wave_npt > npt or (shrunk and n_total >= fewest_blocks * wide_threads * npt)
        if widen and n_total % (wide_threads * wave_npt) == 0:
            return {"strategy": strategy, "threads": wide_threads, "num_per_thread": wave_npt}
    return {"strategy": strategy, "threads": threads, "num_per_thread": npt}


def _register_copy_floor(requested: int, elem_bytes: int) -> int:
    """*requested* elements, but no more than two 16-byte vectors of the dtype.

    An 8-byte dtype has its two vectors at four elements, and eight run slower.
    """
    return min(requested, 2 * _BYTES_PER_THREAD // elem_bytes)


def elementwise_autotune_configs(
    dtype: torch.dtype,
    strategy: str,
    bytes_per_thread: int,
    min_num_per_thread: int,
    register_copy_num_per_thread: int,
) -> list[dict]:
    """Return the launch configs to time for one elementwise specialization.

    The swept elements-per-thread brackets the default the same *bytes_per_thread*
    and *register_copy_num_per_thread* produce, so a kernel can always land back on
    its shipped config, and reaches *min_num_per_thread* where a kernel lowers it.
    """
    autotune_threads = (128, 256, 512)  # block widths the sweep times
    # A direct body takes no num_per_thread: the key would name no parameter to bind,
    # and the sweep would time one kernel three times over.
    if strategy == "direct":
        return [{"threads": t} for t in autotune_threads]
    elem_bytes = dtype.itemsize
    default = max(_MIN_NUM_PER_THREAD, bytes_per_thread // elem_bytes)
    if strategy == "register_copy":
        default = max(default, _register_copy_floor(register_copy_num_per_thread, elem_bytes))
    # An 8-byte dtype's default drops to one 16-byte vector on long launches.
    low = min(min_num_per_thread, _BYTES_PER_THREAD // elem_bytes)
    npts = tuple(sorted({min_num_per_thread, max(low, default // 2), default, default * 2}))
    return [{"threads": t, "num_per_thread": n} for t in autotune_threads for n in npts]


@dataclass(frozen=True)
class _ElementwiseOutputPlan:
    logical_dtype: torch.dtype
    kernel_output_dtype: str
    bool_via_int8: bool = False


def elementwise_output_plan(
    input_dtype: torch.dtype,
    declared_output_dtype: torch.dtype | None = None,
    *,
    strategy: str,
) -> _ElementwiseOutputPlan:
    """Return the dtype the kernel writes, and the dtype the caller sees."""
    logical_dtype = declared_output_dtype or input_dtype
    # Every strategy but `direct` can store the result through an int8 buffer, and
    # wants to: a bool store lowers to one byte per lane, where int8 vectorises.
    bool_via_int8 = declared_output_dtype == torch.bool and strategy != "direct"
    kernel_output_dtype = (
        BOOL_STORAGE_DTYPE if bool_via_int8 else Kernel.dtype_to_str(logical_dtype)
    )
    return _ElementwiseOutputPlan(logical_dtype, kernel_output_dtype, bool_via_int8)


def choose_strategy(
    input_dtype: torch.dtype, declared_output_dtype: torch.dtype | None, same_shape: bool = True
) -> str:
    """``direct`` where a bool operand, or a bool result of a narrow integer, cannot be
    vectorized; else ``register_copy`` for same-shape operands and ``explicit_parallel``
    for broadcast ones."""
    narrow_int = input_dtype in (torch.uint8, torch.int8, torch.int16)
    if input_dtype == torch.bool or (declared_output_dtype == torch.bool and narrow_int):
        return "direct"
    return "register_copy" if same_shape else "explicit_parallel"
