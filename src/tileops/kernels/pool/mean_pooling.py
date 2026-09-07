import functools
from typing import Any, Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import WARP_LANES, get_sm_count

__all__ = ["MeanPoolingFwdKernel"]

# The block widths a `heads * dim` may be cut into.
_BLOCK_WIDTHS = (8192, 4096, 2048, 1024, 512, 256, 128, 64, 32)

# Elements one lane accumulates; eight fp16 of them is a 16-byte load.
_LANE_ELEMS = 8
# The shares a lane is tried at when tuning, either side of `_LANE_ELEMS`.
_TUNED_LANE_ELEMS = (4, 8, 16)


def _block_widths(width: int) -> list[int]:
    """The block widths to try for a `heads * dim` of ``width``, widest first.

    A width the block divides is covered exactly, so those widths come first; a width none
    of them divides — one that is not a multiple of the warp size — is covered with a
    bounds test on the last block instead.
    """
    listed = [w for w in _BLOCK_WIDTHS if w <= width] or [_BLOCK_WIDTHS[-1]]
    return [w for w in listed if width % w == 0] or listed


def _threads_for(bwidth: int, lane_elems: int) -> int:
    """The largest whole-warp thread count dividing ``bwidth`` into ``lane_elems`` or more."""
    threads = min(1024, max(WARP_LANES, bwidth // lane_elems // WARP_LANES * WARP_LANES))
    while threads > WARP_LANES and bwidth % threads:
        threads -= WARP_LANES
    return threads


def _launch(width: int, blocks: int, sm_count: int) -> tuple[int, int]:
    """The widest block whose grid still covers the SMs, and the threads to run it with.

    A block reads one chunk's share of the width, so a narrower block buys more blocks at
    the cost of a shorter contiguous run in each. The width is cut only until the grid
    reaches one block per SM: below that the machine sits idle, above it the reads shorten
    for nothing.
    """
    widths = _block_widths(width)
    bwidth = widths[-1]
    for candidate in widths:
        if blocks * -(-width // candidate) >= sm_count:
            bwidth = candidate
            break
    return bwidth, _threads_for(bwidth, _LANE_ELEMS)


@functools.lru_cache(maxsize=32)
def _mean_pooling_kernel(
    batch_size: int,
    seq_len: int,
    heads: int,
    dim: int,
    chunk_size: int,
    chunks_per_batch: int,
    seq_num: int,
    use_offsets: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    # Neither `heads` nor `dim` is reduced and they are the two innermost axes of a
    # contiguous tensor, so a block reads them as one contiguous width.
    width = heads * dim
    # Every chunk is then `chunk_size` tokens at `i_t * chunk_size`, an index that bound
    # analysis places inside the sequence axis without a per-row guard.
    full_chunks = use_offsets == 0 and seq_len % chunk_size == 0

    @tilelang.jit(out_idx=[1])
    def _mean_pooling_func(bwidth: int, threads: int) -> None:
        # Only a width no block width divides leaves a last block reaching past the axis.
        whole_blocks = width % bwidth == 0

        def in_width(col):
            """The lane's bound, or `True` where no block reaches past the axis.

            A value rather than `whole_blocks or col < width`, whose `or` would drop the
            comparison in Python before TileLang traced it.
            """
            return True if whole_blocks else col < width

        @T.prim_func
        def _mean_pooling_main(
            x: T.Tensor((batch_size, seq_len, width), dtype),
            o: T.Tensor((batch_size, chunks_per_batch, width), dtype),
            offsets: T.Tensor((seq_num + 1,), T.int32),
            indices: T.Tensor(
                (chunks_per_batch, 2), T.int32
            ),  # columns are (seq_id, chunk_id within that sequence)
        ) -> None:
            with T.Kernel(
                T.ceildiv(width, bwidth), chunks_per_batch, batch_size, threads=threads
            ) as (i_w, i_t, i_b):
                total = T.alloc_fragment((bwidth,), accum_dtype)
                start_col = i_w * bwidth
                T.clear(total)

                if full_chunks:
                    for s in T.serial(chunk_size):
                        for j in T.Parallel(bwidth):
                            if in_width(start_col + j):
                                total[j] += T.cast(
                                    x[i_b, i_t * chunk_size + s, start_col + j], accum_dtype
                                )
                    scale = T.cast(1.0 / chunk_size, accum_dtype)
                else:
                    start_token = T.alloc_var(T.int32)
                    end_token = T.alloc_var(T.int32)
                    if use_offsets == 0:
                        start_token = i_t * chunk_size
                        end_token = T.min(start_token + chunk_size, seq_len)
                    else:
                        seq_id = indices[i_t, 0]
                        local_chunk_id = indices[i_t, 1]
                        start_token = offsets[seq_id] + local_chunk_id * chunk_size
                        end_token = T.min(start_token + chunk_size, offsets[seq_id + 1])

                    # The chunk's own token count bounds the loop. Walking `chunk_size`
                    # rows and dropping the ones past the end instead predicates the load,
                    # which is slower on every ragged workload.
                    for s in T.serial(end_token - start_token):
                        for j in T.Parallel(bwidth):
                            if in_width(start_col + j):
                                total[j] += T.cast(
                                    x[i_b, start_token + s, start_col + j], accum_dtype
                                )
                    scale = T.cast(1.0, accum_dtype) / T.cast(end_token - start_token, accum_dtype)

                for j in T.Parallel(bwidth):
                    if in_width(start_col + j):
                        o[i_b, i_t, start_col + j] = T.cast(total[j] * scale, dtype)

        return _mean_pooling_main

    return _mean_pooling_func


@torch.library.custom_op("tileops::mean_pooling_fwd_wrapped_kernel", mutates_args=())
def _mean_pooling_wrapped_kernel(
    batch_size: int,
    seq_len: int,
    heads: int,
    dim: int,
    chunk_size: int,
    chunks_per_batch: int,
    seq_num: int,
    use_offsets: int,
    dtype: str,
    accum_dtype: str,
    bwidth: int,
    threads: int,
    x: torch.Tensor,
    offsets: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    width = heads * dim
    pooled = _mean_pooling_kernel(
        batch_size=batch_size,
        seq_len=seq_len,
        heads=heads,
        dim=dim,
        chunk_size=chunk_size,
        chunks_per_batch=chunks_per_batch,
        seq_num=seq_num,
        use_offsets=use_offsets,
        dtype=dtype,
        accum_dtype=accum_dtype,
    )(bwidth, threads)(x.view(batch_size, seq_len, width), offsets, indices)
    return pooled.view(batch_size, chunks_per_batch, heads, dim)


@_mean_pooling_wrapped_kernel.register_fake
def _(
    batch_size: int,
    seq_len: int,
    heads: int,
    dim: int,
    chunk_size: int,
    chunks_per_batch: int,
    seq_num: int,
    use_offsets: int,
    dtype: str,
    accum_dtype: str,
    bwidth: int,
    threads: int,
    *inputs: tuple[Any],
) -> torch.Tensor:
    _ = (seq_len, chunk_size, seq_num, bwidth, use_offsets, dtype, accum_dtype, threads)
    x = inputs[0]
    return torch.empty(
        (batch_size, chunks_per_batch, heads, dim),
        device=x.device,
        dtype=x.dtype,
    )


class MeanPoolingFwdKernel(Kernel):
    supported_archs: list[int] = [90]

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        heads: int,
        dim: int,
        chunk_size: int,
        chunks_per_batch: int,
        seq_num: int,
        use_offsets: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        config: Optional[dict] = None,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.heads = heads
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunks_per_batch = chunks_per_batch
        self.seq_num = seq_num
        self.use_offsets = use_offsets
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.accum_dtype_str = self.dtype_to_str(self.accum_dtype)
        self.width = heads * dim
        # One block per chunk before the width is cut; what the launch rule divides.
        self.blocks = batch_size * chunks_per_batch

        self.kernel = _mean_pooling_kernel(
            self.batch_size,
            self.seq_len,
            self.heads,
            self.dim,
            self.chunk_size,
            self.chunks_per_batch,
            self.seq_num,
            self.use_offsets,
            self.dtype_str,
            self.accum_dtype_str,
        )

        self.init_config(config, tune)

    @property
    def autotune_supply_prog(self):
        """Supply autotuning the chunk map a real call carries.

        The kernel takes a ragged chunk's token range from ``offsets[seq_id]`` and
        ``offsets[seq_id + 1]`` and divides by that range's length, so random values leave
        it empty or inverted.
        """
        from tilelang.utils.device import get_current_device
        from tilelang.utils.tensor import get_tensor_supply

        default_supply = get_tensor_supply(tilelang.TensorSupplyType.Auto)
        seq_len, seq_num = self.seq_len, self.seq_num
        chunk_size, chunks_per_batch = self.chunk_size, self.chunks_per_batch

        def supply_prog(params):
            device = get_current_device()
            # Sequences split the tokens evenly; a slot past the last sequence
            # clamps onto it and repeats a chunk of the same size.
            bounds = torch.linspace(0, seq_len, seq_num + 1, device=device).to(torch.int32)
            per_seq = max(1, seq_len // seq_num)
            chunks_per_seq = max(1, (per_seq + chunk_size - 1) // chunk_size)
            slots = torch.arange(chunks_per_batch, device=device)
            indices = torch.stack(
                ((slots // chunks_per_seq).clamp(max=seq_num - 1), slots % chunks_per_seq),
                dim=1,
            ).to(torch.int32)

            supplied = []
            matched = 0
            for param in params:
                shape = list(param.shape)
                if str(param.dtype) != "int32":
                    supplied.append(default_supply(param))
                elif shape == [seq_num + 1]:
                    supplied.append(bounds)
                    matched += 1
                elif shape == [chunks_per_batch, 2]:
                    supplied.append(indices)
                    matched += 1
                else:
                    supplied.append(default_supply(param))
            if matched != 2:
                raise RuntimeError(
                    f"autotuning {type(self).__name__} expects int32 offsets "
                    f"[{seq_num + 1}] and indices [{chunks_per_batch}, 2], matched {matched}"
                )
            return supplied

        return supply_prog

    @property
    def default_config(self) -> dict:
        bwidth, threads = _launch(self.width, self.blocks, get_sm_count())
        return {"bwidth": bwidth, "threads": threads}

    @property
    def autotune_configs(self) -> list[dict]:
        """The launch rule's width and its neighbours, at 8, 16 and 32 bytes per lane.

        A band narrow enough that every member is close to the best, with the untuned
        default a member of it.
        """
        widths = _block_widths(self.width)
        chosen = widths.index(self.default_config["bwidth"])
        return [
            {"bwidth": w, "threads": t}
            for w in widths[max(0, chosen - 1) : chosen + 3]
            for t in sorted({_threads_for(w, e) for e in _TUNED_LANE_ELEMS})
        ]

    def forward(
        self, x: torch.Tensor, offsets: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        self._require_cuda(x=x, offsets=offsets, indices=indices)
        return _mean_pooling_wrapped_kernel(
            self.batch_size,
            self.seq_len,
            self.heads,
            self.dim,
            self.chunk_size,
            self.chunks_per_batch,
            self.seq_num,
            self.use_offsets,
            self.dtype_str,
            self.accum_dtype_str,
            self.config["bwidth"],
            self.config["threads"],
            x,
            offsets,
            indices,
        )
