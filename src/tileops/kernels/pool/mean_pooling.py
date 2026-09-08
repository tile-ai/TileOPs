import functools
from typing import Any, Callable, Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import WARP_LANES, get_sm_count

__all__ = ["MeanPoolingFwdKernel"]


class _BlockTiling:
    """How one chunk's ``heads * dim`` is cut into blocks, and each block into lanes.

    The widths and lane shares below describe this kernel's access pattern, not the
    device, and are held here so that a later kernel does not read them as general truths.
    """

    # Widths a block may take. One that divides `heads * dim` covers it exactly; where none
    # does -- a width that is not a whole number of warps -- the last block is bounds-tested.
    _WIDTHS = (8192, 4096, 2048, 1024, 512, 256, 128, 64, 32)
    # Elements a lane accumulates; eight fp16 of them is a 16-byte load.
    _LANE_ELEMS = 8
    # Lane shares a tuning run tries, either side of `_LANE_ELEMS`.
    _TUNED_LANE_ELEMS = (4, 8, 16)
    # Widths a tuning run tries around the default, wider and narrower.
    _TUNED_WIDER = 1
    _TUNED_NARROWER = 2
    # Threads a CUDA block can hold.
    _MAX_THREADS = 1024

    def __init__(self, width: int, blocks: int, sm_count: int) -> None:
        """Fix the tiling for one call's shape.

        Args:
            width: ``heads * dim``, the axis a block takes its share of.
            blocks: One block per chunk, before the width is cut.
            sm_count: Multiprocessors the grid has to cover.
        """
        self._width = width
        self._blocks = blocks
        self._sm_count = sm_count

    def _widths(self) -> list[int]:
        """The widths this ``heads * dim`` admits, widest first."""
        listed = [w for w in self._WIDTHS if w <= self._width] or [self._WIDTHS[-1]]
        return [w for w in listed if self._width % w == 0] or listed

    def _threads(self, bwidth: int, lane_elems: int) -> int:
        """The largest whole-warp count dividing ``bwidth`` into ``lane_elems`` or more."""
        threads = min(
            self._MAX_THREADS,
            max(WARP_LANES, bwidth // lane_elems // WARP_LANES * WARP_LANES),
        )
        while threads > WARP_LANES and bwidth % threads:
            threads -= WARP_LANES
        return threads

    def default(self) -> dict:
        """The widest block whose grid still covers the SMs, and its thread count.

        A block reads one chunk's share of the width, so a narrower block buys more blocks
        at the cost of a shorter contiguous run in each. The width is cut only until the
        grid reaches one block per SM.
        """
        widths = self._widths()
        bwidth = widths[-1]
        for candidate in widths:
            if self._blocks * -(-self._width // candidate) >= self._sm_count:
                bwidth = candidate
                break
        return {"bwidth": bwidth, "threads": self._threads(bwidth, self._LANE_ELEMS)}

    def space(self) -> list[dict]:
        """`default`'s width and its neighbours, at each lane share."""
        widths = self._widths()
        chosen = widths.index(self.default()["bwidth"])
        band = widths[max(0, chosen - self._TUNED_WIDER) : chosen + self._TUNED_NARROWER + 1]
        return [
            {"bwidth": w, "threads": t}
            for w in band
            for t in sorted({self._threads(w, e) for e in self._TUNED_LANE_ELEMS})
        ]


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

            Why: written as `whole_blocks or col < width`, Python's `or` would drop the
            comparison before TileLang traced it.
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

                    # The chunk's own token count bounds the loop; predicating each row
                    # of a `chunk_size` walk instead is slower.
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
        self._tiling = _BlockTiling(
            width=heads * dim,
            blocks=batch_size * chunks_per_batch,
            sm_count=get_sm_count(),
        )

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
            # Sequences split the tokens evenly; a slot past the last sequence clamps
            # onto it, repeating a chunk of the same size.
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
        return self._tiling.default()

    @property
    def autotune_configs(self) -> list[dict]:
        return self._tiling.space()

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
