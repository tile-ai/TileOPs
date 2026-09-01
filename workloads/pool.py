"""Workload definitions for the pool op family."""

from collections.abc import Sequence
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from workloads.workload_base import WorkloadBase  # noqa: F401


def max_pool_ref(ndim: int) -> Callable:
    """Return ``torch.nn.functional.max_pool{ndim}d``."""
    return getattr(F, f"max_pool{ndim}d")


class AvgPool1dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_size: int,
        stride: Optional[int],
        padding: int,
        ceil_mode: bool,
        count_include_pad: bool,
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n, self.c_in, self.l_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)


class AvgPool2dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        kernel_size: tuple[int, int],
        stride: Optional[tuple[int, int]],
        padding: tuple[int, int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n, self.c_in, self.h_in, self.w_in, device="cuda", dtype=self.dtype
        ).contiguous()
        return (x,)


class AvgPool3dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        kernel_size: tuple[int, int, int],
        stride: Optional[tuple[int, int, int]],
        padding: tuple[int, int, int],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n,
            self.c_in,
            self.d_in,
            self.h_in,
            self.w_in,
            device="cuda",
            dtype=self.dtype,
        ).contiguous()
        return (x,)


class MaxPool2dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        kernel_size: tuple[int, int],
        stride: Optional[tuple[int, int]],
        padding: tuple[int, int],
        dilation: tuple[int, int],
        ceil_mode: bool,
        dtype: torch.dtype,
        return_indices: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.return_indices = return_indices

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n, self.c_in, self.h_in, self.w_in, device="cuda", dtype=self.dtype
        ).contiguous()
        return (x,)


class MaxPool1dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        l_in: int,
        kernel_size: tuple[int],
        stride: Optional[tuple[int]],
        padding: tuple[int],
        dilation: tuple[int],
        ceil_mode: bool,
        dtype: torch.dtype,
        return_indices: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.l_in = l_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.return_indices = return_indices

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(self.n, self.c_in, self.l_in, device="cuda", dtype=self.dtype).contiguous()
        return (x,)


class MaxPool3dBenchCase:
    def __init__(
        self,
        n: int,
        c_in: int,
        d_in: int,
        h_in: int,
        w_in: int,
        kernel_size: tuple[int, int, int],
        stride: Optional[tuple[int, int, int]],
        padding: tuple[int, int, int],
        dilation: tuple[int, int, int],
        ceil_mode: bool,
        dtype: torch.dtype,
        return_indices: bool = False,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.d_in = d_in
        self.h_in = h_in
        self.w_in = w_in
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.return_indices = return_indices

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n,
            self.c_in,
            self.d_in,
            self.h_in,
            self.w_in,
            device="cuda",
            dtype=self.dtype,
        ).contiguous()
        return (x,)


class AvgPoolWorkload(WorkloadBase):
    def __init__(
        self,
        ndim: int,
        kernel_size: int | tuple[int, ...],
        stride: Optional[int | tuple[int, ...]],
        padding: int | tuple[int, ...],
        ceil_mode: bool,
        count_include_pad: bool,
        divisor_override: Optional[int],
        dtype: torch.dtype,
    ) -> None:
        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype

    def gen_inputs(self, *shape: int) -> tuple[torch.Tensor]:
        x = torch.randn(*shape, device="cuda", dtype=self.dtype).contiguous()
        return (x,)

    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        kwargs: dict[str, object] = {
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "ceil_mode": self.ceil_mode,
            "count_include_pad": self.count_include_pad,
        }
        if self.ndim > 1:
            kwargs["divisor_override"] = self.divisor_override
        return getattr(F, f"avg_pool{self.ndim}d")(input, **kwargs)


class MaxPoolWorkload(WorkloadBase):
    def __init__(
        self,
        ndim: int,
        kernel_size: tuple[int, ...],
        stride: Optional[tuple[int, ...]],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        ceil_mode: bool,
        dtype: torch.dtype,
        contiguous: bool = True,
        return_indices: bool = False,
    ) -> None:
        self.ndim = ndim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.dtype = dtype
        self.contiguous = contiguous
        self.return_indices = return_indices

    def gen_inputs(self, *shape: int) -> tuple[torch.Tensor]:
        x = torch.randn(*shape, device="cuda", dtype=self.dtype)
        if self.contiguous:
            x = x.contiguous()
        else:
            # Non-contiguous view: transpose the last two dims twice so strides
            # differ but shape semantics stay N,C,<spatial dims>.
            x = x.transpose(-2, -1).contiguous().transpose(-2, -1)
            assert not x.is_contiguous()
        return (x,)

    def ref_program(self, input: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return max_pool_ref(self.ndim)(
            input,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )


class AdaptivePool2dWorkload(WorkloadBase):
    """One NCHW tensor for the adaptive 2D pool family.

    ``output_size`` shapes no input; it rides along because the op needs it.
    """

    def __init__(
        self,
        n: int,
        c_in: int,
        h_in: int,
        w_in: int,
        output_size: int | None | tuple[int | None, int | None],
        dtype: torch.dtype,
    ) -> None:
        self.n = n
        self.c_in = c_in
        self.h_in = h_in
        self.w_in = w_in
        self.output_size = output_size
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n, self.c_in, self.h_in, self.w_in, device="cuda", dtype=self.dtype
        ).contiguous()
        return (x,)


def mean_pooling_chunk_index(
    seq_lens: Sequence[int], chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``offsets`` and ``indices`` a ragged mean-pooling call takes.

    Args:
        seq_lens: Length of each sequence, in tokens.
        chunk_size: Tokens per chunk; a sequence's last chunk may hold fewer.

    Returns:
        ``offsets``, the ``len(seq_lens) + 1`` cumulative boundaries, and ``indices``, one
        ``(sequence, chunk-within-sequence)`` pair per chunk.
    """
    from workloads.nsa_utils import prepare_chunk_indices

    bounds = [0]
    for length in seq_lens:
        bounds.append(bounds[-1] + length)
    offsets = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    return offsets, prepare_chunk_indices(offsets, chunk_size)


class MeanPoolingWorkload(WorkloadBase):
    """One chunked-sequence-mean case: its shape, its dtype, and how it is split.

    ``seq_lens`` selects the ragged split and is what ``offsets`` and ``indices`` are built
    from; without it the split is uniform and the call passes neither tensor.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim: int,
        chunk_size: int,
        dtype: torch.dtype,
        accum_dtype: torch.dtype,
        seq_lens: Optional[Sequence[int]] = None,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim = dim
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.accum_dtype = accum_dtype
        self.seq_lens = None if seq_lens is None else list(seq_lens)

    def gen_inputs(self) -> tuple:
        x = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device="cuda", dtype=self.dtype
        )
        if self.seq_lens is None:
            return (x,)
        return (x, *mean_pooling_chunk_index(self.seq_lens, self.chunk_size))

    def ref_program(
        self,
        x: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = indices
        batch, seq_len, heads, dim = x.shape
        if offsets is None:
            chunks = -(-seq_len // self.chunk_size)
            output = torch.empty(batch, chunks, heads, dim, dtype=x.dtype, device=x.device)
            for chunk_id in range(chunks):
                start = chunk_id * self.chunk_size
                output[:, chunk_id] = x[:, start : start + self.chunk_size].mean(dim=1)
            return output

        lengths = (offsets[1:] - offsets[:-1]).tolist()
        counts = [-(-n // self.chunk_size) for n in lengths]
        output = torch.empty(batch, sum(counts), heads, dim, dtype=x.dtype, device=x.device)
        for b in range(batch):
            # Reset per row: `offsets` partitions every batch row the same way, so each
            # row's chunks fill the same output slots.
            chunk_idx = 0
            for seq_id, chunks_i in enumerate(counts):
                seq_start, seq_end = int(offsets[seq_id]), int(offsets[seq_id + 1])
                for local in range(chunks_i):
                    chunk_start = seq_start + local * self.chunk_size
                    chunk_end = min(chunk_start + self.chunk_size, seq_end)
                    output[b, chunk_idx] = x[b, chunk_start:chunk_end].mean(dim=0)
                    chunk_idx += 1
        return output
