"""Workload definitions for the pool op family."""


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


class MeanPoolingWorkload(WorkloadBase):

    def __init__(self, batch_size: int, seq_len: int, heads: int, dim: int, chunk_size: int,
                 chunks_per_batch: int, seq_num: int, use_offsets: int,
                 dtype: torch.dtype, accum_dtype: torch.dtype,
                 offsets: Optional[torch.Tensor] = None,
                 indices: Optional[torch.Tensor] = None) -> None:
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
        self.offsets = offsets
        self.indices = indices

    def gen_inputs(self) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = torch.randn(
            self.batch_size, self.seq_len, self.heads, self.dim,
            device="cuda", dtype=self.dtype)
        return x, self.offsets, self.indices

    def ref_program(self, x: torch.Tensor, offsets: torch.Tensor,
                    indices: torch.Tensor) -> torch.Tensor:
        _ = indices
        batch_size, seq_len, heads, dim = x.shape

        if self.use_offsets == 0:
            output = torch.empty(
                batch_size, self.chunks_per_batch, heads, dim, dtype=x.dtype, device=x.device)
            for chunk_id in range(self.chunks_per_batch):
                start_token = chunk_id * self.chunk_size
                end_token = min(start_token + self.chunk_size, seq_len)
                output[:, chunk_id] = x[:, start_token:end_token].mean(dim=1)
        else:
            offsets = offsets.to(x.device)
            lengths = offsets[1:] - offsets[:-1]
            chunk_counts = ((lengths + self.chunk_size - 1) // self.chunk_size).tolist()
            total_chunks = sum(chunk_counts)
            output = torch.empty(
                batch_size, total_chunks, heads, dim, dtype=x.dtype, device=x.device)
            chunk_idx = 0
            for b in range(batch_size):
                for seq_id, chunks_i in enumerate(chunk_counts):
                    seq_start = offsets[seq_id].item()
                    seq_end = offsets[seq_id + 1].item()
                    for local_chunk_id in range(chunks_i):
                        chunk_start = seq_start + local_chunk_id * self.chunk_size
                        chunk_end = min(chunk_start + self.chunk_size, seq_end)
                        output[b, chunk_idx] = x[b, chunk_start:chunk_end].mean(dim=0)
                        chunk_idx += 1
        return output
