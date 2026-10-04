"""Workload definitions for the pool op family."""

from collections.abc import Sequence
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase


def max_pool_ref(ndim: int) -> Callable:
    """Return ``torch.nn.functional.max_pool{ndim}d``."""
    return getattr(F, f"max_pool{ndim}d")


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
        x = torch.randn(*shape, device=run_device(), dtype=self.dtype).contiguous()
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
        pool = getattr(F, f"avg_pool{self.ndim}d")
        half = input.dtype in (torch.float16, torch.bfloat16)
        if self.ndim == 3 and half and input.device.type == "cpu":
            # torch has no CPU avg_pool3d for fp16/bf16; its CUDA kernel accumulates in fp32 too.
            return pool(input.float(), **kwargs).to(input.dtype)
        return pool(input, **kwargs)

    def verification(self, *inputs):
        return pool_verification(maximum=False)


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
        x = torch.randn(*shape, device=run_device(), dtype=self.dtype)
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

    def verification(self, *inputs):
        return pool_verification(maximum=True)


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
            self.n, self.c_in, self.h_in, self.w_in, device=run_device(), dtype=self.dtype
        ).contiguous()
        return (x,)


class AdaptiveAvgPool2dWorkload(AdaptivePool2dWorkload):
    """AdaptiveAvgPool2dFwdOp's input and reference."""

    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        # torch rejects a scalar None here; (None, None) means the same.
        size = (None, None) if self.output_size is None else self.output_size
        return F.adaptive_avg_pool2d(input, size)

    def verification(self, *inputs):
        return pool_verification(maximum=False)


class AdaptiveMaxPool2dWorkload(AdaptivePool2dWorkload):
    """AdaptiveMaxPool2dFwdOp's input and reference, or its ``Indices`` variant's."""

    def __init__(self, *args: Any, return_indices: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.return_indices = return_indices

    def ref_program(self, input: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # torch rejects a scalar None here; (None, None) means the same.
        size = (None, None) if self.output_size is None else self.output_size
        return F.adaptive_max_pool2d(input, size, return_indices=self.return_indices)

    def verification(self, *inputs):
        return pool_verification(maximum=True)


def _input_spec(call: Any) -> tuple[tuple[int, ...], torch.dtype]:
    shape, dtype = call.tensors["input"]
    return tuple(shape), getattr(torch, dtype)


class AvgPoolCall(CallWorkload, AvgPoolWorkload):
    """A manifest call of an ``AvgPool{1,2,3}dFwdOp``; the rank is the input's."""

    def __init__(self, call: Any) -> None:
        CallWorkload.__init__(self, call)
        shape, dtype = _input_spec(call)
        params = call.params
        AvgPoolWorkload.__init__(
            self,
            ndim=len(shape) - 2,
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            padding=params["padding"],
            ceil_mode=params["ceil_mode"],
            count_include_pad=params["count_include_pad"],
            divisor_override=params.get("divisor_override"),
            dtype=dtype,
        )

    gen_inputs = CallWorkload.gen_inputs


class MaxPoolCall(CallWorkload, MaxPoolWorkload):
    """A manifest call of a ``MaxPool{1,2,3}d[Indices]FwdOp``; the rank is the input's."""

    def __init__(self, call: Any, return_indices: bool = False) -> None:
        CallWorkload.__init__(self, call)
        shape, dtype = _input_spec(call)
        params = call.params
        MaxPoolWorkload.__init__(
            self,
            ndim=len(shape) - 2,
            kernel_size=params["kernel_size"],
            stride=params["stride"],
            padding=params["padding"],
            dilation=params["dilation"],
            ceil_mode=params["ceil_mode"],
            dtype=dtype,
            return_indices=return_indices,
        )

    gen_inputs = CallWorkload.gen_inputs


class AdaptiveAvgPool2dCall(CallWorkload, AdaptiveAvgPool2dWorkload):
    """A manifest call of AdaptiveAvgPool2dFwdOp.

    The input, batched or not, comes from the call, so only what the reference reads is set.
    """

    def __init__(self, call: Any) -> None:
        CallWorkload.__init__(self, call)
        self.output_size = call.params["output_size"]

    gen_inputs = CallWorkload.gen_inputs


class AdaptiveMaxPool2dCall(CallWorkload, AdaptiveMaxPool2dWorkload):
    """A manifest call of AdaptiveMaxPool2dFwdOp or its ``Indices`` variant.

    The input, batched or not, comes from the call, so only what the reference reads is set.
    """

    def __init__(self, call: Any, return_indices: bool = False) -> None:
        CallWorkload.__init__(self, call)
        self.output_size = call.params["output_size"]
        self.return_indices = return_indices

    gen_inputs = CallWorkload.gen_inputs


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
    from workloads.sequence_metadata import prepare_chunk_indices

    bounds = [0]
    for length in seq_lens:
        bounds.append(bounds[-1] + length)
    offsets = torch.tensor(bounds, dtype=torch.int32, device=run_device())
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
            self.batch, self.seq_len, self.heads, self.dim, device=run_device(), dtype=self.dtype
        )
        if self.seq_lens is None:
            return (x,)
        return (x, *mean_pooling_chunk_index(self.seq_lens, self.chunk_size))

    def chunk_slices(
        self,
        x: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> tuple[tuple[int, int], ...]:
        """Resolve metadata outside a compiled reference; the math stays shared."""
        if offsets is None:
            return tuple(
                (start, min(start + self.chunk_size, x.shape[1]))
                for start in range(0, x.shape[1], self.chunk_size)
            )
        bounds = offsets.tolist()
        return tuple(
            (
                bounds[sequence] + chunk * self.chunk_size,
                min(bounds[sequence] + (chunk + 1) * self.chunk_size, bounds[sequence + 1]),
            )
            for sequence, chunk in indices.tolist()
        )

    @staticmethod
    def reference_slices(x: torch.Tensor, slices: tuple[tuple[int, int], ...]) -> torch.Tensor:
        """Mean of each named chunk in the caller's order, with no padding in its divisor."""
        if not slices:
            return x.new_empty((x.shape[0], 0, *x.shape[2:]))
        return torch.stack([x[:, start:end].mean(1) for start, end in slices], dim=1)

    def ref_program(
        self,
        x: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.reference_slices(x, self.chunk_slices(x, offsets, indices))


class MeanPoolingCallWorkload(CallWorkload, MeanPoolingWorkload):
    """One manifest call of ``MeanPoolingFwdOp``: its tensors, ``offsets`` and ``indices``
    holding the call's generated values, checked against the reference the workload states."""

    def __init__(self, call: Any) -> None:
        CallWorkload.__init__(self, call)
        batch, seq_len, heads, dim = call.tensors["x"][0]
        offsets = call.values("offsets") if call.present("offsets") else None
        MeanPoolingWorkload.__init__(
            self,
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim=dim,
            chunk_size=call.params["chunk_size"],
            dtype=getattr(torch, call.tensors["x"][1]),
            accum_dtype=getattr(torch, call.params["accum_dtype"]),
            seq_lens=None
            if offsets is None
            else [b - a for a, b in zip(offsets, offsets[1:], strict=False)],
        )


def pool_verification(*, maximum):
    """Max pooling selects values and indices exactly; averages incur rounding."""
    from workloads.numerics import Exact

    return Exact(atol=0, rtol=0) if maximum else Exact()
