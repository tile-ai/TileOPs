"""Workload definitions for the pool op family."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.numerics import Exact
from workloads.sequence_metadata import make_cu_seqlens, prepare_chunk_indices
from workloads.workload_base import CallWorkload, WorkloadBase


def max_pool_ref(ndim: int) -> Callable:
    """Return ``torch.nn.functional.max_pool{ndim}d``."""
    return getattr(F, f"max_pool{ndim}d")


@dataclass(eq=False)
class AvgPoolWorkload(WorkloadBase):
    ndim: int
    kernel_size: int | tuple[int, ...]
    stride: int | tuple[int, ...] | None
    padding: int | tuple[int, ...]
    ceil_mode: bool
    count_include_pad: bool
    divisor_override: int | None
    dtype: torch.dtype

    def gen_inputs(self, *shape: int) -> tuple[torch.Tensor]:
        return (torch.randn(*shape, device=run_device(), dtype=self.dtype),)

    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        return self.torch_program(input.float()).to(input.dtype)

    def torch_program(self, input: torch.Tensor) -> torch.Tensor:
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

    def verification(self, *inputs):
        return pool_verification(maximum=False)


@dataclass(eq=False)
class MaxPoolWorkload(WorkloadBase):
    ndim: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...] | None
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    ceil_mode: bool
    dtype: torch.dtype
    contiguous: bool = True
    return_indices: bool = False

    def gen_inputs(self, *shape: int) -> tuple[torch.Tensor]:
        x = torch.randn(*shape, device=run_device(), dtype=self.dtype)
        if not self.contiguous:
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


@dataclass(eq=False)
class AdaptivePool2dWorkload(WorkloadBase):
    """One NCHW tensor for the adaptive 2D pool family.

    ``output_size`` shapes no input; it rides along because the op needs it.
    """

    n: int
    c_in: int
    h_in: int
    w_in: int
    output_size: int | None | tuple[int | None, int | None]
    dtype: torch.dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(
            self.n, self.c_in, self.h_in, self.w_in, device=run_device(), dtype=self.dtype
        )
        return (x,)

    @property
    def _torch_output_size(self) -> int | tuple[int | None, int | None]:
        """``output_size`` as torch takes it: torch rejects a scalar None, and (None, None)
        means the same."""
        return (None, None) if self.output_size is None else self.output_size


class AdaptiveAvgPool2dWorkload(AdaptivePool2dWorkload):
    """AdaptiveAvgPool2dFwdOp's input and reference."""

    def ref_program(self, input: torch.Tensor) -> torch.Tensor:
        return self.torch_program(input.float()).to(input.dtype)

    def torch_program(self, input: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(input, self._torch_output_size)

    def verification(self, *inputs):
        return pool_verification(maximum=False)


class AdaptiveMaxPool2dWorkload(AdaptivePool2dWorkload):
    """AdaptiveMaxPool2dFwdOp's input and reference, or its ``Indices`` variant's."""

    def __init__(self, *args: Any, return_indices: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.return_indices = return_indices

    def ref_program(self, input: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return F.adaptive_max_pool2d(
            input, self._torch_output_size, return_indices=self.return_indices
        )

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


@dataclass(eq=False)
class MeanPoolingWorkload(WorkloadBase):
    """One chunked-sequence-mean case: its shape, its dtype, and how it is split.

    ``seq_lens`` selects the ragged split and is what ``cu_seqlens`` and ``chunk_indices`` are built
    from; without it the split is uniform and the call passes neither tensor.
    """

    batch: int
    seq_len: int
    heads: int
    dim: int
    chunk_size: int
    dtype: torch.dtype
    seq_lens: Sequence[int] | None = None

    def __post_init__(self) -> None:
        self.seq_lens = None if self.seq_lens is None else list(self.seq_lens)

    def gen_inputs(self) -> tuple:
        x = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device=run_device(), dtype=self.dtype
        )
        if self.seq_lens is None:
            return (x,)

        # The ragged split: cumulative boundaries and one (sequence, chunk) pair per chunk.
        cu_seqlens = make_cu_seqlens(self.seq_lens)
        return x, cu_seqlens, prepare_chunk_indices(cu_seqlens, self.chunk_size)

    def chunk_slices(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Resolve metadata outside a compiled reference; the math stays shared."""
        if cu_seqlens is None:
            return tuple(
                (start, min(start + self.chunk_size, x.shape[1]))
                for start in range(0, x.shape[1], self.chunk_size)
            )
        bounds = cu_seqlens.tolist()
        return tuple(
            (
                bounds[sequence] + chunk * self.chunk_size,
                min(bounds[sequence] + (chunk + 1) * self.chunk_size, bounds[sequence + 1]),
            )
            for sequence, chunk in chunk_indices.tolist()
        )

    @staticmethod
    def torch_slices(x: torch.Tensor, slices: tuple[tuple[int, int], ...]) -> torch.Tensor:
        """Mean of each named chunk in the caller's order, with no padding in its divisor."""
        if not slices:
            return x.new_empty((x.shape[0], 0, *x.shape[2:]))
        return torch.stack([x[:, start:end].mean(1) for start, end in slices], dim=1)

    @staticmethod
    def reference_slices(x: torch.Tensor, slices: tuple[tuple[int, int], ...]) -> torch.Tensor:
        """``torch_slices`` in fp32, rounded once to ``x.dtype``."""
        return MeanPoolingWorkload.torch_slices(x.float(), slices).to(x.dtype)

    def ref_program(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.reference_slices(x, self.chunk_slices(x, cu_seqlens, chunk_indices))

    def torch_program(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.torch_slices(x, self.chunk_slices(x, cu_seqlens, chunk_indices))


class MeanPoolingCallWorkload(CallWorkload, MeanPoolingWorkload):
    """One manifest call of ``MeanPoolingFwdOp``: its tensors, ``cu_seqlens`` and ``chunk_indices``
    holding the call's generated values, checked against the reference the workload states."""

    def __init__(self, call: Any) -> None:
        CallWorkload.__init__(self, call)
        batch, seq_len, heads, dim = call.tensors["x"][0]
        cu_seqlens = call.metadata_values("cu_seqlens") if call.present("cu_seqlens") else None
        MeanPoolingWorkload.__init__(
            self,
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            dim=dim,
            chunk_size=call.params["chunk_size"],
            dtype=getattr(torch, call.tensors["x"][1]),
            seq_lens=None
            if cu_seqlens is None
            else [b - a for a, b in zip(cu_seqlens, cu_seqlens[1:], strict=False)],
        )


def pool_verification(*, maximum):
    """Max pooling selects values and indices exactly; averages incur rounding."""
    return Exact(atol=0, rtol=0) if maximum else Exact()
