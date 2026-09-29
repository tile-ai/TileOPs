"""Call records for reduction kernels."""

from __future__ import annotations

import dataclasses
import math

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.utils import get_shared_memory_optin

from ._primitives import edge_axis_split

__all__ = [
    "FOLD_KINDS",
    "NORM_KINDS",
    "SIMPLE_KINDS",
    "WELFORD_KINDS",
    "LogSumExpCall",
    "LogicalReduceCall",
    "ReduceCall",
    "SoftmaxCall",
]


@dataclasses.dataclass(frozen=True)
class LogicalReduceCall(CallSpec):
    """A logical reduction call: the input's shape and dtype, the axes it reduces."""

    shape: tuple[int, ...] = ()
    # Non-negative and ascending.
    axes: tuple[int, ...] = ()
    op_kind: str = ""
    dtype: torch.dtype = torch.float16
    keepdim: bool = False

    @property
    def device_index(self) -> "int | None":
        return self.device.index if self.device is not None else None

    @property
    def edge_kept(self) -> int:
        """The kept extent between a reduced prefix and suffix of axes, or 0 for other layouts."""
        k, j = edge_axis_split(len(self.shape), self.axes)
        return math.prod(self.shape[k : len(self.shape) - j]) if k else 0


@dataclasses.dataclass(frozen=True)
class _SharedMemoryCall(CallSpec):
    """A call whose kernels plan against the device's shared memory.

    ``smem_budget`` is a device fact like ``sm_count``: a record that states none reads
    it when it is built.
    """

    smem_budget: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.smem_budget > 0 or (self.device is not None and self.device.type != "cuda"):
            return
        index = self.device.index if self.device is not None else None
        object.__setattr__(self, "smem_budget", get_shared_memory_optin(index))


@dataclasses.dataclass(frozen=True)
class LogSumExpCall(_SharedMemoryCall):
    """A logsumexp call: the input as the manifest declares it.

    ``axes`` are the reduced axes, ascending and non-negative.
    """

    shape: tuple[int, ...] = ()
    axes: tuple[int, ...] = ()
    keepdim: bool = False
    dtype: torch.dtype = torch.float16

    @property
    def n(self) -> int:
        """Elements each output element reduces."""
        return math.prod(self.shape[a] for a in self.axes)

    @property
    def m(self) -> int:
        """Output elements: the kept extents' product."""
        return math.prod(self.shape) // self.n


@dataclasses.dataclass(frozen=True)
class SoftmaxCall(_SharedMemoryCall):
    """A softmax or log_softmax call: the input as the manifest declares it.

    ``axis`` is the normalized axis, non-negative; ``dtype`` is the input's as the kernel
    reads it and ``out_dtype`` the output's.
    """

    shape: tuple[int, ...] = ()
    axis: int = 0
    op_kind: str = "softmax"
    dtype: torch.dtype = torch.float16
    out_dtype: torch.dtype = torch.float16

    @property
    def n(self) -> int:
        """Length of the normalized axis."""
        return self.shape[self.axis]

    @property
    def m(self) -> int:
        """Rows: the product of every other axis."""
        return math.prod(self.shape) // self.n


SIMPLE_KINDS = frozenset({"sum", "mean", "amax", "amin"})
WELFORD_KINDS = frozenset({"std", "var", "var_mean"})
NORM_KINDS = frozenset({"l1", "l2", "inf"})
# Kinds the register fold serves.
FOLD_KINDS = SIMPLE_KINDS | {"prod"} | NORM_KINDS


@dataclasses.dataclass(frozen=True)
class ReduceCall(_SharedMemoryCall):
    """A reduction or vector-norm call: the manifest's input, ``dim`` and parameters."""

    shape: tuple[int, ...] = ()
    axes: tuple[int, ...] = ()
    keepdim: bool = False
    op_kind: str = ""
    dtype: torch.dtype = torch.float16
    # The ``dtype`` parameter; ``None`` keeps the input's.
    out_dtype: "torch.dtype | None" = None
    # The Welford kinds' correction; 0 where the op divides by zero degrees of freedom itself.
    correction: float = 0

    @property
    def n(self) -> int:
        """Elements each output reduces."""
        return math.prod(self.shape[a] for a in self.axes)

    @property
    def m(self) -> int:
        """Outputs: the elements of the axes the reduction keeps."""
        return math.prod(self.shape) // self.n
