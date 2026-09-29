"""The reduction kernel interfaces and the call specs they take.

An op's reduction kind picks its interface. Kinds share one where every implementation
serves all of them; the call spec's ``op_kind`` then says which one a build computes.
"""

from __future__ import annotations

import dataclasses
import math
from abc import abstractmethod

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface
from tileops.kernels.reduction._primitives import edge_axis_split
from tileops.utils import get_shared_memory_optin

__all__ = [
    "WELFORD_KINDS",
    "ArgreduceCall",
    "ArgreduceFwdInterface",
    "CountNonzeroFwdInterface",
    "CumprodFwdInterface",
    "CumsumFwdInterface",
    "CumulativeCall",
    "LogSumExpCall",
    "LogSumExpFwdInterface",
    "LogicalReduceCall",
    "LogicalReduceFwdInterface",
    "ProdFwdInterface",
    "ReduceCall",
    "ReduceFwdInterface",
    "SoftmaxCall",
    "SoftmaxFwdInterface",
    "VarMeanFwdInterface",
    "VarianceFwdInterface",
    "VectorNormFwdInterface",
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


WELFORD_KINDS = frozenset({"std", "var", "var_mean"})


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


@dataclasses.dataclass(frozen=True)
class ArgreduceCall(CallSpec):
    """An argmax or argmin call: the input's shape and dtype, the axes it reduces.

    ``axes`` is one axis, or every axis for the index into the flattened input.
    """

    shape: tuple[int, ...] = ()
    axes: tuple[int, ...] = ()
    keepdim: bool = False
    op_kind: str = "argmax"
    dtype: torch.dtype = torch.float16

    @property
    def n(self) -> int:
        """Length of the reduced extent."""
        return math.prod(self.shape[a] for a in self.axes)

    @property
    def m(self) -> int:
        """Output elements."""
        return math.prod(self.shape) // self.n

    @property
    def inner_stride(self) -> int:
        """Elements between two neighbours along the reduced axis; 1 for a full reduction."""
        return math.prod(self.shape[self.axes[-1] + 1 :]) if len(self.axes) == 1 else 1


@dataclasses.dataclass(frozen=True)
class CumulativeCall(_SharedMemoryCall):
    """A cumsum or cumprod call: the input's shape and dtype, and the axis it scans."""

    shape: tuple[int, ...] = ()
    axis: int = 0
    op_kind: str = "sum"
    dtype: torch.dtype = torch.float16

    @property
    def n(self) -> int:
        """Length of the scanned axis."""
        return self.shape[self.axis]

    @property
    def m(self) -> int:
        """Rows: the product of every other axis."""
        return math.prod(d for i, d in enumerate(self.shape) if i != self.axis)


class ReduceFwdInterface(KernelInterface):
    """``sum``, ``mean``, ``amax`` or ``amin`` over ``call.axes``, by ``call.op_kind``."""

    request = ReduceCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce ``call.axes`` of *x*; nothing is written in place.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype`` on ``call.device``.

        Returns:
            A new tensor shaped as ``call.axes`` and ``call.keepdim`` leave it, in
            ``call.out_dtype`` (``call.dtype`` when ``None``).
        """


class ProdFwdInterface(KernelInterface):
    """The product over ``call.axes``, one axis."""

    request = ReduceCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multiply along ``call.axes`` of *x*; nothing is written in place.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype`` on ``call.device``.

        Returns:
            A new tensor shaped as ``call.axes`` and ``call.keepdim`` leave it, in
            ``call.out_dtype`` (``call.dtype`` when ``None``).
        """


class VectorNormFwdInterface(KernelInterface):
    """The ``l1``, ``l2`` or ``inf`` vector norm over ``call.axes``, by ``call.op_kind``."""

    request = ReduceCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Norm ``call.axes`` of *x*; a NaN in a reduced run yields NaN.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype`` on ``call.device``.

        Returns:
            A new tensor shaped as ``call.axes`` and ``call.keepdim`` leave it, in
            ``call.out_dtype`` (``call.dtype`` when ``None``).
        """


class VarianceFwdInterface(KernelInterface):
    """``std`` or ``var`` over ``call.axes`` with ``call.correction``, by ``call.op_kind``."""

    request = ReduceCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take the statistic over ``call.axes`` of *x*; nothing is written in place.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype`` on ``call.device``.

        Returns:
            A new tensor in ``call.dtype`` shaped as ``call.axes`` and ``call.keepdim`` leave it.
        """


class VarMeanFwdInterface(KernelInterface):
    """The variance with ``call.correction`` and the mean over ``call.axes``."""

    request = ReduceCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Take both statistics over ``call.axes`` of *x*; nothing is written in place.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype`` on ``call.device``.

        Returns:
            New ``(var, mean)`` in ``call.dtype``, each shaped as ``call.axes`` and
            ``call.keepdim`` leave it.
        """


class ArgreduceFwdInterface(KernelInterface):
    """The index of the maximum or minimum along ``call.axes``, by ``call.op_kind``."""

    request = ArgreduceCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Index the extremum of *x*: the first one, a NaN outranking every number.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype`` on ``call.device``.

        Returns:
            New ``int64`` indices into the reduced extent, shaped as ``call.axes`` and
            ``call.keepdim`` leave it.
        """


class LogicalReduceFwdInterface(KernelInterface):
    """``all`` or ``any`` over ``call.axes``, by ``call.op_kind``."""

    request = LogicalReduceCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Test the elements of *x* for nonzero along ``call.axes``.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype``, any numeric or bool dtype.

        Returns:
            A new ``bool`` tensor shaped as ``call.axes`` and ``call.keepdim`` leave it.
        """


class CountNonzeroFwdInterface(KernelInterface):
    """The count of nonzero elements over ``call.axes``."""

    request = LogicalReduceCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Count the nonzero elements of *x* along ``call.axes``, exactly.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype``, any numeric or bool dtype.

        Returns:
            A new ``int64`` tensor without the reduced axes.
        """


class CumsumFwdInterface(KernelInterface):
    """The inclusive prefix sum along ``call.axis``."""

    request = CumulativeCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan ``call.axis`` of *x*; nothing is written in place.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype`` on ``call.device``.

        Returns:
            A new ``call.shape`` tensor in ``call.dtype``.
        """


class CumprodFwdInterface(KernelInterface):
    """The inclusive prefix product along ``call.axis``."""

    request = CumulativeCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan ``call.axis`` of *x*; nothing is written in place.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype`` on ``call.device``.

        Returns:
            A new ``call.shape`` tensor in ``call.dtype``.
        """


class SoftmaxFwdInterface(KernelInterface):
    """``softmax`` or ``log_softmax`` along ``call.axis``, by ``call.op_kind``."""

    request = SoftmaxCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize ``call.axis`` of *x*; nothing is written in place.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype`` on ``call.device``.

        Returns:
            A new ``call.shape`` tensor in ``call.out_dtype``.
        """


class LogSumExpFwdInterface(KernelInterface):
    """``logsumexp`` over ``call.axes``."""

    request = LogSumExpCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce ``call.axes`` of *x*; an all ``-inf`` run yields ``-inf``, a ``+inf`` ``+inf``.

        Args:
            x: Contiguous ``call.shape`` in ``call.dtype`` on ``call.device``.

        Returns:
            A new tensor in ``call.dtype`` shaped as ``call.axes`` and ``call.keepdim`` leave it.
        """
