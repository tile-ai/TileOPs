"""The facts of one GEMM-family call after the op has inferred its dimensions, and the kernel
interfaces the implementations serving it inherit."""

import dataclasses
from abc import abstractmethod
from typing import Literal, Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = [
    "BmmCall",
    "BmmFp8Call",
    "BmmFp8FwdInterface",
    "BmmFp8TransposeCall",
    "BmmFp8TransposeFwdInterface",
    "BmmFwdInterface",
    "GemmCall",
    "GemmFp8Call",
    "GemmFp8FwdInterface",
    "GemmFwdInterface",
    "GemmW4A16Call",
    "GemmW4A16FwdInterface",
    "W4A16RepackCall",
    "W4A16RepackFwdInterface",
]


@dataclasses.dataclass(frozen=True)
class GemmCall(CallSpec):
    """One dense matmul, as the op knows it after inferring ``(m, n, k)``."""

    m: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None
    trans_a: bool = False
    trans_b: bool = False


@dataclasses.dataclass(frozen=True)
class GemmFp8Call(CallSpec):
    """One FP8 NT matmul, with the scale grids and the bias the call carries.

    The layout is fixed: ``a`` is ``[m, k]`` and ``b`` is ``[n, k]``, so no flags.
    """

    m: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None
    scale_a_shape: Optional[tuple] = None
    scale_b_shape: Optional[tuple] = None
    out_dtype: Optional[torch.dtype] = None
    has_bias: bool = False


@dataclasses.dataclass(frozen=True)
class GemmW4A16Call(CallSpec):
    """One W4A16 NT matmul against a prepacked INT4 weight."""

    m: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None
    # The dequantization group along K its kernels are compiled for.
    group_size: int = 0


@dataclasses.dataclass(frozen=True)
class W4A16RepackCall(CallSpec):
    """One conversion of a row-major packed INT4 weight into the order the GEMM reads."""

    n: int = 0
    # Bytes per weight row, ``k / 2``.
    packed_k: int = 0


@dataclasses.dataclass(frozen=True)
class BmmCall(CallSpec):
    """One batched matmul after the op has inferred ``(batch, m, n, k)``."""

    batch: int = 0
    m: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class BmmFp8Call(CallSpec):
    """One batched FP8 matmul, with the output dtype its epilogue writes."""

    batch: int = 0
    m: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None
    out_dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class BmmFp8TransposeCall(CallSpec):
    """One swap of the last two axes of a contiguous ``[batch, rows, cols]`` tensor."""

    batch: int = 0
    rows: int = 0
    cols: int = 0
    dtype: Optional[torch.dtype] = None


class GemmFwdInterface(KernelInterface):
    """Dense matmul under the ``(trans_a, trans_b)`` layout the call states."""

    request = GemmCall

    @abstractmethod
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply the two matrices; nothing is written in place.

        Both operands are contiguous on ``call.device`` in ``call.dtype``, which is
        ``float16`` or ``bfloat16``; the contraction accumulates in ``float32``.

        Args:
            a: ``(call.m, call.k)``, or ``(call.k, call.m)`` when ``call.trans_a``.
            b: ``(call.n, call.k)`` when ``call.trans_b``, else ``(call.k, call.n)``.

        Returns:
            A new ``(call.m, call.n)`` tensor in ``call.dtype``.
        """


class GemmFp8FwdInterface(KernelInterface):
    """FP8 NT matmul: the scaled product of two ``float8_e4m3fn`` operands, plus a bias."""

    request = GemmFp8Call

    @abstractmethod
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multiply, apply both scales and add the bias; nothing is written in place.

        Every tensor is contiguous on ``call.device``. The scales are ``float32`` and
        their shapes are ``call.scale_a_shape`` and ``call.scale_b_shape``: either both
        ``(1, 1)``, or ``scale_a`` per 1x128 block along K and ``scale_b`` per 1x128 or
        per 128x128 block (:meth:`block_scale_grid`). The program is compiled for one
        side of ``call.has_bias``, so a call passes a bias exactly where the call spec
        said it would.

        Args:
            a: ``(call.m, call.k)`` in ``call.dtype``.
            b: ``(call.n, call.k)`` in ``call.dtype``.
            scale_a: ``float32`` scales of *a*, shaped ``call.scale_a_shape``.
            scale_b: ``float32`` scales of *b*, shaped ``call.scale_b_shape``.
            bias: ``(call.n,)`` in ``call.out_dtype``, or ``None``.

        Returns:
            A new ``(call.m, call.n)`` tensor in ``call.out_dtype``.
        """

    @classmethod
    def block_scale_grid(cls, call: GemmFp8Call) -> Optional[Literal["1d1d", "1d2d"]]:
        """Which block128 grid the scales form, or ``None`` for any other pair.

        Both grids take ``scale_a`` per 1x128 block, ``[m, ceil(k / 128)]``. ``"1d1d"``
        takes ``scale_b`` the same way, ``[n, ceil(k / 128)]``; ``"1d2d"`` takes it per
        128x128 block, ``[ceil(n / 128), ceil(k / 128)]``. At ``n == 1`` the two grids
        coincide and mean the same product; that shape reads as ``"1d1d"``.
        """
        scale_k = -(-call.k // 128)
        if call.scale_a_shape != (call.m, scale_k):
            return None
        if call.scale_b_shape == (call.n, scale_k):
            return "1d1d"
        if call.scale_b_shape == (-(-call.n // 128), scale_k):
            return "1d2d"
        return None


class GemmW4A16FwdInterface(KernelInterface):
    """W4A16 NT matmul with group-wise affine dequantization of a prepacked weight."""

    request = GemmW4A16Call

    @abstractmethod
    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply the activations by the dequantized INT4 weight; nothing is in place.

        Every tensor is contiguous on ``call.device``. Weight element ``(n, k)`` is
        ``(q - weight_zero[n, g]) * weight_scale[n, g]`` with ``g = k // call.group_size``
        and ``q`` its INT4 value. *packed_weight* is in the order
        :class:`W4A16RepackFwdInterface` produces, which has the same shape and dtype as the
        row-major packing and cannot be told from it at runtime.

        Args:
            activation: ``(call.m, call.k)`` in ``call.dtype``.
            packed_weight: ``(call.n, call.k // 2)`` ``uint8``, two INT4 per byte.
            weight_scale: ``(call.n, call.k // call.group_size)`` in ``call.dtype``.
            weight_zero: ``(call.n, call.k // call.group_size)`` ``uint8``.

        Returns:
            A new ``(call.m, call.n)`` tensor in ``call.dtype``.
        """


class W4A16RepackFwdInterface(KernelInterface):
    """The weight order :class:`GemmW4A16FwdInterface` reads, produced once at load time."""

    request = W4A16RepackCall

    @abstractmethod
    def forward(self, packed_weight: torch.Tensor) -> torch.Tensor:
        """Reorder the nibbles of a row-major packed weight; nothing is written in place.

        The order is the one this backend's W4A16 GEMM decodes, so the two are replaced
        together.

        Args:
            packed_weight: ``(call.n, call.packed_k)`` ``uint8`` on ``call.device``,
                contiguous, two INT4 per byte with even K in the low nibble.

        Returns:
            A new ``(call.n, call.packed_k)`` ``uint8`` tensor holding the same nibbles.
        """


class BmmFwdInterface(KernelInterface):
    """Batched dense matmul, one independent GEMM per batch item."""

    request = BmmCall

    @abstractmethod
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply the two batches; nothing is written in place.

        Both operands are contiguous on ``call.device`` in ``call.dtype``, which is
        ``float16`` or ``bfloat16``; the contraction accumulates in ``float32``.

        Args:
            a: ``(call.batch, call.m, call.k)``.
            b: ``(call.batch, call.k, call.n)``.

        Returns:
            A new ``(call.batch, call.m, call.n)`` tensor in ``call.dtype``.
        """


class BmmFp8FwdInterface(KernelInterface):
    """Batched FP8 matmul with one per-tensor scale on each operand."""

    request = BmmFp8Call

    @abstractmethod
    def forward(
        self, a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
    ) -> torch.Tensor:
        """Multiply the two batches and apply both scales; nothing is in place.

        Every tensor is contiguous on ``call.device``. FP8 WGMMA reads *b* K-innermost
        and has no transposed operand mode, so the op hands it that way whatever axis
        order the caller passed.

        Args:
            a: ``(call.batch, call.m, call.k)`` in ``call.dtype``.
            b: ``(call.batch, call.n, call.k)`` in ``call.dtype``.
            scale_a: A one-element ``float32`` tensor scaling every element of *a*.
            scale_b: A one-element ``float32`` tensor scaling every element of *b*.

        Returns:
            A new ``(call.batch, call.m, call.n)`` tensor in ``call.out_dtype``.
        """


class BmmFp8TransposeFwdInterface(KernelInterface):
    """The axis swap that puts a batched FP8 operand K-innermost for the WGMMA kernel."""

    request = BmmFp8TransposeCall

    @abstractmethod
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """Swap the last two axes; nothing is written in place.

        Data movement only: the result is bit-identical to
        ``src.transpose(-2, -1).contiguous()``.

        Args:
            src: Contiguous ``(call.batch, call.rows, call.cols)`` in ``call.dtype`` on
                ``call.device``.

        Returns:
            A new contiguous ``(call.batch, call.cols, call.rows)`` tensor in
            ``call.dtype``.
        """
