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
    "BmmFP8Call",
    "BmmFP8FwdInterface",
    "BmmFP8TransposeCall",
    "BmmFP8TransposeFwdInterface",
    "BmmFwdInterface",
    "GemmCall",
    "GemmFP8Call",
    "GemmFP8FwdInterface",
    "GemmFwdInterface",
    "GemmINT8W8A8Call",
    "GemmINT8W8A8FwdInterface",
    "GemmW4A8Call",
    "GemmW4A8FwdInterface",
    "GemmW4A16Call",
    "GemmW4A16FwdInterface",
    "W4A8RepackCall",
    "W4A8RepackFwdInterface",
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
class GemmFP8Call(CallSpec):
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
class GemmINT8W8A8Call(CallSpec):
    """One INT8 NT matmul, with the granularity of each scale and the bias the call carries.

    ``a`` is ``[m, k]`` and ``b`` is ``[n, k]``. ``per_token`` says ``scale_a`` is
    ``[m, 1]`` rather than ``[1, 1]``, ``per_channel`` that ``scale_b`` is ``[1, n]``
    rather than ``[1, 1]``.
    """

    m: int = 0
    n: int = 0
    k: int = 0
    per_token: bool = False
    per_channel: bool = False
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
class GemmW4A8Call(CallSpec):
    """One W4A8 NT matmul: FP8 activations against a prepacked signed INT4 weight."""

    m: int = 0
    n: int = 0
    k: int = 0
    # The weight scale group along K its kernels are compiled for.
    group_size: int = 0


@dataclasses.dataclass(frozen=True)
class W4A8RepackCall(CallSpec):
    """One conversion of a row-major packed signed INT4 weight into the order the GEMM reads."""

    n: int = 0
    k: int = 0


@dataclasses.dataclass(frozen=True)
class BmmCall(CallSpec):
    """One batched matmul after the op has inferred ``(batch, m, n, k)``."""

    batch: int = 0
    m: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class BmmFP8Call(CallSpec):
    """One batched FP8 matmul, with the output dtype its epilogue writes."""

    batch: int = 0
    m: int = 0
    n: int = 0
    k: int = 0
    dtype: Optional[torch.dtype] = None
    out_dtype: Optional[torch.dtype] = None


@dataclasses.dataclass(frozen=True)
class BmmFP8TransposeCall(CallSpec):
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


class GemmFP8FwdInterface(KernelInterface):
    """FP8 NT matmul: the scaled product of two ``float8_e4m3fn`` operands, plus a bias."""

    request = GemmFP8Call

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
    def block_scale_grid(cls, call: GemmFP8Call) -> Optional[Literal["1d1d", "1d2d"]]:
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


class GemmINT8W8A8FwdInterface(KernelInterface):
    """INT8 NT matmul accumulated in int32, dequantized by a row and a column scale."""

    request = GemmINT8W8A8Call

    @abstractmethod
    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multiply exactly in int32, scale in fp32, add the bias; nothing is written in place.

        Every tensor is contiguous on ``call.device``. The program is compiled for one side
        of ``call.has_bias``, so a call passes a bias exactly where the call spec said it
        would.

        Args:
            a: ``(call.m, call.k)`` ``int8``.
            b: ``(call.n, call.k)`` ``int8``.
            scale_a: ``float32``, ``(call.m, 1)`` when ``call.per_token``, else ``(1, 1)``.
            scale_b: ``float32``, ``(1, call.n)`` when ``call.per_channel``, else ``(1, 1)``.
            bias: ``(call.n,)`` in ``call.out_dtype``, or ``None``.

        Returns:
            A new ``(call.m, call.n)`` tensor in ``call.out_dtype``.
        """


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


class GemmW4A8FwdInterface(KernelInterface):
    """W4A8 NT matmul: FP8 activations times a group-scaled signed INT4 weight."""

    request = GemmW4A8Call

    @abstractmethod
    def forward(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_group_scale: torch.Tensor,
        weight_channel_scale: torch.Tensor,
        activation_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Multiply the FP8 activations by the scaled INT4 weight; nothing is in place.

        Every tensor is contiguous on ``call.device``. Weight element ``(n, k)`` enters the
        FP8 tensor-core product as ``float8_e4m3fn(q * weight_group_scale[n, g])``, rounded
        to nearest even, with ``g = k // call.group_size`` and ``q`` its signed INT4 value;
        output element ``(m, n)`` is ``bfloat16(acc * weight_channel_scale[n] *
        activation_scale[m])``. *packed_weight* is in the order
        :class:`W4A8RepackFwdInterface` produces, which has the same shape and dtype as the
        row-major packing and cannot be told from it at runtime.

        Args:
            activation: ``(call.m, call.k)`` ``float8_e4m3fn``.
            packed_weight: ``(call.n, call.k // 2)`` ``uint8``, two INT4 per byte.
            weight_group_scale: ``(call.n, call.k // call.group_size)`` ``float8_e4m3fn``.
            weight_channel_scale: ``(call.n,)`` ``float32``.
            activation_scale: ``(call.m,)`` ``float32``.

        Returns:
            A new ``(call.m, call.n)`` ``bfloat16`` tensor.
        """


class W4A8RepackFwdInterface(KernelInterface):
    """The weight order :class:`GemmW4A8FwdInterface` reads, produced once at load time."""

    request = W4A8RepackCall

    @abstractmethod
    def forward(self, packed_weight: torch.Tensor) -> torch.Tensor:
        """Reorder and re-encode the nibbles of a row-major packed weight; nothing is in place.

        The order and nibble code are the ones this backend's W4A8 GEMMs decode, so the
        three are replaced together.

        Args:
            packed_weight: ``(call.n, call.k // 2)`` ``uint8`` on ``call.device``,
                contiguous, two signed two's-complement INT4 per byte with even K in the
                low nibble.

        Returns:
            A new ``(call.n, call.k // 2)`` ``uint8`` tensor holding the same weights.
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


class BmmFP8FwdInterface(KernelInterface):
    """Batched FP8 matmul with one per-tensor scale on each operand."""

    request = BmmFP8Call

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


class BmmFP8TransposeFwdInterface(KernelInterface):
    """The axis swap that puts a batched FP8 operand K-innermost for the WGMMA kernel."""

    request = BmmFP8TransposeCall

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
