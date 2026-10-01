"""The elementwise kernel interfaces and the call specs they take.

The family's programs are built from an element count, or from the two operand shapes a
broadcast coalesces, plus the element type and the scalar parameters the op bakes into the
body. A call spec holds those facts; an interface publishes the tensors one program is
handed and what it returns.
"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = [
    "AlibiCall",
    "AlibiFwdInterface",
    "AlphaScaledCall",
    "AlphaScaledBinaryFwdInterface",
    "BinaryElementwiseFwdInterface",
    "BinaryPredicateFwdInterface",
    "BoundedUnaryFwdInterface",
    "BoundsCall",
    "BroadcastCall",
    "ClampTensorCall",
    "ClampTensorFwdInterface",
    "ElementwiseCall",
    "EluCall",
    "EluFwdInterface",
    "FusedGatedCall",
    "FusedGatedFwdInterface",
    "LeakyReluCall",
    "LeakyReluFwdInterface",
    "LerpCall",
    "LerpFwdInterface",
    "LerpTensorFwdInterface",
    "MaskedFillCall",
    "MaskedFillFwdInterface",
    "MaskedFillTensorValueFwdInterface",
    "NanToNumCall",
    "NanToNumFwdInterface",
    "PreluCall",
    "PreluFwdInterface",
    "ReciprocalFwdInterface",
    "RoundCall",
    "RoundFwdInterface",
    "SinusoidalCall",
    "SinusoidalFwdInterface",
    "SoftplusCall",
    "SoftplusFwdInterface",
    "UnaryElementwiseFwdInterface",
    "UnaryPredicateFwdInterface",
    "WhereFwdInterface",
]


@dataclasses.dataclass(frozen=True)
class ElementwiseCall(CallSpec):
    """An elementwise call over one flattened extent.

    ``n_total`` is the element count of the output, which every operand broadcasts to, and
    ``dtype`` the element type the op was called with. A bool call keeps ``torch.bool``
    here; an implementation computing it in another storage type says so in ``entry_for``.
    """

    n_total: int = 0
    dtype: torch.dtype = torch.float16


@dataclasses.dataclass(frozen=True)
class BroadcastCall(CallSpec):
    """A binary elementwise call: the two operand shapes as the manifest declares them.

    The shapes broadcast against each other under the PyTorch rules; coalescing them into
    a loop nest is the implementation's own work, so both shapes are facts of the call.
    """

    a_shape: tuple[int, ...] = ()
    b_shape: tuple[int, ...] = ()
    dtype: torch.dtype = torch.float16


@dataclasses.dataclass(frozen=True)
class AlphaScaledCall(BroadcastCall):
    """A binary call whose second operand is scaled by ``alpha`` before combining."""

    alpha: "int | float" = 1


@dataclasses.dataclass(frozen=True)
class LerpCall(BroadcastCall):
    """A scalar-weight lerp call; ``weight`` is the op's construction parameter."""

    weight: float = 0.5


@dataclasses.dataclass(frozen=True)
class FusedGatedCall(CallSpec):
    """A fused gated call: the input is ``(m, 2 * n)`` and the output ``(m, n)``."""

    m: int = 0
    n: int = 0
    dtype: torch.dtype = torch.float16


@dataclasses.dataclass(frozen=True)
class LeakyReluCall(ElementwiseCall):
    """A leaky ReLU call; the slope is the op's construction parameter."""

    negative_slope: float = 0.01


@dataclasses.dataclass(frozen=True)
class EluCall(ElementwiseCall):
    """An ELU call; the scale of the negative part is the op's construction parameter."""

    alpha: float = 1.0


@dataclasses.dataclass(frozen=True)
class BoundsCall(ElementwiseCall):
    """A clamping call; ``None`` on either side leaves that side unbounded."""

    min_val: Optional[float] = None
    max_val: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class SoftplusCall(ElementwiseCall):
    """A softplus call; both scalars are the op's construction parameters."""

    beta: float = 1.0
    threshold: float = 20.0


@dataclasses.dataclass(frozen=True)
class NanToNumCall(ElementwiseCall):
    """A nan_to_num call.

    ``posinf`` and ``neginf`` are ``None`` where the op left them unset, which stands for
    the largest and smallest finite value of ``dtype``; resolving them is the
    implementation's, since it is the one that knows what it bakes into the body.
    """

    nan: float = 0.0
    posinf: Optional[float] = None
    neginf: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class RoundCall(ElementwiseCall):
    """A round call; ``decimals`` is the op's construction parameter."""

    decimals: int = 0


@dataclasses.dataclass(frozen=True)
class MaskedFillCall(ElementwiseCall):
    """A scalar-value masked_fill call; ``value`` is the op's construction parameter."""

    value: "bool | int | float" = 0.0


@dataclasses.dataclass(frozen=True)
class ClampTensorCall(ElementwiseCall):
    """A Tensor-bound clamp call; the flags say which bounds this call passes."""

    has_min: bool = False
    has_max: bool = False


@dataclasses.dataclass(frozen=True)
class PreluCall(ElementwiseCall):
    """A PReLU call.

    ``num_channels`` is the weight's length and ``inner_size`` the elements one channel
    spans in a row, so a flat index names its channel.
    """

    num_channels: int = 1
    inner_size: int = 1


@dataclasses.dataclass(frozen=True)
class AlibiCall(CallSpec):
    """An ALiBi call: the extents and element type of the tensor it generates."""

    seq_len: int = 0
    num_heads: int = 0
    dtype: torch.dtype = torch.float32


@dataclasses.dataclass(frozen=True)
class SinusoidalCall(CallSpec):
    """A sinusoidal-encoding call: the extents and element type of the tensor it generates."""

    seq_len: int = 0
    d_model: int = 0
    dtype: torch.dtype = torch.float32


class UnaryElementwiseFwdInterface(KernelInterface):
    """One tensor in, one tensor of the same shape and element type out.

    Every unary program of the family whose body needs nothing but the element type: the
    exp/log/root/rounding/trigonometry set, the parameter-free activations, the bitwise
    negation.
    """

    request = ElementwiseCall

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the implementation's function to each element; nothing is written in place.

        Args:
            input: ``call.n_total`` elements in ``call.dtype`` on ``call.device``,
                contiguous in whatever shape the caller passed. The op makes it contiguous
                before the call.

        Returns:
            A new contiguous tensor on ``call.device``, shaped and typed as *input*.
        """


class UnaryPredicateFwdInterface(KernelInterface):
    """One tensor in, one bool tensor of the same shape out."""

    request = ElementwiseCall

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Test each element; nothing is written in place.

        Args:
            input: ``call.n_total`` elements in ``call.dtype`` on ``call.device``,
                contiguous in whatever shape the caller passed.

        Returns:
            A new contiguous ``torch.bool`` tensor on ``call.device``, shaped as *input*.
        """


class ReciprocalFwdInterface(KernelInterface):
    """The reciprocal, which promotes an integral input the way torch does."""

    request = ElementwiseCall

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``1 / input`` elementwise; nothing is written in place.

        Args:
            input: ``call.n_total`` elements in ``call.dtype`` on ``call.device``,
                contiguous in whatever shape the caller passed.

        Returns:
            A new contiguous tensor on ``call.device``, shaped as *input*, in
            ``call.dtype`` for a floating input and in ``torch.float32`` for an integral
            one, as ``torch.reciprocal`` promotes it.
        """


class LeakyReluFwdInterface(KernelInterface):
    """Leaky ReLU with the slope the call names."""

    request = LeakyReluCall

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``input`` where it is positive and ``call.negative_slope * input`` elsewhere.

        Nothing is written in place.

        Args:
            input: ``call.n_total`` elements in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, shaped and typed as *input*.
        """


class EluFwdInterface(KernelInterface):
    """ELU with the negative-part scale the call names."""

    request = EluCall

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``input`` where it is positive and ``call.alpha * (exp(input) - 1)`` elsewhere.

        Nothing is written in place.

        Args:
            input: ``call.n_total`` elements in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, shaped and typed as *input*.
        """


class BoundedUnaryFwdInterface(KernelInterface):
    """Clamping to the scalar bounds the call names, which hardtanh and clamp both are."""

    request = BoundsCall

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Clamp each element into ``[call.min_val, call.max_val]``; nothing is written in place.

        A bound the call leaves as ``None`` is not applied on that side. A NaN input or
        bound gives NaN, as ``torch.clamp`` does.

        Args:
            input: ``call.n_total`` elements in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, shaped and typed as *input*.
        """


class SoftplusFwdInterface(KernelInterface):
    """Softplus with the scale and linear-regime threshold the call names."""

    request = SoftplusCall

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Return ``log1p(exp(beta * input)) / beta``, and ``input`` above the threshold.

        The switch is ``beta * input > call.threshold``, and nothing is written in place.

        Args:
            input: ``call.n_total`` elements in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, shaped and typed as *input*.
        """


class NanToNumFwdInterface(KernelInterface):
    """Replacement of the non-finite values by the ones the call names."""

    request = NanToNumCall

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Replace NaN, ``+inf`` and ``-inf``; nothing is written in place.

        NaN becomes ``call.nan``. An infinity becomes ``call.posinf`` or ``call.neginf``,
        or, where the call left that side ``None``, the largest or smallest finite value of
        ``call.dtype``. A finite value is unchanged.

        Args:
            input: ``call.n_total`` elements in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, shaped and typed as *input*.
        """


class RoundFwdInterface(KernelInterface):
    """Rounding to ``call.decimals`` decimal places."""

    request = RoundCall

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Round each element half to even; nothing is written in place.

        ``call.decimals`` places are kept, and an integral ``call.dtype`` is returned
        unchanged.

        Args:
            input: ``call.n_total`` elements in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, shaped and typed as *input*.
        """


class BinaryElementwiseFwdInterface(KernelInterface):
    """Two tensors broadcast against each other, one tensor of their element type out."""

    request = BroadcastCall

    @abstractmethod
    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Combine the two operands elementwise; nothing is written in place.

        Args:
            input: shape ``call.a_shape`` in ``call.dtype`` on ``call.device``, contiguous.
            other: shape ``call.b_shape`` in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, in ``call.dtype``, shaped as the
            broadcast of the two operands.
        """


class BinaryPredicateFwdInterface(KernelInterface):
    """Two tensors broadcast against each other, one bool tensor out."""

    request = BroadcastCall

    @abstractmethod
    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Test the two operands elementwise; nothing is written in place.

        Args:
            input: shape ``call.a_shape`` in ``call.dtype`` on ``call.device``, contiguous.
            other: shape ``call.b_shape`` in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous ``torch.bool`` tensor on ``call.device``, shaped as the
            broadcast of the two operands.
        """


class AlphaScaledBinaryFwdInterface(KernelInterface):
    """Two tensors combined after the second is scaled by ``call.alpha``."""

    request = AlphaScaledCall

    @abstractmethod
    def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        """Combine ``input`` with ``call.alpha * other``; nothing is written in place.

        The scale follows the PyTorch coercion of a scalar to the operand type: an integral
        ``call.dtype`` wraps ``alpha`` into its own range, and bool takes its truth value.

        Args:
            input: shape ``call.a_shape`` in ``call.dtype`` on ``call.device``, contiguous.
            other: shape ``call.b_shape`` in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, in ``call.dtype``, shaped as the
            broadcast of the two operands.
        """


class LerpFwdInterface(KernelInterface):
    """Linear interpolation by the scalar weight the call names."""

    request = LerpCall

    @abstractmethod
    def forward(self, input: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        """Return ``input + call.weight * (end - input)``; nothing is written in place.

        Args:
            input: shape ``call.a_shape`` in ``call.dtype`` on ``call.device``, contiguous.
            end: shape ``call.b_shape`` in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, in ``call.dtype``, shaped as the
            broadcast of the two operands.
        """


class FusedGatedFwdInterface(KernelInterface):
    """An activation of the first half of each row, multiplied by the second half."""

    request = FusedGatedCall

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``activation(x[:, :n]) * x[:, n:]``; nothing is written in place.

        Args:
            x: ``(call.m, 2 * call.n)`` in ``call.dtype`` on ``call.device``, contiguous
                and row-major.

        Returns:
            A new contiguous ``(call.m, call.n)`` tensor on ``call.device``, in ``call.dtype``.
        """


class WhereFwdInterface(KernelInterface):
    """Elementwise selection between two operands by a bool condition."""

    request = ElementwiseCall

    @abstractmethod
    def forward(
        self, condition: torch.Tensor, input: torch.Tensor, other: torch.Tensor
    ) -> torch.Tensor:
        """Take ``input`` where ``condition`` holds and ``other`` elsewhere.

        Nothing is written in place. The three operands broadcast together to
        ``call.n_total`` elements.

        Args:
            condition: any shape broadcasting with the others, ``torch.bool`` on
                ``call.device``, contiguous.
            input: any shape broadcasting with the others, ``call.dtype`` on
                ``call.device``, contiguous.
            other: any shape broadcasting with the others, ``call.dtype`` on
                ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, in ``call.dtype``, shaped as the
            broadcast of the three operands.
        """


class LerpTensorFwdInterface(KernelInterface):
    """Linear interpolation by a tensor weight."""

    request = ElementwiseCall

    @abstractmethod
    def forward(self, input: torch.Tensor, end: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Return ``input + weight * (end - input)``; nothing is written in place.

        The three operands broadcast together to ``call.n_total`` elements.

        Args:
            input: any shape broadcasting with the others, ``call.dtype`` on
                ``call.device``, contiguous.
            end: any shape broadcasting with the others, ``call.dtype`` on
                ``call.device``, contiguous.
            weight: any shape broadcasting with the others, ``call.dtype`` on
                ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, in ``call.dtype``, shaped as the
            broadcast of the three operands.
        """


class ClampTensorFwdInterface(KernelInterface):
    """Clamping to tensor bounds that broadcast against the input."""

    request = ClampTensorCall

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        min: Optional[torch.Tensor],
        max: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Clamp each element between the bounds; nothing is written in place.

        A bound the call does not pass arrives as ``None`` and is not applied; the flags
        ``call.has_min`` and ``call.has_max`` say which ones the call passes, and at least
        one is set. A NaN in any operand gives NaN, as ``torch.clamp`` does. The operands
        broadcast together to ``call.n_total`` elements.

        Args:
            input: any shape broadcasting with the bounds passed, ``call.dtype`` on
                ``call.device``, contiguous.
            min: the same, or ``None`` where ``call.has_min`` is unset.
            max: the same, or ``None`` where ``call.has_max`` is unset.

        Returns:
            A new contiguous tensor on ``call.device``, in ``call.dtype``, shaped as the
            broadcast of the operands passed.
        """


class MaskedFillFwdInterface(KernelInterface):
    """Filling the masked positions with the scalar the call names."""

    request = MaskedFillCall

    @abstractmethod
    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return ``call.value`` where ``mask`` holds and ``input`` elsewhere.

        Nothing is written in place. The two operands broadcast together to
        ``call.n_total`` elements.

        Args:
            input: any shape broadcasting with ``mask``, ``call.dtype`` on
                ``call.device``, contiguous.
            mask: any shape broadcasting with ``input``, ``torch.bool`` on
                ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, in ``call.dtype``, shaped as the
            broadcast of the two operands.
        """


class MaskedFillTensorValueFwdInterface(KernelInterface):
    """Filling the masked positions with a value read from a 0-dim tensor."""

    request = ElementwiseCall

    @abstractmethod
    def forward(self, input: torch.Tensor, mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Return ``value`` where ``mask`` holds and ``input`` elsewhere.

        Nothing is written in place. ``input`` and ``mask`` broadcast together to
        ``call.n_total`` elements.

        Args:
            input: any shape broadcasting with ``mask``, ``call.dtype`` on
                ``call.device``, contiguous.
            mask: any shape broadcasting with ``input``, ``torch.bool`` on
                ``call.device``, contiguous.
            value: a 0-dim tensor in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, in ``call.dtype``, shaped as the
            broadcast of ``input`` and ``mask``.
        """


class PreluFwdInterface(KernelInterface):
    """PReLU with a per-channel slope."""

    request = PreluCall

    @abstractmethod
    def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Return ``input`` where it is positive and ``weight[channel] * input`` elsewhere.

        Nothing is written in place. A flat index names its channel through
        ``call.inner_size``: channel ``(index // call.inner_size) % call.num_channels``.

        Args:
            input: ``call.n_total`` elements in ``call.dtype`` on ``call.device``, contiguous.
            weight: ``(call.num_channels,)`` in ``call.dtype`` on ``call.device``, contiguous.

        Returns:
            A new contiguous tensor on ``call.device``, shaped and typed as *input*.
        """


class AlibiFwdInterface(KernelInterface):
    """Generation of the ALiBi position-bias tensor."""

    request = AlibiCall

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Return ``bias[h, i, j] = -slope_h * abs(i - j)``.

        The slopes are the geometric sequence ALiBi defines for ``call.num_heads`` heads.
        The call reads no tensor and writes none in place.

        Returns:
            A new contiguous ``(call.num_heads, call.seq_len, call.seq_len)`` tensor in
            ``call.dtype`` on ``call.device``.
        """


class SinusoidalFwdInterface(KernelInterface):
    """Generation of the sinusoidal positional encoding."""

    request = SinusoidalCall

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Return the encoding of "Attention Is All You Need".

        Column ``2k`` of row ``p`` is ``sin(p / 10000 ** (2k / d_model))`` and column
        ``2k + 1`` its cosine. The call reads no tensor and writes none in place.

        Returns:
            A new contiguous ``(call.seq_len, call.d_model)`` tensor in ``call.dtype`` on
            ``call.device``.
        """
