"""The normalization kernel interfaces and the call specs they take."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod

import torch

from tileops._kernels.call_spec import CallSpec
from tileops._kernels.kernel_base import KernelInterface

__all__ = [
    "AdaLayerNormFwdInterface",
    "AdaLayerNormZeroFwdInterface",
    "BatchNormBwdInterface",
    "BatchNormCall",
    "BatchNormInferFwdInterface",
    "BatchNormTrainFwdInterface",
    "FusedAddLayerNormFwdInterface",
    "FusedAddRMSNormFwdInterface",
    "FusedAddRMSNormPerTokenQuantFwdInterface",
    "FusedQKNormRoPECall",
    "FusedQKNormRoPEFwdInterface",
    "GroupNormCall",
    "GroupNormFwdInterface",
    "InstanceNormFwdInterface",
    "InstanceNormInferFwdInterface",
    "InstanceNormTrainFwdInterface",
    "LayerNormCall",
    "LayerNormFwdInterface",
    "RMSNormCall",
    "RMSNormFwdInterface",
    "RMSNormGatedCall",
    "RMSNormGatedFwdInterface",
    "RMSNormQuantCall",
    "whole_row_refusal",
    "whole_row_scale_refusal",
]


@dataclasses.dataclass(frozen=True)
class BatchNormCall(CallSpec):
    """The facts that select a batch or instance normalization implementation and build it.

    The input is ``(n, c, *spatial)``; ``spatial`` is the product of the trailing axes.
    ``eps`` and ``momentum`` are the op's construction parameters, which the programs
    compile in. ``input_dtype_params`` is set where the affine is in the input dtype and
    the running statistics are read rounded to it, as ``instance_norm`` reads them;
    ``has_weight`` and ``has_bias`` say which affine tensors are passed, and
    ``has_running_stats`` whether the running statistics are.
    """

    n: int = 0
    c: int = 0
    spatial: int = 0
    dtype: torch.dtype = torch.float16
    eps: float = 1e-5
    momentum: float = 0.1
    input_dtype_params: bool = False
    has_weight: bool = True
    has_bias: bool = True
    has_running_stats: bool = True

    @property
    def passes_affine(self) -> bool:
        """Whether ``weight`` or ``bias`` is passed."""
        return self.has_weight or self.has_bias


@dataclasses.dataclass(frozen=True)
class LayerNormCall(CallSpec):
    """The facts that select an implementation normalizing trailing rows and build it.

    Layer, RMS, fused-add and adaptive layer normalization take it. ``n`` is the row's
    element count, ``eps`` the op's epsilon and ``rows`` the rows the call normalizes, ``0``
    where the op does not state them.
    """

    n: int = 0
    eps: float = 1e-5
    dtype: torch.dtype = torch.float16
    rows: int = 0


@dataclasses.dataclass(frozen=True)
class RMSNormCall(LayerNormCall):
    """The facts that select an RMS normalization implementation and build it.

    RMS and fused-add RMS normalization take it. ``zero_centered`` scales by ``1 + weight``
    instead of ``weight``; ``group_size`` is the run of elements each root mean square
    covers, ``None`` for the whole row.
    """

    zero_centered: bool = False
    group_size: int | None = None


def whole_row_refusal(call: RMSNormCall) -> str | None:
    """Why a kernel normalizing each whole row does not serve *call*, or ``None`` when it
    does."""
    if call.group_size not in (None, call.n):
        return "the kernel normalizes each whole row; group_size is not served"
    return None


def whole_row_scale_refusal(call: RMSNormCall) -> str | None:
    """Why a kernel normalizing each whole row and scaling it by ``weight`` does not serve
    *call*, or ``None`` when it does."""
    if call.zero_centered:
        return "the kernel scales by the weight itself; zero_centered is not served"
    return whole_row_refusal(call)


@dataclasses.dataclass(frozen=True)
class RMSNormQuantCall(CallSpec):
    """The facts that select an implementation normalizing and quantizing trailing rows.

    ``n`` is the row's element count and ``rows`` the rows the call normalizes. ``eps``,
    ``quant_dtype`` and ``scale_ub`` are the op's construction parameters, which the
    programs compile in; ``scale_ub`` is ``None`` when the amax is not clamped.
    """

    n: int = 0
    rows: int = 0
    eps: float = 1e-6
    dtype: torch.dtype = torch.bfloat16
    quant_dtype: torch.dtype = torch.float8_e4m3fn
    scale_ub: float | None = None


@dataclasses.dataclass(frozen=True)
class RMSNormGatedCall(CallSpec):
    """The facts that select a gated RMS normalization implementation and build it.

    ``n`` is the row's element count and ``group`` the element count each norm runs over,
    which divides ``n``. ``eps``, ``norm_before_gate`` and ``activation`` are the op's
    construction parameters, which the programs compile in. ``bias_dtype`` and ``z_dtype``
    are ``None`` where the call passes no ``bias`` or no ``z``.
    """

    n: int = 0
    group: int = 0
    eps: float = 1e-6
    norm_before_gate: bool = True
    activation: str = "silu"
    dtype: torch.dtype = torch.bfloat16
    weight_dtype: torch.dtype = torch.bfloat16
    bias_dtype: torch.dtype | None = None
    z_dtype: torch.dtype | None = None


class BatchNormTrainFwdInterface(KernelInterface):
    """Batch normalization by the batch statistics, updating the running ones."""

    request = BatchNormCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor | None,
        running_var: torch.Tensor | None,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize *x* per channel by its batch statistics.

        Every tensor passed is contiguous on ``call.device``, and none aliases another. A
        tensor is passed exactly where ``call`` says it is.

        Args:
            x: ``(n, c, spatial)`` in ``call.dtype``.
            running_mean: ``float32`` ``(c,)``, updated in place with ``call.momentum``;
                ``None`` where ``call.has_running_stats`` is false.
            running_var: ``float32`` ``(c,)``, updated in place with the unbiased variance;
                ``None`` exactly where ``running_mean`` is.
            weight: ``float32`` ``(c,)`` scale, or ``None`` for one where
                ``call.has_weight`` is false.
            bias: ``float32`` ``(c,)`` shift, or ``None`` for zero where
                ``call.has_bias`` is false.

        Returns:
            ``(y, mean, rstd)``: a new ``(n, c, spatial)`` output in ``call.dtype``, and the
            new ``float32`` ``(c,)`` batch mean and reciprocal standard deviation.
        """


class BatchNormInferFwdInterface(KernelInterface):
    """Batch normalization by the running statistics."""

    request = BatchNormCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalize *x* per channel by the running statistics; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            x: ``(n, c, spatial)`` in ``call.dtype``.
            running_mean: ``float32`` ``(c,)``.
            running_var: ``float32`` ``(c,)``.
            weight: ``float32`` ``(c,)`` scale.
            bias: ``float32`` ``(c,)`` shift.

        Returns:
            A new ``(n, c, spatial)`` output in ``call.dtype``.
        """


class BatchNormBwdInterface(KernelInterface):
    """The gradients of batch normalization's training forward."""

    request = BatchNormCall

    @abstractmethod
    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiate the training forward; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            grad_out: ``(n, c, spatial)`` in ``call.dtype``.
            x: The forward's input, ``(n, c, spatial)`` in ``call.dtype``.
            weight: ``float32`` ``(c,)`` scale.
            mean: ``float32`` ``(c,)`` batch mean the forward computed.
            rstd: ``float32`` ``(c,)`` reciprocal standard deviation the forward computed.

        Returns:
            New ``(grad_x, grad_weight, grad_bias)``: ``(n, c, spatial)`` in ``call.dtype``,
            then two ``float32`` ``(c,)``.
        """


class InstanceNormFwdInterface(KernelInterface):
    """Instance normalization by each instance's own statistics, with no running statistics."""

    request = BatchNormCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor | None = None,
        running_var: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalize each row of *x* by its own statistics; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            x: ``(n * c, spatial)`` in ``call.dtype``; row ``m`` is channel ``m % c``.
            running_mean: ``None``; the interface reads no running statistics.
            running_var: ``None``.
            weight: ``(c,)`` scale in ``call.dtype``, passed exactly when
                ``call.passes_affine``.
            bias: ``(c,)`` shift in ``call.dtype``, passed exactly when ``weight`` is.

        Returns:
            A new ``(n * c, spatial)`` output in ``call.dtype``.
        """


class InstanceNormTrainFwdInterface(KernelInterface):
    """Instance normalization by each instance's statistics, updating the running ones."""

    request = BatchNormCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalize *x* per instance and update the running statistics in place.

        Every tensor is contiguous on ``call.device``, and none aliases another.

        Args:
            x: ``(n, c, spatial)`` in ``call.dtype``.
            running_mean: ``(c,)``; moves to the batch mean of the instance means with
                ``call.momentum``.
            running_var: ``(c,)``; the same with the unbiased instance variances.
            weight: ``(c,)`` scale in ``call.dtype``, passed exactly when ``call.has_weight``.
            bias: ``(c,)`` shift in ``call.dtype``, passed exactly when ``call.has_bias``.

        Returns:
            A new ``(n, c, spatial)`` output in ``call.dtype``.
        """


class InstanceNormInferFwdInterface(KernelInterface):
    """Instance normalization by the running statistics."""

    request = BatchNormCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalize *x* per channel by the running statistics; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            x: ``(n, c, spatial)`` in ``call.dtype``.
            running_mean: ``(c,)``, read rounded to ``call.dtype``.
            running_var: ``(c,)``, read rounded to ``call.dtype``.
            weight: ``(c,)`` scale in ``call.dtype``, passed exactly when ``call.has_weight``.
            bias: ``(c,)`` shift in ``call.dtype``, passed exactly when ``call.has_bias``.

        Returns:
            A new ``(n, c, spatial)`` output in ``call.dtype``.
        """


@dataclasses.dataclass(frozen=True)
class GroupNormCall(CallSpec):
    """The facts that select a group normalization implementation and build it.

    The input is ``(n, c, *spatial)``; ``spatial`` is the product of the trailing axes.
    ``passes_affine`` says whether ``weight`` and ``bias`` are passed, as
    ``BatchNormCall.passes_affine`` does.
    """

    c: int = 0
    spatial: int = 0
    num_groups: int = 1
    eps: float = 1e-5
    dtype: torch.dtype = torch.float16
    passes_affine: bool = True


class LayerNormFwdInterface(KernelInterface):
    """Layer normalization over the trailing ``call.n`` elements."""

    request = LayerNormCall

    @abstractmethod
    def forward(
        self, x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None
    ) -> torch.Tensor:
        """Normalize each run of ``call.n`` trailing elements; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            x: Any shape whose trailing axes hold ``call.n`` elements, in ``call.dtype``.
            weight: ``call.n`` elements of scale in ``call.dtype``, or ``None`` for one.
            bias: ``call.n`` elements of shift in ``call.dtype``, or ``None`` for zero.

        Returns:
            A new output shaped like *x*, in ``call.dtype``.
        """


class RMSNormFwdInterface(KernelInterface):
    """Root mean square normalization over the trailing ``call.n`` elements."""

    request = RMSNormCall

    @abstractmethod
    def forward(self, x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        """Normalize each run of ``call.n`` trailing elements; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            x: Any shape whose trailing axes hold ``call.n`` elements, in ``call.dtype``.
            weight: ``call.n`` elements of scale in ``call.dtype``, or ``None`` for one.

        Returns:
            A new output shaped like *x*, in ``call.dtype``.
        """


class RMSNormGatedFwdInterface(KernelInterface):
    """RMS normalization of each group of ``call.group`` trailing elements, gated by
    ``act(z)``."""

    request = RMSNormGatedCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        z: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalize, scale, shift and gate each row; nothing is written in place.

        Every tensor is contiguous on ``call.device``. A tensor is passed exactly where
        ``call`` names its dtype.

        Args:
            x: ``(*leading, call.n)`` in ``call.dtype``.
            weight: ``(call.n,)`` scale in ``call.weight_dtype``.
            bias: ``(call.n,)`` shift in ``call.bias_dtype``, or ``None``.
            z: Shaped like *x*, in ``call.z_dtype``, or ``None`` for no gate.

        Returns:
            A new output shaped like *x*, in ``call.dtype``.
        """


class FusedAddLayerNormFwdInterface(KernelInterface):
    """A residual add, then layer normalization of the sum over the last ``call.n`` elements."""

    request = LayerNormCall

    @abstractmethod
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> list[torch.Tensor]:
        """Normalize ``x + residual``; nothing is written in place.

        Every tensor is contiguous on ``call.device``, in ``call.dtype``.

        Args:
            x: ``(*leading, call.n)``.
            residual: Shaped like *x*.
            weight: ``(call.n,)`` scale.
            bias: ``(call.n,)`` shift.

        Returns:
            New ``[y, residual_out]`` shaped like *x*, ``residual_out`` being ``x + residual``.
        """


class FusedAddRMSNormFwdInterface(KernelInterface):
    """A residual add, then RMS normalization of the sum over the last ``call.n`` elements."""

    request = RMSNormCall

    @abstractmethod
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> list[torch.Tensor]:
        """Normalize ``x + residual``; nothing is written in place.

        Every tensor is contiguous on ``call.device``, in ``call.dtype``.

        Args:
            x: ``(*leading, call.n)``.
            residual: Shaped like *x*.
            weight: ``(call.n,)`` scale.

        Returns:
            New ``[y, residual_out]`` shaped like *x*, ``residual_out`` being ``x + residual``.
        """


class FusedAddRMSNormPerTokenQuantFwdInterface(KernelInterface):
    """A residual add, RMS normalization of the sum and a per-row quantization of the result."""

    request = RMSNormQuantCall

    @abstractmethod
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> list[torch.Tensor]:
        """Normalize ``x + residual`` and quantize each row; nothing is written in place.

        Every tensor is contiguous on ``call.device``; ``x`` and ``residual`` are
        ``(call.rows, call.n)`` and ``weight`` is ``(call.n,)``, all in ``call.dtype``.

        Returns:
            New ``[q, scale, residual_out]``: ``q`` ``(call.rows, call.n)`` in
            ``call.quant_dtype``, ``scale`` ``(call.rows, 1)`` float32 and ``residual_out``,
            ``x + residual``, shaped and typed like *x*.
        """


class AdaLayerNormFwdInterface(KernelInterface):
    """Layer normalization over the last ``call.n`` elements, then a per-element scale and shift."""

    request = LayerNormCall

    @abstractmethod
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Normalize *x* and modulate it; nothing is written in place.

        Every tensor is contiguous on ``call.device``, in ``call.dtype``, shaped
        ``(*leading, call.n)``.

        Returns:
            A new ``scale * norm(x) + shift`` shaped like *x*.
        """


class AdaLayerNormZeroFwdInterface(KernelInterface):
    """Adaptive layer normalization over the last ``call.n`` elements, then a per-element gate."""

    request = LayerNormCall

    @abstractmethod
    def forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, gate: torch.Tensor
    ) -> torch.Tensor:
        """Normalize *x*, modulate and gate it; nothing is written in place.

        Every tensor is contiguous on ``call.device``, in ``call.dtype``, shaped
        ``(*leading, call.n)``.

        Returns:
            A new ``gate * (scale * norm(x) + shift)`` shaped like *x*.
        """


class GroupNormFwdInterface(KernelInterface):
    """Group normalization: each ``call.num_groups``-th of the channels with its spatial extent."""

    request = GroupNormCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalize each group of *x* by its own statistics; nothing is written in place.

        Every tensor is contiguous on ``call.device``, in ``call.dtype``.

        Args:
            x: ``(n, call.c, *spatial)`` whose trailing axes hold ``call.spatial`` elements.
            weight: ``(call.c,)`` scale, passed exactly when ``call.passes_affine``.
            bias: ``(call.c,)`` shift, passed exactly when ``weight`` is.

        Returns:
            A new output shaped like *x*.
        """


@dataclasses.dataclass(frozen=True)
class FusedQKNormRoPECall(CallSpec):
    """One in-place q/k normalization and rotation; the token and position counts are runtime
    data.

    ``zero_centered`` scales by ``1 + weight``. ``mrope_section`` is the op's multimodal
    rotary sections, ``None`` for one position a token, and ``mrope_interleaved`` its
    interleaved assignment.
    """

    heads: int = 0
    kv_heads: int = 0
    dim: int = 0
    rot: int = 0
    interleaved: bool = False
    eps: float = 1e-6
    dtype: torch.dtype | None = None
    cache_dtype: torch.dtype | None = None
    zero_centered: bool = False
    mrope_section: tuple[int, ...] | None = None
    mrope_interleaved: bool = False


class FusedQKNormRoPEFwdInterface(KernelInterface):
    """RMSNorm of every q and k head of a packed projection, then RoPE at each token's position."""

    request = FusedQKNormRoPECall

    @abstractmethod
    def forward(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Normalize and rotate the q and k heads of *qkv* in place; the v heads are not touched.

        Every tensor is on ``call.device`` and contiguous; *qkv* is written in place, no other
        input is.

        Args:
            qkv: ``(tokens, (call.heads + 2 * call.kv_heads) * call.dim)`` in ``call.dtype``.
            q_weight: ``(call.dim,)`` in ``call.dtype``.
            k_weight: ``(call.dim,)`` in ``call.dtype``.
            cos_sin_cache: ``(positions, call.rot)`` in ``call.cache_dtype``, cosines then sines.
            positions: ``int64`` ``(tokens,)``, each a row of *cos_sin_cache*.
        """
