"""The normalization kernel interfaces and the call specs they take."""

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from typing import Optional

import torch

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import KernelInterface

__all__ = [
    "BatchNormBwdInterface",
    "BatchNormCall",
    "BatchNormFwdInferInterface",
    "BatchNormFwdTrainInterface",
    "InstanceNormFwdInferInterface",
    "InstanceNormFwdInterface",
    "InstanceNormFwdTrainInterface",
    "LayerNormCall",
    "LayerNormFwdInterface",
]


@dataclasses.dataclass(frozen=True)
class BatchNormCall(CallSpec):
    """The facts that select a batch or instance normalization implementation and build it.

    The input is ``(n, c, *spatial)``; ``spatial`` is the product of the trailing axes.
    ``eps`` and ``momentum`` are the op's construction parameters, which the programs
    compile in. ``input_dtype_params`` is set where the affine is in the input dtype and
    the running statistics are read rounded to it, as ``instance_norm`` reads them;
    ``has_weight`` and ``has_bias`` say which affine tensors are passed.
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

    @property
    def passes_affine(self) -> bool:
        """Whether ``weight`` or ``bias`` is passed."""
        return self.has_weight or self.has_bias


@dataclasses.dataclass(frozen=True)
class LayerNormCall(CallSpec):
    """The facts that select a layer normalization implementation and build it.

    ``n`` is the product of ``normalized_shape`` and ``eps`` the op's epsilon.
    """

    n: int = 0
    eps: float = 1e-5
    dtype: torch.dtype = torch.float16


class BatchNormFwdTrainInterface(KernelInterface):
    """Batch normalization by the batch statistics, updating the running ones."""

    request = BatchNormCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize *x* per channel by its batch statistics.

        Every tensor is contiguous on ``call.device``, and none aliases another.

        Args:
            x: ``(n, c, spatial)`` in ``call.dtype``.
            running_mean: ``float32`` ``(c,)``; updated in place with ``call.momentum``.
            running_var: ``float32`` ``(c,)``; updated in place with the unbiased variance.
            weight: ``float32`` ``(c,)`` scale.
            bias: ``float32`` ``(c,)`` shift.

        Returns:
            ``(y, mean, rstd)``: a new ``(n, c, spatial)`` output in ``call.dtype``, and the
            new ``float32`` ``(c,)`` batch mean and reciprocal standard deviation.
        """


class BatchNormFwdInferInterface(KernelInterface):
    """Batch normalization by the running statistics."""

    request = BatchNormCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
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
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
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


class InstanceNormFwdTrainInterface(KernelInterface):
    """Instance normalization by each instance's statistics, updating the running ones."""

    request = BatchNormCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
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


class InstanceNormFwdInferInterface(KernelInterface):
    """Instance normalization by the running statistics."""

    request = BatchNormCall

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
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


class LayerNormFwdInterface(KernelInterface):
    """Layer normalization over the trailing ``call.n`` elements."""

    request = LayerNormCall

    @abstractmethod
    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Normalize each run of ``call.n`` trailing elements; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            x: Any shape whose trailing axes hold ``call.n`` elements, in ``call.dtype``.
            weight: ``call.n`` elements of scale in ``call.dtype``.
            bias: ``call.n`` elements of shift in ``call.dtype``.

        Returns:
            A new output shaped like *x*, in ``call.dtype``.
        """
