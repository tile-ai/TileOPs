"""Batch Normalization Op.

Wraps BatchNormFwdTrainKernel, BatchNormFwdInferKernel, and BatchNormBwdKernel
in a standard TileOPs Op interface.

User-facing API mirrors torch.nn.functional.batch_norm:

    fwd_op = BatchNormFwdOp(N, C, *spatial, dtype=dtype, momentum=0.1, eps=1e-5)
    y, mean, rstd = fwd_op(x, weight, bias, running_mean, running_var,
                           training=True)

    bwd_op = BatchNormBwdOp(N, C, *spatial, dtype=dtype)
    grad_x, grad_weight, grad_bias = bwd_op(grad_out, x, weight, mean, rstd)

Input tensors accept any shape (N, C, *spatial); the op reshapes to (C, L)
internally.  L = N * prod(spatial) must be divisible by the kernel's block_l
(chosen automatically by the kernel's default_config).
"""

import math
from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.norm.batch_norm import (
    BatchNormBwdKernel,
    BatchNormFwdInferKernel,
    BatchNormFwdTrainKernel,
)

from ..op import Op

__all__ = ["BatchNormFwdOp", "BatchNormBwdOp"]


def _reshape_to_CL(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
    """Reshape (N, C, *spatial) → (C, L) and return the original shape."""
    orig_shape = x.shape
    C = orig_shape[1]
    L = x.numel() // C
    # Bring C to front: (N, C, *spatial) → (C, N, *spatial) → (C, L)
    x_cl = x.permute(1, 0, *range(2, x.ndim)).reshape(C, L).contiguous()
    return x_cl, orig_shape


def _restore_shape(y_cl: torch.Tensor, orig_shape: Tuple) -> torch.Tensor:
    """Reshape (C, L) back to orig_shape."""
    N = orig_shape[0]
    C = orig_shape[1]
    spatial = orig_shape[2:]
    y_reshaped = y_cl.reshape(C, N, *spatial)
    y = y_reshaped.permute(1, 0, *range(2, y_reshaped.ndim)).contiguous()
    return y


class BatchNormFwdOp(Op):
    """Batch Normalization forward operator (training and inference).

    Computes batch normalization over the channel dimension:

    .. math::

        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
            \\cdot \\gamma + \\beta

    where the mean and variance are computed per channel over ``(N, *spatial)``
    elements.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Input tensors accept any shape ``(N, C, *spatial)``; the op reshapes
        to ``(C, L)`` internally where ``L = N * prod(spatial)``.

    Args:
        N: Batch size.
        C: Number of channels.
        *spatial: Spatial dimensions (H, W, ...).
        dtype: Input/output data type.
        eps: Epsilon for numerical stability.
        momentum: Running-stat update momentum (used in training mode).
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        N: int,
        C: int,
        *spatial: int,
        dtype: torch.dtype = torch.float16,
        eps: float = 1e-5,
        momentum: float = 0.1,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.L = N * math.prod(spatial) if spatial else N
        self.dtype = dtype
        self.eps = eps
        self.momentum = momentum

        self.dispatch_kernel(kernel_map)

        self.train_kernel: BatchNormFwdTrainKernel = self.kernel_map["fwd_train_kernel"](
            C, self.L, dtype, eps, momentum, tune=tune)
        self.infer_kernel: BatchNormFwdInferKernel = self.kernel_map["fwd_infer_kernel"](
            C, self.L, dtype, eps, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "fwd_train_kernel": BatchNormFwdTrainKernel,
            "fwd_infer_kernel": BatchNormFwdInferKernel,
        }

    def _prepare(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Reshape x to (C, L)."""
        return _reshape_to_CL(x)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run batch normalization forward pass.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.
            weight: Affine scale (gamma) of shape ``(C,)``.
            bias: Affine shift (beta) of shape ``(C,)``.
            running_mean: Running mean of shape ``(C,)``, updated in-place
                during training.
            running_var: Running variance of shape ``(C,)``, updated in-place
                during training.
            training: If ``True``, compute batch statistics and update
                running stats; otherwise use running stats directly.

        Returns:
            Tuple of ``(y, mean, rstd)`` where *y* is the normalized output
            (same shape as *x*), *mean* is the per-channel batch mean, and
            *rstd* is the per-channel reciprocal standard deviation. In
            inference mode *mean* and *rstd* are ``None``.
        """
        x_cl, orig_shape = self._prepare(x)

        if training:
            y_cl, mean, rstd = self.train_kernel(
                x_cl, weight.float(), bias.float(),
                running_mean, running_var)
            y = _restore_shape(y_cl, orig_shape)
            return y, mean, rstd
        else:
            y_cl = self.infer_kernel(
                x_cl, weight.float(), bias.float(),
                running_mean, running_var)
            y = _restore_shape(y_cl, orig_shape)
            return y, None, None


class BatchNormBwdOp(Op):
    """Batch Normalization backward operator.

    Computes gradients with respect to input, scale, and shift for batch
    normalization:

    .. math::

        \\frac{\\partial \\mathcal{L}}{\\partial x},\\;
        \\frac{\\partial \\mathcal{L}}{\\partial \\gamma},\\;
        \\frac{\\partial \\mathcal{L}}{\\partial \\beta}

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Input tensors accept any shape ``(N, C, *spatial)``; the op reshapes
        to ``(C, L)`` internally where ``L = N * prod(spatial)``.

    Args:
        N: Batch size.
        C: Number of channels.
        *spatial: Spatial dimensions (H, W, ...).
        dtype: Data type of ``grad_out``, ``x``, and ``grad_x``.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        N: int,
        C: int,
        *spatial: int,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.L = N * math.prod(spatial) if spatial else N
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)

        self.bwd_kernel: BatchNormBwdKernel = self.kernel_map["bwd_kernel"](
            C, self.L, dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"bwd_kernel": BatchNormBwdKernel}

    def _prepare(self, t: torch.Tensor) -> torch.Tensor:
        t_cl, _ = _reshape_to_CL(t)
        return t_cl

    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run batch normalization backward pass.

        Args:
            grad_out: Upstream gradient of shape ``(N, C, *spatial)``.
            x: Original input tensor of shape ``(N, C, *spatial)``.
            weight: Affine scale (gamma) of shape ``(C,)``.
            mean: Per-channel batch mean from the forward pass, shape ``(C,)``.
            rstd: Per-channel reciprocal std from the forward pass,
                shape ``(C,)``.

        Returns:
            Tuple of ``(grad_x, grad_weight, grad_bias)`` where *grad_x*
            has the same shape as *x*, *grad_weight* has shape ``(C,)``,
            and *grad_bias* has shape ``(C,)``.
        """
        orig_shape = grad_out.shape
        go_cl = self._prepare(grad_out)
        x_cl = self._prepare(x)

        grad_x_cl, grad_weight, grad_bias = self.bwd_kernel(
            go_cl, x_cl, weight.float(), mean, rstd)

        grad_x = _restore_shape(grad_x_cl, orig_shape)
        return grad_x, grad_weight, grad_bias
