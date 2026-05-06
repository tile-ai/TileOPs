"""Batch Normalization Op.

Wraps BatchNormFwdTrainKernel, BatchNormFwdInferKernel, and BatchNormBwdKernel
in a standard TileOPs Op interface.

User-facing API mirrors :func:`torch.nn.functional.batch_norm`:

    fwd_op = BatchNormFwdOp(N, C, *spatial, dtype=dtype, momentum=0.1, eps=1e-5)
    y = fwd_op(x, running_mean, running_var, weight, bias, training=False)

    bwd_op = BatchNormBwdOp(N, C, *spatial, dtype=dtype)
    grad_x, grad_weight, grad_bias = bwd_op(grad_out, x, weight, mean, rstd)

Forward returns the normalized output only (manifest contract); ``mean`` and
``rstd`` from the training path stay internal. Callers needing them for the
backward pass can recompute on the original input.

Input tensors accept any shape ``(N, C, *spatial)``; the op reshapes to
``(C, L)`` internally.  ``L = N * prod(spatial)`` must be divisible by the
kernel's block_l (chosen automatically by the kernel's default_config).
"""

import functools
import math
import weakref
from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm.batch_norm import (
    BatchNormBwdKernel,
    BatchNormFwdInferKernel,
    BatchNormFwdTrainKernel,
)

from ..op_base import Op

__all__ = ["BatchNormBwdOp", "BatchNormFwdOp"]


def _reshape_to_CL(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
    """Reshape (N, C, *spatial) to (C, L) and return the original shape."""
    orig_shape = x.shape
    C = orig_shape[1]
    L = x.numel() // C
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

    Mirrors :func:`torch.nn.functional.batch_norm`: ``forward`` accepts
    ``(input, running_mean, running_var, weight, bias)`` in PyTorch's
    positional order and returns only the normalized output. Internal
    mean/rstd computed in training mode stay private; callers needing them
    for the backward pass recompute on the original input.

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
        training: Default ``training`` flag for ``forward()``; per the
            manifest the default is ``False``.
        momentum: Running-stat update momentum (used in training mode).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        N: int,
        C: int,
        *spatial: int,
        dtype: torch.dtype = torch.float16,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.L = N * math.prod(spatial) if spatial else N
        self.dtype = dtype
        self.training = training
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

    def eval_roofline(self) -> tuple[int, int]:
        elem_bytes = self.dtype.itemsize
        return (
            10 * self.C * self.L,
            2 * self.C * self.L * elem_bytes + 4 * self.C * 4,
        )

    def _prepare(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Reshape x to (C, L)."""
        return _reshape_to_CL(x)

    def _validate_inputs(self, x: torch.Tensor) -> None:
        """Validate device, dtype, and shape of the input tensor."""
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.dtype != self.dtype:
            raise ValueError(
                f"Expected x.dtype {self.dtype}, got {x.dtype}"
            )
        expected_shape = (self.N, self.C, *self.spatial)
        if x.shape != expected_shape:
            raise ValueError(
                f"Expected input shape {expected_shape}, got {x.shape}"
            )

    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        training: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run batch normalization forward pass.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)`` on CUDA.
            running_mean: Running mean of shape ``(C,)`` on the same CUDA
                device as ``x``, with dtype ``torch.float32``. Updated
                in-place during training.
            running_var: Running variance of shape ``(C,)`` on the same
                CUDA device as ``x``, with dtype ``torch.float32``. Updated
                in-place during training.
            weight: Affine scale (gamma) of shape ``(C,)`` on the same CUDA
                device as ``x``.
            bias: Affine shift (beta) of shape ``(C,)`` on the same CUDA
                device as ``x``.
            training: Override the ctor-bound ``training`` flag for this
                call. ``None`` uses the ctor-bound default.

        Returns:
            Normalized output tensor with the same shape as ``x``.
        """
        train_flag = self.training if training is None else training
        self._validate_inputs(x)
        x_cl, orig_shape = self._prepare(x)

        if train_flag:
            y_cl, _mean, _rstd = self.train_kernel(
                x_cl, weight.float(), bias.float(),
                running_mean, running_var)
        else:
            y_cl = self.infer_kernel(
                x_cl, weight.float(), bias.float(),
                running_mean, running_var)
        return _restore_shape(y_cl, orig_shape)


class BatchNormBwdOp(Op):
    """Batch Normalization backward operator.

    Computes gradients with respect to input, scale, and shift for batch
    normalization.

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

    def eval_roofline(self) -> tuple[int, int]:
        elem_bytes = self.dtype.itemsize
        return (
            8 * self.C * self.L,
            3 * self.C * self.L * elem_bytes + 3 * self.C * 4,
        )

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

        All inputs must reside on the same CUDA device.

        Args:
            grad_out: Upstream gradient of shape ``(N, C, *spatial)``.
            x: Original input tensor of shape ``(N, C, *spatial)``.
            weight: Affine scale (gamma) of shape ``(C,)`` on the same CUDA
                device as ``x``. Internally cast to ``torch.float32`` for the
                backward kernel.
            mean: Per-channel batch mean from the forward pass, shape
                ``(C,)``. Expected as ``torch.float32``.
            rstd: Per-channel reciprocal std from the forward pass,
                shape ``(C,)``. Expected as ``torch.float32``.

        Returns:
            Tuple of ``(grad_x, grad_weight, grad_bias)`` where ``grad_x``
            has the same shape as ``x``, ``grad_weight`` has shape ``(C,)``,
            and ``grad_bias`` has shape ``(C,)``.
        """
        orig_shape = grad_out.shape
        go_cl = self._prepare(grad_out)
        x_cl = self._prepare(x)

        grad_x_cl, grad_weight, grad_bias = self.bwd_kernel(
            go_cl, x_cl, weight.float(), mean, rstd)

        grad_x = _restore_shape(grad_x_cl, orig_shape)
        return grad_x, grad_weight, grad_bias


# ---------------------------------------------------------------------------
# torch.compile registration -- BatchNormBwdOp
# ---------------------------------------------------------------------------

_OP_REGISTRY: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _register_instance(op_instance: object) -> int:
    """Register an op instance and return its integer key."""
    key = id(op_instance)
    _OP_REGISTRY[key] = op_instance
    return key


@torch.library.custom_op("top::batch_norm_bwd", mutates_args=())
def _batch_norm_bwd_wrapped(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    instance_key: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    instance = _OP_REGISTRY[instance_key]
    return instance._eager_forward(grad_out, x, weight, mean, rstd)


@_batch_norm_bwd_wrapped.register_fake
def _batch_norm_bwd_fake(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    instance_key: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    C = weight.shape[0]
    return (
        torch.empty_like(grad_out),
        torch.empty(C, device=grad_out.device, dtype=torch.float32),
        torch.empty(C, device=grad_out.device, dtype=torch.float32),
    )


BatchNormBwdOp._wrapped = _batch_norm_bwd_wrapped


def _batchnorm_bwd_eager_forward(
    self,
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct kernel call (no torch.compile wrapping)."""
    orig_shape = grad_out.shape
    go_cl = self._prepare(grad_out)
    x_cl = self._prepare(x)
    grad_x_cl, grad_weight, grad_bias = self.bwd_kernel(
        go_cl, x_cl, weight.float(), mean, rstd)
    grad_x = _restore_shape(grad_x_cl, orig_shape)
    return grad_x, grad_weight, grad_bias


BatchNormBwdOp._eager_forward = _batchnorm_bwd_eager_forward

_orig_bwd_init = BatchNormBwdOp.__init__


@functools.wraps(_orig_bwd_init)
def _patched_bwd_init(self, *args, **kwargs):
    _orig_bwd_init(self, *args, **kwargs)
    self._instance_key = _register_instance(self)


BatchNormBwdOp.__init__ = _patched_bwd_init


def _patched_bwd_forward(self, grad_out, x, weight, mean, rstd):
    return _batch_norm_bwd_wrapped(
        grad_out, x, weight, mean, rstd, self._instance_key)


BatchNormBwdOp.forward = _patched_bwd_forward
