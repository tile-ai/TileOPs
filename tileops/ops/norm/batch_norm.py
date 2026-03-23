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
import weakref
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
    """Batch normalization forward operator (training + inference).

    Args:
        N: Batch size.
        C: Number of channels.
        *spatial: Spatial dimensions (H, W, ...).
        dtype: Input/output dtype.
        eps: Numerical stability constant.
        momentum: Running-stat update momentum (used in training).
        kernel_map: Optional kernel override dict.
        tune: If True, autotune tile configs.
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
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run forward pass.

        Returns:
            y: Normalized output (same shape as x).
            mean: Per-channel batch mean (None in inference mode).
            rstd: Per-channel reciprocal std (None in inference mode).
        """
        self._validate_inputs(x)
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
    """Batch normalization backward operator.

    Args:
        N: Batch size.
        C: Number of channels.
        *spatial: Spatial dimensions (H, W, ...).
        dtype: grad_out/x/grad_x dtype.
        kernel_map: Optional kernel override dict.
        tune: If True, autotune tile config.
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
        """Run backward pass.

        Returns:
            grad_x: Gradient w.r.t. input (same shape as x).
            grad_weight: Gradient w.r.t. affine scale gamma (shape: C).
            grad_bias: Gradient w.r.t. affine shift beta (shape: C).
        """
        orig_shape = grad_out.shape
        go_cl = self._prepare(grad_out)
        x_cl = self._prepare(x)

        grad_x_cl, grad_weight, grad_bias = self.bwd_kernel(
            go_cl, x_cl, weight.float(), mean, rstd)

        grad_x = _restore_shape(grad_x_cl, orig_shape)
        return grad_x, grad_weight, grad_bias


# ---------------------------------------------------------------------------
# torch.compile registration — BatchNormFwdOp
# ---------------------------------------------------------------------------

_OP_REGISTRY: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _register_instance(op_instance: object) -> int:
    """Register an op instance and return its integer key."""
    key = id(op_instance)
    _OP_REGISTRY[key] = op_instance
    return key


# -- Fwd custom_op ----------------------------------------------------------

@torch.library.custom_op("top::batch_norm_fwd", mutates_args=("running_mean", "running_var"))
def _batch_norm_fwd_wrapped(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool,
    instance_key: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    instance = _OP_REGISTRY[instance_key]
    y, mean, rstd = instance._eager_forward(
        x, weight, bias, running_mean, running_var, training=training)
    # custom_op must return concrete tensors; replace None with empty
    if mean is None:
        mean = torch.empty(0, device=x.device, dtype=torch.float32)
    if rstd is None:
        rstd = torch.empty(0, device=x.device, dtype=torch.float32)
    return y, mean, rstd


@_batch_norm_fwd_wrapped.register_fake
def _batch_norm_fwd_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool,
    instance_key: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    C = weight.shape[0]
    return (
        torch.empty_like(x),
        torch.empty(C, device=x.device, dtype=torch.float32),
        torch.empty(C, device=x.device, dtype=torch.float32),
    )


# Patch BatchNormFwdOp to support torch.compile
BatchNormFwdOp._wrapped = _batch_norm_fwd_wrapped


def _batchnorm_fwd_eager_forward(
    self,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Direct kernel call (no torch.compile wrapping)."""
    self._validate_inputs(x)
    x_cl, orig_shape = self._prepare(x)
    if training:
        y_cl, mean, rstd = self.train_kernel(
            x_cl, weight.float(), bias.float(), running_mean, running_var)
        return _restore_shape(y_cl, orig_shape), mean, rstd
    else:
        y_cl = self.infer_kernel(
            x_cl, weight.float(), bias.float(), running_mean, running_var)
        return _restore_shape(y_cl, orig_shape), None, None


BatchNormFwdOp._eager_forward = _batchnorm_fwd_eager_forward

# Override forward to go through custom_op when instance is registered
_orig_fwd_init = BatchNormFwdOp.__init__


def _patched_fwd_init(self, *args, **kwargs):
    _orig_fwd_init(self, *args, **kwargs)
    self._instance_key = _register_instance(self)


BatchNormFwdOp.__init__ = _patched_fwd_init


def _patched_fwd_forward(self, x, weight, bias, running_mean, running_var, training=True):
    y, mean, rstd = _batch_norm_fwd_wrapped(
        x, weight, bias, running_mean, running_var, training, self._instance_key)
    if mean.numel() == 0:
        mean = None
    if rstd.numel() == 0:
        rstd = None
    return y, mean, rstd


BatchNormFwdOp.forward = _patched_fwd_forward


# -- Bwd custom_op ----------------------------------------------------------

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


def _patched_bwd_init(self, *args, **kwargs):
    _orig_bwd_init(self, *args, **kwargs)
    self._instance_key = _register_instance(self)


BatchNormBwdOp.__init__ = _patched_bwd_init


def _patched_bwd_forward(self, grad_out, x, weight, mean, rstd):
    return _batch_norm_bwd_wrapped(
        grad_out, x, weight, mean, rstd, self._instance_key)


BatchNormBwdOp.forward = _patched_bwd_forward
