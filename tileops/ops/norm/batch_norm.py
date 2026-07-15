"""Batch Normalization Op.

Wraps BatchNormFwdTrainKernel, BatchNormFwdInferKernel, and BatchNormBwdKernel
in a standard TileOPs Op interface.

User-facing API mirrors :func:`torch.nn.functional.batch_norm`:

    fwd_op = BatchNormFwdOp(training=False, momentum=0.1, eps=1e-5)
    y = fwd_op(x, running_mean, running_var, weight, bias)

    bwd_op = BatchNormBwdOp()
    grad_x, grad_weight, grad_bias = bwd_op(grad_out, x, weight, mean, rstd)

Forward returns the normalized output only (manifest contract); ``mean`` and
``rstd`` from the training path stay internal. Callers needing them for the
backward pass can recompute on the original input.

Input tensors accept any shape ``(N, C, *spatial)``; the op reshapes to
``(C, L)`` internally.  ``L = N * prod(spatial)`` must be divisible by the
kernel's block_l (chosen automatically by the kernel's default_config).
"""

import functools
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
        training: Default ``training`` flag for ``forward()``; per the
            manifest the default is ``False``.
        momentum: Running-stat update momentum (used in training mode).
        eps: Epsilon for numerical stability.
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.N: Optional[int] = None
        self.C: Optional[int] = None
        self.spatial: Optional[Tuple[int, ...]] = None
        self.L: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None
        self.training = training
        self.eps = eps
        self.momentum = momentum
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel: Optional[Kernel] = None
        self.train_kernel: Optional[BatchNormFwdTrainKernel] = None
        self.infer_kernel: Optional[BatchNormFwdInferKernel] = None
        self._last_roofline_spec: Optional[tuple[int, int, torch.dtype]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "fwd_train_kernel": BatchNormFwdTrainKernel,
            "fwd_infer_kernel": BatchNormFwdInferKernel,
        }

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "BatchNormFwdOp.eval_roofline() requires a prior forward() call"
            )
        C, L, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return (
            10 * C * L,
            2 * C * L * elem_bytes + 4 * C * 4,
        )

    def _prepare(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """Reshape x to (C, L)."""
        return _reshape_to_CL(x)

    def _resolve_spec(self, x: torch.Tensor) -> Tuple[int, int, Tuple[int, ...], int, torch.dtype]:
        """Validate input metadata and return (N, C, spatial, L, dtype)."""
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if x.ndim < 2:
            raise ValueError("x must have shape (N, C, *spatial)")
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                "x.dtype must be float32, float16, or bfloat16, "
                f"got {x.dtype}"
            )
        N, C, *spatial_list = x.shape
        spatial = tuple(spatial_list)
        L = x.numel() // C
        return N, C, spatial, L, x.dtype

    @staticmethod
    def _validate_channel_tensor(
        name: str,
        tensor: torch.Tensor,
        C: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != device:
            raise ValueError(f"Expected {name} on {device}, got {tensor.device}")
        if tensor.dtype != dtype:
            raise ValueError(f"Expected {name}.dtype {dtype}, got {tensor.dtype}")
        if tensor.ndim != 1 or tensor.shape[0] != C:
            raise ValueError(f"Expected {name} shape ({C},), got {tuple(tensor.shape)}")

    def _bind_spec(
        self,
        N: int,
        C: int,
        spatial: Tuple[int, ...],
        L: int,
        dtype: torch.dtype,
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.L = L
        self.dtype = dtype
        self._last_roofline_spec = (C, L, dtype)

    def _get_kernel(
        self,
        C: int,
        L: int,
        dtype: torch.dtype,
        device_index: Optional[int],
    ) -> Kernel:
        mode = "train" if self.training else "infer"
        key = (mode, C, L, dtype, device_index, self.eps, self.momentum, self.tune)
        if key not in self._kernel_cache:
            if self.training:
                self._kernel_cache[key] = self.kernel_map["fwd_train_kernel"](
                    C, L, dtype, self.eps, self.momentum, tune=self.tune,
                )
            else:
                self._kernel_cache[key] = self.kernel_map["fwd_infer_kernel"](
                    C, L, dtype, self.eps, tune=self.tune,
                )
        kernel = self._kernel_cache[key]
        self.kernel = kernel
        if self.training:
            self.train_kernel = kernel
        else:
            self.infer_kernel = kernel
        return kernel

    def _forward_impl(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        N, C, spatial, L, dtype = self._resolve_spec(x)
        self._validate_channel_tensor("running_mean", running_mean, C, x.device, torch.float32)
        self._validate_channel_tensor("running_var", running_var, C, x.device, torch.float32)
        self._validate_channel_tensor("weight", weight, C, x.device, torch.float32)
        self._validate_channel_tensor("bias", bias, C, x.device, torch.float32)
        self._bind_spec(N, C, spatial, L, dtype)
        x_cl, orig_shape = self._prepare(x)
        kernel = self._get_kernel(C, L, dtype, x.device.index)

        if self.training:
            y_cl, _mean, _rstd = kernel(
                x_cl, weight.float(), bias.float(),
                running_mean, running_var)
        else:
            y_cl = kernel(
                x_cl, weight.float(), bias.float(),
                running_mean, running_var)
        return _restore_shape(y_cl, orig_shape)

    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """Run batch normalization forward pass.

        The ``training`` mode is bound at ctor time. Construct a separate
        op instance to switch between training and inference.

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

        Returns:
            Normalized output tensor with the same shape as ``x``.
        """
        return self._forward_impl(x, running_mean, running_var, weight, bias)


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
        kernel_map: Optional kernel override dictionary.
        tune: If ``True``, autotune tile configurations.
    """

    def __init__(
        self,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        self.N: Optional[int] = None
        self.C: Optional[int] = None
        self.spatial: Optional[Tuple[int, ...]] = None
        self.L: Optional[int] = None
        self.dtype: Optional[torch.dtype] = None
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self._kernel_cache: Dict[tuple, Kernel] = {}
        self.kernel: Optional[Kernel] = None
        self.bwd_kernel: Optional[BatchNormBwdKernel] = None
        self._last_roofline_spec: Optional[tuple[int, int, torch.dtype]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"bwd_kernel": BatchNormBwdKernel}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError(
                "BatchNormBwdOp.eval_roofline() requires a prior forward() call"
            )
        C, L, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return (
            8 * C * L,
            3 * C * L * elem_bytes + 3 * C * 4,
        )

    def _prepare(self, t: torch.Tensor) -> torch.Tensor:
        t_cl, _ = _reshape_to_CL(t)
        return t_cl

    def _resolve_spec(
        self, grad_out: torch.Tensor, x: torch.Tensor
    ) -> Tuple[int, int, Tuple[int, ...], int, torch.dtype]:
        if not grad_out.is_cuda:
            raise ValueError("grad_out must be a CUDA tensor")
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if grad_out.device != x.device:
            raise ValueError(
                f"Expected grad_out and x on the same device, got "
                f"{grad_out.device} and {x.device}"
            )
        if grad_out.shape != x.shape:
            raise ValueError(f"Expected x shape {grad_out.shape}, got {x.shape}")
        if grad_out.dtype != x.dtype:
            raise ValueError(
                f"Expected x.dtype {grad_out.dtype}, got {x.dtype}"
            )
        if grad_out.ndim < 2:
            raise ValueError("grad_out must have shape (N, C, *spatial)")
        if grad_out.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                "grad_out.dtype must be float32, float16, or bfloat16, "
                f"got {grad_out.dtype}"
            )
        N, C, *spatial_list = grad_out.shape
        spatial = tuple(spatial_list)
        L = grad_out.numel() // C
        return N, C, spatial, L, grad_out.dtype

    def _bind_spec(
        self,
        N: int,
        C: int,
        spatial: Tuple[int, ...],
        L: int,
        dtype: torch.dtype,
    ) -> None:
        self.N = N
        self.C = C
        self.spatial = spatial
        self.L = L
        self.dtype = dtype
        self._last_roofline_spec = (C, L, dtype)

    def _get_kernel(
        self,
        C: int,
        L: int,
        dtype: torch.dtype,
        device_index: Optional[int],
    ) -> Kernel:
        key = (C, L, dtype, device_index, self.tune)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = self.kernel_map["bwd_kernel"](
                C, L, dtype, tune=self.tune,
            )
        kernel = self._kernel_cache[key]
        self.kernel = kernel
        self.bwd_kernel = kernel
        return kernel

    @staticmethod
    def _validate_channel_tensor(
        name: str,
        tensor: torch.Tensor,
        C: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != device:
            raise ValueError(f"Expected {name} on {device}, got {tensor.device}")
        if tensor.dtype != dtype:
            raise ValueError(f"Expected {name}.dtype {dtype}, got {tensor.dtype}")
        if tensor.ndim != 1 or tensor.shape[0] != C:
            raise ValueError(f"Expected {name} shape ({C},), got {tuple(tensor.shape)}")

    def _forward_impl(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, C, spatial, L, dtype = self._resolve_spec(grad_out, x)
        self._validate_channel_tensor("weight", weight, C, grad_out.device, torch.float32)
        self._validate_channel_tensor("mean", mean, C, grad_out.device, torch.float32)
        self._validate_channel_tensor("rstd", rstd, C, grad_out.device, torch.float32)
        self._bind_spec(N, C, spatial, L, dtype)
        orig_shape = grad_out.shape
        go_cl = self._prepare(grad_out)
        x_cl = self._prepare(x)
        kernel = self._get_kernel(C, L, dtype, grad_out.device.index)

        grad_x_cl, grad_weight, grad_bias = kernel(
            go_cl, x_cl, weight.float(), mean, rstd)

        grad_x = _restore_shape(grad_x_cl, orig_shape)
        return grad_x, grad_weight, grad_bias

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
        return self._forward_impl(grad_out, x, weight, mean, rstd)


# ---------------------------------------------------------------------------
# torch.compile registration -- BatchNormFwdOp / BatchNormBwdOp
# ---------------------------------------------------------------------------

_OP_REGISTRY: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _register_instance(op_instance: object) -> int:
    """Register an op instance and return its integer key."""
    key = id(op_instance)
    _OP_REGISTRY[key] = op_instance
    return key


@torch.library.custom_op(
    "top::norm_batch_norm_fwd",
    mutates_args=("running_mean", "running_var"),
)
def _batch_norm_fwd_wrapped(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    instance_key: int,
) -> torch.Tensor:
    instance = _OP_REGISTRY[instance_key]
    return instance._eager_forward(
        x, running_mean, running_var, weight, bias)


@_batch_norm_fwd_wrapped.register_fake
def _batch_norm_fwd_fake(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    instance_key: int,
) -> torch.Tensor:
    return torch.empty_like(x)


BatchNormFwdOp._wrapped = _batch_norm_fwd_wrapped


def _batchnorm_fwd_eager_forward(
    self,
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Direct kernel call (no torch.compile wrapping)."""
    return self._forward_impl(x, running_mean, running_var, weight, bias)


BatchNormFwdOp._eager_forward = _batchnorm_fwd_eager_forward

_orig_fwd_init = BatchNormFwdOp.__init__


@functools.wraps(_orig_fwd_init)
def _patched_fwd_init(self, *args, **kwargs):
    _orig_fwd_init(self, *args, **kwargs)
    self._instance_key = _register_instance(self)


BatchNormFwdOp.__init__ = _patched_fwd_init


def _patched_fwd_forward(
    self,
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _batch_norm_fwd_wrapped(
        x, running_mean, running_var, weight, bias, self._instance_key)


BatchNormFwdOp.forward = _patched_fwd_forward


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
    return self._forward_impl(grad_out, x, weight, mean, rstd)


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
