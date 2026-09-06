"""Batch Normalization Op.

Wraps BatchNormFwdTrainKernel, BatchNormFwdInferKernel, and BatchNormBwdKernel
in a standard TileOPs Op interface.

User-facing API mirrors `torch.nn.functional.batch_norm`:

    fwd_op = BatchNormFwdOp(training=False, momentum=0.1, eps=1e-5)
    y = fwd_op(x, running_mean, running_var, weight, bias)

    bwd_op = BatchNormBwdOp()
    grad_x, grad_weight, grad_bias = bwd_op(grad_out, x, weight, mean, rstd)

Forward returns the normalized output only (manifest contract); ``mean`` and
``rstd`` from the training path stay internal. Callers needing them for the
backward pass can recompute on the original input.

Input tensors accept any shape ``(N, C, *spatial)``; the kernel moves them into its
$[C \\times L]$ layout. ``L = N * prod(spatial)`` must be divisible by the kernel's block_l
(chosen automatically by the kernel's default_config).
"""

import math
from typing import ClassVar, Dict, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm.batch_norm import (
    BatchNormBwdKernel,
    BatchNormFwdInferKernel,
    BatchNormFwdTrainKernel,
)

from ..compile_boundary import get_instance
from ..op_base import Op

__all__ = ["BatchNormBwdOp", "BatchNormFwdOp"]


class BatchNormFwdOp(Op):
    """Batch Normalization forward operator (training and inference).

    Computes batch normalization over the channel dimension:

    $$
    y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
    \\cdot \\gamma + \\beta
    $$

    where the mean and variance are computed per channel over ``(N, *spatial)``
    elements.

    Mirrors `torch.nn.functional.batch_norm`: ``forward`` accepts
    ``(input, running_mean, running_var, weight, bias)`` in PyTorch's
    positional order and returns only the normalized output. Internal
    mean/rstd computed in training mode stay private; callers needing them
    for the backward pass recompute on the original input.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    """

    compile_op_names: ClassVar[Tuple[str, ...]] = ("tileops::norm_batch_norm_fwd",)

    def __init__(
        self,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            training: Whether the batch statistics come from this call's input, which is also
                what decides whether the running statistics are written (manifest
                ``params.training``).
            momentum: Running-stat update momentum (used in training mode).
            eps: Epsilon for numerical stability.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dictionary.
            tune: If ``True``, autotune tile configurations.
        """
        self.dtype: Optional[torch.dtype] = None
        self.training = training
        self.eps = eps
        self.momentum = momentum
        self.target = target
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel: Optional[Kernel] = None
        self._last_roofline_spec: Optional[tuple[int, int, torch.dtype]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "fwd_train_kernel": BatchNormFwdTrainKernel,
            "fwd_infer_kernel": BatchNormFwdInferKernel,
        }

    def _infer_output_shapes(
        self,
        x_shape: Tuple[int, ...],
        running_mean_shape: Tuple[int, ...],
        running_var_shape: Tuple[int, ...],
        weight_shape: Tuple[int, ...],
        bias_shape: Tuple[int, ...],
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules``: ``output.shape == x.shape``."""
        return {"output": tuple(x_shape)}

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError("BatchNormFwdOp.eval_roofline() requires a prior forward() call")
        C, L, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return (
            10 * C * L,
            2 * C * L * elem_bytes + 4 * C * 4,
        )

    def _resolve_spec(self, x: torch.Tensor) -> Tuple[int, int, torch.dtype]:
        """Validate input metadata and return (C, L, dtype)."""
        if x.ndim < 2:
            raise ValueError("x must have shape (N, C, *spatial)")
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"x.dtype must be float32, float16, or bfloat16, got {x.dtype}")
        C = x.shape[1]
        L = x.numel() // C
        if self.training and L == 1:
            # Bessel's correction divides by L - 1. torch refuses the same call.
            raise ValueError(
                f"Expected more than 1 value per channel when training, got input size {x.shape}"
            )
        return C, L, x.dtype

    @staticmethod
    def _validate_channel_tensor(
        name: str,
        tensor: torch.Tensor,
        C: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if tensor.device != device:
            raise ValueError(f"Expected {name} on {device}, got {tensor.device}")
        if tensor.dtype != dtype:
            raise ValueError(f"Expected {name}.dtype {dtype}, got {tensor.dtype}")
        if tensor.ndim != 1 or tensor.shape[0] != C:
            raise ValueError(f"Expected {name} shape ({C},), got {tuple(tensor.shape)}")

    def _bind_spec(self, C: int, L: int, dtype: torch.dtype) -> None:
        """Bind what ``eval_roofline`` reads off the call that just ran."""
        self.dtype = dtype
        self._last_roofline_spec = (C, L, dtype)

    def _eager_forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        C, L, dtype = self._resolve_spec(x)
        self._validate_channel_tensor("running_mean", running_mean, C, x.device, torch.float32)
        self._validate_channel_tensor("running_var", running_var, C, x.device, torch.float32)
        self._validate_channel_tensor("weight", weight, C, x.device, torch.float32)
        self._validate_channel_tensor("bias", bias, C, x.device, torch.float32)
        self._bind_spec(C, L, dtype)

        # Handed over as the manifest declares it; the layout a kernel wants is its own business.
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        # The running statistics are written, so normalizing them is not enough: whoever
        # serves this op writes the tensor it was handed, and a copy would swallow that
        # write. ``contiguous()`` returns the same object when it has nothing to do, so
        # what came back tells us whether a write-back is owed.
        stats = (running_mean, running_var)
        running_mean, running_var = (stat.contiguous() for stat in stats)

        spatial = math.prod(x.shape[2:])

        # ``training`` decides which implementation serves the call, so it belongs in the
        # key; both are fetched under one name, which is what a target is asked to serve.
        slot = "fwd_train_kernel" if self.training else "fwd_infer_kernel"
        kernel = self.get_or_build_kernel(
            "batch_norm_fwd",
            (x, running_mean, running_var, weight, bias),
            # Both paths index the caller's layout, so the spatial extent
            # changes the kernel that is built.
            key=(C, L, dtype, self.training, spatial),  # this instance's in-tree cache key
            build=lambda: (
                self.kernel_map[slot](
                    C, L, dtype, self.eps, self.momentum, tune=self.tune, S=spatial
                )
                if self.training
                else self.kernel_map[slot](C, L, dtype, self.eps, tune=self.tune, S=spatial)
            ),
        )
        self.kernel = kernel

        # The training kernel also returns the batch statistics, which the manifest keeps
        # out of this op's outputs.
        if not self.training:
            return kernel(x, running_mean, running_var, weight, bias)

        y, _mean, _rstd = kernel(x, running_mean, running_var, weight, bias)
        for original, handed_over in zip(stats, (running_mean, running_var), strict=True):
            if handed_over is not original:
                original.copy_(handed_over)
        return y

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
            running_mean: Running mean of shape $[C]$ on the same CUDA
                device as ``x``, with dtype ``torch.float32``. Updated
                in-place during training.
            running_var: Running variance of shape $[C]$ on the same
                CUDA device as ``x``, with dtype ``torch.float32``. Updated
                in-place during training.
            weight: Affine scale (gamma) of shape $[C]$ on the same CUDA
                device as ``x``.
            bias: Affine shift (beta) of shape $[C]$ on the same CUDA
                device as ``x``.

        Returns:
            Normalized output tensor with the same shape as ``x``.
        """
        return _batch_norm_fwd_wrapped(
            x, running_mean, running_var, weight, bias, self._instance_key
        )


class BatchNormBwdOp(Op):
    """Batch Normalization backward operator.

    Computes gradients with respect to input, scale, and shift for batch
    normalization.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    """

    compile_op_names: ClassVar[Tuple[str, ...]] = ("tileops::norm_batch_norm_bwd",)

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dictionary.
            tune: If ``True``, autotune tile configurations.
        """
        self.dtype: Optional[torch.dtype] = None
        self.target = target
        self.tune = tune

        self.dispatch_kernel(kernel_map)
        self.kernel: Optional[Kernel] = None
        self._last_roofline_spec: Optional[tuple[int, int, torch.dtype]] = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"bwd_kernel": BatchNormBwdKernel}

    def _infer_output_shapes(
        self,
        grad_out_shape: Tuple[int, ...],
        x_shape: Tuple[int, ...],
        weight_shape: Tuple[int, ...],
        mean_shape: Tuple[int, ...],
        rstd_shape: Tuple[int, ...],
    ) -> Dict[str, Tuple[int, ...]]:
        """Manifest ``shape_rules``: ``grad_x`` follows *x*, the two others the channels."""
        channels = (grad_out_shape[1],)
        return {
            "grad_x": tuple(x_shape),
            "grad_weight": channels,
            "grad_bias": channels,
        }

    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_spec is None:
            raise RuntimeError("BatchNormBwdOp.eval_roofline() requires a prior forward() call")
        C, L, dtype = self._last_roofline_spec
        elem_bytes = dtype.itemsize
        return (
            8 * C * L,
            3 * C * L * elem_bytes + 3 * C * 4,
        )

    def _resolve_spec(
        self, grad_out: torch.Tensor, x: torch.Tensor
    ) -> Tuple[int, int, torch.dtype]:
        if grad_out.device != x.device:
            raise ValueError(
                f"Expected grad_out and x on the same device, got {grad_out.device} and {x.device}"
            )
        if grad_out.shape != x.shape:
            raise ValueError(f"Expected x shape {grad_out.shape}, got {x.shape}")
        if grad_out.dtype != x.dtype:
            raise ValueError(f"Expected x.dtype {grad_out.dtype}, got {x.dtype}")
        if grad_out.ndim < 2:
            raise ValueError("grad_out must have shape (N, C, *spatial)")
        if grad_out.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                f"grad_out.dtype must be float32, float16, or bfloat16, got {grad_out.dtype}"
            )
        C = grad_out.shape[1]
        return C, grad_out.numel() // C, grad_out.dtype

    def _bind_spec(self, C: int, L: int, dtype: torch.dtype) -> None:
        """Bind what ``eval_roofline`` reads off the call that just ran."""
        self.dtype = dtype
        self._last_roofline_spec = (C, L, dtype)

    @staticmethod
    def _validate_channel_tensor(
        name: str,
        tensor: torch.Tensor,
        C: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if tensor.device != device:
            raise ValueError(f"Expected {name} on {device}, got {tensor.device}")
        if tensor.dtype != dtype:
            raise ValueError(f"Expected {name}.dtype {dtype}, got {tensor.dtype}")
        if tensor.ndim != 1 or tensor.shape[0] != C:
            raise ValueError(f"Expected {name} shape ({C},), got {tuple(tensor.shape)}")

    def _eager_forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        C, L, dtype = self._resolve_spec(grad_out, x)
        self._validate_channel_tensor("weight", weight, C, grad_out.device, torch.float32)
        self._validate_channel_tensor("mean", mean, C, grad_out.device, torch.float32)
        self._validate_channel_tensor("rstd", rstd, C, grad_out.device, torch.float32)
        self._bind_spec(C, L, dtype)
        grad_out = grad_out.contiguous()
        x = x.contiguous()
        weight = weight.contiguous()
        mean = mean.contiguous()
        rstd = rstd.contiguous()
        kernel = self.get_or_build_kernel(
            "batch_norm_bwd",
            (grad_out, x, weight, mean, rstd),
            key=(C, L, dtype),  # this instance's in-tree cache key
            build=lambda: self.kernel_map["bwd_kernel"](C, L, dtype, tune=self.tune),
        )
        self.kernel = kernel
        return kernel(grad_out, x, weight, mean, rstd)

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
            weight: Affine scale (gamma) of shape $[C]$ on the same CUDA
                device as ``x``. Internally cast to ``torch.float32`` for the
                backward kernel.
            mean: Per-channel batch mean from the forward pass, shape
                ``(C,)``. Expected as ``torch.float32``.
            rstd: Per-channel reciprocal std from the forward pass,
                shape $[C]$. Expected as ``torch.float32``.

        Returns:
            Tuple of ``(grad_x, grad_weight, grad_bias)`` where ``grad_x``
            has the same shape as ``x``, ``grad_weight`` has shape $[C]$,
            and ``grad_bias`` has shape $[C]$.
        """
        return _batch_norm_bwd_wrapped(grad_out, x, weight, mean, rstd, self._instance_key)


# torch.compile dispatch boundary (see src/tileops/ops/compile_boundary.py)


@torch.library.custom_op(
    "tileops::norm_batch_norm_fwd",
    mutates_args=("running_mean", "running_var"),
)
def _batch_norm_fwd_wrapped(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    instance = get_instance(instance_key)
    return instance._eager_forward(x, running_mean, running_var, weight, bias)


@_batch_norm_fwd_wrapped.register_fake
def _batch_norm_fwd_fake(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(
        tuple(x.shape),
        tuple(running_mean.shape),
        tuple(running_var.shape),
        tuple(weight.shape),
        tuple(bias.shape),
    )
    return x.new_empty(shapes["output"])


@torch.library.custom_op("tileops::norm_batch_norm_bwd", mutates_args=())
def _batch_norm_bwd_wrapped(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    instance_key: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    instance = get_instance(instance_key)
    return instance._eager_forward(grad_out, x, weight, mean, rstd)


@_batch_norm_bwd_wrapped.register_fake
def _batch_norm_bwd_fake(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    instance_key: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(
        tuple(grad_out.shape),
        tuple(x.shape),
        tuple(weight.shape),
        tuple(mean.shape),
        tuple(rstd.shape),
    )
    return (
        x.new_empty(shapes["grad_x"]),
        weight.new_empty(shapes["grad_weight"], dtype=torch.float32),
        weight.new_empty(shapes["grad_bias"], dtype=torch.float32),
    )
