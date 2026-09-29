"""Batch Normalization Op.

Training forward, inference forward and backward are separate kernel slots, each
selecting among its implementations by the call's ``BatchNormCall``.

User-facing API follows `torch.nn.functional.batch_norm`:

    fwd_op = BatchNormFwdOp(training=False, momentum=0.1, eps=1e-5)
    y = fwd_op(x, running_mean, running_var, weight, bias)

    bwd_op = BatchNormBwdOp()
    grad_x, grad_weight, grad_bias = bwd_op(grad_out, x, weight, mean, rstd)

Forward returns the normalized output only (manifest contract); ``mean`` and
``rstd`` from the training path stay internal. Callers needing them for the
backward pass can recompute on the original input.

Input tensors accept any shape ``(N, C, *spatial)``; the kernels take its contiguous
``(N, C, S)`` view, with S the product of the spatial axes.
"""

import math
from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.norm.batch_norm import (
    BatchNormBwdKernel,
    BatchNormBwdSplitKernel,
    BatchNormBwdWideKernel,
    BatchNormFwdInferKernel,
    BatchNormFwdTrainKernel,
    BatchNormFwdTrainSplitKernel,
    BatchNormFwdTrainWholeKernel,
    BatchNormFwdTrainWideKernel,
)
from tileops.kernels.norm.call_spec import (
    BatchNormBwdInterface,
    BatchNormCall,
    BatchNormFwdInferInterface,
    BatchNormFwdTrainInterface,
)
from tileops.ops.norm.norm_base import affine_or_constant
from tileops.ops.op_base import Op

__all__ = ["BatchNormBwdOp", "BatchNormFwdOp"]


class BatchNormFwdOp(Op):
    """Batch Normalization forward operator (training and inference).

    Computes batch normalization over the channel dimension:

    $$
    y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
    \\cdot \\gamma + \\beta
    $$

    where the mean and variance are computed per channel over ``(N, *spatial)``
    elements in training, and read from the running statistics in inference.

    Follows `torch.nn.functional.batch_norm`: ``forward`` takes
    ``(input, running_mean, running_var, weight, bias)`` in PyTorch's positional order
    and returns only the normalized output. The running statistics are optional in
    training, where they are updated in place when passed, and required in inference.
    An absent ``weight`` scales by one and an absent ``bias`` shifts by zero.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fwd_train_whole": BatchNormFwdTrainWholeKernel,
        "fwd_train_wide": BatchNormFwdTrainWideKernel,
        "fwd_train_split": BatchNormFwdTrainSplitKernel,
        "fwd_train_kernel": BatchNormFwdTrainKernel,
        "fwd_infer_kernel": BatchNormFwdInferKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "batch_norm_fwd_train": BatchNormFwdTrainInterface,
        "batch_norm_fwd_infer": BatchNormFwdInferInterface,
    }

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
            training: Whether the batch statistics come from this call's input, which is
                also what decides whether passed running statistics are written.
            momentum: Running-stat update momentum (used in training mode).
            eps: Epsilon for numerical stability.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel override dictionary.
            tune: If ``True``, autotune tile configurations.
        """
        self.training = training
        self.eps = eps
        self.momentum = momentum
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    def _eager_forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        if x.numel() == 0:
            # torch leaves the running statistics as they are.
            return torch.empty_like(x)
        batch, channels = x.shape[0], x.shape[1]
        spatial = math.prod(x.shape[2:])
        weight = affine_or_constant(weight, (channels,), 1.0, torch.float32, x.device)
        bias = affine_or_constant(bias, (channels,), 0.0, torch.float32, x.device)
        x_ncs = x.contiguous().view(batch, channels, spatial)
        # The running statistics are written, so normalizing them is not enough: whoever
        # serves this op writes the tensor it was handed, and a copy would swallow that
        # write. ``contiguous()`` returns the same object when it has nothing to do, so
        # what came back tells us whether a write-back is owed. Absent statistics in
        # training take scratch buffers whose update nobody reads.
        stats = (running_mean, running_var)
        if running_mean is None:
            handed = (
                torch.zeros(channels, dtype=torch.float32, device=x.device),
                torch.ones(channels, dtype=torch.float32, device=x.device),
            )
        else:
            handed = tuple(stat.contiguous() for stat in stats)

        call = BatchNormCall(
            device=x.device,
            n=batch,
            c=channels,
            spatial=spatial,
            dtype=x.dtype,
            eps=self.eps,
            momentum=self.momentum,
        )
        interface = "batch_norm_fwd_train" if self.training else "batch_norm_fwd_infer"
        kernel = self.kernel_for(interface, (x_ncs, *handed, weight, bias), call)
        self.kernel = kernel

        # The training kernel also returns the batch statistics, which the manifest keeps
        # out of this op's outputs.
        if not self.training:
            return kernel(x_ncs, *handed, weight, bias).view(x.shape)

        y, _mean, _rstd = kernel(x_ncs, *handed, weight, bias)
        if running_mean is not None:
            for original, written in zip(stats, handed, strict=True):
                if written is not original:
                    original.copy_(written)
        return y.view(x.shape)

    def forward(
        self,
        x: torch.Tensor,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run batch normalization forward pass.

        The ``training`` mode is bound at ctor time. Construct a separate
        op instance to switch between training and inference.

        Args:
            x: Input tensor of shape ``(N, C, *spatial)``.
            running_mean: Running mean of shape $[C]$, ``torch.float32``; updated in place
                in training, required in inference.
            running_var: Running variance, the same; passed exactly when ``running_mean`` is.
            weight: Affine scale (gamma) of shape $[C]$, ``torch.float32``, or ``None``.
            bias: Affine shift (beta) of shape $[C]$, ``torch.float32``, or ``None``.

        Returns:
            Normalized output tensor with the same shape as ``x``.
        """
        return self._call_boundary(x, running_mean, running_var, weight, bias)


class BatchNormBwdOp(Op):
    """Batch Normalization backward operator.

    Computes the gradients of the training forward with respect to its input, scale and
    shift, from the per-channel ``mean`` and ``rstd`` that forward computed.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.
    """

    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "bwd_wide": BatchNormBwdWideKernel,
        "bwd_split": BatchNormBwdSplitKernel,
        "bwd_kernel": BatchNormBwdKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "batch_norm_bwd": BatchNormBwdInterface
    }

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
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    def _eager_forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder, which dynamo cannot follow.
        """
        channels = grad_out.shape[1]
        if grad_out.numel() == 0:
            # An empty channel sums to zero.
            zeros = torch.zeros(channels, dtype=torch.float32, device=grad_out.device)
            return torch.empty_like(x), zeros, zeros.clone()
        batch = grad_out.shape[0]
        spatial = math.prod(grad_out.shape[2:])
        grad_out_ncs = grad_out.contiguous().view(batch, channels, spatial)
        x_ncs = x.contiguous().view(batch, channels, spatial)
        weight = weight.contiguous()
        mean = mean.contiguous()
        rstd = rstd.contiguous()
        call = BatchNormCall(device=x.device, n=batch, c=channels, spatial=spatial, dtype=x.dtype)
        kernel = self.kernel_for("batch_norm_bwd", (grad_out_ncs, x_ncs, weight, mean, rstd), call)
        self.kernel = kernel
        grad_x, grad_weight, grad_bias = kernel(grad_out_ncs, x_ncs, weight, mean, rstd)
        return grad_x.view(x.shape), grad_weight, grad_bias

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
            x: Original input tensor of the same shape.
            weight: Affine scale (gamma) of shape $[C]$, ``torch.float32``.
            mean: Per-channel batch mean from the forward pass, $[C]$, ``torch.float32``.
            rstd: Per-channel reciprocal std from the forward pass, $[C]$, ``torch.float32``.

        Returns:
            Tuple of ``(grad_x, grad_weight, grad_bias)`` where ``grad_x``
            has the same shape as ``x``, ``grad_weight`` has shape $[C]$,
            and ``grad_bias`` has shape $[C]$.
        """
        return self._call_boundary(grad_out, x, weight, mean, rstd)
