"""Batch Normalization Op.

Training forward, inference forward and backward are separate kernel slots, each
selecting among its implementations by the call's ``BatchNormCall``.

User-facing API follows `torch.nn.functional.batch_norm`:

    fwd_op = BatchNormFwdOp(training=True, momentum=0.1, eps=1e-5)
    y, save_mean, save_invstd = fwd_op(input, running_mean, running_var, weight, bias)

    bwd_op = BatchNormBwdOp()
    dinput, dweight, dbias = bwd_op(doutput, input, weight, save_mean, save_invstd)

Forward returns ``(output, save_mean, save_invstd)`` as
``torch.ops.aten.native_batch_norm`` does; without training the last two are ``None``.

Input tensors accept any shape ``(N, C, *spatial)``; the kernels take its contiguous
``(N, C, S)`` view, with S the product of the spatial axes.
"""

import math
from typing import ClassVar, Mapping, Optional, Tuple

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.norm.batch_norm import (
    BatchNormBwdKernel,
    BatchNormBwdSplitKernel,
    BatchNormBwdWideKernel,
    BatchNormFwdInferKernel,
    BatchNormFwdTrainKernel,
    BatchNormFwdTrainSplitKernel,
    BatchNormFwdTrainWholeKernel,
    BatchNormFwdTrainWideKernel,
)
from tileops._kernels.norm.call_spec import (
    BatchNormBwdInterface,
    BatchNormCall,
    BatchNormInferFwdInterface,
    BatchNormTrainFwdInterface,
)
from tileops._ops.norm.affine import affine_or_constant
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["BatchNormBwdOp", "BatchNormFwdOp"]


class BatchNormFwdOp(Op):
    """Batch Normalization forward; in training it updates the running statistics in place.

    Computes batch normalization over the channel dimension:

    $$
    y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}}
    \\cdot \\gamma + \\beta
    $$

    where the mean and variance are computed per channel over ``(N, *spatial)``
    elements in training, and read from the running statistics in inference.

    Follows `torch.nn.functional.batch_norm`: ``forward`` takes
    ``(input, running_mean, running_var, weight, bias)`` in PyTorch's positional order.
    It returns ``(output, save_mean, save_invstd)``, the structure of
    ``torch.ops.aten.native_batch_norm``: in training the last two are the per-channel batch
    mean and reciprocal standard deviation that ``BatchNormBwdOp`` takes; without training
    they are ``None``. The running statistics are optional in
    training, where they are updated in place when passed, and required in inference.
    An absent ``weight`` scales by one and an absent ``bias`` shifts by zero.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fwd_train_whole": BatchNormFwdTrainWholeKernel,
        "fwd_train_wide": BatchNormFwdTrainWideKernel,
        "fwd_train_split": BatchNormFwdTrainSplitKernel,
        "fwd_train_kernel": BatchNormFwdTrainKernel,
        "fwd_infer_kernel": BatchNormFwdInferKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "batch_norm_fwd_train": BatchNormTrainFwdInterface,
        "batch_norm_fwd_infer": BatchNormInferFwdInterface,
    }

    def __init__(
        self,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            training: Whether the batch statistics come from this call's input, which is
                also what decides whether passed running statistics are written.
            momentum: Running-stat update momentum (used in training mode).
            eps: Epsilon for numerical stability.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.training = training
        self.eps = eps
        self.momentum = momentum
        super().__init__(target=target)

    def forward(
        self,
        input: torch.Tensor,
        running_mean: Optional[torch.Tensor] = None,
        running_var: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run batch normalization forward pass.

        The ``training`` mode is bound at ctor time. Construct a separate
        op instance to switch between training and inference.

        Args:
            input: Input tensor of shape ``(N, C, *spatial)``.
            running_mean: Running mean of shape $[C]$, ``torch.float32``; updated in place
                in training, required in inference.
            running_var: Running variance, the same; passed exactly when ``running_mean`` is.
            weight: Affine scale (gamma) of shape $[C]$, ``torch.float32``, or ``None``.
            bias: Affine shift (beta) of shape $[C]$, ``torch.float32``, or ``None``.

        Returns:
            ``(output, save_mean, save_invstd)``. ``output`` has the shape of ``input``. In
            training, ``save_mean`` and ``save_invstd`` are the batch mean and
            $1 / \\sqrt{\\mathrm{Var}[x] + \\epsilon}$ (biased variance) per channel, $[C]$,
            ``torch.float32``; without training both are ``None``.
        """
        if input.numel() == 0:
            # torch leaves the running statistics as they are; an empty channel has no mean.
            if not self.training:
                return torch.empty_like(input), None, None
            nan = torch.full((input.shape[1],), math.nan, dtype=torch.float32, device=input.device)
            return torch.empty_like(input), nan, nan.clone()
        batch, channels = input.shape[0], input.shape[1]
        spatial = math.prod(input.shape[2:])
        x_ncs = input.contiguous().view(batch, channels, spatial)
        # The inference program reads a weight and a bias, so an absent one is filled
        # before the call records what is passed.
        if not self.training:
            weight = affine_or_constant(weight, (channels,), 1.0, torch.float32, input.device)
            bias = affine_or_constant(bias, (channels,), 0.0, torch.float32, input.device)

        def call() -> BatchNormCall:
            return BatchNormCall(
                device=input.device,
                n=batch,
                c=channels,
                spatial=spatial,
                dtype=input.dtype,
                eps=self.eps,
                momentum=self.momentum,
                has_weight=weight is not None,
                has_bias=bias is not None,
                has_running_stats=running_mean is not None,
            )

        if not self.training:
            kernel = self.kernel_for("batch_norm_fwd_infer", call)
            stats = (running_mean.contiguous(), running_var.contiguous())
            return kernel(x_ncs, *stats, weight, bias).view(input.shape), None, None

        # A training program is built for the tensors passed and reads no stand-in for an
        # absent one. The running statistics are written, so normalizing them is not
        # enough: whoever serves this op writes the tensor it was handed, and a copy would
        # swallow that write. ``contiguous()`` returns the same object when it has nothing
        # to do, so what came back tells us whether a write-back is owed.
        stats = (running_mean, running_var)
        handed = stats if running_mean is None else tuple(stat.contiguous() for stat in stats)
        weight = None if weight is None else weight.contiguous()
        bias = None if bias is None else bias.contiguous()
        kernel = self.kernel_for("batch_norm_fwd_train", call)

        y, save_mean, save_invstd = kernel(x_ncs, *handed, weight, bias)
        if running_mean is not None:
            for original, written in zip(stats, handed, strict=True):
                if written is not original:
                    original.copy_(written)
        return y.view(input.shape), save_mean, save_invstd


class BatchNormBwdOp(Op):
    """Batch Normalization backward operator.

    Computes the gradients of the training forward with respect to its input, scale and
    shift, from the per-channel ``save_mean`` and ``save_invstd`` that forward computed.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.
    """

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
    ) -> None:
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        super().__init__(target=target)

    def forward(
        self,
        doutput: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        save_mean: torch.Tensor,
        save_invstd: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run batch normalization backward pass.

        Args:
            doutput: Upstream gradient of shape ``(N, C, *spatial)``.
            input: Original input tensor of the same shape.
            weight: Affine scale (gamma) of shape $[C]$, ``torch.float32``.
            save_mean: Per-channel batch mean from the forward pass, $[C]$, ``torch.float32``.
            save_invstd: Per-channel reciprocal std from the forward pass, $[C]$,
                ``torch.float32``.

        Returns:
            Tuple of ``(dinput, dweight, dbias)`` where ``dinput``
            has the same shape as ``input``, ``dweight`` has shape $[C]$,
            and ``dbias`` has shape $[C]$.
        """
        channels = doutput.shape[1]
        if doutput.numel() == 0:
            # An empty channel sums to zero.
            zeros = torch.zeros(channels, dtype=torch.float32, device=doutput.device)
            return torch.empty_like(input), zeros, zeros.clone()
        batch = doutput.shape[0]
        spatial = math.prod(doutput.shape[2:])
        grad_out_ncs = doutput.contiguous().view(batch, channels, spatial)
        x_ncs = input.contiguous().view(batch, channels, spatial)
        weight = weight.contiguous()
        save_mean = save_mean.contiguous()
        save_invstd = save_invstd.contiguous()
        kernel = self.kernel_for(
            "batch_norm_bwd",
            lambda: BatchNormCall(
                device=input.device, n=batch, c=channels, spatial=spatial, dtype=input.dtype
            ),
        )
        dinput, dweight, dbias = kernel(grad_out_ncs, x_ncs, weight, save_mean, save_invstd)
        return dinput.view(input.shape), dweight, dbias
