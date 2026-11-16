"""Gated RMS normalization operator (L2 Op layer).

Provides:
  - RMSNormGatedFwdOp: y = rmsnorm(x) * w + b, gated by silu(z) or sigmoid(z) after or
    before the norm
"""

from collections.abc import Mapping
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.norm import RMSNormGatedKernel
from tileops._kernels.norm.call_spec import RMSNormGatedCall, RMSNormGatedFwdInterface
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = ["RMSNormGatedFwdOp"]


class RMSNormGatedFwdOp(Op):
    """RMS normalization over groups of the last axis, gated by ``act(z)``.

    Follows ``mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn``, with fla's sigmoid gate.
    Over each group of ``group_size`` columns of a row (the whole row when ``None``):

    $$
    y = \\frac{h}{\\sqrt{\\mathrm{mean}(h^2) + \\epsilon}} \\cdot w + b
    $$

    With ``norm_before_gate``, ``h = x`` and ``y`` is then multiplied by ``act(z)``;
    otherwise ``h = x \\cdot act(z)``. ``act`` is silu or sigmoid, by ``activation``.
    Without ``z`` there is no gate; ``bias`` ``None`` shifts by zero.

    The arithmetic is float32 and ``y`` rounds once to the dtype of ``x``, as vLLM's and
    fla's kernels round it. The reciprocal square root and the gate's reciprocal are the
    hardware's approximate ones, within two float32 units in the last place. The gate's
    exponential is ``2^-64 exp2(64 - |z| log2 e)``, which keeps a gate down to the smallest
    float32 subnormal; the float32 rounding of that exponent makes its relative error larger
    than the reciprocal's.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"rms_norm_gated": RMSNormGatedKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "rms_norm_gated": RMSNormGatedFwdInterface
    }

    def __init__(
        self,
        eps: float = 1e-6,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        activation: str = "silu",
        *,
        target: Target = None,
    ) -> None:
        """Build the op. Shapes and dtypes are taken from each call.

        Args:
            eps: Added to the mean square before the root.
            group_size: Columns each norm runs over, dividing the row; ``None`` for the
                whole row.
            norm_before_gate: Gate the normalized row when true, the input when false.
            activation: ``"silu"`` or ``"sigmoid"``, the gate's activation.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.eps = eps
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.activation = activation
        super().__init__(target=target)

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalize, scale, shift and gate each row of ``x``.

        Args:
            x: ``(*B, N)``, float16, bfloat16 or float32.
            weight: ``(N,)`` scale, float16, bfloat16 or float32.
            bias: ``(N,)`` shift, float16, bfloat16 or float32, or ``None``.
            z: The gate, shaped like ``x``, float16, bfloat16 or float32, or ``None``.

        Returns:
            A new tensor shaped like ``x``, in its dtype.
        """
        n = x.shape[-1]
        x = x.contiguous()
        weight = weight.contiguous()
        bias = None if bias is None else bias.contiguous()
        z = None if z is None else z.contiguous()
        return self.kernel_for(
            "rms_norm_gated",
            lambda: RMSNormGatedCall(
                device=x.device,
                n=n,
                group=n if self.group_size is None else self.group_size,
                eps=self.eps,
                norm_before_gate=self.norm_before_gate,
                activation=self.activation,
                dtype=x.dtype,
                weight_dtype=weight.dtype,
                bias_dtype=None if bias is None else bias.dtype,
                z_dtype=None if z is None else z.dtype,
            ),
        )(x, weight, bias, z)
