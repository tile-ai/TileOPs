from typing import ClassVar, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Kernel, KernelInterface
from tileops.kernels.norm import FusedAddRMSNormPerTokenQuantKernel
from tileops.kernels.norm.call_spec import (
    FusedAddRMSNormPerTokenQuantFwdInterface,
    RMSNormQuantCall,
)
from tileops.ops.op_base import Op

__all__ = ["FusedAddRMSNormPerTokenQuantFwdOp"]


class FusedAddRMSNormPerTokenQuantFwdOp(Op):
    """Fused residual addition, RMS normalization and dynamic per-token quantization.

    One pass over the activations computes the residual sum, both row reductions (the sum
    of squares and the absolute maximum) and the codes:

    $$
    \\begin{aligned}
    r &= x + \\mathrm{residual} \\\\
    y &= \\frac{r}{\\sqrt{\\mathrm{mean}(r^2) + \\epsilon}} \\cdot w \\\\
    s &= \\max\\left(\\frac{\\min(\\max|y|, \\mathrm{scale\\_ub})}{q_{\\max}},
          s_{\\min}\\right) \\\\
    q &= \\mathrm{sat}(y \\cdot s^{-1})
    \\end{aligned}
    $$

    It follows vLLM's ``rms_norm_dynamic_per_token_quant``: ``r`` enters the reduction in
    float32 unrounded, ``y`` is rounded to the input dtype before and after the weight
    multiply, $q_{\\max}$ is 448 for ``float8_e4m3fn`` and 127 for ``int8``, $s_{\\min}$ is
    $1 / (448 \\cdot 512)$ and the float32 epsilon respectively, and ``int8`` codes round
    half to even. ``scale_ub`` applies to ``float8_e4m3fn`` codes only, as in vLLM. A NaN is
    skipped by the absolute maximum, and a NaN code becomes +448 in ``float8_e4m3fn`` and 0
    in ``int8``.

    The row reduction's order differs from vLLM's, so the float32 $\\mathrm{rrms}$ may differ
    by one ulp. Then the scale may differ by up to two ulps of the input dtype, an FP8 code
    by one representable step and an INT8 code by $\\lceil 4 \\epsilon_T \\cdot 127 \\rceil$
    (4 for ``bfloat16``, 1 for ``float16``); ``residual_out`` is exact.

    Returns ``(q, scale, residual_out)``: the codes, the float32 per-token scale shaped
    ``(*B, 1)`` and ``x + residual`` in the input dtype.

    Supported dtypes:
        ``torch.float16``, ``torch.bfloat16`` inputs; ``torch.float8_e4m3fn``,
        ``torch.int8`` codes.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fused_add_rms_norm_per_token_quant": FusedAddRMSNormPerTokenQuantKernel
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "fused_add_rms_norm_per_token_quant": FusedAddRMSNormPerTokenQuantFwdInterface
    }

    def __init__(
        self,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
        eps: float = 1e-6,
        scale_ub: Optional[float] = None,
        *,
        target: Target = None,
    ):
        """Build the op. Shapes and dtype are taken from each call.

        Args:
            quant_dtype: Dtype of the codes, ``torch.float8_e4m3fn`` or ``torch.int8``.
            eps: Epsilon for numerical stability.
            scale_ub: Upper bound on a token's absolute maximum before it sets the scale,
                or ``None`` for no bound.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for the
                in-tree kernels, or ``None`` to decide from the input device.
        """
        self.quant_dtype = quant_dtype
        self.eps = eps
        self.scale_ub = scale_ub
        super().__init__(target=target)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add, normalize and quantize each token.

        Args:
            x: Input tensor of shape ``(*B, N)``.
            residual: Residual tensor of the same shape as *x*.
            weight: Affine scale of shape $[N]$.

        Returns:
            ``(q, scale, residual_out)``: ``q`` shaped like *x* in ``quant_dtype``,
            ``scale`` of shape ``(*B, 1)`` in float32 and ``residual_out``, ``x + residual``,
            shaped and typed like *x*.

        Raises:
            ValueError: Dtypes or shapes disagree. Raised by the generated signature checks.
        """
        n = x.shape[-1]
        leading = x.shape[:-1]
        rows = x.reshape(-1, n).contiguous()
        call = RMSNormQuantCall(
            device=x.device,
            n=n,
            rows=rows.shape[0],
            eps=self.eps,
            dtype=x.dtype,
            quant_dtype=self.quant_dtype,
            scale_ub=self.scale_ub,
        )
        kernel = self.kernel_for("fused_add_rms_norm_per_token_quant", call)
        q, scale, residual_out = kernel(
            rows, residual.reshape(-1, n).contiguous(), weight.contiguous()
        )
        return (
            q.reshape(*leading, n),
            scale.reshape(*leading, 1),
            residual_out.reshape(*leading, n),
        )
