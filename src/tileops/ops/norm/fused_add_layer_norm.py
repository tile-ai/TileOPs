from typing import Optional, Tuple

import torch

from ..op_base import Op

__all__ = ["FusedAddLayerNormFwdOp"]


class FusedAddLayerNormFwdOp(Op):
    """Fused residual addition and Layer Normalization operator.

    Computes the residual sum followed by layer normalization in a single
    fused kernel:

    .. math::

        \\begin{aligned}
        r &= x + \\mathrm{residual} \\\\
        y &= \\frac{r - \\mathrm{E}[r]}{\\sqrt{\\mathrm{Var}[r] + \\epsilon}}
            \\cdot w + b
        \\end{aligned}

    Returns dual outputs ``(y, residual_out)`` so downstream residual connections can
    reuse the pre-norm sum without recomputation.

    Supported dtypes:
        ``torch.float32``, ``torch.float16``, ``torch.bfloat16``.

    Note:
        Supports arbitrary leading dimensions (3-D+) via flatten/unflatten.
        Handles non-contiguous inputs and non-power-of-two hidden dims
        by padding to 256-element alignment.

    Args:
        M: Optional committed row count for strict compatibility. Preferred
            API infers it from ``x.shape[:-1]``.
        N: Optional committed hidden dimension. Preferred API infers it from
            ``x.shape[-1]``.
        eps: Epsilon for numerical stability.
        target: Which backend serves this op, or ``None`` to detect from the input device.
        tune: If ``True``, autotune tile configurations.
    """

    OP_NAME = "FusedAddLayerNormFwdOp"

    def __init__(
        self,
        M: Optional[int] = None,
        N: Optional[int] = None,
        eps: float = 1e-5,
        *,
        target: Optional[str] = None,
        tune: bool = False,
    ):
        self.M = M
        self.N = N
        self._committed_M = M
        self._committed_N = N
        self.eps = eps
        self.target = target
        self.tune = tune
        self.dispatch_kernel()
        self._last_roofline_mn: Optional[tuple[int, int]] = None


    def eval_roofline(self) -> tuple[int, int]:
        if self._last_roofline_mn is None or self.dtype is None:
            raise RuntimeError(
                f"{type(self).__name__}.eval_roofline() requires a prior "
                "forward() call to bind input shape and dtype"
            )
        M, N = self._last_roofline_mn
        elem_bytes = self.dtype.itemsize
        return (
            6 * M * N,
            (4 * M * N + 2 * N) * elem_bytes,
        )


    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply fused residual addition and layer normalization.

        Args:
            x: Input tensor of shape ``(*leading, N)`` on CUDA.
            residual: Residual tensor of the same shape as *x* on CUDA.
            weight: Affine scale of shape ``(N,)`` on CUDA.
            bias: Affine shift of shape ``(N,)`` on CUDA.

        Returns:
            Tuple of ``(y, residual_out)`` where *y* is the normalized
            output and *residual_out* is ``x + residual``, both of the
            same shape as *x*.

        Raises:
            ValueError: If tensors are not on CUDA, dtypes mismatch,
                or shapes are incompatible with the configured dimensions.
        """
        (x, residual, weight, bias), params = self._bind_call(x, residual, weight, bias)
        expected_dtype = x.dtype
        for name, tensor in [("x", x), ("residual", residual), ("weight", weight), ("bias", bias)]:
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
            if tensor.dtype != expected_dtype:
                raise ValueError(
                    f"Expected {name}.dtype {expected_dtype}, got {tensor.dtype}"
                )
        if weight.ndim != 1:
            raise ValueError(
                f"Expected weight to be 1D, got {weight.ndim}D"
            )
        if bias.ndim != 1:
            raise ValueError(
                f"Expected bias to be 1D, got {bias.ndim}D"
            )
        N = x.shape[-1]
        if self._committed_N is not None and self._committed_N != N:
            raise ValueError(
                f"Expected hidden dim {self._committed_N}, got {N}"
            )
        if residual.shape != x.shape:
            raise ValueError(
                f"Expected residual shape {x.shape}, got {residual.shape}"
            )
        if weight.shape[0] != N:
            raise ValueError(
                f"Expected weight dim {N}, got {weight.shape[0]}"
            )
        if bias.shape[0] != N:
            raise ValueError(
                f"Expected bias dim {N}, got {bias.shape[0]}"
            )

        orig_shape = x.shape
        x = x.contiguous().reshape(-1, N)
        residual = residual.contiguous().reshape(-1, N)
        M_actual = x.shape[0]
        if self._committed_M is not None and M_actual != self._committed_M:
            raise ValueError(
                f"Expected M={self._committed_M} (product of leading dims), got {M_actual}"
            )
        self.M = M_actual
        self.N = N
        dtype = expected_dtype
        assert dtype is not None

        kernel = self.backend_kernel(x, residual, weight, bias, **params)
        y, residual_out = kernel(x, residual, weight, bias)
        self._last_roofline_mn = (M_actual, N)
        self.dtype = expected_dtype

        return y.reshape(orig_shape), residual_out.reshape(orig_shape)
