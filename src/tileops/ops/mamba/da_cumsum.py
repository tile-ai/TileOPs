from typing import ClassVar, Dict, Mapping, Optional, Tuple

import torch

from tileops.backend import Target
from tileops.kernels.kernel_base import Entry, Kernel
from tileops.kernels.mamba import DaCumsumFwdKernel

from ..op_base import Op

__all__ = ["DaCumsumFwdOp"]


class DaCumsumFwdOp(Op):
    """Mamba-2 dA_cumsum forward operator.

    Applies optional per-head bias, optional softplus activation, and clamping to
    raw dt values, then computes the chunk-local inclusive prefix sum of dA = dt * A.

    Note: dt_out is cast to the target dtype for storage efficiency, but dA_cumsum
    is computed from the fp32 dt values before casting, ensuring numerical precision.
    """

    compile_boundary = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"da_cumsum_fwd": DaCumsumFwdKernel}

    def __init__(
        self,
        chunk_len: int,
        out_dtype: torch.dtype = torch.float32,
        dt_softplus: bool = False,
        dt_min: float = 0.0,
        dt_max: float = float("inf"),
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes are taken from each call.

        Args:
            chunk_len: Tokens per chunk.
            out_dtype: Storage dtype of ``dt_out``.
            dt_softplus: Whether to apply softplus (with bypass for dt > 20) to dt.
            dt_min: Lower clamp bound applied after bias and softplus.
            dt_max: Upper clamp bound applied after bias and softplus.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional override for kernel dispatch.
            tune: Whether to autotune the tile config when a kernel is first built.
        """
        self.chunk_len = chunk_len
        self.out_dtype = out_dtype
        self.dt_softplus = dt_softplus
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)

    def entry_for(self, role: str, call: tuple) -> Entry:
        """One implementation, built per shape, bias presence and device."""
        batch, seq_len, n_heads, has_dt_bias, _device = call
        return call, lambda: self.kernel_map["da_cumsum_fwd"](
            batch,
            seq_len // self.chunk_len,
            self.chunk_len,
            n_heads,
            seq_len,
            self.out_dtype,
            dt_softplus=self.dt_softplus,
            has_dt_bias=has_dt_bias,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            tune=self.tune,
        )

    def forward(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the dA_cumsum forward pass.

        Args:
            dt: (batch, seq_len, n_heads) float32 — raw dt values.
            A:  (n_heads,) float32 — SSM decay parameters.
            dt_bias: (n_heads,) float32, optional — per-head dt bias.

        Returns:
            dt_out: (batch, n_heads, num_chunks, chunk_len) ``out_dtype`` — processed dt.
            dA_cumsum: (batch, n_heads, num_chunks, chunk_len) float32 — inclusive prefix sum
                of dA = dt_val * A, computed from fp32 dt_val before casting dt_out.
        """
        return self._call_boundary(dt, A, dt_bias)

    def _eager_forward(
        self,
        dt: torch.Tensor,
        A: torch.Tensor,
        dt_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the kernel and launch, inside the operator.

        Never traced: kernel construction enters a TileLang builder.
        """
        batch, seq_len, n_heads = dt.shape
        dt = dt.contiguous()
        A = A.contiguous()
        kernel = self.kernel_for(
            "da_cumsum_fwd",
            (dt, A, dt_bias),
            (batch, seq_len, n_heads, dt_bias is not None, dt.device.index),
        )
        return kernel(dt, A, dt_bias)
