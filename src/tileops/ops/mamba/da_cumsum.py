import functools
from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.kernels.mamba import DaCumsumFwdKernel
from tileops.manifest import load_manifest

from ..op_base import Op

__all__ = ["DaCumsumFwdOp"]


@functools.lru_cache(maxsize=1)
def _dt_out_dtypes() -> tuple:
    """Storage dtypes ``dt_out`` may take, read from the manifest.

    No input tensor supplies this dtype — ``dt`` and ``A`` are always float32 —
    so ``_validate_dtypes`` cannot be the gate.
    """
    expr = load_manifest()["DaCumsumFwdOp"]["signature"]["outputs"]["dt_out"]["dtype"]
    return tuple(getattr(torch, name.strip()) for name in expr.split("|"))


class DaCumsumFwdOp(Op):
    """Mamba-2 dA_cumsum forward operator.

    Applies optional per-head bias, optional softplus activation, and clamping to
    raw dt values, then computes the chunk-local inclusive prefix sum of dA = dt * A.

    Note: dt_out is cast to the target dtype for storage efficiency, but dA_cumsum
    is computed from the fp32 dt values before casting, ensuring numerical precision.

    Args:
        chunk_len:    Tokens per chunk.
        dt_softplus:  Whether to apply softplus (with bypass for dt > 20) to dt.
        dt_min:       Lower clamp bound applied after bias and softplus.
        dt_max:       Upper clamp bound applied after bias and softplus.
        tune:         Whether to autotune tile config on construction.
    """

    def __init__(
        self,
        chunk_len: int,
        dtype: torch.dtype = torch.float32,
        dt_softplus: bool = False,
        dt_min: float = 0.0,
        dt_max: float = float("inf"),
        tune: bool = False,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        declared = _dt_out_dtypes()
        if dtype not in declared:
            supported = ", ".join(str(dt) for dt in declared)
            raise ValueError(
                f"{type(self).__name__} dt_out dtype must be one of [{supported}], got {dtype}"
            )
        self.batch = None
        self.num_chunks = None
        self.chunk_len = chunk_len
        self.n_heads = None
        self.seq_len = None
        self.dtype = dtype
        self.dt_softplus = dt_softplus
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"da_cumsum_fwd": DaCumsumFwdKernel}

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        num_chunks: int,
        n_heads: int,
        seq_len: int,
        has_dt_bias: bool,
        device_index: int | None,
    ) -> Kernel:
        key = (
            batch,
            num_chunks,
            self.chunk_len,
            n_heads,
            seq_len,
            self.dtype,
            self.dt_softplus,
            has_dt_bias,
            self.dt_min,
            self.dt_max,
            device_index,
            self.tune,
        )
        return self.get_or_build_kernel(
            "da_cumsum_fwd",
            inputs,
            key=key,
            build=lambda: self.kernel_map["da_cumsum_fwd"](
                batch,
                num_chunks,
                self.chunk_len,
                n_heads,
                seq_len,
                self.dtype,
                dt_softplus=self.dt_softplus,
                has_dt_bias=has_dt_bias,
                dt_min=self.dt_min,
                dt_max=self.dt_max,
                tune=self.tune,
            ),
        )

    def _infer_output_shapes(
        self,
        dt_shape: tuple[int, ...],
        A_shape: tuple[int, ...],
        dt_bias_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: ``[B, H, NC, chunk_len]``, with ``NC = S // chunk_len``."""
        b, s, h = dt_shape
        chunked = (b, h, s // self.chunk_len, self.chunk_len)
        return {"dt_out": chunked, "dA_cumsum": chunked}

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
            dt_out: (batch, n_heads, num_chunks, chunk_len) dtype — processed dt in target dtype.
            dA_cumsum: (batch, n_heads, num_chunks, chunk_len) float32 — inclusive prefix sum
                of dA = dt_val * A, computed from fp32 dt_val before casting dt_out.
        """
        if not dt.is_cuda:
            raise ValueError("dt must be a CUDA tensor")
        if dt.dtype != torch.float32:
            raise ValueError(f"Expected float32 dt, got {dt.dtype}")
        if dt.ndim != 3:
            raise ValueError("dt must have shape [batch, seq_len, n_heads]")
        batch, seq_len, n_heads = dt.shape
        if seq_len % self.chunk_len != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by chunk_len ({self.chunk_len})"
            )
        if A.shape != (n_heads,):
            raise ValueError("A must have shape [n_heads]")
        if dt_bias is not None and dt_bias.shape != (n_heads,):
            raise ValueError("dt_bias must have shape [n_heads]")

        self.batch = batch
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.num_chunks = seq_len // self.chunk_len
        self.dt_bias_shape = None if dt_bias is None else tuple(dt_bias.shape)
        self.kernel = self._get_kernel(
            (dt, A, dt_bias),
            batch,
            self.num_chunks,
            n_heads,
            seq_len,
            dt_bias is not None,
            dt.device.index,
        )

        dt = dt.contiguous()
        A = A.contiguous()

        return self.kernel(dt, A, dt_bias)
