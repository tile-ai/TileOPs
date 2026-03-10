"""Variable-length GQA forward with sliding window attention."""
from typing import Dict, Optional

import torch

from tileops.kernels.deepseek_nsa import (
    GqaSlidingWindowVarlenFwdKernel,
    GqaSlidingWindowVarlenFwdWgmmaPipelinedKernel,
)
from tileops.kernels.kernel import Kernel
from tileops.ops.op import Op
from tileops.utils import is_hopper


class GqaSlidingWindowVarlenFwdOp(Op):
    """Variable-length GQA forward with sliding window attention.

    Inputs are packed (no padding); per-sample boundaries are given via
    cu_seqlens arrays.  seqlen_q and seqlen_k may differ per sample:

      offset = seqlen_k - seqlen_q  (per sample, FA3 bottom-right convention)

    A token at local q_pos attends to local k_pos when ALL conditions hold:
      k_pos <= q_pos + offset                      (is_causal=True)
      k_pos >= q_pos + offset - window_size_left   (window_size_left >= 0)
      k_pos <= q_pos + offset + window_size_right  (window_size_right >= 0)

    Args:
        batch: Number of sequences in the batch.
        heads: Number of query heads.
        heads_kv: Number of KV heads (must divide heads evenly).
        dim: Head dimension.
        is_causal: Whether to apply causal masking.
        window_size_left: Left window size (-1 = unlimited).
        window_size_right: Right window size (-1 = unlimited).
        dtype: Tensor data type.
        accum_dtype: Accumulator data type for intermediate computations.
        kernel_map: Optional override for hardware-specific kernel dispatch.
        tune: Whether to run autotuning on kernel instantiation.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        dim: int,
        is_causal: bool,
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: torch.dtype = torch.float16,
        accum_dtype: torch.dtype = torch.float32,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
        if window_size_left != -1 and window_size_left < 0:
            raise ValueError(
                f"window_size_left must be -1 (unlimited) or >= 0, got {window_size_left}")
        if window_size_right != -1 and window_size_right < 0:
            raise ValueError(
                f"window_size_right must be -1 (unlimited) or >= 0, got {window_size_right}")
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype
        self.accum_dtype = accum_dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_sliding_window_varlen_fwd"](
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            dim=dim,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            dtype=dtype,
            accum_dtype=accum_dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        kernel = (GqaSlidingWindowVarlenFwdWgmmaPipelinedKernel
                  if is_hopper() else GqaSlidingWindowVarlenFwdKernel)
        return {"gqa_sliding_window_varlen_fwd": kernel}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        """Run variable-length GQA sliding window forward.

        Args:
            q: Query tensor, shape [total_q, heads, dim].
            k: Key tensor, shape [total_k, heads_kv, dim].
            v: Value tensor, shape [total_k, heads_kv, dim].
            cu_seqlens_q: Cumulative Q lengths, shape [batch+1], dtype int32.
            cu_seqlens_k: Cumulative K lengths, shape [batch+1], dtype int32.
            max_seqlen_q: Maximum Q sequence length across the batch.

        Returns:
            Output tensor, shape [total_q, heads, dim].
        """
        for t, name in [(q, 'q'), (k, 'k'), (v, 'v')]:
            if t.device.type != 'cuda':
                raise ValueError(
                    f"{name} must be on a cuda device, got {t.device}")
            if t.dtype != self.dtype:
                raise ValueError(
                    f"{name} dtype {t.dtype} does not match op dtype {self.dtype}")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        if q.shape != (q.shape[0], self.heads, self.dim):
            raise ValueError(
                f"q shape {q.shape} incompatible with heads={self.heads}, "
                f"dim={self.dim}")
        if k.shape != (k.shape[0], self.heads_kv, self.dim):
            raise ValueError(
                f"k shape {k.shape} incompatible with heads_kv={self.heads_kv}"
                f", dim={self.dim}")
        if v.shape != (v.shape[0], self.heads_kv, self.dim):
            raise ValueError(
                f"v shape {v.shape} incompatible with heads_kv={self.heads_kv}"
                f", dim={self.dim}")
        if cu_seqlens_q.shape[0] != self.batch + 1:
            raise ValueError(
                f"cu_seqlens_q.shape[0] ({cu_seqlens_q.shape[0]}) must equal "
                f"batch+1 ({self.batch + 1})")
        if cu_seqlens_k.shape[0] != self.batch + 1:
            raise ValueError(
                f"cu_seqlens_k.shape[0] ({cu_seqlens_k.shape[0]}) must equal "
                f"batch+1 ({self.batch + 1})")

        output, _ = self.kernel.forward(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
        return output

    @property
    def total_flops(self) -> int:
        raise NotImplementedError(
            "total_flops is not defined for varlen ops; "
            "compute per-sample from cu_seqlens at call time.")

    @property
    def total_memory(self) -> int:
        raise NotImplementedError(
            "total_memory is not defined for varlen ops; "
            "compute per-sample from cu_seqlens at call time.")
