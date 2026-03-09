"""Fixed-length GQA forward with sliding window attention."""
from typing import Dict, Optional

import torch

from tileops.kernels.deepseek_nsa import (
    GqaSlidingWindowFwdKernel,
    GqaSlidingWindowFwdWgmmaPipelinedKernel,
)
from tileops.kernels.kernel import Kernel
from tileops.ops.op import Op


class GqaSlidingWindowFwdOp(Op):
    """Fixed-length GQA forward with sliding window attention.

    Token at q_pos attends to k_pos when ALL applicable conditions hold:
      - k_pos <= q_pos                          (is_causal=True)
      - k_pos >= q_pos - window_size_left       (window_size_left >= 0)
      - k_pos <= q_pos + window_size_right      (window_size_right >= 0)

    Use window_size_left=-1 / window_size_right=-1 for no restriction.

    Args:
        batch: Batch size.
        heads: Number of query heads.
        heads_kv: Number of KV heads (must divide heads evenly).
        seq_len: Sequence length (same for Q, K, V).
        dim: Head dimension.
        is_causal: Whether to apply causal masking.
        window_size_left: Left window size (-1 = unlimited).
        window_size_right: Right window size (-1 = unlimited).
        dtype: Tensor data type.
        kernel_map: Optional override for hardware-specific kernel dispatch.
        tune: Whether to run autotuning on kernel instantiation.
    """

    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        window_size_left: int = -1,
        window_size_right: int = -1,
        dtype: torch.dtype = torch.float16,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ) -> None:
        assert heads % heads_kv == 0, "heads must be divisible by heads_kv"
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.window_size_left = window_size_left
        self.window_size_right = window_size_right
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_sliding_window_fwd"](
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            seq_len=seq_len,
            dim=dim,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            dtype=dtype,
            tune=tune,
        )

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_sliding_window_fwd": GqaSlidingWindowFwdKernel}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Run fixed-length GQA sliding window forward.

        Args:
            q: Query tensor, shape [batch, seq_len, heads, dim].
            k: Key tensor, shape [batch, seq_len, heads_kv, dim].
            v: Value tensor, shape [batch, seq_len, heads_kv, dim].

        Returns:
            Output tensor, shape [batch, seq_len, heads, dim].
        """
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        if q.shape != (self.batch, self.seq_len, self.heads, self.dim):
            raise ValueError(
                f"q shape {q.shape} does not match expected "
                f"({self.batch}, {self.seq_len}, {self.heads}, {self.dim})")
        if k.shape != (self.batch, self.seq_len, self.heads_kv, self.dim):
            raise ValueError(
                f"k shape {k.shape} does not match expected "
                f"({self.batch}, {self.seq_len}, {self.heads_kv}, {self.dim})")
        if v.shape != (self.batch, self.seq_len, self.heads_kv, self.dim):
            raise ValueError(
                f"v shape {v.shape} does not match expected "
                f"({self.batch}, {self.seq_len}, {self.heads_kv}, {self.dim})")

        output, _ = self.kernel.forward(q, k, v)
        return output

    @property
    def total_flops(self) -> int:
        """Approximate FLOPs for QK^T and PV GEMMs."""
        if self.is_causal:
            eff_kv = self.seq_len // 2
        elif self.window_size_left >= 0 or self.window_size_right >= 0:
            wl = self.window_size_left if self.window_size_left >= 0 else self.seq_len
            wr = self.window_size_right if self.window_size_right >= 0 else self.seq_len
            eff_kv = min(self.seq_len, wl + wr + 1)
        else:
            eff_kv = self.seq_len
        return 4 * self.batch * self.heads * self.seq_len * eff_kv * self.dim

    @property
    def total_memory(self) -> int:
        """Approximate bytes accessed: read Q/K/V, write O."""
        elem = torch.tensor([], dtype=self.dtype).element_size()
        return (3 * self.batch * self.seq_len * self.heads_kv * self.dim +
                self.batch * self.seq_len * self.heads * self.dim) * elem
