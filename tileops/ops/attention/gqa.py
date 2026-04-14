from typing import Dict, Optional

import torch
import torch.nn.functional as F

from tileops.kernels.attention import (
    FlashAttnBwdPostprocessKernel,
    FlashAttnBwdPreprocessKernel,
    GQABwdKernel,
    GQABwdWgmmaPipelinedKernel,
    GQADecodeKernel,
    GQADecodePagedKernel,
    GQAFwdKernel,
    GQAFwdWgmmaPipelinedKernel,
    GQASlidingWindowFwdKernel,
    GQASlidingWindowFwdWgmmaPipelinedKernel,
    GQASlidingWindowVarlenFwdKernel,
    GQASlidingWindowVarlenFwdWgmmaPipelinedKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.utils import is_hopper

from ..op_base import Op

__all__ = [
    "GroupedQueryAttentionFwdOp",
    "GroupedQueryAttentionBwdOp",
    "GroupedQueryAttentionDecodeWithKVCacheFwdOp",
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp",
    "GqaSlidingWindowFwdOp",
    "GqaSlidingWindowVarlenFwdOp",
]


class GroupedQueryAttentionFwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_fwd_kernel"](
            batch, heads, heads_kv, seq_len, dim, is_causal, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_fwd_kernel": GQAFwdWgmmaPipelinedKernel if is_hopper() else GQAFwdKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.kernel(q, k, v)


class GroupedQueryAttentionBwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len  # TODO: support s_q != s_kv
        self.dim = dim
        self.is_causal = is_causal

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.prep_kernel = self.kernel_map["gqa_bwd_preprocess_kernel"](batch, heads, seq_len, dim,
                                                                        self.dtype)
        self.kernel = self.kernel_map["gqa_bwd_kernel"](
            batch, heads, heads_kv, seq_len, dim, is_causal, self.dtype, tune=tune)
        if not is_hopper():
            self.post_kernel = self.kernel_map["gqa_bwd_postprocess_kernel"](batch, heads, seq_len,
                                                                             dim, self.dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "gqa_bwd_preprocess_kernel":
                FlashAttnBwdPreprocessKernel,
            "gqa_bwd_kernel":
                GQABwdWgmmaPipelinedKernel if is_hopper() else GQABwdKernel,
            "gqa_bwd_postprocess_kernel":
                FlashAttnBwdPostprocessKernel if not is_hopper() else None,
        }

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                do: torch.Tensor,
                lse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        do = do.contiguous()
        delta = self.prep_kernel(o, do)
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        self.kernel(q, k, v, do, lse, delta, dq, dk, dv)
        dq = dq.to(self.dtype) if is_hopper() else self.post_kernel(dq)
        dk, dv = dk.to(self.dtype), dv.to(self.dtype)
        return dq, dk, dv


class GroupedQueryAttentionDecodeWithKVCacheFwdOp(Op):
    """Layout: BSHD"""

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seqlen_kv: int,
                 dim: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim

        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_decode_kernel"](
            batch, heads, heads_kv, seqlen_kv, dim, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_decode_kernel": GQADecodeKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        real_seqlen_kv = k.shape[1]
        if real_seqlen_kv < self.seqlen_kv:
            k = F.pad(
                k, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)
            v = F.pad(
                v, pad=(0, 0, 0, 0, 0, self.seqlen_kv - real_seqlen_kv), mode='constant', value=0)

        return self.kernel(q, k, v, real_seqlen_kv)


class GroupedQueryAttentionDecodePagedWithKVCacheFwdOp(Op):
    """Paged GQA decode with dynamic KV cache. Layout: Q [batch, heads, dim] (BHD);
    K, V physical cache [seqlen_kv, heads_kv, dim]; real_seqlen_kv [batch]; block_table [batch, num_pages].
    """

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seqlen_kv: int,
                 dim: int,
                 page_size: int,
                 dtype: torch.dtype = torch.float16,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
                 tune: bool = False) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["gqa_decode_paged_kernel"](
            batch, heads, heads_kv, seqlen_kv, dim, page_size, self.dtype, tune=tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"gqa_decode_paged_kernel": GQADecodePagedKernel}

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        return self.kernel(q, k, v, real_seqlen_kv, block_table)


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
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
        if window_size_left != -1 and window_size_left < 0:
            raise ValueError(
                f"window_size_left must be -1 (unlimited) or >= 0, got {window_size_left}")
        if window_size_right != -1 and window_size_right < 0:
            raise ValueError(
                f"window_size_right must be -1 (unlimited) or >= 0, got {window_size_right}")
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
        kernel = GQASlidingWindowFwdWgmmaPipelinedKernel if is_hopper() else GQASlidingWindowFwdKernel
        return {"gqa_sliding_window_fwd": kernel}

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
        for t, name in [(q, 'q'), (k, 'k'), (v, 'v')]:
            if t.device.type != 'cuda':
                raise ValueError(
                    f"{name} must be on a cuda device, got {t.device}")
            if t.dtype != self.dtype:
                raise ValueError(
                    f"{name} dtype {t.dtype} does not match op dtype {self.dtype}")
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
        S = self.seq_len
        wl = self.window_size_left
        wr = self.window_size_right
        total_attended = 0
        for q in range(S):
            hi = q if self.is_causal else (min(S - 1, q + wr) if wr >= 0 else S - 1)
            lo = max(0, q - wl) if wl >= 0 else 0
            total_attended += hi - lo + 1
        return 4 * self.batch * self.heads * total_attended * self.dim

    @property
    def total_memory(self) -> int:
        """Approximate bytes accessed: read Q/K/V, write O."""
        elem = torch.tensor([], dtype=self.dtype).element_size()
        return 2 * self.batch * self.seq_len * (self.heads + self.heads_kv) * self.dim * elem


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
        if heads % heads_kv != 0:
            raise ValueError("heads must be divisible by heads_kv")
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
        kernel = (GQASlidingWindowVarlenFwdWgmmaPipelinedKernel
                  if is_hopper() else GQASlidingWindowVarlenFwdKernel)
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

        if q.ndim != 3 or q.shape[1] != self.heads or q.shape[2] != self.dim:
            raise ValueError(
                f"q shape {q.shape} incompatible with heads={self.heads}, "
                f"dim={self.dim}")
        if k.ndim != 3 or k.shape[1] != self.heads_kv or k.shape[2] != self.dim:
            raise ValueError(
                f"k shape {k.shape} incompatible with heads_kv={self.heads_kv}"
                f", dim={self.dim}")
        if v.ndim != 3 or v.shape[1] != self.heads_kv or v.shape[2] != self.dim:
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
        for cu, name in [(cu_seqlens_q, 'cu_seqlens_q'),
                         (cu_seqlens_k, 'cu_seqlens_k')]:
            if cu.device.type != 'cuda':
                raise ValueError(
                    f"{name} must be on a cuda device, got {cu.device}")
            if cu.dtype != torch.int32:
                raise ValueError(
                    f"{name} must have dtype int32, got {cu.dtype}")
            if not cu.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if cu_seqlens_q[0].item() != 0:
            raise ValueError(
                f"cu_seqlens_q[0] must be 0, got {cu_seqlens_q[0].item()}")
        if cu_seqlens_k[0].item() != 0:
            raise ValueError(
                f"cu_seqlens_k[0] must be 0, got {cu_seqlens_k[0].item()}")
        if not torch.all(cu_seqlens_q[1:] >= cu_seqlens_q[:-1]):
            raise ValueError("cu_seqlens_q must be non-decreasing")
        if not torch.all(cu_seqlens_k[1:] >= cu_seqlens_k[:-1]):
            raise ValueError("cu_seqlens_k must be non-decreasing")
        if cu_seqlens_q[-1].item() > q.shape[0]:
            raise ValueError(
                f"cu_seqlens_q[-1] ({cu_seqlens_q[-1].item()}) exceeds "
                f"q.shape[0] ({q.shape[0]})")
        if cu_seqlens_k[-1].item() > k.shape[0]:
            raise ValueError(
                f"cu_seqlens_k[-1] ({cu_seqlens_k[-1].item()}) exceeds "
                f"k.shape[0] ({k.shape[0]})")
        actual_max_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
        if max_seqlen_q < actual_max_q:
            raise ValueError(
                f"max_seqlen_q ({max_seqlen_q}) must be >= actual max Q "
                f"sequence length ({actual_max_q})")

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
