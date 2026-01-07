from typing import Any, Optional, Tuple

import flash_attn_interface
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.benchmark import Benchmark
from top.ops import GroupQueryAttentionBwdOp, GroupQueryAttentionFwdOp


class GroupQueryAttentionFwdBenchmark(Benchmark):

    op_type = GroupQueryAttentionFwdOp

    def __init__(self, batch: int, heads: int, heads_kv: int, seq_len: int, dim: int,
                 is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    @property
    def total_flops(self) -> float:
        flops_per_matmul = 2.0 * self.batch * self.heads * self.seq_len * self.seq_len * self.dim
        flops = flops_per_matmul * 2
        return flops / 2 if self.is_causal else flops

    @property
    def total_memory(self) -> int:
        return 2 * self.batch * self.seq_len * self.dim * (self.heads +
                                                           self.heads_kv) * self.dtype.itemsize

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        K = torch.randn(
            self.batch, self.seq_len, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        V = torch.randn(
            self.batch, self.seq_len, self.heads_kv, self.dim, device='cuda',
            dtype=self.dtype).contiguous()
        return Q, K, V

    def ref_program(self, q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output, None  # do not check lse

    def baseline_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        out = flash_attn_interface.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=None,  # use default 1 / sqrt(head_dim)
            causal=self.is_causal,
        )

        # Be robust to different return types.
        if isinstance(out, tuple):
            out = out[0]

        return out

    def baseline_profile(self,
                         *inputs: Any,
                         warmup: int = 100,
                         rep: int = 10,
                         device: str = "cuda:0") -> Any:
        return super().baseline_profile(
            self.baseline_program, *inputs, backend="FA3", warmup=warmup, rep=rep, device=device)


class GroupQueryAttentionBwdBenchmark(Benchmark):

    op_type = GroupQueryAttentionBwdOp

    def __init__(self, batch: int, heads: int, heads_kv: int, seq_len: int, dim: int,
                 is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    @property
    def total_flops(self) -> float:
        flops_per_matmul = 2.0 * self.batch * self.heads * self.seq_len * self.seq_len * self.dim
        flops = flops_per_matmul * 5
        return flops / 2 if self.is_causal else flops

    @property
    def total_memory(self) -> int:
        return self.batch * (3 * self.heads +
                             4 * self.heads_kv) * self.seq_len * self.dim * self.dtype.itemsize

    def gen_inputs(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        K = torch.randn(
            self.batch,
            self.seq_len,
            self.heads_kv,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        V = torch.randn(
            self.batch,
            self.seq_len,
            self.heads_kv,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        dO = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device='cuda')

        fwd_op = GroupQueryAttentionFwdOp(self.batch, self.heads, self.heads_kv, self.seq_len,
                                          self.dim, self.is_causal, self.dtype)
        with torch.no_grad():
            O, lse = fwd_op(Q, K, V)

        return Q, K, V, O, dO, lse

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                    do: torch.Tensor,
                    lse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(do)
        return q.grad, k.grad, v.grad

    def baseline_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                         do: torch.Tensor,
                         lse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        softmax_scale = q.shape[-1]**(-0.5)

        dQ = torch.empty_like(q)
        dK = torch.empty_like(k)
        dV = torch.empty_like(v)
        dQ, dK, dV, _ = flash_attn_interface._flash_attn_backward(do, q, k, v, o, lse, None, None,
                                                                  None, None, None, None, dQ, dK,
                                                                  dV, softmax_scale, self.is_causal)
        return dQ, dK, dV

    def baseline_profile(self,
                         *inputs: Any,
                         warmup: int = 100,
                         rep: int = 10,
                         device: str = "cuda:0") -> Any:

        print("===== Profiling GQA FA3 backend =====")
        return super().baseline_profile(
            self.baseline_program, *inputs, backend="FA3", warmup=warmup, rep=rep, device=device)


class GroupQueryAttentionBenchmark(Benchmark):

    def __init__(self,
                 batch: int,
                 heads: int,
                 heads_kv: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 grad: bool = True) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.grad = grad

        self.gqa_fwd_bench = GroupQueryAttentionFwdBenchmark(batch, heads, heads_kv, seq_len, dim,
                                                             is_causal, dtype)
        self.gqa_bwd_bench = GroupQueryAttentionBwdBenchmark(batch, heads, heads_kv, seq_len, dim,
                                                             is_causal, dtype)

    @property
    def total_flops(self) -> float:
        return self.gqa_fwd_bench.total_flops + self.gqa_bwd_bench.total_flops

    @property
    def total_memory(self) -> int:
        return self.gqa_fwd_bench.total_memory + self.gqa_bwd_bench.total_memory

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.grad:
            Q, K, V, _, _, _ = self.gqa_bwd_bench.gen_inputs()
            return Q, K, V
        else:
            return self.gqa_fwd_bench.gen_inputs()

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Any:

        output = self.gqa_fwd_bench.ref_program(q, k, v)[0]
        if not self.grad:
            return output
        else:
            loss = output.sum()
            loss.backward()
            return output, q.grad, k.grad, v.grad
