from typing import Optional, Tuple, Union

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.benchmark import Benchmark
from top.ops import MultiHeadAttentionBwdOp, MultiHeadAttentionFwdOp


class MultiHeadAttentionFwdBenchmark(Benchmark):

    op_type = MultiHeadAttentionFwdOp

    def __init__(self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool,
                 dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
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
        return 4 * self.batch * self.heads * self.seq_len * self.dim * self.dtype.itemsize

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        k = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        v = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        return q, k, v

    def ref_program(self, q: torch.Tensor, k: torch.Tensor,
                    v: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output, None  # do not check lse

    def baseline_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        try:
            import flash_attn_interface
        except ImportError as e:
            raise ImportError(
                "flash-attn-3 not installed. To run baseline, install flash-attn-3") from e

        return flash_attn_interface.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=None,  # use default 1 / sqrt(head_dim)
            causal=self.is_causal,
        )

    def baseline_profile(self,
                         *inputs: Union[torch.Tensor, Tuple],
                         warmup: int = 100,
                         rep: int = 100,
                         device: str = "cuda:0") -> None:

        print("===== Profiling MHA FA3 backend =====")
        return super().baseline_profile(
            self.baseline_program, *inputs, backend="FA3", warmup=warmup, rep=rep, device=device)


class MultiHeadAttentionBwdBenchmark(Benchmark):

    op_type = MultiHeadAttentionBwdOp

    def __init__(self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool,
                 dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
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
        return 7 * self.batch * self.heads * self.seq_len * self.dim * self.dtype.itemsize

    def gen_inputs(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        k = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        v = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        grad_output = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device='cuda')

        fwd_op = MultiHeadAttentionFwdOp(self.batch, self.heads, self.seq_len, self.dim,
                                         self.is_causal, self.dtype)
        with torch.no_grad():
            o, lse = fwd_op(q, k, v)

        return q, k, v, o, grad_output, lse

    def ref_program(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            o: torch.Tensor,  # noqa: U100
            grad_output: torch.Tensor,
            lse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: U100
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal)
        output = output_bhsd.transpose(1, 2).contiguous()

        # from IPython import embed; embed()
        output.backward(grad_output)
        return q.grad, k.grad, v.grad

    def baseline_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor,
                         grad_output: torch.Tensor,
                         lse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            import flash_attn_interface
        except ImportError as e:
            raise ImportError(
                "flash-attn-3 not installed. To run baseline, install flash-attn-3") from e

        softmax_scale = q.shape[-1]**(-0.5)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dq, dk, dv, _ = flash_attn_interface._flash_attn_backward(grad_output, q, k, v, o, lse,
                                                                  None, None, None, None, None,
                                                                  None, dq, dk, dv, softmax_scale,
                                                                  self.is_causal)
        return dq, dk, dv

    def baseline_profile(self,
                         *inputs: Union[torch.Tensor, Tuple],
                         warmup: int = 100,
                         rep: int = 100,
                         device: str = "cuda:0") -> None:

        print("===== Profiling MHA FA3 backend =====")
        return super().baseline_profile(
            self.baseline_program, *inputs, backend="FA3", warmup=warmup, rep=rep, device=device)


class MultiHeadAttentionBenchmark(Benchmark):

    def __init__(self,
                 batch: int,
                 heads: int,
                 seq_len: int,
                 dim: int,
                 is_causal: bool,
                 dtype: torch.dtype,
                 grad: bool = True) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.grad = grad

        self.mha_fwd_bench = MultiHeadAttentionFwdBenchmark(batch, heads, seq_len, dim, is_causal,
                                                            dtype)
        self.mha_bwd_bench = MultiHeadAttentionBwdBenchmark(batch, heads, seq_len, dim, is_causal,
                                                            dtype)

    @property
    def total_flops(self) -> float:
        return self.mha_fwd_bench.total_flops + self.mha_bwd_bench.total_flops

    @property
    def total_memory(self) -> int:
        return self.mha_fwd_bench.total_memory + self.mha_bwd_bench.total_memory

    def gen_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=self.grad)
        k = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=self.grad)
        v = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=self.grad)

        return q, k, v

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal)
        output = output_bhsd.transpose(1, 2).contiguous()

        if not self.grad:
            return output

        loss = output.sum()
        loss.backward()
        return output, q.grad, k.grad, v.grad
