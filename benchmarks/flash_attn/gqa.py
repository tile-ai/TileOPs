from benchmarks.benchmark import Benchmark
from top.ops import gqa_fwd, gqa_bwd
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Tuple, Any, Optional
import flash_attn_interface


class GroupQueryAttentionFwdBenchmark(Benchmark):

    op_type = gqa_fwd

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

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor,
                    V: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q_bhsd = Q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = K.transpose(1, 2)
        v_bhsd = V.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output, None  # do not check lse

    def baseline_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:

        out = flash_attn_interface.flash_attn_func(
            Q,
            K,
            V,
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

    op_type = gqa_bwd

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

        fwd_op = gqa_fwd(self.batch, self.heads, self.heads_kv, self.seq_len, self.dim,
                         self.is_causal, self.dtype)
        with torch.no_grad():
            O, lse = fwd_op(Q, K, V)

        return Q, K, V, O, dO, lse

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
                    dO: torch.Tensor,
                    lse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bhsd = Q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = K.transpose(1, 2)
        v_bhsd = V.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(dO)
        return Q.grad, K.grad, V.grad

    def baseline_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
                         dO: torch.Tensor,
                         lse: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        softmax_scale = Q.shape[-1]**(-0.5)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        dQ, dK, dV, _ = flash_attn_interface._flash_attn_backward(dO, Q, K, V, O, lse, None, None,
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

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> Any:

        output = self.gqa_fwd_bench.ref_program(Q, K, V)[0]
        if not self.grad:
            return output
        else:
            loss = output.sum()
            loss.backward()
            return output, Q.grad, K.grad, V.grad
