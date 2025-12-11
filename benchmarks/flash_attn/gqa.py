from benchmarks.benchmark import Benchmark
from top.ops import gqa_fwd, gqa_bwd
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


class gqa_fwd_benchmark(Benchmark):

    op_type = gqa_fwd

    def __init__(self, batch, heads, heads_kv, seq_len, dim, is_causal, dtype):
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    @property
    def total_flops(self):
        flops_per_matmul = 2.0 * self.batch * self.heads * self.seq_len * self.seq_len * self.dim
        flops = flops_per_matmul * 2
        return flops / 2 if self.is_causal else flops

    @property
    def total_memory(self):
        return 2 * self.batch * self.seq_len * self.dim * (self.heads +
                                                           self.heads_kv) * self.dtype.itemsize

    def gen_inputs(self):
        Q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len, self.heads_kv, self.dim, device='cuda', dtype=self.dtype)
        V = torch.randn(
            self.batch, self.seq_len, self.heads_kv, self.dim, device='cuda', dtype=self.dtype)
        return Q, K, V

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        q_bhsd = Q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = K.transpose(1, 2)
        v_bhsd = V.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output, None  # do not check lse


class gqa_bwd_benchmark(Benchmark):

    op_type = gqa_bwd

    def __init__(self, batch, heads, heads_kv, seq_len, dim, is_causal, dtype):
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    @property
    def total_flops(self):
        flops_per_matmul = 2.0 * self.batch * self.heads * self.seq_len * self.seq_len * self.dim
        flops = flops_per_matmul * 5
        return flops / 2 if self.is_causal else flops

    @property
    def total_memory(self):
        return self.batch * (3 * self.heads +
                             4 * self.heads_kv) * self.seq_len * self.dim * self.dtype.itemsize

    def gen_inputs(self):
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
                    dO: torch.Tensor, lse: torch.Tensor):
        dim = Q.size(-1)
        groups = self.heads // self.heads_kv
        # Expand K and V to match Q's head dimension for computation
        K_expanded = K.repeat_interleave(groups, dim=2)
        V_expanded = V.repeat_interleave(groups, dim=2)
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K_expanded)
        scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
        if self.is_causal:
            seq_len = Q.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V_expanded)

        output.backward(dO)
        return Q.grad, K.grad, V.grad


class gqa_benchmark(Benchmark):

    def __init__(self, batch, heads, heads_kv, seq_len, dim, is_causal, dtype, grad=True):
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.grad = grad

        self.gqa_fwd_bench = gqa_fwd_benchmark(batch, heads, heads_kv, seq_len, dim, is_causal,
                                               dtype)
        self.gqa_bwd_bench = gqa_bwd_benchmark(batch, heads, heads_kv, seq_len, dim, is_causal,
                                               dtype)

    @property
    def total_flops(self):
        return self.gqa_fwd_bench.total_flops + self.gqa_bwd_bench.total_flops

    @property
    def total_memory(self):
        return self.gqa_fwd_bench.total_memory + self.gqa_bwd_bench.total_memory

    def gen_inputs(self):
        if self.grad:
            Q, K, V, _, _, _ = self.gqa_bwd_bench.gen_inputs()
            return Q, K, V
        else:
            return self.gqa_fwd_bench.gen_inputs()

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):

        output = self.gqa_fwd_bench.ref_program(Q, K, V)[0]
        if not self.grad:
            return output
        else:
            loss = output.sum()
            loss.backward()
            return output, Q.grad, K.grad, V.grad
