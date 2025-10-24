from top.benchmarks.benchmark import Benchmark
from top.ops import mha_fwd, mha_bwd
import torch
from torch.nn import functional as F


class mha_fwd_benchmark(Benchmark):

    op_type = mha_fwd

    def __init__(self, batch, heads, seq_len, dim, is_causal, dtype):
        self.batch = batch
        self.heads = heads
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
        return 4 * self.batch * self.heads * self.seq_len * self.dim * self.dtype.itemsize

    def gen_inputs(self):
        Q = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        K = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        V = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, device='cuda', dtype=self.dtype)
        return Q, K, V

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        dim = Q.size(-1)
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
        scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
        if self.is_causal:
            seq_len = Q.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)
        return output, None  # do not check lse


class mha_bwd_benchmark(Benchmark):

    op_type = mha_bwd

    def __init__(self, batch, heads, seq_len, dim, is_causal, dtype):
        self.batch = batch
        self.heads = heads
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
        return 7 * self.batch * self.heads * self.seq_len * self.dim * self.dtype.itemsize

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
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        V = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=True)
        dO = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device='cuda')

        fwd_op = mha_fwd(self.batch, self.heads, self.seq_len, self.dim, self.is_causal, self.dtype)
        with torch.no_grad():
            O, lse = fwd_op(Q, K, V)

        return Q, K, V, O, dO, lse

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
                    dO: torch.Tensor, lse: torch.Tensor):
        dim = Q.size(-1)
        scores = torch.einsum('bqhd,bkhd->bhqk', Q, K)
        scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
        if self.is_causal:
            seq_len = Q.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V)

        output.backward(dO)
        return Q.grad, K.grad, V.grad
