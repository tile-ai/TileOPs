from benchmarks.benchmark import Benchmark
from top.ops import mha_fwd, mha_bwd
import flash_attn_interface
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


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
        q_bhsd = Q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = K.transpose(1, 2)
        v_bhsd = V.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal)
        output = output_bhsd.transpose(1, 2).contiguous()
        return output, None  # do not check lse

    def baseline_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
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

    def baseline_profile(self, *inputs, warmup=100, rep=100, device="cuda:0"):

        print("===== Profiling MHA FA3 backend =====")
        return super().baseline_profile(
            self.baseline_program, *inputs, backend="FA3", warmup=warmup, rep=rep, device=device)


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
        q_bhsd = Q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = K.transpose(1, 2)
        v_bhsd = V.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal)
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(dO)
        return Q.grad, K.grad, V.grad

    def baseline_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor,
                         dO: torch.Tensor, lse: torch.Tensor):
        softmax_scale = Q.shape[-1]**(-0.5)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        dQ, dK, dV, _ = flash_attn_interface._flash_attn_backward(dO, Q, K, V, O, lse, None, None,
                                                                  None, None, None, None, dQ, dK,
                                                                  dV, softmax_scale, self.is_causal)
        return dQ, dK, dV

    def baseline_profile(self, *inputs, warmup=100, rep=100, device="cuda:0"):

        print("===== Profiling MHA FA3 backend =====")
        return super().baseline_profile(
            self.baseline_program, *inputs, backend="FA3", warmup=warmup, rep=rep, device=device)


class mha_benchmark(Benchmark):

    def __init__(self, batch, heads, seq_len, dim, is_causal, dtype, grad=True):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.grad = grad

        self.mha_fwd_bench = mha_fwd_benchmark(batch, heads, seq_len, dim, is_causal, dtype)
        self.mha_bwd_bench = mha_bwd_benchmark(batch, heads, seq_len, dim, is_causal, dtype)

    @property
    def total_flops(self):
        return self.mha_fwd_bench.total_flops + self.mha_bwd_bench.total_flops

    @property
    def total_memory(self):
        return self.mha_fwd_bench.total_memory + self.mha_bwd_bench.total_memory

    def gen_inputs(self):
        Q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=self.grad)
        K = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=self.grad)
        V = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device='cuda',
            requires_grad=self.grad)

        return Q, K, V

    def ref_program(self,
                    Q: torch.Tensor,
                    K: torch.Tensor,
                    V: torch.Tensor,
                    dO: torch.Tensor = None):
        q_bhsd = Q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = K.transpose(1, 2)
        v_bhsd = V.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal)
        output = output_bhsd.transpose(1, 2).contiguous()

        if not self.grad:
            return output
        else:
            loss = output.sum()
            loss.backward()
            return output, Q.grad, K.grad, V.grad
