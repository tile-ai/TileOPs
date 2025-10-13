import torch
from torch.nn import functional as F
from tilelang.profiler import do_bench
from .function import Function
from top.kernels import mha_fwd_kernel_sm80

class mha_fwd(Function):
    """Layout: BSHD"""

    def __init__(self, batch, heads, seq_len, dim, is_causal):
        #TODO: support s_q != s_kv and more dtypes
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal

        flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
        self.total_flops = 2 * flops_per_matmul
        if is_causal:
            self.total_flops *= 0.5

        # TODO: dispatch to different kernels based on archs and input shapes
        self.kernel = mha_fwd_kernel_sm80(batch, heads, seq_len, dim, is_causal)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        return self.kernel(Q, K, V)

    def ref_program(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
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
        return output

    def gen_inputs(self):
        return (torch.randn([self.batch, self.seq_len, self.heads, self.dim], 
            dtype=torch.float16, device='cuda') for _ in range(3))

    def check(self):
        Q, K, V = self.gen_inputs()
        o, _ = self.forward(Q, K, V)  # lse is only used for bwd
        o_ref = self.ref_program(Q, K, V)
        assert torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2), f'o max err: {(o - o_ref).abs().max()}'
        print("All checks passed.✅")

    def profile(self, warmup=100, rep=100):
        # TODO: support cupti backend for better accuracy (avaiable in tilelang v0.1.7)
        Q, K, V = self.gen_inputs()
        with torch.no_grad():
            tl_latency = do_bench(lambda: self.forward(Q, K, V), warmup=warmup, rep=rep)
        
        print(f"Tilelang latency: {tl_latency:.2f} ms")
        print(f"Tilelang TFlops: {self.total_flops / tl_latency * 1e-9:.2f} TFlops")