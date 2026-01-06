import torch
from top import MLAKernel

device = "cuda"
dtype = torch.float16

batch = 128
heads = 64
kv_heads = 1
kv_ctx = 8192
dim = 512
pe_dim = 64

# Query input: [batch, heads, dim]
q = torch.randn(batch, heads, dim, device=device, dtype=dtype)

# Query positional encoding: [batch, heads, pe_dim]
q_pe = torch.randn(batch, heads, pe_dim, device=device, dtype=dtype)

# KV cache input: [batch, kv_ctx, kv_heads, dim]
kv = torch.randn(batch, kv_ctx, kv_heads, dim, device=device, dtype=dtype)

# KV positional encoding: [batch, kv_ctx, kv_heads, pe_dim]
k_pe = torch.randn(batch, kv_ctx, kv_heads, pe_dim, device=device, dtype=dtype)

# Use MLA kernel
block_N = 64
block_H = 64
num_split = 1

mla = MLAKernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim, block_N, block_H, num_split)

out = mla(q, q_pe, kv, k_pe)
