# TileAttention (TLA)

**TileAttention (TLA)** is a high-performance attention kernel library built to run on the [TileLang](https://github.com/tile-ai/tilelang) backend. It offers efficient, modular, and composable attention implementations optimized for AI workloads.

---


## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch >= 2.1
- [TileLang](https://github.com/tilelang/tilelang)
- Triton (optional, for selected fast kernels)

### Install (editable mode for development)

```bash
git clone https://github.com/tile-ai/TileAttention
cd TileAttention
git submodule update --init --recursive
pip install -e .
```

## 🚀 Quick Usage

```python
import torch
import tla
from tla import MLA_kernel

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

mla = MLA_kernel(batch, heads, kv_heads, kv_ctx, dim, pe_dim, block_N, block_H, num_split)

out = mla(q, q_pe, kv, k_pe)
```

## Acknowledgments