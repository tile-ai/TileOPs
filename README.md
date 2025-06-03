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

# Query input: [batch, heads, seq_len, dim]
q = torch.randn(batch, heads, 1, dim, device=device, dtype=dtype)

# Query positional encoding: [batch, heads, 1, pe_dim]
q_pe = torch.randn(batch, heads, 1, pe_dim, device=device, dtype=dtype)

# KV cache input: [batch, kv_heads, kv_ctx, dim]
kv = torch.randn(batch, kv_heads, kv_ctx, dim, device=device, dtype=dtype)

# KV positional encoding: [batch, kv_heads, kv_ctx, pe_dim]
k_pe = torch.randn(batch, kv_heads, kv_ctx, pe_dim, device=device, dtype=dtype)

# Use MLA kernel
mla = MLA_kernel()

out = mla(q, q_pe, kv, k_pe)
```

## Acknowledgments