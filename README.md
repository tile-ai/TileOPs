# TileOPs (TOP)

**TileOPs (TOP)** is a high-performance machine learning operator collections built on top of [TileLang](https://github.com/tile-ai/tilelang). It offers efficient, modular, and composable implementations optimized for AI workloads.

Note: TileOPs is still under rapid development.

---

![DeepSeek-V3.2-Exp DeepSeek Sparse Attention (DSA) performance on H800 SXM](https://raw.githubusercontent.com/tile-ai/TileOPs/main/docs/figures/sparse_mla_perf.png)
*DeepSeek-V3.2-Exp DeepSeek Sparse Attention (DSA) performance on H800 SXM*

## 📦 Installation

### Requirements

- Python 3.8+
- PyTorch >= 2.1
- GLIBCXX_3.4.32
- [TileLang](https://github.com/tile-ai/tilelang)

### Method 1: Install with Pip

```bash
pip install tileops
```

### Method 2: Install from source (editable mode for development)

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
pip install -e '.[dev]' -v # remove -e option if you don't want to install in editable mode, -v for verbose output
```

## 🚀 Quick Usage

### Sparse MLA

```python
import torch
from top import SparseMLAKernel

batch_size = 1
seq_len = 1024
seq_len_kv = 2048
q_start_index_s = 1024
n_heads = 128
head_dim = 512
tail_dim = 64
topk = 2048
kv_stride = 1
kv_group = 1
sm_scale = None

sparse_mla = SparseMLAKernel(
    batch=batch_size,
    seq_len=seq_len,
    seq_len_kv=seq_len_kv,
    q_start_index_s=q_start_index_s,
    heads=n_heads,
    dim=head_dim,
    tail_dim=tail_dim,
    topk=topk,
    kv_stride=kv_stride,
    kv_group=kv_group,
    sm_scale=sm_scale,
    is_casual=True,
    dtype=torch.bfloat16,
    device='cuda',
)

# Evaluate the Sparse MLA kernel performance
sparse_mla.check()
latency = sparse_mla.profile()
print(f"Latency: {latency:.4f} ms")
print(f'fwd tflops = ',
        (batch_size * seq_len * (head_dim + tail_dim + head_dim) * topk * 2 * n_heads) / (latency * 1e-3) / 1e12)
```

### MLA

```python
import torch
import top
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
```
## Design

### Hierarchical Concepts
There are 4 hierarchical key concepts in TileOPs, each layer stands for a different level of abstraction and is built from or delegates to the next lower level.

- **Layer**: **Op**s that are equipped with runtime elements (e.g. Weights, Buffers, etc.) 
- **Op**: defines computation and is implemented **Function**s for forward and backward.
- **Function**: determines the implementation for a given shape and hardware, dispatching to the correct **Kernel** and providing unit test and benchmark.
- **Kernel**: Tilelang-based kernels for hardware-specific logic.

## Acknowledgments
