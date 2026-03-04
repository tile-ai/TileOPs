<div align="center">
  <img src="https://raw.githubusercontent.com/tile-ai/TileOPs/main/assets/logo.png" width="350"/>
  <h1>TileOPs: Operator Library for LLMs Built on TileLang</h1>
  <!-- <p>
    <a href="https://pypi.org/project/tileops/"><img src="https://img.shields.io/badge/PyPI-tileops-1E90FF" alt="PyPI version" height="20"></a>
  </p> -->
  <p>
    <a href="#-install-with-pip"><b>Installation</b></a> |
    <a href="#-quick-start"><b>Getting Started</b></a> |
    <a href="#documents"><b>Documents</b></a>
  </p>
</div>

**TileOPs** is a high-performance operator library for large language models (LLMs) built on **[TileLang](https://github.com/tile-ai/tilelang)**. It offers efficient, modular, and composable implementations for AI workloads, especially for LLMs.

> ⚠️ **Status**: TileOPs is under active and rapid development. APIs and features may change.

What TileOPs is for:

- **Out-of-the-box Operator Library**: A growing collection of production-ready operators commonly used in LLM workloads, designed with clear abstractions and modular building blocks. These operators can be used directly or easily extended for custom research and system integration.
- **Efficient Attention Kernels for LLMs**: Highly optimized attention implementations, including MHA/GQA (implemented FA2 on Ampere-like GPUs and FA3 on Hopper), DeepSeek-MLA, and DeepSeek-DSA.
- **Reference Implementation for TileLang**: TileOPs acts as a **canonical reference implementation** for writing performant and maintainable kernels in **TileLang**. It demonstrates best practices in tiling strategies, memory hierarchy utilization, and warp-/block-level coordination, making it a practical learning resource for compiler and kernel developers.

The core features of TileOPs include:

- **Auto-Tuning**: Built-in auto-tuning support to explore tile sizes, pipelines, and scheduling parameters, enabling kernels to adapt efficiently to different GPU architectures and workload characteristics with minimal manual effort.
- **CUDA-Graph and torch.compile Compatibility**: TileOPs APIs are fully compatible with CUDA-Graph capture and PyTorch `torch.compile`, allowing seamless integration into modern training and inference pipelines with reduced launch overhead and improved end-to-end performance.
- **Lightweight Dependencies**: TileOPs depends only on TileLang, PyTorch, and einops, keeping the software stack minimal and easy to integrate.

## 📦 Install with pip

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.1
- CUDA Toolkit (required — this is a GPU kernel project)
- A CUDA-capable NVIDIA GPU
  - Tested architectures: **Ampere** (SM_80, SM_86) and **Hopper** (SM_90)
  - Other architectures may work but are not tested
- [TileLang](https://github.com/tile-ai/tilelang) == 0.1.8

### Method 1: Install from PyPI

```bash
pip install tileops
```

### Method 2: Install from source (for development)

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
pip install -e '.[dev]' -v
```

> [!NOTE]
> If you have CUDA and TileLang already installed system-wide and encounter build issues, try:
> `PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev]' -v`
> This disables pip's build isolation so it can find your existing CUDA/TileLang installation.

After installing, set up the pre-commit hooks and verify with a test run:

```bash
pre-commit install          # enables lint checks before each commit
python -m pytest tests/ -q  # run the test suite (requires a CUDA GPU)
```

## 🚀 Quick Start

```python
import torch
from tileops.ops import GroupQueryAttentionDecodeWithKVCacheOp

# Define shapes and data type
B, H, G, S_kv, D = 1, 32, 4, 1024, 128  # batch, heads, groups, seq_len, dim
dtype = torch.float16

# Instantiate the op
op = GroupQueryAttentionDecodeWithKVCacheOp(B, H, G, S_kv, D, dtype=dtype)

# Generate inputs
q = torch.randn(B, H, D, device="cuda", dtype=dtype)
k_cache = torch.randn(B, S_kv, G, D, device="cuda", dtype=dtype)
v_cache = torch.randn(B, S_kv, G, D, device="cuda", dtype=dtype)

# Run the operator
output = op(q, k_cache, v_cache)
```

## Documents

### Hierarchical APIs

TileOPs is structured around two hierarchical key concepts, each representing a distinct level of abstraction. Higher-level components are composed from, or delegate execution to, the next lower level.

- **Op**: determines the implementation for a given shape and hardware, dispatching to the correct **Kernel** and providing unit test and benchmark. Ops are fully compatible with CUDA-Graph capture and `torch.compile`.
- **Kernel**: TileLang-based kernels with hardware-specific optimizations.
