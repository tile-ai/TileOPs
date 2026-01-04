<div align="center">
  <!-- <img src="assets/logo.png" width="350"/> -->
  <h1>TileOPs (TOP): Operator Library for LLMs Built on TileLang</h1>
  <!-- <p>
    <a href="https://pypi.org/project/tileops/"><img src="https://img.shields.io/badge/PyPI-tileops-1E90FF" alt="PyPI version" height="20"></a>
  </p> -->
  <p>
    <a href="#-install-with-pip"><b>Installation</b></a> |
    <a href="#-quick-start"><b>Getting Started</b></a> |
    <a href="#documents"><b>Documents</b></a>
  </p>
</div>

**TileOPs (TOP)** is a high-performance operator library for large language models (LLMs) built on **[TileLang](https://github.com/tile-ai/tilelang)**. It offers efficient, modular, and composable implementations for AI workloads, especially for LLMs.

**Note: TileOPs is still under rapid development.**

What TileOPs is for:

* **Out-of-the-box Operator Library**: A growing collection of production-ready operators commonly used in LLM workloads, designed with clear abstractions and modular building blocks. These operators can be used directly or easily extended for custom research and system integration.
* **Efficient Attention Kernels for LLMs**: Highly optimized attention implementations, including MHA/GQA (implemented FA2 on Ampere-like GPUs and FA3 on Hopper), DeepSeek-MLA, and DeepSeek-DSA.
* **Reference Implementation and Learning Ground of TileLang**: TileOPs serves as a canonical example of how to write performant, maintainable kernels using TileLang. It demonstrates best practices in tiling strategies, memory hierarchy utilization, and warp-/block-level coordination, making it a practical learning resource for compiler and kernel developers.

The core features of TileOPs include:

* **Auto-Tuning**: Built-in auto-tuning support to explore tile sizes, pipelines, and scheduling parameters, enabling kernels to adapt efficiently to different GPU architectures and workload characteristics with minimal manual effort.
* **CUDA-Graph and torch.compile Compatibility**: TileOPs APIs are fully compatible with CUDA-Graph capture and PyTorch ``torch.compile``, allowing seamless integration into modern training and inference pipelines with reduced launch overhead and improved end-to-end performance.
<!-- * **torch.autograd Compatibility**: -->
* **Lightweight Dependencies**: TileOPs depends only on TileLang and PyTorch, keeping the software stack minimal and easy to integrate.



## Benchmark Summary

TODO

<!-- DeepSeek Sparse Attention Decode

![DeepSeek-V3.2-Exp DeepSeek Sparse Attention (DSA) performance on H800 SXM](https://raw.githubusercontent.com/tile-ai/TileOPs/main/docs/figures/sparse_mla_perf.png)
*DeepSeek-V3.2-Exp DeepSeek Sparse Attention (DSA) performance on H800 SXM* -->



## Support Matrix

TODO

<!-- **Legend**: ✔ Optimized in tests / ● Functional support / ✗ Not supported yet / N/A Not Applicable

| **Operator** | **FWD** | **BWD** | **Data Type** | **Tested Platform** |
|:------------:|:-------:|:-------:|:-------------:|:-------------------:|
| DeepSeek Sparse Attention (DSA) Decode | ✔ | N/A | BF16/FP16 | Hopper(SM_90) |
| DeepSeek Multi-Head Latent Attention (MLA) Decode | ✔ | N/A | BF16/FP16 | Hopper(SM_90) |
| Multi-Head Attention | ✔ | ● | BF16/FP16 | Hopper(SM_90)/Ampere(SM_80/SM_86) |
| Group Query Attention | ✔ | ● | BF16/FP16 | Hopper(SM_90)/Ampere(SM_80/SM_86) |
| Multi-Head Attention Decode | ✔ | N/A | BF16/FP16 | Hopper(SM_90)/Ampere(SM_80/SM_86) |
| Group Query Attention Decode | ✔ | N/A | BF16/FP16 | Hopper(SM_90)/Ampere(SM_80/SM_86) |
| MatMul | ✔ | ● | BF16/FP16 | Hopper(SM_90)/Ampere(SM_80/SM_86) | -->



## 📦 Install with pip

### Requirements

- Python >= 3.8
- Torch >= 2.1
- [TileLang](https://github.com/tile-ai/tilelang) >= 0.1.7

### Method 1: Install from PyPI

```bash
pip install tileops
```

### Method 2: Install from source (editable mode for development)

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
pip install -e '.[dev]' -v # remove -e option if you don't want to install in editable mode, -v for verbose output
```

## 🚀 Quick Start

TODO

<!-- ### Example: Group-Query Attention Decode

```Python
import torch
from top.functions import gqa_decode_with_kvcache

# Generate inputs
B, H, G, S_kv, D = 1, 32, 4, 1024, 128  # batch, heads, groups, seq_len, dim
dtype = torch.float16
q = torch.randn(B, H, D, device='cuda', dtype=dtype)
k_cache = torch.randn(B, S_kv, G, D, device='cuda', dtype=dtype)
v_cache = torch.randn(B, S_kv, G, D, device='cuda', dtype=dtype)

# Call function, k_cache and v_cache are updated in-place
output = gqa_decode_with_kvcache(q, k_cache, v_cache)
``` -->

## Documents

### Hierarchical APIs

TileOPs is structured around four hierarchical key concepts, each representing a distinct level of abstraction. Higher-level components are composed from, or delegate execution to, the next lower level.

- **Layer**: A high-level, user-facing abstraction analogous to ``torch.nn.Module``. It manages stateful parameters.
- **Function**: A stateless, functional abstraction analogous to ``torch.nn.functional``. Functions are fully compatible with CUDA-Graph capture, ``torch.compile``, and ``torch.autograd``.
- **Op**: determines the implementation for a given shape and hardware, dispatching to the correct **Kernel** and providing unit test and benchmark. Ops are fully compatible with CUDA-Graph capture and ``torch.compile``.
- **Kernel**: Tilelang-based kernels with hardware-specific optimizations.


