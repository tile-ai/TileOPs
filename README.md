<div align="center">
  <img src="https://raw.githubusercontent.com/tile-ai/TileOPs/main/assets/logo.png" width="350"/>
  <h1>TileOPs</h1>
  <p><strong>GPU operator library for LLMs, built on <a href="https://github.com/tile-ai/tilelang">TileLang</a></strong></p>
  <!-- <p>
    <a href="https://pypi.org/project/tileops/"><img src="https://img.shields.io/badge/PyPI-tileops-1E90FF" alt="PyPI version" height="20"></a>
  </p> -->
  <p>
    <a href="#installation"><b>Installation</b></a> |
    <a href="#quick-start"><b>Quick Start</b></a> |
    <a href="#operator-families"><b>Operators</b></a> |
    <a href="#documentation"><b>Docs</b></a>
  </p>
</div>

> **Status**: TileOPs is under active development. APIs may change.

## Overview

TileOPs provides ~160 GPU operators across the op families listed below, targeting LLM training and inference workloads. All operators follow a two-layer design:

- **Op** (L2) — stateless Python entry point. Handles validation, dtype casting, and memory layout. Compatible with CUDA-Graph and `torch.compile`.
- **Kernel** (L1) — TileLang GPU implementation with hardware-specific optimizations (Ampere, Hopper).

Key properties:

- **Auto-tuning** — built-in search over tile sizes, pipelines, and scheduling parameters
- **Lightweight** — depends only on TileLang, PyTorch, and einops
- **Reference for TileLang** — demonstrates tiling strategies, memory hierarchy usage, and warp-/block-level coordination

## Operator Families

| Family               | Examples                                                          |
| -------------------- | ----------------------------------------------------------------- |
| **Attention**        | MHA, GQA (FA2/FA3), sliding-window, varlen, paged KV-cache decode |
| **DeepSeek**         | MLA, NSA, DSA                                                     |
| **Linear Attention** | GLA, DeltaNet, Gated DeltaNet (fwd/bwd/decode)                    |
| **State Space**      | Mamba, SSD (chunk scan, chunk state, state passing)               |
| **Normalization**    | RMSNorm, LayerNorm, GroupNorm, BatchNorm, fused variants          |
| **Reduction**        | Sum, Mean, Softmax, LogSumExp, Argmax, cumulative ops             |
| **Elementwise**      | Unary, binary, fused gated (SiLU-gate, GeGLU, etc.)               |
| **MoE**              | Qwen3MoE, permute/unpermute, TopK routing                         |
| **GEMM**             | FP16/BF16 matrix multiply                                         |
| **Other**            | FFT, Engram, MHC                                                  |

## Installation

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.1
- CUDA Toolkit
- NVIDIA GPU: **Ampere** (SM_80, SM_86) or **Hopper** (SM_90)
- [TileLang](https://github.com/tile-ai/tilelang) == 0.1.8

### From PyPI

```bash
pip install tileops
```

### From source

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
make install    # dev dependencies + pre-commit hooks
```

> [!NOTE]
> If CUDA and TileLang are already installed system-wide and you encounter build issues:
> `PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev]' -v && pre-commit install`

Verify:

```bash
python -m pytest tests/ -q    # requires a CUDA GPU
```

## Quick Start

```python
import torch
from tileops.ops import GemmOp

M, N, K = 1024, 1024, 512
dtype = torch.float16

gemm = GemmOp(M, N, K, dtype=dtype)

A = torch.randn(M, K, device="cuda", dtype=dtype)
B = torch.randn(K, N, device="cuda", dtype=dtype)

C = gemm(A, B)
```

## Documentation

| Document                                | Description                                          |
| --------------------------------------- | ---------------------------------------------------- |
| [architecture.md](docs/architecture.md) | System modules, data flow, directory structure       |
| [ops-design.md](docs/ops-design.md)     | Op/Kernel interface design and inheritance hierarchy |
| [manifest.md](docs/manifest.md)         | `ops_manifest.yaml` spec format                      |
| [roofline.md](docs/roofline.md)         | Performance evaluation methodology                   |
| [testing.md](docs/testing.md)           | Test/benchmark framework and tolerances              |
| [workflow.md](docs/workflow.md)         | Development workflow, coding standards, PR process   |

## Contributing

See [workflow.md](docs/workflow.md) for branch naming, commit conventions, and the PR process.

## License

TileOPs is released under the [Apache 2.0 License](LICENSE).
