<div align="center">
  <img src="https://raw.githubusercontent.com/tile-ai/TileOPs/main/assets/logo.png" width="350"/>
  <h1>TileOPs</h1>
  <p><strong>Spec-driven GPU operator library for LLMs — designed for AI agents to build, evaluate, and optimize</strong></p>
  <p>Built on <a href="https://github.com/tile-ai/tilelang">TileLang</a></p>
  <!-- <p>
    <a href="https://pypi.org/project/tileops/"><img src="https://img.shields.io/badge/PyPI-tileops-1E90FF" alt="PyPI version" height="20"></a>
  </p> -->
  <p>
    <a href="https://github.com/tile-ai/TileOPs/tree/main/tileops/manifest"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-implemented.json" alt="Spec coverage"></a>
    <a href="https://github.com/tile-ai/TileOPs/tree/main/benchmarks"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-benchmark.json" alt="Bench coverage"></a>
  </p>
  <p>
    <a href="#installation"><b>Installation</b></a> |
    <a href="#quick-start"><b>Quick Start</b></a> |
    <a href="#documentation"><b>Docs</b></a>
  </p>
</div>

> **Status**: TileOPs is under active development. APIs may change.

## Overview

TileOPs is a GPU operator library for LLM training and inference, built on [TileLang](https://github.com/tile-ai/tilelang). Beyond providing a growing collection of production-quality operators, TileOPs explores a **spec-driven development model** where AI agents can read declarative operator specifications, generate kernel implementations, and evaluate them against hardware-theoretical performance bounds — with minimal human scaffolding.

### Architecture

Every operator is split into two layers with a strict boundary:

- **Op** (L2) — stateless Python entry point. Handles validation, dtype casting, and memory layout. Compatible with CUDA-Graph and `torch.compile`.
- **Kernel** (L1) — TileLang GPU implementation with hardware-specific optimizations (Ampere, Hopper).

This separation keeps user-facing behavior independent of GPU strategy, allowing agents and developers to modify either layer without side effects on the other.

### Key Properties

- **Spec-driven** — each operator is declared in a machine-readable manifest (`tileops/manifest/`) that specifies signatures, workloads, and roofline formulas, serving as the entry point for both agent code generation and automated validation
- **Roofline-evaluated** — kernel performance is measured against Speed-of-Light hardware bounds, not relative baselines
- **Auto-tuning** — built-in search over tile sizes, pipelines, and scheduling parameters
- **Lightweight** — depends only on TileLang, PyTorch, and einops

## Installation

TileOPs can be installed from PyPI or built from source. A CUDA-capable GPU is required.

### Prerequisites

- Python >= 3.10
- PyTorch >= 2.1
- CUDA Toolkit
- NVIDIA GPU: **Hopper** (SM_90)
- [TileLang](https://github.com/tile-ai/tilelang) — pinned to a `main` commit, built from source (see the note below)

### From source

> [!NOTE]
> Installing from PyPI (`pip install tileops`) is temporarily unavailable: TileOPs
> pins TileLang to a `main` commit (see the note below), and a published package
> cannot carry a git dependency. Build from source until TileLang has a release again.

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
make install    # dev dependencies + pre-commit hooks
```

> [!NOTE]
> If CUDA and TileLang are already installed system-wide and you encounter build issues:
> `PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev]' -v && pre-commit install`

> [!IMPORTANT]
> **TileOPs currently pins TileLang to a `main` commit** because it depends on a fix not
> yet in a TileLang release. While this holds:
>
> - Installing **compiles TileLang from source** — a git commit has no prebuilt wheel, so
>   `pip` builds TileLang and its submodules. A CUDA build toolchain is required and the
>   first build is slow; the pinned commit is cached and reused afterward.
> - **`pip install tileops` from PyPI is unavailable** during this phase: a published
>   package cannot carry a git dependency. Install from source.
> - On import, TileOPs preloads the CUDA driver (`libcuda.so.1`) into the global symbol
>   namespace, working around a TileLang source-build gap where `libtvm_runtime.so` leaves
>   driver symbols (e.g. `cuModuleLoadData`) unresolved. If you `import tilelang` **directly**
>   (not via `tileops`) and hit `undefined symbol: cuModuleLoadData`, run with
>   `LD_PRELOAD=/path/to/libcuda.so.1`.
>
> This reverts to a normal version pin once TileLang cuts a release.

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

Design docs and development guides are in [`docs/`](docs/). The full API reference and performance tables are published at [TileOPs.github.io](https://github.com/tile-ai/TileOPs.github.io).

## Contributing

See [docs/](docs/) for design docs. Branch and commit conventions are in [`.claude/conventions/types.sh`](.claude/conventions/types.sh).

## License

TileOPs is released under the [MIT License](LICENSE).
