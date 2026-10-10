<div align="center">
  <img src="https://raw.githubusercontent.com/tile-ai/TileOPs/main/assets/logo.png" width="360"/>

<h3>Spec-driven LLM operators across backends — by agents, for agents</h3>

<p>A library agents can write, and keep writing.</p>

<p>
    <a href="https://tile-ai.github.io/TileOPs.github.io/manifest/"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-implemented.json" alt="Spec coverage"></a>
    <a href="https://tile-ai.github.io/TileOPs.github.io/benchmarks/"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-benchmark.json" alt="Bench coverage"></a>
    <a href="https://github.com/tile-ai/TileFoundry"><img src="https://img.shields.io/github/issues-search/tile-ai/TileOPs?query=is%3Apr%20is%3Amerged%20label%3Afoundry&label=Forged%20by%20TileFoundry&color=0891b2&logo=github" alt="Kernels forged by TileFoundry"></a>
    <!-- Uncomment with the first release. <a href="https://pypi.org/project/tileops/"><img src="https://img.shields.io/pypi/v/tileops?color=1E90FF" alt="PyPI version"></a> -->
  </p>

<p>
    <a href="#quick-start"><b>Quick Start</b></a> ·
    <a href="#installation"><b>Installation</b></a> ·
    <a href="https://tile-ai.github.io/TileOPs.github.io/benchmarks/"><b>Benchmarks</b></a> ·
    <a href="https://tile-ai.github.io/TileOPs.github.io/"><b>Docs</b></a>
  </p>
</div>

## Built for agents

An implementation can be regenerated from its spec; a spec cannot be recovered from an
implementation. TileOPs is built on that asymmetry. Agents build the whole library, not just
its kernels, so the library has to stay coherent as it grows: a consistent structure, no
drift, no bloat, code that stays maintainable. The design serves three goals:

- **Maintainable.** Every op starts as a self-contained YAML spec, and code generation reads
  the spec and nothing else. Ops in a family share interfaces and rules, and checks enforce
  the boundary between the Python op and the TileLang kernel, because a convention nobody
  enforces does not survive automated edits.
- **Verifiable.** Correctness is checked against the reference implementation the spec names,
  performance against the modelled roofline bound. Both criteria are declared in the spec
  before the code exists, and CI validates the spec, the generated code, the tests and the
  benchmarks.
- **Tunable.** The roofline model reports each kernel's gap to its bound, and nightly
  benchmarks compare each kernel with the fastest other implementation on the same GPU.

## Built with TileFoundry

TileOPs kernels are forged with [TileFoundry](https://github.com/tile-ai/TileFoundry), an agentic
platform for high-performance kernel generation.

## Quick Start

```python
import torch
from tileops.gemm import GemmFwdOp

gemm = GemmFwdOp()  # shapes and dtype are inferred at call time

a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 512, device="cuda", dtype=torch.float16)

d = gemm(a, b)  # equals a @ b.T
```

Operators autotune after `op.autotune()`, are CUDA-Graph compatible, and every op whose
manifest entry has a call-time tensor input and no composition supports
`torch.compile(fullgraph=True)`.

## How it works

Each operator is declared before it is implemented, as an entry in
[`src/tileops/manifest/spec/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest/spec/), one file per family. The entry drives
code generation, testing, and benchmarking. Abridged from `gemm.yaml`:

```yaml
GemmFwdOp:
  ref_api: "torch.matmul"
  family: gemm
  signature:
    forall: {M: Dim, N: Dim, K: Dim, T: "DType[float16 | bfloat16]"}
    inputs: {a: {dtype: T, shape: "[M, K]"}, b: {dtype: T, shape: "[N, K]"}}
    outputs: {d: {dtype: T, shape: "[M, N]"}}
  workloads: [{M: 1024, N: 1024, K: 1024, dtype_cases: [{T: float16}, {T: bfloat16}], label: square-1k}]
  roofline: {flops: "2 * M * N * K"}
```

| Field       | Role                                                                                    |
| ----------- | --------------------------------------------------------------------------------------- |
| `ref_api`   | The API the operator follows semantically, when it has one.                             |
| `signature` | Tensor types over named indices. Every call is checked against it before it dispatches. |
| `workloads` | The concrete calls the contract tests and the nightly benchmarks run.                   |
| `roofline`  | The performance model. `eval_roofline()` evaluates it on each checked call.             |

A validator checks every entry against its implementation in CI, so the declaration and the code
stay in step.

Each operator has two layers: the **Op** (L2), the Python entry point that owns the caller-facing
contract, and the **Kernel** (L1), the TileLang implementation.
[architecture.md](https://github.com/tile-ai/TileOPs/blob/main/docs/design/architecture.md#two-layer-separation-m2) defines the boundary
between them.

## Installation

TileOPs installs from source; a PyPI release lands with the first stable version. A
CUDA-capable GPU is required.

**Prerequisites** — the one combination a release is verified on and declares:

- Python 3.12
- PyTorch 2.13
- CUDA Toolkit 13.2
- A GPU of compute capability 9.0 (SM90), tested on H200
- [TileLang](https://github.com/tile-ai/tilelang) 0.1.12 — see [development.md](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md#dev-docker-image)

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
pip install -e '.[dev]' -c constraints.txt   # constraints.txt pins what CI validates
pre-commit install

python -m pytest -q tests -m smoke           # verify; requires a CUDA GPU
```

A prebuilt Docker image carries the whole stack and is the environment CI runs in — see
[development.md](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md#dev-docker-image), along with test tiers, benchmarks, and
build troubleshooting.

## Documentation

|                                                                                                     |                                                                 |
| --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [CONTRIBUTING.md](https://github.com/tile-ai/TileOPs/blob/main/CONTRIBUTING.md)                     | Naming, PR shape, what a review checks                          |
| [development.md](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md)                  | Build, test, benchmark, dev image                               |
| [architecture.md](https://github.com/tile-ai/TileOPs/blob/main/docs/design/architecture.md)         | Module map and the agent production loop                        |
| [manifest.md](https://github.com/tile-ai/TileOPs/blob/main/docs/design/manifest.md)                 | The spec format every operator starts from                      |
| [ops-design.md](https://github.com/tile-ai/TileOPs/blob/main/docs/design/ops-design.md)             | Adding an operator, step by step                                |
| [roofline.md](https://github.com/tile-ai/TileOPs/blob/main/docs/design/roofline.md)                 | How performance is scored against Speed-of-Light                |
| [layer-boundaries.md](https://github.com/tile-ai/TileOPs/blob/main/docs/design/layer-boundaries.md) | What each layer owns, and the interfaces layers compose through |

The rendered site carries what this table cannot: the
[API reference](https://tile-ai.github.io/TileOPs.github.io/api/) generated from the operator
signatures, and the [performance tables](https://tile-ai.github.io/TileOPs.github.io/benchmarks/)
from the nightly run, each operator against the tuned libraries it competes with.

## Contributing

Operators are added through the loop above — start from [ops-design.md](https://github.com/tile-ai/TileOPs/blob/main/docs/design/ops-design.md),
which walks the path from a manifest entry to a merged kernel.

## License

TileOPs is released under the [MIT License](https://github.com/tile-ai/TileOPs/blob/main/LICENSE).
