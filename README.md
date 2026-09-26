<div align="center">
  <img src="https://raw.githubusercontent.com/tile-ai/TileOPs/main/assets/logo.png" width="360"/>

<h3>Spec-driven LLM operators across backends — built by agents</h3>

<p>The spec is the source; kernels are derived from it and judged against it.</p>

<p>
    <a href="https://tile-ai.github.io/TileOPs.github.io/manifest/"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-implemented.json" alt="Spec coverage"></a>
    <a href="https://tile-ai.github.io/TileOPs.github.io/benchmarks/"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-benchmark.json" alt="Bench coverage"></a>
    <a href="https://github.com/tile-ai/TileFoundry"><img src="https://img.shields.io/github/issues-search/tile-ai/TileOPs?query=is%3Apr%20is%3Amerged%20label%3Afoundry&label=Forged%20by%20TileFoundry&color=0891b2&logo=github" alt="Kernels forged by TileFoundry"></a>
    <!-- <a href="https://pypi.org/project/src/tileops/"><img src="https://img.shields.io/badge/PyPI-tileops-1E90FF" alt="PyPI version"></a> -->
  </p>

<p>
    <a href="#quick-start"><b>Quick Start</b></a> ·
    <a href="#built-for-agents"><b>Why it's different</b></a> ·
    <a href="#how-it-works"><b>How it works</b></a> ·
    <a href="#installation"><b>Installation</b></a> ·
    <a href="https://tile-ai.github.io/TileOPs.github.io/benchmarks/"><b>Benchmarks</b></a> ·
    <a href="https://tile-ai.github.io/TileOPs.github.io/"><b>Docs</b></a>
  </p>
</div>

## Quick Start

```python
import torch
from tileops.gemm import GemmFwdOp

gemm = GemmFwdOp()  # shapes and dtype are inferred at call time

a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 512, device="cuda", dtype=torch.float16)

d = gemm(a, b)  # equals a @ b.T
```

Operators autotune when built with `tune=True`, are CUDA-Graph compatible, and each declares
whether it supports `torch.compile(fullgraph=True)`.

## Built for agents

An implementation can be regenerated from its spec; a spec cannot be recovered from an
implementation. The project is organised around the spec rather than around the kernels:

- **The spec is self-contained.** Generation reads it and nothing else, so every constraint on
  the implementation is declared rather than assumed.
- **Acceptance is decidable.** Correctness settles against a declared reference, performance
  against a modelled bound — neither is a judgement call.
- **The operator/kernel split is enforced.** The boundary is checked rather than agreed, because
  an unenforced convention does not survive automated edits.
- **Conformance is validated at every stage.** Spec, generated code, tests and benchmarks each
  answer to a validator, so an operator is certified as it is produced rather than reviewed once
  at the end.

## How it works

Each operator is declared in [`src/tileops/manifest/`](src/tileops/manifest/) before it is implemented.
The entry drives code generation, testing, and benchmarking:

```yaml
GemmFwdOp:
  ref_api: "torch.matmul"
  signature:
    types: {Mat: ...}            # a matrix stored as written, or transposed
    forall: {M: Dim, N: Dim, K: Dim, T: "DType[float16 | bfloat16]"}
    params: {trans_a: {type: bool, default: false}, trans_b: {type: bool, default: true}}
    inputs: {a: {dtype: T, shape: "Mat[trans_a, M, K]"}, b: {dtype: T, shape: "Mat[trans_b, K, N]"}}
    outputs: {d: {dtype: T, shape: "[M, N]"}}
  workloads: [{M: 1024, N: 1024, K: 1024, dtype_cases: [{T: float16}, {T: bfloat16}], label: square-1k}]
  roofline: {flops: "2 * M * N * K"}
```

| Field       | Role                                                                          |
| ----------- | ----------------------------------------------------------------------------- |
| `ref_api`   | The API the operator follows semantically, when it has one.                   |
| `signature` | Tensor types over named indices; the call checks are generated from it.       |
| `workloads` | The calls the manifest contract cases and the benchmarks take.                |
| `roofline`  | Performance model. Efficiency is achieved throughput over the modelled bound. |

Three things are derived from the entry rather than written per operator:

- **Call checks.** Every call is checked against the signature before it dispatches: shapes
  unify to the named indices, dtypes and refinements hold, and the output shapes follow from the
  same types.
- **Roofline.** `eval_roofline()` evaluates the formula on the checked call; the byte count
  follows from the signature unless the entry writes one.
- **Workloads.** Each row instantiates to a concrete call, with metadata tensors drawn from
  declared generators. The contract tests and the benchmarks run exactly these calls, and each
  call's case id keys its nightly history.

A validator checks every entry against its implementation in CI, so the declaration and the
code stay in step.

The implementation is split in two layers. **L2**, the Python entry point, owns the
caller-facing contract: validation, dtype casting, and memory layout. **L1**, the TileLang
kernel, owns the GPU implementation. [architecture.md](docs/design/architecture.md#two-layer-separation-m2) defines the
boundary between them.

## Installation

TileOPs installs from source; a PyPI release lands with the first stable version. A
CUDA-capable GPU is required.

**Prerequisites**

- Python >= 3.10 (CI validates 3.12)
- PyTorch >= 2.1, < 2.14 (CI validates 2.13)
- CUDA Toolkit 13.2
- A GPU of compute capability 9.0 (SM90)
- [TileLang](https://github.com/tile-ai/tilelang) >= 0.1.9, < 0.2.0 (CI validates 0.1.11 at a
  pinned main snapshot — see [development.md](docs/development.md#dev-docker-image))

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
pip install -e '.[dev]' -c constraints.txt   # constraints.txt pins what CI validates
pre-commit install

python -m pytest -q tests -m smoke           # verify; requires a CUDA GPU
```

A prebuilt Docker image carries the whole stack and is the environment CI runs in — see
[development.md](docs/development.md#dev-docker-image), along with test tiers, benchmarks, and
build troubleshooting.

## Documentation

|                                                        |                                                                 |
| ------------------------------------------------------ | --------------------------------------------------------------- |
| [CONTRIBUTING.md](CONTRIBUTING.md)                     | Naming, PR shape, what a review checks                          |
| [development.md](docs/development.md)                  | Build, test, benchmark, dev image                               |
| [architecture.md](docs/design/architecture.md)         | Module map and the agent production loop                        |
| [manifest.md](docs/design/manifest.md)                 | The spec format every operator starts from                      |
| [ops-design.md](docs/design/ops-design.md)             | Adding an operator, step by step                                |
| [roofline.md](docs/design/roofline.md)                 | How performance is scored against Speed-of-Light                |
| [layer-boundaries.md](docs/design/layer-boundaries.md) | What each layer owns, and the interfaces layers compose through |

The rendered site carries what this table cannot: the
[API reference](https://tile-ai.github.io/TileOPs.github.io/api/) generated from the operator
signatures, and the [performance tables](https://tile-ai.github.io/TileOPs.github.io/benchmarks/)
from the nightly run, each operator against the tuned libraries it competes with.

## Contributing

Operators are added through the loop above — start from [ops-design.md](docs/design/ops-design.md),
which walks the path from a manifest entry to a merged kernel.

## License

TileOPs is released under the [MIT License](LICENSE).
