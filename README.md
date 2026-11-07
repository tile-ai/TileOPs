<p align="center">
  <img src="https://raw.githubusercontent.com/tile-ai/TileOPs/main/assets/readme-banner.png" alt="TileOPs × TileFoundry — From Kernels to a Library: By Agents, for Agents" width="100%">
</p>

<h3 align="center">Spec-driven LLM operators across backends — by agents, for agents</h3>

<p align="center">
  <i>TileOPs defines what each operator must do.<br>
  Coding agents and TileFoundry, an AI compiler built for agents, iterate to grow the library.</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/tileops/"><img src="https://img.shields.io/pypi/v/tileops?color=1E90FF" alt="PyPI version"></a>
  <a href="https://tile-ai.github.io/TileOPs.github.io/user-guide/manifest/"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-implemented.json" alt="Spec coverage"></a>
  <a href="https://tile-ai.github.io/TileOPs.github.io/benchmarks/"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-benchmark.json&label=bench" alt="Bench coverage"></a>
  <a href="https://tile-ai.github.io/TileOPs.github.io/benchmarks/"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs-nightly%2Fsnapshots%2Fnightly-vs-baseline.json&label=vs%20best" alt="Rows ahead of or level with the fastest baseline, from the latest nightly"></a>
  <a href="https://github.com/tile-ai/TileFoundry"><img src="https://img.shields.io/github/issues-search/tile-ai/TileOPs?query=is%3Apr%20is%3Amerged%20label%3Afoundry&label=Forged%20by%20TileFoundry&color=0891b2&logo=github" alt="Kernels forged by TileFoundry"></a>
</p>

<p align="center">
  <a href="#quick-start"><b>Quick Start</b></a> ·
  <a href="https://tile-ai.github.io/TileOPs.github.io/benchmarks/"><b>Benchmarks</b></a> ·
  <a href="https://tile-ai.github.io/TileOPs.github.io/"><b>Docs</b></a> ·
  <a href="https://github.com/tile-ai/TileFoundry"><b>TileFoundry</b></a>
</p>

______________________________________________________________________

## Agent and compiler, working together

| **TileOPs**<br>agents answer to specs                                                                                                                                                     | **[TileFoundry](https://github.com/tile-ai/TileFoundry)**<br>the compiler answers to agents                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Each op starts with a YAML spec. Tests and benchmarks use the reference implementation, workloads and roofline formula declared in the spec to check correctness and measure performance. | A tile-based AI compiler that coding agents use to generate high-performance programs across hardware platforms. |

TileOPs shows how a coding agent and TileFoundry can work together in a loop to build an
operator library. Tests and benchmarks driven by each op's spec report failures and the gap
to the roofline bound. The agent and compiler iterate on that feedback. Kernels that pass the
checks become part of the library, where later work can build on them. This approach supports
a growing, maintainable library rather than a few one-off kernels.

```text
spec ──▶ coding agent ◀──▶ TileFoundry ──▶ kernel
              ▲            (AI compiler)     │
              │                              ▼
              └─ failures, roofline gap ◀── tests + benchmarks
                                             │ pass
                                             ▼
                                      TileOPs library
```

PRs whose kernels were generated with TileFoundry carry the `foundry` label. The badge above
counts merged PRs with this label. These PRs go through the same CI as every other PR.

## Quick Start

```python
import torch
from tileops.gemm import GEMMFwdOp

gemm = GEMMFwdOp()  # shapes and dtype are inferred at call time

a = torch.randn(1024, 512, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 512, device="cuda", dtype=torch.float16)

d = gemm(a, b)  # equals a @ b.T
```

Operators autotune after `op.request_tune()`. CUDA Graph support on H200 is validated,
after warmup, for the calls in the [graph test group](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md#cuda-graph-validation).
Every op whose manifest entry has a call-time tensor input and no composition supports
`torch.compile(fullgraph=True)`.

### Release Docker image

The release image contains TileOPs 0.1.0, [TileFoundry](https://github.com/tile-ai/TileFoundry)
0.0.3 and the benchmark baselines, on CUDA 13.2, PyTorch 2.13 and TileLang 0.1.12.

```bash
docker run --rm -it --gpus all \
  ghcr.io/tile-ai/tileops-release:tileops0.1.0-tilefoundry0.0.3-cuda13.2-torch2.13-tilelang0.1.12
```

To check the image on a GPU, run from a clone of this repository:

```bash
scripts/verify_release_image.sh
scripts/verify_release_image.sh --bench benchmarks/ops/bench_rope.py  # also run a benchmark
```

## How the library grows

An implementation can be regenerated from its spec, but a spec cannot be recovered from an
implementation. Agents work on the whole library, not only its kernels, so it needs to stay
consistent as it grows.

- **Maintainable.** Each op starts with a self-contained YAML spec, which is the sole input
  to code generation. Checks enforce the boundary between the Python op and the
  TileLang kernel.
- **Verifiable.** The spec declares correctness and performance criteria before code is
  written. CI checks the spec, generated code, tests and benchmarks.
- **Tunable.** The roofline model reports each kernel's gap to its bound, and nightly
  [benchmarks](https://tile-ai.github.io/TileOPs.github.io/benchmarks/) compare each kernel with
  the fastest other implementation on the same GPU.

<details>
<summary><b>Inside an op spec</b></summary>

Each op has an entry in
[`src/tileops/manifest/spec/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest/spec/),
with one file per family. Abridged from `gemm.yaml`:

```yaml
GEMMFwdOp:
  ref_api: nvmath.linalg.advanced.Matmul.execute
  family: gemm
  signature:
    forall: {M: Dim, N: Dim, K: Dim, T: "DType[float16 | bfloat16]"}
    params: {trans_a: {type: bool, default: false}, trans_b: {type: bool, default: true}}
    inputs: {a: {dtype: T, shape: "Mat[trans_a, M, K]"}, b: {dtype: T, shape: "Mat[trans_b, K, N]"}}
    outputs: {d: {dtype: T, shape: "[M, N]"}}
  workloads: [{M: 1024, N: 1024, K: 1024, trans_a: false, trans_b: false, dtype_cases: [{T: float16}, {T: bfloat16}], label: square-1k}]
  roofline: {flops: "2 * M * N * K"}
```

`Mat[t, R, C]` is a type the entry declares: the shape `[R, C]`, or `[C, R]` when `t` is true.

| Field       | Role                                                                              |
| ----------- | --------------------------------------------------------------------------------- |
| `ref_api`   | The API whose semantics the operator follows, when applicable.                    |
| `signature` | Tensor types over named indices. Each call is checked against it before dispatch. |
| `workloads` | The concrete calls used by contract tests and nightly benchmarks.                 |
| `roofline`  | The performance model. `eval_roofline()` evaluates it for each checked call.      |

A validator checks each entry against its implementation in CI.
[layer-boundaries.md](https://github.com/tile-ai/TileOPs/blob/main/docs/design/layer-boundaries.md)
states what each layer of an op owns.

</details>

## Installation

```bash
pip install tileops
```

pip resolves PyTorch 2.13, [TileLang](https://github.com/tile-ai/tilelang) 0.1.12 and PyYAML
from PyPI. The machine provides the rest:

- Linux and an NVIDIA GPU with compute capability 9.0 (SM90), with its driver
- CUDA Toolkit 13.2, with `nvcc` on `PATH`: TileLang compiles kernels at run time

The release is verified on H200 with Python 3.12 and 3.13.

Some kernels also have SM80, SM86 or SM89 paths. They are tested on those devices when run
there, and are outside the release verification.

To develop TileOPs, install it from source:

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
pip install -e . --group dev -c constraints.txt   # constraints.txt pins what CI validates
pre-commit install

python -m pytest -q tests -m smoke           # verify; requires a CUDA GPU
```

A prebuilt Docker image includes the full stack and provides the CI environment. See
[development.md](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md#dev-docker-image)
for details, test tiers, benchmarks and build troubleshooting.

## Documentation

- [Docs site](https://tile-ai.github.io/TileOPs.github.io/): API reference, user guides and nightly benchmarks.
- [Development guide](https://tile-ai.github.io/TileOPs.github.io/user-guide/development/): environment, tests and pull requests.
- [CONTRIBUTING.md](https://github.com/tile-ai/TileOPs/blob/main/CONTRIBUTING.md): naming, PR structure and review criteria.
- [Design docs](https://github.com/tile-ai/TileOPs/tree/main/docs/design): architecture, manifest, roofline and layer boundaries.
- [API stability](https://github.com/tile-ai/TileOPs/blob/main/docs/stability.md): public modules, compatibility promise, support scope and writing an op outside TileOPs.

## License

TileOPs is released under the [MIT License](https://github.com/tile-ai/TileOPs/blob/main/LICENSE).
