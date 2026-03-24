# Architecture

TileOPs is a spec-driven GPU operator platform built on TileLang. Every operator has a declarative specification in `ops_manifest.yaml` before code is written. The spec drives code generation, test validation, performance evaluation, and documentation — but the runtime interface remains plain Python (`from tileops.ops import RmsNormOp`).

## Modules

The platform consists of 8 modules:

```mermaid
graph TD
    M1["M1: Spec<br/>ops_manifest.yaml"]
    M2["M2: Op + Kernel<br/>user-facing API + TileLang GPU kernels"]
    M3["M3: Correctness<br/>tests/"]
    M4["M4: Benchmark<br/>raw time"]
    M5["M5: Roofline<br/>efficiency"]
    M6["M6: HW Profile<br/>GPU parameters"]
    M7["M7: CI Gate<br/>correctness + perf regression"]
    M8["M8: Docs<br/>auto-generated"]

    M1 -- defines --> M2
    M2 --> M3
    M2 --> M4
    M2 -- docstring --> M8
    M4 -- raw time --> M5
    M6 --> M5
    M5 --> M8
    M1 --> M8
    M3 --> M7
    M4 --> M7
```

| Module              | Responsibility                                            | Key Artifact                       |
| ------------------- | --------------------------------------------------------- | ---------------------------------- |
| **M1: Spec**        | Declare op interface, workloads, roofline formulas        | `ops_manifest.yaml`                |
| **M2: Kernel + Op** | GPU kernel implementations and user-facing Python API     | `tileops/kernels/`, `tileops/ops/` |
| **M3: Correctness** | Numerical correctness against PyTorch reference           | `tests/`                           |
| **M4: Benchmark**   | Raw execution time measurement                            | `benchmarks/`                      |
| **M5: Roofline**    | Hardware efficiency from raw time + formulas + HW profile | `tileops/perf/`                    |
| **M6: HW Profile**  | GPU hardware parameters (bandwidth, FLOPS)                | `tileops/perf/profiles/`           |
| **M7: CI Gate**     | Correctness and performance regression guard per PR       | CI pipeline                        |
| **M8: Docs**        | Auto-generated API reference, perf tables, support matrix | TileOPs.github.io                  |

## Two-Layer Separation

Every operator is split into exactly two layers:

| Layer  |    Name    | Description                                                                                          |
| :----: | :--------: | :--------------------------------------------------------------------------------------------------- |
| **L2** |   **Op**   | Stateless dispatcher. Hardware-agnostic entry point. Compatible with CUDA-Graph and `torch.compile`. |
| **L1** | **Kernel** | TileLang implementation optimized for specific hardware (Hopper, Ampere, etc.).                      |

The Op layer never contains TileLang code. The Kernel layer never validates user input. See [ops-design.md](ops-design.md) for the full boundary specification.

## Data Flow

```mermaid
graph LR
    M1["M1: Spec"] --> M2["M2: Code"]
    M2 --> M3["M3: Test"]
    M3 --> M4["M4: Benchmark"]
    M2 -- docstring --> M8["M8: Docs"]
    M4 -- raw time --> M5["M5: Roofline"]
    M6["M6: HW Profile"] --> M5
    M5 --> M8
```

## Agent Production Loop

1. Read spec from M1 (manifest)
1. Write kernel (M2), op (M2), test (M3), docstring
1. Run tests (M3) — if fail, iterate on code
1. Run benchmark (M4) — raw time output
1. Roofline tool (M5) computes efficiency from raw time + manifest formulas + GPU profile
1. If efficiency is insufficient, optimize kernel and repeat from step 2
1. Submit PR → CI (M7) checks correctness and regression → merge → docs auto-update (M8)

## Documentation System

Documentation is an automatic output of the production pipeline, not a manually maintained artifact.

| Content                        | Data Source                    | Generation    |
| ------------------------------ | ------------------------------ | ------------- |
| API reference                  | Code docstrings (Google style) | sphinx/mkdocs |
| Performance tables             | Benchmark raw data             | Script        |
| Bound type per workload        | Roofline analysis output       | Script        |
| Support matrix (dtype × shape) | Manifest workloads             | Script        |
| Op list and status             | Manifest + test pass status    | Script        |

Design documents and tutorials are authored in TileOpsGov and published manually.

## Directory Structure

```
TileOPs/
├── ops_manifest.yaml                 # Op registry (agent entry point)
├── tileops/
│   ├── kernels/                      # L1: GPU kernel implementations
│   ├── ops/                          # L2: User-facing Op classes
│   ├── utils/
│   └── perf/                         # Roofline evaluation
│       ├── roofline.py               # efficiency = sol_bound / actual
│       ├── formulas.py               # Complex op roofline functions
│       └── profiles/                 # GPU hardware parameters
│           ├── h100.yaml
│           └── h200.yaml
├── benchmarks/
│   ├── hardware/                     # Microbench (GPU characterization)
│   │   ├── memory/
│   │   ├── compute/
│   │   └── system/
│   ├── ops/                          # Op-level benchmarks
│   └── kernels/                      # Kernel-level benchmarks
├── tests/                            # Correctness tests
├── docs/
└── scripts/
```
