# Architecture

TileOPs is a spec-driven GPU operator platform built on TileLang. Every operator has a declarative specification in `ops_manifest.yaml` before code is written. The spec drives code generation, test validation, performance evaluation, and documentation — but the runtime interface remains plain Python imports.

## Modules

The platform consists of 8 modules (M1–M8). Four data flows connect them into end-to-end pipelines.

### System topology

One diagram, all modules. Edge color = which flow owns that edge.

```mermaid
graph TD
    M1["M1: Spec<br/>ops_manifest.yaml"]
    M2["M2: Op + Kernel"]
    M3["M3: Correctness"]
    M6["M6: HW Profile"]
    HW["HW Microbench"]
    M7["M7: CI Gate"]
    M8["M8: Docs<br/>design + API + perf"]

    subgraph loop ["Kernel Tuning Loop"]
        M4["M4: Perf Tuning"]
        M5["M5: Roofline"]
    end

    M1 -- "spec" --> M2
    M1 -- "workloads" --> M4
    M1 -- "formulas" --> M5
    M2 --> M3

    M2 --> M4
    M4 -- "raw time" --> M5
    M5 -- "optimize" --> M2

    HW --> M6
    M6 -- "GPU profile" --> M5

    M3 --> M7

    M2 -- "design docs" --> M8
    M7 --> M8

    style loop fill:none,stroke:#94a3b8,stroke-width:1.5px,stroke-dasharray:5,rx:8,ry:8

    linkStyle 0 stroke:#0d9488,stroke-width:2px
    linkStyle 1,2 stroke:#1d4ed8,stroke-width:1.5px,stroke-dasharray:6
    linkStyle 3 stroke:#0d9488,stroke-width:2px
    linkStyle 4,5,6 stroke:#1d4ed8,stroke-width:3px
    linkStyle 7,8 stroke:#ca8a04,stroke-width:2px
    linkStyle 9 stroke:#0d9488,stroke-width:2px
    linkStyle 10,11 stroke:#9333ea,stroke-width:2px
```

🟢 Op Delivery 🔵 Perf Tuning 🟠 HW Calibration 🟣 Publish

### Flow status

| Flow                  | Status      | What works                    | Gap                                                                 |
| :-------------------- | :---------- | :---------------------------- | :------------------------------------------------------------------ |
| 🟢 **Op Delivery**    | done        | M1 → M2 → M3 → M7 (CI gate)   | —                                                                   |
| 🔵 **Perf Tuning**    | broken      | M4 produces raw time          | `roofline.py` + `formulas.py` missing; optimization loop not closed |
| 🟠 **HW Calibration** | partial     | HBM microbench + h200 profile | tensor core calibration missing; h100 profile missing               |
| 🟣 **Publish**        | not started | —                             | no doc-gen scripts, no build system                                 |

### Module reference

| Module                                       | Responsibility                                                                                                                         | Key Artifact                       |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| **M1: Spec**                                 | Declare op interface, workloads, roofline formulas                                                                                     | `ops_manifest.yaml`                |
| **M2: Kernel + Op**                          | GPU kernel implementations and user-facing Python API                                                                                  | `tileops/kernels/`, `tileops/ops/` |
| **M3: Correctness**                          | Numerical correctness against PyTorch reference                                                                                        | `tests/`                           |
| **M4: Perf Tuning**                          | Benchmark execution time and drive kernel optimization loop                                                                            | `benchmarks/`                      |
| **M5: Roofline**                             | Hardware efficiency from raw time + formulas + HW profile                                                                              | `tileops/perf/`                    |
| **M6: HW Profile**                           | GPU hardware parameters (bandwidth, FLOPS) from offline calibration                                                                    | `tileops/perf/profiles/`           |
| **M7: CI Gate**                              | Correctness and performance regression guard per PR                                                                                    | CI pipeline                        |
| **M8: Docs**                                 | Design docs, API reference, perf tables — agent artifacts published alongside auto-generated content                                   | TileOPs.github.io                  |
| **Workloads** _(shared layer, not a module)_ | Shared input generation + parametrize decorators consumed by M3 and M4. See [trust-model.md](trust-model.md#workloads-layer-contract). | `workloads/`                       |

## Data Contracts

Modules communicate through data contracts. The topology diagram above is simplified for clarity — this table is the complete contract list.

| From | To  | Artifact                         | Format                           |
| ---- | --- | -------------------------------- | -------------------------------- |
| M1   | M2  | signature, workloads             | `ops_manifest.yaml`              |
| M1   | M4  | workloads (shapes, dtypes)       | `ops_manifest.yaml`              |
| M1   | M5  | roofline formulas (flops, bytes) | `ops_manifest.yaml`              |
| M2   | M3  | Op callable                      | Python import                    |
| M2   | M4  | Op callable                      | Python import                    |
| M2   | M8  | design docs, docstrings          | Markdown, Google-style in source |
| M3   | M7  | pass/fail                        | pytest exit code                 |
| M4   | M5  | raw time per workload            | JSON/CSV                         |
| M6   | M5  | GPU profile                      | YAML (`tileops/perf/profiles/`)  |
| M7   | M8  | gate status                      | CI pipeline                      |

## Two-Layer Separation (M2)

Every operator is split into exactly two layers:

| Layer  |    Name    | Description                                                                                          |
| :----: | :--------: | :--------------------------------------------------------------------------------------------------- |
| **L2** |   **Op**   | Stateless dispatcher. Hardware-agnostic entry point. Compatible with CUDA-Graph and `torch.compile`. |
| **L1** | **Kernel** | TileLang implementation optimized for specific hardware (Hopper, Ampere, etc.).                      |

The Op layer never contains TileLang code. The Kernel layer never validates user input. See [ops-design.md](ops-design.md) for the full boundary specification.

## Agent Production Loop

1. Read spec from M1 (manifest)
1. Write kernel (M2), op (M2), test (M3), docstring
1. Run tests (M3) — if fail, iterate on code
1. Run perf tuning (M4) — benchmark raw time, feed to roofline (M5)
1. M5 computes efficiency from raw time + manifest formulas + GPU profile
1. If efficiency is insufficient, optimize kernel and repeat from step 2
1. Submit PR → CI (M7) checks correctness and regression → merge → docs auto-update (M8)

## Documentation System

Documentation combines auto-generated content (API reference, perf tables) with design artifacts produced during agent-driven development.

| Content                        | Data Source                    | Generation    |
| ------------------------------ | ------------------------------ | ------------- |
| API reference                  | Code docstrings (Google style) | sphinx/mkdocs |
| Performance tables             | Benchmark raw data             | Script        |
| Bound type per workload        | Roofline analysis output       | Script        |
| Support matrix (dtype × shape) | Manifest workloads             | Script        |
| Op list and status             | Manifest + test pass status    | Script        |

Design documents are authored during development and published alongside auto-generated content.

## Directory Structure

```
TileOPs/
├── tileops/
│   ├── ops_manifest.yaml             # Op registry (agent entry point, packaged)
│   ├── kernels/                      # L1: GPU kernel implementations
│   ├── ops/                          # L2: User-facing Op classes
│   ├── utils/
│   └── perf/                         # Roofline evaluation
│       ├── roofline.py               # efficiency = sol_time / actual_time
│       ├── formulas.py               # Complex op roofline functions
│       └── profiles/                 # GPU hardware parameters
│           ├── h100.yaml
│           └── h200.yaml
├── workloads/                        # Shared workload definitions (WorkloadBase, FixtureBase)
│   ├── base.py                       # WorkloadBase, FixtureMeta, FixtureBase
│   └── ops/                          # Per-op workload subclasses
├── benchmarks/
│   ├── benchmark.py                  # BenchmarkBase[W], capability protocols, ManifestBenchmark
│   ├── hardware/                     # Microbench (GPU characterization)
│   │   ├── memory/
│   │   ├── compute/
│   │   └── system/
│   ├── ops/                          # Op-level benchmarks
│   ├── kernels/                      # Kernel-level benchmarks
│   └── tests/                        # Benchmark contract tests
├── tests/                            # Correctness tests
├── docs/
└── scripts/
```
