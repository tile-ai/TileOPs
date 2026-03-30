# Architecture

TileOPs is a spec-driven GPU operator platform built on TileLang. Every operator has a declarative specification in `ops_manifest.yaml` before code is written. The spec drives code generation, test validation, performance evaluation, and documentation — but the runtime interface remains plain Python imports.

## Modules

The platform consists of 8 modules:

```mermaid
graph TD
    M1["M1: Spec<br/>ops_manifest.yaml"]
    M2["M2: Op + Kernel"]
    M6["M6: HW Profile"]

    subgraph Validate
        M3["M3: Correctness"]
        M4["M4: Benchmark"]
    end

    M5["M5: Roofline"]
    M7["M7: CI Gate"]
    M8["M8: Docs"]

    M1 -- "signature<br/>workloads" --> M2
    M1 -- "workloads<br/>(shapes, dtypes)" --> M4
    M1 -- "roofline formulas<br/>(flops, bytes)" --> M5
    M2 -- "Op callable" --> M3
    M2 -- "Op callable" --> M4
    M2 -- "docstring" --> M8
    M3 -- "pass/fail" --> M7
    M4 -- "raw time<br/>(JSON/CSV)" --> M5
    M4 -- "pass/fail<br/>latency delta" --> M7
    M5 -- "SOL%<br/>bound type" --> M8
    M6 -- "GPU profile<br/>(YAML)" --> M5
    M7 -- "gate status<br/>benchmark results" --> M8
```

| Module              | Responsibility                                                         | Key Artifact                       |
| ------------------- | ---------------------------------------------------------------------- | ---------------------------------- |
| **M1: Spec**        | Declare op interface, workloads, roofline formulas                     | `ops_manifest.yaml`                |
| **M2: Kernel + Op** | GPU kernel implementations and user-facing Python API                  | `tileops/kernels/`, `tileops/ops/` |
| **M3: Correctness** | Numerical correctness against PyTorch reference                        | `tests/`                           |
| **M4: Benchmark**   | Performance guard — CI benchmarks to detect regression on existing ops | `benchmarks/`                      |
| **M5: Roofline**    | Hardware efficiency from raw time + formulas + HW profile              | `tileops/perf/`                    |
| **M6: HW Profile**  | GPU hardware parameters (bandwidth, FLOPS) from offline calibration    | `tileops/perf/profiles/`           |
| **M7: CI Gate**     | Correctness and performance regression guard per PR                    | CI pipeline                        |
| **M8: Docs**        | Auto-generated API reference, perf tables, support matrix              | TileOPs.github.io                  |

## Data Contracts

Modules communicate through **data contracts** — explicit, versioned agreements on what artifact one module produces and another consumes. Every arrow in the module graph corresponds to exactly one row in the table below. If an edge has no contract, it does not exist. No implicit dependencies.

| From | To  | Artifact                         | Format                                |
| ---- | --- | -------------------------------- | ------------------------------------- |
| M1   | M2  | signature, workloads             | `ops_manifest.yaml`                   |
| M1   | M4  | workloads (shapes, dtypes)       | `ops_manifest.yaml`                   |
| M1   | M5  | roofline formulas (flops, bytes) | `ops_manifest.yaml`                   |
| M2   | M3  | Op callable                      | Python import                         |
| M2   | M4  | Op callable                      | Python import                         |
| M2   | M8  | docstring                        | Google-style in source                |
| M3   | M7  | pass/fail                        | pytest exit code                      |
| M4   | M5  | raw time per workload            | JSON/CSV                              |
| M4   | M7  | pass/fail, latency delta         | pytest exit code + JUnit properties   |
| M5   | M8  | SOL%, bound type per workload    | structured output                     |
| M6   | M5  | GPU profile                      | YAML (`tileops/perf/profiles/`)       |
| M7   | M8  | gate status, benchmark results   | CI scheduled job → perf tables update |

Note: M3 (Correctness) and M4 (Benchmark) have no data dependency. The development workflow runs correctness first, but this is a process convention, not a data contract.

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
├── tileops/
│   ├── ops_manifest.yaml             # Op registry (agent entry point, packaged)
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
