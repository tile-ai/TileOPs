# Architecture

TileOPs is a spec-driven GPU operator platform built on TileLang. Every operator has a declarative specification in `ops_manifest.yaml` before code is written. The spec drives code generation, test validation, performance evaluation, and documentation — but the runtime interface remains plain Python imports.

## Modules

The platform consists of 8 modules connected by five end-to-end data flows:

| Flow                  | Trigger                 | Path            | Outcome                        |
| :-------------------- | :---------------------- | :-------------- | :----------------------------- |
| **🟩 New Op**         | new manifest entry      | M1 → M2 → M3    | correct op ready for benchmark |
| **🟪 Perf Tuning**    | op callable             | M2/M1 → M4 → M5 | SOL% meets threshold           |
| **🟥 HW Calibration** | new GPU / driver update | HW → M6 → M5    | GPU profile YAML               |
| **🟧 CI Guard**       | PR push                 | M3/M4 → M7      | gate pass or block             |
| **🟦 Publish**        | merge                   | M2/M5/M7 → M8   | auto-generated docs            |

```mermaid
graph TD
    subgraph Develop [" Develop"]
        M1["M1: Spec<br/>ops_manifest.yaml"]
        M2["M2: Op + Kernel"]
    end

    subgraph Validate [" Validate"]
        M3["M3: Correctness"]
        M4["M4: Benchmark"]
    end

    subgraph Evaluate [" Evaluate"]
        HW["HW Microbench<br/>benchmarks/hardware/"]
        M6["M6: HW Profile"]
        M5["M5: Roofline"]
    end

    subgraph Deliver [" Deliver"]
        M7["M7: CI Gate"]
        M8["M8: Docs"]
    end

    %% Flow 1: New Op (emerald)
    M1 -- "signature<br/>workloads" --> M2
    M2 -- "Op callable" --> M3

    %% Flow 2: Perf Tuning (violet)
    M2 -- "Op callable" --> M4
    M1 -- "workloads<br/>(shapes, dtypes)" --> M4
    M4 -- "raw time<br/>(JSON/CSV)" --> M5
    M1 -- "roofline formulas<br/>(flops, bytes)" --> M5

    %% Flow 3: HW Calibration (rose)
    HW -- "measured BW<br/>measured FLOPS" --> M6
    M6 -- "GPU profile<br/>(YAML)" --> M5

    %% Flow 4: CI Guard (amber)
    M3 -- "pass/fail" --> M7
    M4 -- "pass/fail<br/>latency delta" --> M7

    %% Flow 5: Publish (sky)
    M2 -- "docstring" --> M8
    M5 -- "SOL%<br/>bound type" --> M8
    M7 -- "gate status<br/>benchmark results" --> M8

    %% Node styles by layer
    classDef dev fill:#fef9c3,stroke:#ca8a04,color:#713f12
    classDef val fill:#e0e7ff,stroke:#6366f1,color:#312e81
    classDef eval fill:#ccfbf1,stroke:#0d9488,color:#134e4a
    classDef del fill:#fce7f3,stroke:#db2777,color:#831843

    class M1,M2 dev
    class M3,M4 val
    class HW,M6,M5 eval
    class M7,M8 del

    %% Subgraph styles
    style Develop fill:#fefce8,stroke:#eab308,stroke-width:1.5px,rx:8,ry:8
    style Validate fill:#eef2ff,stroke:#818cf8,stroke-width:1.5px,rx:8,ry:8
    style Evaluate fill:#f0fdfa,stroke:#14b8a6,stroke-width:1.5px,rx:8,ry:8
    style Deliver fill:#fdf2f8,stroke:#ec4899,stroke-width:1.5px,rx:8,ry:8

    %% Edge colors by flow
    linkStyle 0,1 stroke:#059669,stroke-width:2px
    linkStyle 2,3,4,5 stroke:#7c3aed,stroke-width:2px
    linkStyle 6,7 stroke:#e11d48,stroke-width:2px
    linkStyle 8,9 stroke:#d97706,stroke-width:2px
    linkStyle 10,11,12 stroke:#2563eb,stroke-width:2px
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
