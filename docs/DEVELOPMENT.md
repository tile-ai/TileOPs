# TileOps Development Guide

This document outlines the software engineering standards, architecture, and development workflow for the TileOps project. All contributors must adhere to these guidelines to ensure code quality, maintainability, and performance.

## 1. Architecture Overview

TileOps follows a strict **4-Layer Hierarchical Architecture**. This separation of concerns ensures that hardware-specific optimizations (Kernels) are decoupled from user-facing APIs (Layers).

| Layer | Name | Analog | Description |
|:---:|:---:|:---:|:---|
| **L4** | **Layer** | `torch.nn.Module` | **User Interface**: Manages state (weights), input validation, and provides a Pythonic API. |
| **L3** | **Function** | `torch.nn.functional` | **Function API**: Stateless, autograd-compatible. Handles forward/backward passes via `torch.autograd.Function`. |
| **L2** | **Op** | `torch.ops` | **Stateless Dispatcher**: Hardware-agnostic entry point. Dispatches to specific kernels. **No Autograd**. Compatible with **CUDA-Graph** & **torch.compile**. |
| **L1** | **Kernel** | TileLang | **Implementation**: Raw TileLang kernels optimized for specific hardware (e.g., Hopper, Ampere). |

---

## 2. Development Workflow (The "V-Model")

Developing a new operator involves a bottom-up approach, moving from Kernel implementation to Layer abstraction.

### Step 0: Create Tracking Issue
*   **Action**: Create a new issue using the **"New Operator Request"** template.
*   **Goal**: Define scope and track progress across the 4 layers.

### Step 1: Kernel Implementation (L1)
*   **Location**: `top/kernels/{operator_name}/`
*   **Goal**: Implement the core logic using TileLang.
*   **Definition of Done**: The kernel compiles and runs correctly.

### Step 2: Op Definition & Verification (L2)
*   **Location**: `top/ops/{operator_name}.py`
*   **Responsibilities**:
    *   Wrap the kernel in a Python function.
    *   **Unit Test**: Compare output against a pure PyTorch reference implementation (required).
    *   **Benchmark**: Measure Latency, TFLOPS (required) and DRAM Bandwidth (required).
*   **Standards**:
    *   Use `torch.testing.assert_close` for verification.
        *   **FP16**: `rtol=1e-3`, `atol=1e-3`
        *   **BF16**: `rtol=1.6e-2`, `atol=1.6e-2`
    *   Benchmark results must be reproducible.

### Step 3: Functional API (L3)
*   **Location**: `top/functions/{operator_name}.py`
*   **Responsibilities**:
    *   Implement `torch.autograd.Function`.
    *   Define `forward()` and `backward()` static methods.
*   **Verification**: Pass `torch.autograd.gradcheck`.

### Step 4: Layer Wrapper (L4)
*   **Location**: `top/layers/{operator_name}.py`
*   **Description**: Expose the functionality as an `nn.Module` (e.g., `class FlashAttention(nn.Module)`).

---

## 3. Coding Standards

We enforce high standards for code quality and consistency.

### Python Code Style
*   **Style Guide**: **[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)**.
    *   We strictly follow the Google style for formatting and docstrings.
*   **Formatter**: `yapf`
    *   Configuration: `based_on_style = "google"` in `pyproject.toml`.
*   **Linter**: `ruff`
*   **Docstrings**: All public functions and classes must use **Google-style docstrings**.

### Improvements & Type Safety
*   **Type Hints**: All function signatures (inputs and outputs) must be type-hinted.
*   **Strict Typing**: L2 (Op), L3 (Function) and L4 (Layer) APIs are checked with `mypy` in strict mode.

---

## 4. Testing & Benchmarking Strategy

### Unit Tests
*   **Framework**: `pytest`
*   **Location**: `tests/`
*   **Requirement**:
    *   Tests must cover `FP16` and `BF16` data types.
    *   Tests must parameterize over common Shapes (Batch size, Heads, Sequence length).
    *   Use `torch.testing.assert_close` with appropriate tolerances.

### Benchmarks
*   **Framework**: `triton.testing.do_bench` or internal benchmark utilities.
*   **Location**: `benchmarks/` or inside `top/ops/`.
*   **Metrics**:
    *   Latency (ms)
    *   TFLOPS (Terra Floating-point Operations Per Second)
    *   Speedup vs PyTorch Native or FlashAttention-2.

---

## 5. Directory Structure Reference

```text
d:\projects\tileops
├── top/
│   ├── kernels/   # L1: TileLang kernels
│   ├── ops/       # L2: Dispatcher + Tests + Benchmarks
│   ├── functions/ # L3: Autograd Functions
│   └── layers/    # L4: nn.Module
├── tests/         # Integration tests
├── benchmarks/    # Performance scripts
└── docs/          # Project documentation
```
