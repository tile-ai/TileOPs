# TileOPs Development Guide

This document outlines the software engineering standards, architecture, and development workflow for the TileOPs project. All contributors must adhere to these guidelines to ensure code quality, maintainability, and performance.

## 1. Architecture Overview

TileOPs follows a strict **4-Layer Hierarchical Architecture**. This separation of concerns ensures that hardware-specific optimizations (Kernels) are decoupled from user-facing APIs (Layers).

| Layer | Name | Analog | Description |
|:---:|:---:|:---:|:---|
| **L4** | **Layer** | `torch.nn.Module` | **User Interface**: Manages state (weights), input validation, and provides a Pythonic API. |
| **L3** | **Function** | `torch.nn.functional` | **Function API**: Stateless, autograd-compatible. Handles forward/backward passes via `torch.autograd.Function`. |
| **L2** | **Op** | `torch.ops` | **Stateless Dispatcher**: Hardware-agnostic entry point. Dispatches to specific kernels. **No Autograd**. Compatible with **CUDA-Graph** & **torch.compile**. |
| **L1** | **Kernel** | TileLang | **Implementation**: Raw TileLang kernels optimized for specific hardware (e.g., Hopper, Ampere). |

---

## 2. Development Workflow

Developing a new operator involves a bottom-up approach, moving from Kernel implementation to Layer abstraction.

### Step 0: Create Tracking Issue
*   **Action**: Create a new issue using the **"New Operator Request"** template.
*   **Goal**: Define scope and track progress across the 4 layers.
*   **Task Decomposition**: For new operators, **break down the checklist items into detailed sub-issues** (i.e., **Kernel Implementation**, **Op Implementation**, **Function Implementation**, **Layer Implementation**, **Benchmark Results**). This allows new contributors to pick up smaller, well-defined tasks and submit smaller PRs.
*   **Definition of Done**: The issue is closed when the operator is fully implemented and verified.

### Step 1: Kernel Implementation (L1)
*   **Location**: `top/kernels/{operator_name}/`
*   **Goal**: Implement the core logic using TileLang.
*   **Docstrings**: Detailed description of arguments and return values.
*   **Definition of Done**: The kernel compiles and runs correctly.

### Step 2: Op Definition & Verification (L2)
*   **Location**: `top/ops/{operator_name}.py`
*   **Responsibilities**:
    *   Wrap the kernel in a Python function.
    *   **Docstrings**: Google Style (Args, Returns, Example).
    *   **Unit Test**: Compare output against a pure PyTorch reference implementation (required).
    *   **Benchmark**: Measure Latency, TFLOPS (required) and DRAM Bandwidth (required).
*   **Standards**:
    *   Use `torch.testing.assert_close` for verification.
        *   **FP16**: `rtol=1e-3`, `atol=1e-3`
        *   **BF16**: `rtol=1.6e-2`, `atol=1.6e-2`
    *   Benchmark results must be reproducible.
*   **Definition of Done**: The op is verified in unit tests, and benchmarks run correctly.

### Step 3: Functional API (L3)
*   **Location**: `top/functions/{operator_name}.py`
*   **Responsibilities**:
    *   Implement `torch.autograd.Function`.
    *   Define `forward()` and `backward()` static methods.
    *   **Docstrings**: Google Style (Args, Returns, Gradients).
*   **Verification**: Pass `torch.autograd.gradcheck` for ops with backward().
*   **Definition of Done**: The op is verified in unit tests.

### Step 4: Layer Wrapper (L4)
*   **Location**: `top/layers/{operator_name}.py`
*   **Description**: Expose the functionality as an `nn.Module` (e.g., `class FlashAttention(nn.Module)`).
*   **Docstrings**: Google Style (Class description, Init args, Forward args).
*   **Definition of Done**: The op is exposed as an `nn.Module` and verified in unit tests.

### Step 5: Benchmark Results
*   **Location**: `benchmarks/{operator_name}.py`
*   **Goal**: Measure Latency, TFLOPS (required) and DRAM Bandwidth (required).
*   **Definition of Done**: Benchmark the op and put the results in the issue.

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

### Unified Benchmark Object Pattern
To ensure consistency between tests and benchmarks, we use a **Unified Benchmark Object** pattern.
1.  **Define**: Create a subclass of `benchmarks.Benchmark` in `benchmarks/`.
    *   Implement `gen_inputs()`: Returns random inputs for testing.
    *   Implement `ref_program()`: The pure PyTorch reference implementation.
2.  **Verify**: In `tests/`, instantiate this object and call `.check(op, inputs)` or `.check_fn(fn, inputs)`.
3.  **Profile**: In `benchmarks/`, instantiate this object and call `.profile(op, inputs)`.

### Unit Tests
*   **Framework**: `pytest`
*   **Location**: `tests/`
*   **Requirement**:
    *   **Reuse**: Tests **MUST** reuse the `gen_inputs` and `check` methods from the Benchmark object.
    *   Tests must cover `FP16` and `BF16` data types.
    *   Tests must parameterize over common Shapes (Batch size, Heads, Sequence length).

### Benchmarks
*   **Framework**: `benchmarks.benchmark.Benchmark`.
*   **Location**: `benchmarks/`.
*   **Metrics**:
    *   Latency (ms)
    *   TFLOPS (Terra Floating-point Operations Per Second)
    *   DRAM Bandwidth (GB/s)
    *   Speedup vs PyTorch Native or other baselines.

---

## 5. Directory Structure Reference

```text
TileOPs/
├── top/
│   ├── kernels/   # L1: TileLang kernels
│   ├── ops/       # L2: Dispatcher + Tests + Benchmarks
│   ├── functions/ # L3: Autograd Functions
│   └── layers/    # L4: nn.Module
├── tests/         # Integration tests
├── benchmarks/    # Performance scripts
└── docs/          # Project documentation
```

## 6. Pull Request Process

### Before Submitting a PR
1.  **Format Code**: Run pre-commit hooks to ensure code style compliance.
    ```bash
    pre-commit run --all-files
    ```
2.  **Run Tests**: Ensure all relevant unit tests pass locally.
    ```bash
    pytest tests/ops/test_<op_name>.py
    ```

### CI/CD Checks
When you open a PR, the following automated checks will run:
*   **Lint**: Checks code style (Google Style), imports sorting, and spelling.
*   **Test**: Runs unit tests and benchmarks on GPU runners.
*   **Build**: Verifies the package builds successfully.

**Note**:
*   Merging is blocked until all CI checks pass.
*   **Approval**: At least **2 reviewers** must approve the PR before merging.
