---
type: FEAT
component: Reduce
labels: [feature]
target_repo: tile-ai/TileOPs
replaces: "#424"
---

# [FEAT][REDUCE] implement linalg_vector_norm sub-category (l1/l2/inf norm)

## Description

### Symptom / Motivation
Part of the reduce operator family (#398). The linalg_vector_norm sub-category covers 3 vector norm computations: l1_norm, l2_norm, inf_norm. These are used for gradient clipping, regularization, and numerical stability checks.

### Root Cause Analysis
N/A — new feature implementation.

### Related Files
- `tileops/kernels/reduction/_primitives.py` — shared utilities (`align_up`, `DEFAULT_ALIGNMENT`); already landed via #418
- `tileops/kernels/reduction/reduce/fwd.py` — closest structural reference (same `(M,)` output shape, `op_kind`-parameterized kernel)
- `tileops/kernels/reduction/__init__.py` — has commented-out placeholder `from .vector_norm import VectorNormKernel` (line 25)
- `tileops/ops/reduction/__init__.py` — has commented-out placeholders for `L1NormOp`, `L2NormOp`, `InfNormOp` (lines 44-47)
- `tests/test_reduction_init_files.py` — validates init file placeholders; must move classes from `PLACEHOLDER_*` to `IMPLEMENTED_*` lists

## Goal
Implement 3 vector norm ops (l1_norm, l2_norm, inf_norm) with a parameterized TileLang kernel, Op wrappers, correctness tests, and benchmarks. All norms reduce along dim=-1.

## Plan
**Plan type: proposal**
1. Create `tileops/kernels/reduction/vector_norm/fwd.py`:
   - `_vector_norm_kernel(M, N, op_kind, dtype)` — parameterized kernel factory:
     - `"l1"` (ord=1): compute `abs(x)` in fp32, then `T.reduce_sum`
     - `"l2"` (ord=2): compute `x*x` in fp32, then `T.reduce_sum`, then `sqrt`
     - `"inf"` (ord=inf): compute `abs(x)` in fp32, then `T.reduce_max`
   - `@torch.library.custom_op("top::vector_norm_fwd")` + `register_fake` wrapper
   - `VectorNormKernel(Kernel)` class with `supported_archs = [80, 86, 89, 90]`, `default_config`, `autotune_configs`
   - Import `align_up`, `DEFAULT_ALIGNMENT` from `._primitives`; padding value `0.0` (neutral for all three ops)
2. Create `tileops/kernels/reduction/vector_norm/__init__.py`:
   - Explicit `__all__ = ["VectorNormKernel"]` and `from .fwd import VectorNormKernel`
3. Create `tileops/ops/reduction/l1_norm.py`, `l2_norm.py`, `inf_norm.py`:
   - Each contains one Op class (`L1NormOp`, `L2NormOp`, `InfNormOp`)
   - Follow validate → reshape → pad(0.0) → kernel → reshape pattern (reference: `reduce.py` `_SimpleReduceOp`)
   - `op_kind` parameter passed to `VectorNormKernel`: `"l1"`, `"l2"`, `"inf"` respectively
4. Update init files:
   - `tileops/kernels/reduction/__init__.py`: uncomment `from .vector_norm import VectorNormKernel` and its `__all__` entry
   - `tileops/ops/reduction/__init__.py`: uncomment `L1NormOp`, `L2NormOp`, `InfNormOp` imports and `__all__` entries
   - `tests/test_reduction_init_files.py`: move `VectorNormKernel` from `PLACEHOLDER_KERNEL_CLASSES` to `IMPLEMENTED_KERNEL_CLASSES`; move `L1NormOp`, `L2NormOp`, `InfNormOp` from `PLACEHOLDER_OP_CLASSES` to `IMPLEMENTED_OP_CLASSES`
5. Create `tests/ops/test_vector_norm.py`:
   - `FixtureBase` subclass with `pytest.mark.smoke` / `pytest.mark.full` markers
   - `TestBase` subclass with `gen_inputs()` and `ref_program()` using `torch.linalg.vector_norm`
   - Coverage: fp32/fp16/bf16, non-aligned N, tail-M, multi-dim (1D/3D/4D), non-contiguous inputs
6. Create `benchmarks/ops/bench_vector_norm.py`:
   - `BenchmarkBase` subclass with `calculate_flops()` and `calculate_memory()`
   - Profile TileOPs vs PyTorch baseline for l1/l2/inf across representative shapes
   - Use `BenchmarkReport.record()` (same structure as `bench_reduce.py`)

## Constraints
- Depends on #418 (shared primitives) — already merged.
- Uses `T.reduce_sum` and `T.reduce_max` directly (no custom macro needed).
- Output dtype follows PyTorch semantics (same dtype as input); internal computation in fp32.
- Padding value is `0.0` for all three ops: neutral for `sum(abs)`, `sum(sq)`, and `max(abs)`.
- Kernel directory is `vector_norm/` (matching the existing placeholder `from .vector_norm import ...`), not `linalg_vector_norm/`.
- Test/benchmark filenames follow existing convention: `test_vector_norm.py`, `bench_vector_norm.py`.

## Acceptance Criteria
- [ ] AC-1: `VectorNormKernel` declares `supported_archs`, `custom_op` + `register_fake`, `default_config`, `autotune_configs`
- [ ] AC-2: All 3 Op classes follow validate → reshape → pad → kernel → reshape pattern
- [ ] AC-3: Output dtype matches input dtype; internal computation in fp32
- [ ] AC-4: All 3 ops pass correctness tests against `torch.linalg.vector_norm` for fp32/fp16/bf16
- [ ] AC-5: Non-contiguous, multi-dimensional (1D, 3D, 4D), pow2 and non-pow2 last-dim input tests pass
- [ ] AC-6: `vector_norm/__init__.py` has explicit `__all__` and proper re-exports; uncomment placeholder lines in `tileops/kernels/reduction/__init__.py` and `tileops/ops/reduction/__init__.py`; update `tests/test_reduction_init_files.py` to move `VectorNormKernel`, `L1NormOp`, `L2NormOp`, `InfNormOp` from `PLACEHOLDER_*` to `IMPLEMENTED_*` lists; `pytest tests/test_reduction_init_files.py` passes
- [ ] AC-7: Benchmark file `benchmarks/ops/bench_vector_norm.py` exists, uses `BenchmarkBase`/`BenchmarkReport` framework, profiles both TileOPs and PyTorch baseline (same structure as `bench_reduce.py`)
- [ ] AC-8: PR description includes a `## Benchmark` section with markdown tables showing real measured data (latency_ms, tflops, bandwidth_tbs) for l1_norm, l2_norm, and inf_norm, with TileOPs vs baseline comparison and speedup summary
- [ ] AC-9: Test fixtures use `pytest.mark.smoke` and `pytest.mark.full` markers consistently
- [ ] AC-10: Modified files pass existing tests (`pytest tests/`)
- [ ] AC-11: Lint passes (`ruff check`, `ruff format --check`)
- [ ] AC-12: TileOPs throughput (bandwidth_tbs or tflops) for all three ops reaches ≥ 80% of the PyTorch baseline on at least one supported architecture (sm80/sm86/sm89/sm90), as measured by the benchmark file
