# NPU Benchmark

Standalone benchmark framework for TileLang kernels on Ascend NPU.

Extracted from the TileOPs project — does not depend on any TileOPs interfaces.

## Architecture

```
manifest (YAML, authoritative spec)
   │  load_workloads()
   ▼
workloads (gen_inputs → NPU tensors)
   │
   ▼
bench test (parametrize from manifest)
   ├── Op → dispatches Kernel (TileLang JIT)
   └── ManifestBenchmark.profile() → bench_kernel() (event timing)
        │
        ├── op.eval_roofline() → perf.formulas (FLOPs / bytes)
        │
        ▼
   BenchmarkReport.record() → conftest → profile_run.log (markdown)
```

### Layer-by-layer

| Layer | File | Responsibility |
|-------|------|----------------|
| **Manifest** | `manifest/*.yaml` | Op signature, workloads, roofline, source paths |
| **Manifest loader** | `manifest/__init__.py` | Merge YAML files, `load_workloads()` |
| **Workload** | `workloads/*.py` | `gen_inputs()` — create NPU tensors |
| **Kernel** | `kernels/*.py` | TileLang JIT prim_func + config/autotune |
| **Op** | `ops/*.py` | Input validation, kernel dispatch/cache, `eval_roofline()` |
| **Perf** | `perf/formulas.py` | Roofline FLOP/byte formulas |
| **Benchmark** | `benchmarks/benchmark_base.py` | `bench_kernel()`, `ManifestBenchmark`, `BenchmarkReport` |
| **Conftest** | `benchmarks/conftest.py` | Session hooks, JUnit properties, report dump |
| **Device** | `utils/device.py` | NPU/CUDA abstraction |

### Timing protocol

`bench_kernel()` follows a SOL-ExecBench-style protocol:
1. L2 cache flush (configurable buffer) before each iteration.
2. 10 warmup iterations (untimed).
3. 3 trials × 50 repeats, each timed with backend events.
4. Median of trial means reported (robust to outliers).
5. Inputs cloned from a 3-element pool (fresh addresses per iteration).

NPU timing uses `torch.npu.Event(enable_timing=True)` (via `utils.device.timing_event()`).
Set `NPU_BENCHMARK_FORCE_CUDA=1` to use CUDA events for local NVIDIA development.

## Usage

### Install

```bash
cd npu_benchmark
pip install -r requirements.txt
# For Ascend: also install torch_npu matching your torch/CANN version
```

### Run benchmarks

```bash
make bench           # all workloads
make bench-smoke     # first workload per op (fast sanity)
```

### Run correctness tests

```bash
make test
```

### Output

- `profile_run.log` — markdown report with latency, TFLOPS, bandwidth, and
  kernel-vs-torch ratio for each workload.

## Adding a new kernel

1. **Manifest**: add an entry to `manifest/<family>.yaml` (or a new YAML file).
2. **Workload**: create `workloads/<op>.py` implementing `gen_inputs()`.
3. **Kernel**: create `kernels/<op>.py` with a `Kernel` subclass.
4. **Op**: create `ops/<op>.py` with an `Op` subclass (dispatch + `eval_roofline()`).
5. **Roofline**: add the formula to `perf/formulas.py`.
6. **Bench**: create `benchmarks/ops/bench_<op>.py` (parametrize from manifest).
7. **Test** (optional): create `tests/ops/test_<op>.py` for correctness.

See `TopkSelectorOp` for a complete reference.

## Current ops

- **TopkSelectorOp** — radix-based top-k index selection over `[B, S, S_kv, G]`.

### Kernel implementations

| Kernel | File | Backend | Status |
|--------|------|---------|--------|
| `TopkSelectorTorchKernel` | `kernels/topk_selector_torch.py` | NPU/CUDA/CPU | **Default** — runs everywhere via `torch.topk` |
| `TopkSelectorKernel` | `kernels/topk_selector.py` | TileLang (CUDA/future NPU) | Reference — uses CUDA SIMT primitives (`alloc_shared`, `sync_threads`, `atomic_add`) not yet supported by the TileLang Ascend backend |

The Op defaults to `TopkSelectorTorchKernel` so the framework runs end-to-end on NPU.
To use the TileLang kernel (on a backend that supports it):

```python
from kernels import TopkSelectorKernel
op = TopkSelectorOp(topk=32, kernel_map={"topk_selector_kernel": TopkSelectorKernel})
```

### NPU backend limitation

The TileLang Ascend backend (tilelang-mlir-fork v0.1.2) does not yet support:
- `T.alloc_shared()` / `T.alloc_var()` / `T.alloc_local()` / `T.alloc_ub()` — segfault
- `T.sync_threads()` / `T.block_barrier()` / `T.subblock_barrier()` — segfault

Only `T.alloc_L1()` works for buffer allocation. Kernels that rely on the CUDA SIMT
model (shared memory histograms, thread-level synchronization, intra-block atomics)
cannot be compiled on NPU until these primitives are implemented.
