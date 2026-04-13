# Benchmark Workload Interface

Resolved in PR #923. Option B (capability protocols) was chosen and implemented.

## Design Decision

The benchmark interface uses capability-based structural typing instead of nominal `WorkloadBase` inheritance.

- Helpers depend on the smallest protocol they need
- Benchmark classes declare the exact workload capability they require
- `WorkloadBase` remains the default in-repo implementation, not the public benchmark API

## Capability Protocols

Defined in `benchmarks/benchmark.py`:

| Protocol                  | Provides          | Use when                                                 |
| ------------------------- | ----------------- | -------------------------------------------------------- |
| `ShapeDtypeWorkload`      | `shape`, `dtype`  | Code only reads shape and dtype (e.g. `roofline_vars()`) |
| `InputGeneratingWorkload` | `gen_inputs()`    | Code only needs input generation                         |
| `BenchmarkWorkload`       | Both of the above | Code needs shape/dtype and input generation              |

For benchmark-specific metadata (e.g. `m/n/k` for GEMM), define a dedicated protocol for that benchmark family.

## Benchmark Base

`BenchmarkBase` is generic over workload type:

```python
class BenchmarkBase(Generic[W], ABC):
    def __init__(self, workload: W):
        self.workload = workload
```

`ManifestBenchmark` accepts `ShapeDtypeWorkload` — matching its actual dependency on workload metadata for roofline evaluation.

## Layering

- Protocol = public benchmark interface contract
- `WorkloadBase` = default in-repo implementation style

This keeps the trust boundary clear: workloads define inputs, benchmarks define profiling and metric logic.
