# Op Manifest Specification

`ops_manifest.yaml` is the central op registry and the agent entry point. Every operator has a declarative spec here before code is written. The spec drives code generation, test validation, performance evaluation, and documentation — but does not affect runtime. The user-facing API remains plain Python classes.

```
ops_manifest.yaml (spec)
       │
       ├──→ Agent reads spec, generates minimal Python code
       ├──→ Test generator reads workloads + tolerance
       ├──→ Benchmark reads workloads
       ├──→ Roofline reads flops/bytes formulas
       └──→ Docs generator reads signatures
```

## Fields

Each manifest entry lives under the top-level `ops:` key and has a `family` field plus four sections: signature, workloads, roofline, and source.

`family` is an explicit, machine-readable field for doc generation, API grouping, and tooling — not derived from source paths, which can change.

### Signature

Declares inputs, outputs, parameters, and optional shape rules:

```yaml
ops:
  rmsnorm_fwd:
    family: norm

    signature:
      inputs:
        x: {dtype: "float16 | bfloat16"}
        weight: {dtype: "same_as(x)"}
      outputs:
        y: {dtype: "same_as(x)"}
      params:
        dim: {type: int, default: -1}
        eps: {type: float, default: 1e-6}
      shape_rules:
        - "weight.shape == (x.shape[dim],)"
        - "y.shape == x.shape"
```

Conventions:

- **Signature uses dict, not list.** Name is identity — making it a key (`x: {dtype: ...}`) is more concise than list items (`- name: x`).
- **No per-tensor shape.** Tensor rank is intentionally unconstrained (DNN ops accept 1D, 2D, 3D, etc.). Shape relationships are expressed through `shape_rules`, not by fixing shape on each tensor.
- **`dtype`** uses `|` for alternatives, `same_as(x)` for dependent types. Concrete entries may list dtypes explicitly.
- **`shape_rules`** use Python expression syntax, are optional and best-effort.
- **Params declare the full interface.** If an op mathematically supports a parameter (e.g., `dim` for norm), it belongs in the manifest even if the current kernel only supports the default value.

### Workloads

Representative shape/dtype combinations for benchmarking:

```yaml
workloads:
  - x_shape: [2048, 4096]
    dtypes: [float16, bfloat16]
    label: "llama-3.1-8b-prefill"
  - x_shape: [1, 4096]
    dtypes: [bfloat16]
    label: "llama-3.1-8b-decode"
```

- `x_shape` and `dtypes` are required — they drive benchmark execution and code generation. Kernel-level parameters (e.g., `M`/`N`) are derivable from shapes and should not be repeated.
- `label` is optional — a human-readable tag for reports and dashboards. Tools auto-generate from shape + dtype when omitted.
- Op-specific parameters (e.g., `dim` for norm, `causal` for attention) can be added per workload entry.

Shapes are chosen by the op developer based on target model architectures.

### Roofline

Two modes — inline expression for simple ops, Python function reference for complex ops:

```yaml
roofline:
  flops: "4 * M * N"
  bytes: "2 * (M * N + N + M * N)"
```

```yaml
roofline:
  func: "tileops.perf.formulas.gqa_prefill_fwd"
```

Referenced functions live in `tileops/perf/formulas.py` and return `{"flops": int, "bytes": int}`.

The field is `bytes` (total bytes moved), not `memory` — maps directly to `bytes_moved` in the roofline formula `memory_time = bytes_moved / hbm_bandwidth`.

### Source

Pointers to implementation files for navigation and CI validation:

```yaml
source:
  kernel: tileops/kernels/norm/rms_norm.py
  op: tileops/ops/norm/rms_norm.py
  test: tests/ops/test_rms_norm.py
  bench: benchmarks/ops/bench_rms_norm.py
```

## What Is NOT in the Manifest

- **Reference implementations** — stay in test files as PyTorch code.
- **Kernel implementation details** — tile sizes, memory strategies, `num_per_thread` are implementation choices.
- **Autotuning configuration** — handled by the kernel layer.
