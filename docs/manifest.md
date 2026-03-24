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

Each manifest entry has four sections: signature, workloads, roofline, and source.

### Signature

Declares inputs, outputs, parameters, and optional shape rules:

```yaml
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

- **Tensor rank is unconstrained** — DNN tensors can be 1D, 2D, 3D, etc.
- **`shape_rules`** use Python expression syntax, are optional and best-effort.
- **`dtype`** uses `|` for alternatives, `same_as(x)` for dependent types.

### Workloads

Concrete shape/dtype combinations for benchmarking, based on real model architectures:

```yaml
  workloads:
    # Llama-3.1-8B (hidden=4096)
    - {x_shape: [1, 4096, 4096], dim: -1, dtypes: [float16, bfloat16]}
    - {x_shape: [32, 1, 4096], dim: -1, dtypes: [bfloat16]}
    # Llama-3.1-70B (hidden=8192)
    - {x_shape: [1, 4096, 8192], dim: -1, dtypes: [float16, bfloat16]}
    - {x_shape: [32, 1, 8192], dim: -1, dtypes: [bfloat16]}
    # Llama-3.1-405B (hidden=16384)
    - {x_shape: [1, 2048, 16384], dim: -1, dtypes: [float16, bfloat16]}
    - {x_shape: [32, 1, 16384], dim: -1, dtypes: [bfloat16]}
```

Shapes are chosen by the op developer based on target model architectures. No centralized shape source is mandated.

### Roofline

Two modes — inline expression for simple ops, Python function reference for complex ops:

```yaml
  # Simple op: inline
  roofline:
    flops: "2 * M * N"
    bytes: "(2 * M * N + N) * sizeof(dtype)"

  # Complex op (e.g., flash attention): function reference
  roofline:
    func: "tileops.perf.formulas.gqa_prefill_fwd"
```

Referenced functions live in `tileops/perf/formulas.py` and return `{"flops": int, "bytes": int}`.

### Source

Pointers to implementation files for navigation and CI validation:

```yaml
  source:
    kernel: "tileops/kernels/norm/rms_norm.py"
    op: "tileops/ops/norm/rms_norm.py"
    test: "tests/ops/test_rms_norm.py"
```

## What Is NOT in the Manifest

- **Reference implementations** — stay in test files as PyTorch code.
- **Kernel implementation details** — tile sizes, memory strategies, `num_per_thread` are implementation choices.
- **Autotuning configuration** — handled by the kernel layer.
