# Roofline Evaluation

TileOPs evaluates kernel performance against hardware-theoretical Speed-of-Light (SOL) bounds, not relative to PyTorch baselines.

## Efficiency Ratio

```
efficiency = sol_bound / actual_time
```

Where `sol_bound` is the theoretical minimum execution time:

```
memory_time  = bytes_moved / hbm_bandwidth
compute_time = total_flops / peak_flops
sol_bound    = max(memory_time, compute_time)
```

An efficiency of 90% means the kernel runs within 10% of the hardware theoretical limit.

## Bound Type

Whichever of `memory_time` or `compute_time` is larger determines the bound type (memory-bound vs compute-bound). This is **not** a static property of an op — it varies by shape (e.g., a small GEMM may be memory-bound while a large GEMM is compute-bound). Bound type is computed per-workload by the roofline tool and displayed in auto-generated documentation. It is **not** declared in the manifest.

## GPU Profile

Hardware parameters use theoretical values with calibration factors from one-time microbenchmark measurements. YAML files store only `theoretical` and `calibration`; `effective = theoretical × calibration` is computed by `load_profile()`:

```yaml
# tileops/perf/profiles/h200.yaml
hbm:
  theoretical: 4800e9       # bytes/s, from spec sheet
  calibration: 0.94         # from microbench
tensor_core:
  fp16:
    theoretical: 989.5e12   # FLOPS, from spec sheet
    calibration: 0.75       # from microbench (cuBLAS peak)
```

Profiles are stored in `tileops/perf/profiles/`. Microbenchmarks for calibration live in `benchmarks/hardware/`.

## Benchmark / Roofline Decoupling

Benchmark (M4) produces raw time (JSON/CSV). Roofline (M5) is a separate tool that reads raw time + manifest formulas + GPU profile to compute efficiency. This separation enables:

- Re-analyzing historical data when GPU profiles are updated
- Multiple consumers of raw benchmark data (roofline, regression detection, dashboards)
- Benchmark module has no third-party dependencies beyond the project itself

## Roofline Formulas

Defined per-op in `ops_manifest.yaml` (see [manifest.md](manifest.md)). Simple ops use inline expressions; complex ops reference functions in `tileops/perf/formulas.py` that return `{"flops": int, "bytes": int}`.

### Design Gap: Formula Semantics

The current manifest stores simple `flops` / `bytes` formulas as strings:

```yaml
roofline:
  flops: "4 * M * N"
  bytes: "(2 * M * N + N) * elem_bytes"
```

Those strings are not free-form Python, but the supported expression
language is not currently defined as a single contract. This has created
multiple evaluator surfaces with different behavior:

- manifest inline `roofline.flops` / `roofline.bytes`
- manifest `roofline.vars` expressions
- generated Op-level `eval_roofline()` proposals

These surfaces must not grow independent expression languages. A formula
that the manifest accepts must either lower to generated Op code without
semantic changes or be rejected before codegen. Conversely, generated Op
code must not accept formulas that the manifest validator cannot reason
about.

### Two Distinct Expression Layers

Roofline metadata has two different expression problems and they should
remain separate.

**Variable resolution** derives named roofline variables from runtime
inputs, shapes, and manifest params. This layer is manifest-analysis
only. It may need shape access, slicing, `product()`, `range()`, and small
comprehensions:

```yaml
roofline:
  vars:
    M: "product(x.shape[:dim])"
    N: "x.shape[dim]"
```

The result of variable resolution is ordinary data:

```python
{"M": 1024, "N": 4096, "elem_bytes": 2}
```

**Arithmetic formulas** compute final `flops` and `bytes` from resolved
variables:

```yaml
roofline:
  flops: "4 * M * N"
  bytes: "(2 * M * N + N) * elem_bytes"
```

This layer should be much smaller than the variable-resolution language.
It should not contain tensor access, shape slicing, comprehensions,
attributes, or arbitrary calls. It is the only expression layer that
generated Op code should consume.

### Target Model

The target design is a single roofline arithmetic expression model with
multiple frontends and shared backends:

```text
manifest string formula ─┐
                         ├─> RooflineExpr IR ─> validate
structured YAML formula ─┘                    ├─> evaluate
                                              └─> emit Python code
```

The IR is a small algebraic tree, not a general Python AST. It should
support only the operations required by roofline cost formulas:

- constants: integer / float literals
- variables: names from resolved roofline variables plus `elem_bytes`
- arithmetic: `add`, `sub`, `mul`, `div`, `floor_div`, `mod`
- domain primitives when needed: `ceil_div`, `ceil`, `floor`, `log2`

The exact primitive set must be explicit. Adding a primitive is a schema
and codegen decision, not a local evaluator tweak.

### Structured Formula Syntax

New manifest entries should be able to express formulas without embedding
Python-like strings. A structured expression avoids parser ambiguity and
can be validated directly from YAML:

```yaml
roofline:
  flops:
    expr:
      mul:
        - 4
        - var: M
        - var: N
  bytes:
    expr:
      mul:
        - add:
            - mul:
                - 2
                - var: M
                - var: N
            - var: N
        - var: elem_bytes
```

Common roofline patterns should use explicit domain primitives rather than
spelling them as arbitrary function calls. For example, prefer `ceil_div`
over `ceil(x / y)` when the intent is integer block rounding:

```yaml
roofline:
  bytes:
    expr:
      mul:
        - ceil_div:
            - var: tokens
            - var: block_size
        - var: elem_bytes
```

This is more verbose than `"ceil(tokens / block_size) * elem_bytes"`, but
it gives the validator and code generator an exact, typed operation.

### Codegen Contract

Generated Op code must not invent or reinterpret formula semantics. It
has two valid lowering strategies:

1. Emit Python arithmetic directly from `RooflineExpr`:

   ```python
   def eval_roofline(self) -> tuple[int, int]:
       return (
           4 * self.M * self.N,
           (2 * self.M * self.N + self.N) * self.elem_bytes,
       )
   ```

1. Store structured class-level declarations and let the base class
   evaluate the same `RooflineExpr` model:

   ```python
   _flops_expr = {"mul": [4, {"var": "M"}, {"var": "N"}]}
   _bytes_expr = {
       "mul": [
           {
               "add": [
                   {"mul": [2, {"var": "M"}, {"var": "N"}]},
                   {"var": "N"},
               ]
           },
           {"var": "elem_bytes"},
       ]
   }
   ```

In both cases, the accepted language is defined by `RooflineExpr`, not by
an Op-local string parser.

### Migration Rules

Existing string formulas remain supported during migration, but they are a
compatibility frontend. They must parse into the same `RooflineExpr` IR
used by structured formulas.

Migration should proceed in this order:

1. Factor manifest inline formula evaluation into a shared
   `RooflineExpr` parser/evaluator.
1. Keep `roofline.vars` resolution separate because it has a broader,
   manifest-only language.
1. Add structured YAML formula support for new or migrated entries.
1. Make validator checks compare both string and structured formulas by
   lowering them to the same IR.
1. Make Op codegen consume only the IR, either by emitting Python
   arithmetic or by storing structured declarations.
1. Deprecate string formulas only after all existing entries have a
   structured equivalent and codegen no longer depends on string parsing.

Until this migration exists, new Op-base `eval_roofline()` work should
reuse the shared manifest inline evaluator or be limited to a documented
temporary path. It should not introduce another independent formula
language.
