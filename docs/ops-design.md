# Op Interface Design

## Class Hierarchy

```
Op (base)
  └── FamilyBase (e.g., RowNormOp, _ReduceOpBase, ...)
        └── ConcreteOp (declaration only)
```

- **Op** — abstract base. Defines the `forward()` contract.
- **FamilyBase** — per-family intermediate base. Owns shared `forward()` flow: validation, reshape, padding, kernel dispatch, trim. One per op family. Current families: `RowNormOp` (norm ops), `_ReduceOpBase` (reduce ops).
- **ConcreteOp** — leaf class. Pure declaration: kernel class, supported dtypes, input wiring. No logic override.

> **Reduction-specific note:** `_ReduceOpBase` has two sub-bases — `_SimpleReduceOp` (overrides `_pad_value`) and `_WelfordReduceOp` (adds `correction` kwarg and owns single-output `forward()`). This is a reduction-specific arrangement, not a general hierarchy pattern. Only `VarMeanOp` overrides `forward()` for tuple output.

For trust boundaries (what implementation OWNS, MUST NOT do, and MAY READ), see [trust-model.md -- Implementation](trust-model.md#implementation).

## Principle 1: Two-Layer Boundary

Every operator splits into Op (L2) and Kernel (L1):

| Concern                   | Owner  | Examples                                    |
| ------------------------- | ------ | ------------------------------------------- |
| Input validation          | Op     | CUDA check, dtype check, shape check        |
| Memory layout             | Op     | `.contiguous()`, reshape, alignment padding |
| Dtype casting             | Op     | fp8 pre/post cast, bool output cast         |
| Output reshape            | Op     | Trim padding, restore original shape        |
| TileLang program          | Kernel | `T.prim_func`, shared memory, `T.copy`      |
| Tile configuration        | Kernel | `block_m`, `threads`, `num_stages`          |
| Autotuning                | Kernel | Config search space, `tilelang.autotuner`   |
| JIT compilation + caching | Kernel | `@functools.lru_cache`                      |

Either layer can be modified independently.

## Principle 2: Base Classes Follow Forward Flow, Not Math

Create an intermediate base class when **multiple ops share the same `forward()` control flow**, the shared boilerplate is substantial, and per-op differences fit into class variables or hooks.

Do NOT create one when only 1 op uses the pattern, ops share math but differ in flow, or a common base would need excessive `if/else`.

## Principle 3: Concrete Ops Are Declarations

A concrete Op should be short and declarative: which kernel, which dtypes, how to wire inputs. Shared mechanics (validation, reshape, padding, trimming) are inherited from the base class.

## Principle 4: Conventions in Code, Not Documentation

| Convention                             | Enforced By                                           |
| -------------------------------------- | ----------------------------------------------------- |
| Non-contiguous → `.contiguous()`       | Per-family base `forward()` or family-specific helper |
| 256-element alignment padding          | Per-family base `forward()` or family-specific helper |
| CUDA device check                      | Per-family base `forward()` or per-op implementation  |
| dtype validation                       | Per-family base `forward()` via `SUPPORTED_DTYPES`    |
| `torch.library.custom_op` registration | Per-op module or shared registration utility          |
| Docstring format (Google style)        | Linter / CI check                                     |

Contiguous conversion is the family base class's responsibility. Concrete ops should not handle stride or memory layout unless explicitly documented.

## Principle 5: Class Variable Protocol

| Variable           | Required?  | Defined At              | Purpose                                            |
| ------------------ | ---------- | ----------------------- | -------------------------------------------------- |
| `SUPPORTED_DTYPES` | Yes        | Every concrete Op       | Runtime dtype check + manifest validation          |
| `ALIGNMENT`        | Per-family | Intermediate base class | Padding alignment (256 for row-reduction/row-norm) |
| `_op_name`         | Yes        | Every concrete Op       | `torch.library.custom_op` registration, logging    |

Single-kernel ops declare `_kernel_key` and `_kernel_cls`. Multi-kernel ops define `default_kernel_map` returning a dict. Dispatch is determined by the family base class.

Adding a new protocol variable requires updating: (1) the base class, (2) all concrete ops, (3) the manifest schema if applicable.

## Naming Conventions

### Op Classes

Op class names use PascalCase with a mandatory direction suffix and `Op` suffix:

```
{PascalCaseName}{Direction}Op
```

- **PascalCaseName** — descriptive name (e.g., `RMSNorm`, `BatchNorm`, `Softmax`). No mechanical abbreviation rules are enforced — the manifest author determines the name.
- **Direction** — mandatory suffix: `Fwd` or `Bwd`.
- **Op** — literal suffix.

Examples: `RMSNormFwdOp`, `SoftmaxFwdOp`, `LinearFwdOp`, `BatchNormFwdOp`.

The manifest key must exactly equal `cls.__name__`. The validator enforces this via direct equality check — there is no heuristic snake_case-to-PascalCase resolution.

### Kernel Classes

Kernel classes use PascalCase with a `Kernel` suffix:

```
{PascalCaseName}{Direction}Kernel
```

Examples: `RMSNormFwdKernel`, `SoftmaxFwdKernel`.

### Kernel Map Keys

`kernel_map` keys are **stable snake_case dispatch identifiers**. They are protocol-level names and MUST NOT be derived from `cls.__name__`. Renaming a Kernel class does not by itself require renaming the dispatch key.

**`_kernel_key` (family base dispatch):** A bare snake_case name identifying the kernel within the family. No `_kernel` suffix — this is the family's internal dispatch token.

```python
_kernel_key = "rms_norm"  # RowNormOp family
_kernel_key = "softmax_fwd"  # _ReduceOpBase family
```

**`default_kernel_map` (per-op dispatch):** Keys use descriptive snake_case with a `_kernel` suffix.

```python
# Single-kernel op
def default_kernel_map(self):
    return {"fp8_quant_kernel": FP8QuantKernel}


# Multi-kernel op
def default_kernel_map(self):
    return {
        "mha_bwd_preprocess_kernel": MhaBwdPreprocessKernel,
        "mha_bwd_kernel": MhaBwdKernel,
    }
```

### Builder Functions

Kernel builder functions (that construct TileLang programs) remain `snake_case`:

```python
def rms_norm_fwd(M, N, dtype, ...): ...
```

## Adding a New Intermediate Base Class

1. **Implement 2-3 concrete ops inheriting `Op` directly** — understand the pattern before abstracting
1. **Identify shared steps** — which parts of forward() are identical?
1. **Extract the base class** — shared steps into base, per-op differences as hooks
1. **Migrate existing ops** — verify tests pass unchanged
1. **Register the pattern** — update this hierarchy

**Abstraction follows implementation, never the reverse.**
