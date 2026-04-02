# Op Interface Design

## Class Hierarchy (target architecture)

```
Op (base)
  └── FamilyBase (e.g., RowNormOp → NormBase, ReductionBase, ...)
        └── ConcreteOp (declaration only)
```

- **Op** — abstract base. Defines the `forward()` contract.
- **FamilyBase** — per-family intermediate base. Owns shared `forward()` flow: validation, reshape, padding, kernel dispatch, trim. One per op family. Current: `RowNormOp` serves norm ops; migration to `NormBase` tracked in #741.
- **ConcreteOp** — leaf class. Pure declaration: kernel class, supported dtypes, input wiring. No logic override.

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

## Adding a New Intermediate Base Class

1. **Implement 2-3 concrete ops inheriting `Op` directly** — understand the pattern before abstracting
1. **Identify shared steps** — which parts of forward() are identical?
1. **Extract the base class** — shared steps into base, per-op differences as hooks
1. **Migrate existing ops** — verify tests pass unchanged
1. **Register the pattern** — update this hierarchy

**Abstraction follows implementation, never the reverse.**
