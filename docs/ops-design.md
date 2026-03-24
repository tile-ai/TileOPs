# Op Interface Design

This document defines how TileOPs operators are structured. The design ensures agents and developers can reason about any op by following consistent patterns.

## Principle 1: Two-Layer Boundary

Every operator is split into Op (L2) and Kernel (L1) with a strict responsibility boundary:

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

An agent can modify a kernel's GPU strategy without touching user-facing behavior, and vice versa.

## Principle 2: Base Classes Follow Forward Flow, Not Math Taxonomy

The criterion for creating an intermediate base class: **do N ops share the same `forward()` control flow?**

Example: `RowNormOp` and `RowReductionOp` share validate → reshape to `(M, N)` → pad to 256-alignment → kernel → trim → reshape. Different math, same Op-layer structure — they share a base class.

Counter-example: `GroupNormOp` and `BatchNormOp` are both "normalization" but have different forward() flows (spatial dimensions, reduction axes, padding). They do NOT share a base class.

**Create a new intermediate base class when:**

- 3+ ops share the same validate/reshape/pad/kernel/trim/reshape sequence
- The pattern is stable and unlikely to diverge
- Per-op differences fit into class variables and abstract hooks

**Do NOT create one when:**

- Only 1-2 ops share the pattern
- Ops share math similarity but differ in forward() flow
- A common base would require excessive `if/else` or optional hooks

## Principle 3: Concrete Ops Are Declarations

A well-designed concrete Op reads like configuration:

```python
class RmsNormOp(RowNormOp):
    """y = x * rsqrt(mean(x², dim=-1) + eps) * weight"""

    _op_name = "rmsnorm"
    kernel_cls = RmsNormKernel
    SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

    def _init_kernel(self, tune):
        self.kernel = self.kernel_map["rms_norm"](
            self.M, self.N, self.eps, self.dtype, tune=tune
        )

    def _get_input_tensors(self, x, weight):
        return [x, weight]

    def _call_kernel(self, x, weight):
        return self.kernel(x, weight)
```

The per-op logic is: which kernel, which dtypes, how to wire inputs. Everything else is inherited.

## Principle 4: Conventions in Code, Not Documentation

If a convention applies to all ops (or all ops in a family), it lives in the base class:

| Convention                             | Enforced By                                   |
| -------------------------------------- | --------------------------------------------- |
| Non-contiguous → `.contiguous()`       | Base class `forward()`                        |
| 256-element alignment padding          | Base class `forward()`                        |
| CUDA device check                      | Base class `forward()`                        |
| dtype validation                       | Base class `forward()` via `SUPPORTED_DTYPES` |
| `torch.library.custom_op` registration | Base class or shared utility                  |
| Docstring format (Google style)        | Linter / CI check                             |

Non-contiguous tensors: all ops silently convert via `.contiguous()` in the base class `forward()`. Individual ops do not handle stride or memory layout.

## Principle 5: Class Variable Protocol

Every Op declares capabilities through class variables for both runtime behavior and static analysis:

| Variable           | Required?  | Defined At              | Purpose                                            |
| ------------------ | ---------- | ----------------------- | -------------------------------------------------- |
| `SUPPORTED_DTYPES` | Yes        | Every concrete Op       | Runtime dtype check + manifest validation          |
| `ALIGNMENT`        | Per-family | Intermediate base class | Padding alignment (256 for row-reduction/row-norm) |
| `_op_name`         | Yes        | Every concrete Op       | `torch.library.custom_op` registration, logging    |
| `kernel_cls`       | Yes        | Every concrete Op       | Maps Op to its Kernel implementation               |

Adding a new class variable requires updating: (1) the base class that reads it, (2) all existing concrete ops, (3) the manifest schema if applicable.

## Inheritance Hierarchy

```
Op (ABC)                                 # tileops/ops/op.py
 │
 ├── PointwiseOp                         # elementwise family
 │   ├── UnaryOp
 │   ├── BinaryOp
 │   └── FusedGatedOp
 │
 ├── RowReductionOp                      # reduce along last dim
 │   ├── SumOp, MeanOp, AmaxOp, ...     #   simple reduce
 │   ├── SoftmaxOp, LogSoftmaxOp        #   shape-preserving
 │   ├── ArgmaxOp, ArgminOp             #   output dtype differs
 │   └── CumsumOp, CumprodOp           #   cumulative
 │
 ├── RowNormOp                           # normalize along last dim
 │   ├── RmsNormOp, LayerNormOp
 │   ├── FusedAddRmsNormOp, ...         #   multi-output variant
 │   └── AdaLayerNormOp, ...
 │
 └── (ops with unique patterns inherit Op directly)
     ├── GroupNormOp                     # spatial reduction
     ├── BatchNormFwdOp / BwdOp          # spatial + running stats
     ├── FlashAttention variants         # complex multi-tensor
     └── fused_moe                       # routing + compute
```

This is a taxonomy of `forward()` control flow, not mathematics. Ops that don't fit any shared pattern inherit `Op` directly.

## Adding a New Intermediate Base Class

When a new op family emerges (e.g., SSM ops with shared scan patterns):

1. **Implement 2-3 concrete ops inheriting `Op` directly** — understand the actual forward() pattern before abstracting
1. **Identify shared steps** — which parts of forward() are identical?
1. **Extract the base class** — shared steps into base, per-op differences as abstract hooks
1. **Migrate existing ops** — rewrite to inherit the new base, verify tests pass unchanged
1. **Register the pattern** — add the base class to this hierarchy

**Abstraction follows implementation, never the reverse.**
