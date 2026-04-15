# Op Interface Design

## How to Add a New Op

### 1. Implement Two Classes: Op and Kernel

Every operator requires an **Op** (host-side orchestration) and a **Kernel** (device-side computation). Op validates inputs, prepares layout, dispatches to Kernel, assembles output. Kernel owns the TileLang program, tile config, and JIT compilation. The two layers are independently modifiable.

See [Op/Kernel Interface](ops-design-reference.md#opkernel-interface) for full interface tables and [Skeleton Code Examples](ops-design-reference.md#skeleton-code-examples) for implementation templates.

### 2. Determine Hierarchy Position

Target architecture is three layers: Op (L1) → FamilyBase (L2) → ConcreteOp (L3). New ops start by inheriting L1 directly. When a family accumulates 2-3 ops with identical `forward()` flow, extract an L2 base.

See [Op Class Hierarchy](ops-design-reference.md#op-class-hierarchy) and [Development Path](ops-design-reference.md#development-path).

### 3. Source Parameters from Manifest

Every `__init__` and `forward` parameter must trace back to a manifest declaration:

| Manifest source                      | Goes to    | Examples                |
| ------------------------------------ | ---------- | ----------------------- |
| `signature.inputs` (tensors)         | `forward`  | `x`, `weight`           |
| `signature.params` (non-tensor)      | `__init__` | `dim`, `eps`, `keepdim` |
| `dtype`                              | `__init__` | `torch.float16`         |
| `shape` dimension names (fixed-rank) | `__init__` | `M`, `K`, `N`           |
| `init_dims` (arbitrary-rank)         | `__init__` | `N` (see below)         |

Information not declared in the manifest MUST NOT appear in `__init__`. No exceptions.

Op `__init__` parameters use **keyword-only arguments**. See [manifest.md](manifest.md) for the full manifest specification.

### 4. Fixed-Rank vs Arbitrary-Rank

**Fixed-rank** — manifest declares `shape` with dimension names → all dimensions become `__init__` keywords → kernel constructed at init.

**Arbitrary-rank** — manifest does not declare `shape`. Use `init_dims` to declare which derived dimensions users must provide at init. Undeclared dimensions are derived from tensors at forward time.

See [Fixed-Rank Ops](ops-design-reference.md#fixed-rank-ops) and [Arbitrary-Rank Ops and init_dims](ops-design-reference.md#arbitrary-rank-ops-and-init_dims) for spec and examples.

### 5. Generate Codegen Methods

Agent generates three methods from manifest declarations:

| Method                   | Manifest source          | Purpose                 |
| ------------------------ | ------------------------ | ----------------------- |
| `_infer_output_shapes()` | `shape_rules`            | Output shape derivation |
| `_validate_dtypes()`     | `dtype` / `dtype_combos` | Input dtype validation  |
| `eval_roofline()`        | `roofline`               | Performance model       |

See [Codegen Methods](ops-design-reference.md#codegen) for calling conventions, inheritance rules, and examples.

### 6. Follow Naming Conventions

- Op: `{PascalCaseName}{Direction}Op` (e.g., `RMSNormFwdOp`). Elementwise ops omit direction.
- Kernel: `{PascalCaseName}{Direction}Kernel`. Elementwise kernels omit direction.
- `kernel_map` keys: `snake_case`, decoupled from class names.
- Manifest key must exactly equal `cls.__name__`.

See [Naming Conventions](ops-design-reference.md#naming-conventions) for full rules.
