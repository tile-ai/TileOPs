## Boundary

- **OWNS**: `tileops/ops/`, `tileops/kernels/`
- **MUST NOT modify**: `tests/`, `benchmarks/`, `workloads/`, `tileops/ops_manifest.yaml`
- **MAY READ**: `tests/` (behavior understanding), `workloads/`, `tileops/ops_manifest.yaml`

→ [trust-model.md §Implementation](../../docs/trust-model.md#implementation) | [ops-design.md](../../docs/ops-design.md)

______________________________________________________________________

- Op class names use PascalCase: `{PascalCaseName}{Direction}Op` (e.g., `RMSNormFwdOp`). Direction suffix is mandatory. No abbreviation rules — the manifest author determines the name.
- Kernel class names use PascalCase with `Kernel` suffix: `{PascalCaseName}{Direction}Kernel` (e.g., `RMSNormFwdKernel`). Builder functions remain snake_case.
- `kernel_map` is the Op→Kernel dispatch registration table. Keys are snake_case dispatch identifiers, decoupled from class names (convention for new ops — some existing ops use PascalCase keys). Values are Kernel class names. The manifest declares this table; agents implement the listed Kernels. See [ops-design-reference.md § Kernel Dispatch](../../docs/ops-design-reference.md#kernel-dispatch-kernel_map).
- Op `__init__` parameters must be keyword-only (`def __init__(self, *, ...)`). Parameter names come from manifest: `shape` dimension names (fixed-rank), `init_dims` keys (arbitrary-rank), and `params` keys. Information not declared in the manifest must not appear in `__init__`.
- For arbitrary-rank ops, use `init_dims` in the manifest to declare dimensions users must provide at construction time. Dimensions not in `init_dims` are derived from tensors at forward time. See [manifest.md R20](../../docs/manifest.md).
- When adding or modifying an intermediate base class, changing kernel dispatch patterns, or introducing new class variable protocols, update `docs/ops-design.md` to reflect the change.
- When adding a new op family that inherits `Op` directly, evaluate whether it shares `forward()` flow with an existing family before creating a new base class. Document the decision in the PR.
