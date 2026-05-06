## Boundary

- **OWNS**: `tileops/ops/`, `tileops/kernels/`
- **MUST NOT WRITE**: `tests/`, `benchmarks/`, `workloads/`, `tileops/manifest/`

→ [trust-model.md §Implementation](../../docs/design/trust-model.md#implementation) | [ops-design.md](../../docs/design/ops-design.md)

______________________________________________________________________

- Op class names: PascalCase `{Name}{Direction}Op`, direction suffix mandatory. Manifest author chooses `{Name}`; no abbreviation rules.
- Kernel class names: PascalCase `{Name}{Direction}Kernel`. Builder functions remain snake_case.
- `kernel_map` is the Op→Kernel dispatch registration table. Keys are snake_case dispatch identifiers, decoupled from class names (convention for new ops — some existing ops use PascalCase keys). Values are Kernel class names. The manifest declares this table; agents implement the listed Kernels. See [ops-design-reference.md § Kernel Dispatch](../../docs/design/ops-design-reference.md#kernel-dispatch-kernel_map).
- Op `__init__` parameters must be keyword-only (`def __init__(self, *, ...)`). Parameter names come from manifest: `shape` dimension names (fixed-rank), `static_dims` keys (arbitrary-rank), and `params` keys. Information not declared in the manifest must not appear in `__init__`.
- For arbitrary-rank ops, use `static_dims` in the manifest to declare values the user commits to at construction time. Each entry is a single-axis reference `<tensor>.shape[<const_or_param>]`. Dimensions not in `static_dims` are derived from tensors at forward time. See [manifest.md R20](../../docs/design/manifest.md).
- When adding or modifying an intermediate base class, changing kernel dispatch patterns, or introducing new class variable protocols, update `docs/design/ops-design.md` to reflect the change.
- When adding a new op family that inherits `Op` directly, evaluate whether it shares `forward()` flow with an existing family before creating a new base class. Document the decision in the PR.
