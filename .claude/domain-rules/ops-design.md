## Boundary

- **OWNS**: `tileops/ops/`, `tileops/kernels/`
- **MUST NOT WRITE**: `tests/`, `benchmarks/`, `workloads/`, `tileops/manifest/`

→ [trust-model.md §Implementation](../../docs/design/trust-model.md#implementation) | [ops-design.md](../../docs/design/ops-design.md)

______________________________________________________________________

- Class names: PascalCase `{Name}{Direction}Op` (Op layer) or `{Name}{Direction}Kernel` (Kernel layer); direction suffix mandatory. Manifest author chooses `{Name}`. Builder functions stay snake_case.
- `kernel_map` is the Op→Kernel dispatch registration table: snake_case dispatch keys (decoupled from class names) → Kernel class names. Manifest declares it; agents implement the listed Kernels. See [ops-design-reference.md § Kernel Dispatch](../../docs/design/ops-design-reference.md#kernel-dispatch-kernel_map).
- Op `__init__` is keyword-only (`def __init__(self, *, ...)`). Parameter names come from the manifest: `shape` dim names (fixed-rank), `static_dims` keys (arbitrary-rank), `params` keys. Only manifest-declared information belongs in `__init__`.
- Arbitrary-rank ops declare construction-time values via manifest `static_dims`. Each entry is a single-axis reference `<tensor>.shape[<const_or_param>]`; other dims come from tensors at forward time. See [manifest.md R20](../../docs/design/manifest.md).
- Update `docs/design/ops-design.md` whenever you add/modify an intermediate base class, change a kernel-dispatch pattern, or introduce a new class-variable protocol.
- A new op family inheriting `Op` directly: first check whether an existing family's `forward()` flow already fits before creating a new base class. Record the decision in the PR.
- Per-op workarounds MUST NOT be promoted to a base-class shared mechanism (mixin, class attribute, shared method, opt-out flag) within the same op-family migration PR — even when multiple ops share the workaround. Promote only via a separate design PR that shows the mechanism is a genuine family invariant (would belong in the base even if no op had taken a shortcut), not a shared shortcut.
