## Boundary

- **OWNS**: `tileops/ops/`, `tileops/kernels/`
- **MUST NOT modify**: `tests/`, `benchmarks/`, `workloads/`, `tileops/ops_manifest.yaml`
- **MAY READ**: `tests/` (behavior understanding), `workloads/`, `tileops/ops_manifest.yaml`

→ [trust-model.md §Implementation](../../docs/trust-model.md#implementation) | [ops-design.md](../../docs/ops-design.md)

______________________________________________________________________

- **Post-#860**: Op class names use PascalCase: `{PascalCaseName}{Direction}Op` (e.g., `RMSNormFwdOp`). Direction suffix is mandatory. No abbreviation rules — the manifest author determines the name. Until #860 lands, not all classes follow this convention.
- **Post-#860**: Kernel class names use PascalCase with `Kernel` suffix: `{PascalCaseName}{Direction}Kernel` (e.g., `RMSNormFwdKernel`). Builder functions remain snake_case.
- When adding or modifying an intermediate base class, changing kernel dispatch patterns, or introducing new class variable protocols, update `docs/ops-design.md` to reflect the change.
- When adding a new op family that inherits `Op` directly, evaluate whether it shares `forward()` flow with an existing family before creating a new base class. Document the decision in the PR.
