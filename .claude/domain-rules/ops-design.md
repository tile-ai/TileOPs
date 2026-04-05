## Boundary

- **OWNS**: `tileops/ops/`, `tileops/kernels/`
- **MUST NOT modify**: `tests/`, `benchmarks/`, `workloads/`, `tileops/ops_manifest.yaml`
- **MAY READ**: `tests/` (behavior understanding), `workloads/`, `tileops/ops_manifest.yaml`

→ [trust-model.md §Implementation](../../docs/trust-model.md#implementation) | [ops-design.md](../../docs/ops-design.md)

______________________________________________________________________

- When adding or modifying an intermediate base class, changing kernel dispatch patterns, or introducing new class variable protocols, update `docs/ops-design.md` to reflect the change.
- When adding a new op family that inherits `Op` directly, evaluate whether it shares `forward()` flow with an existing family before creating a new base class. Document the decision in the PR.
