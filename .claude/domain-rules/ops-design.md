## Boundary

- **OWNS**: `tileops/ops/`, `tileops/kernels/`
- **MUST NOT modify**: `tests/`, `benchmarks/`, `workloads/`, `tileops/ops_manifest.yaml`
- **MAY READ**: `tests/` (behavior understanding), `workloads/`, `tileops/ops_manifest.yaml`

→ [trust-model.md §Implementation](../../docs/trust-model.md#implementation) | [ops-design.md](../../docs/ops-design.md)

______________________________________________________________________

- Op class names use PascalCase: `{PascalCaseName}{Direction}Op` (e.g., `RMSNormFwdOp`). Direction suffix is mandatory. No abbreviation rules — the manifest author determines the name.
- Kernel class names use PascalCase with `Kernel` suffix: `{PascalCaseName}{Direction}Kernel` (e.g., `RMSNormFwdKernel`). Builder functions remain snake_case.
- `kernel_map` keys are stable snake_case dispatch identifiers (protocol-level). They MUST NOT be derived from `cls.__name__`. Renaming a Kernel class does not require renaming its dispatch key. `_kernel_key` uses bare snake_case (e.g., `rms_norm`); `default_kernel_map` keys use descriptive snake_case with `_kernel` suffix (e.g., `fp8_quant_kernel`, `mha_bwd_preprocess_kernel`).
- When adding or modifying an intermediate base class, changing kernel dispatch patterns, or introducing new class variable protocols, update `docs/ops-design.md` to reflect the change.
- When adding a new op family that inherits `Op` directly, evaluate whether it shares `forward()` flow with an existing family before creating a new base class. Document the decision in the PR.
