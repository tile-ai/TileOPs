## Boundary

- **OWNS**: `tileops/ops/`, `tileops/kernels/`
- **MUST NOT modify**: `tests/`, `benchmarks/`, `workloads/`, `tileops/ops_manifest.yaml`
- **MAY READ**: `tests/` (behavior understanding), `workloads/`, `tileops/ops_manifest.yaml`

→ [trust-model.md §Implementation](../../docs/trust-model.md#implementation) | [ops-design.md](../../docs/ops-design.md)

______________________________________________________________________

- Op class names use PascalCase: `{PascalCaseName}{Direction}Op` (e.g., `RMSNormFwdOp`). Direction suffix is mandatory. No abbreviation rules — the manifest author determines the name.
- Kernel class names use PascalCase with `Kernel` suffix: `{PascalCaseName}{Direction}Kernel` (e.g., `RMSNormFwdKernel`). Builder functions remain snake_case.
- `kernel_map` keys are stable snake_case dispatch identifiers (protocol-level). They SHOULD NOT be derived from `cls.__name__` (target convention; some ops like `GLAFwdKernel` still use PascalCase keys pending migration). Renaming a Kernel class does not require renaming its dispatch key. Family-based ops define keys via their family base's convention — `_SoftmaxBaseOp` uses `_kernel_key`/`_kernel_class`, `_ReduceOpBase` uses a fixed `"reduce"` key in `default_kernel_map`. Standalone ops define `default_kernel_map` directly, often with `_kernel`-suffixed keys (e.g., `fp8_quant_kernel`), but some use established non-suffixed keys (e.g., `"dropout"`).
- When adding or modifying an intermediate base class, changing kernel dispatch patterns, or introducing new class variable protocols, update `docs/ops-design.md` to reflect the change.
- When adding a new op family that inherits `Op` directly, evaluate whether it shares `forward()` flow with an existing family before creating a new base class. Document the decision in the PR.
