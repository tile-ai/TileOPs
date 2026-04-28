## Boundary

- **OWNS**: `tests/`, `workloads/` (test stage creates workload definitions first)
- **MUST NOT modify**: `tileops/ops/`, `tileops/kernels/`, `benchmarks/`, `tileops/manifest/`
- **MAY READ**: `workloads/`, `tileops/manifest/`

→ [trust-model.md §Test](../../docs/trust-model.md#test) | [testing.md §Tests](../../docs/testing.md#tests)

______________________________________________________________________

- Every test case must trace back to a specific code path, dtype dispatch, or regression. Do not add cases for combinatorial confidence.

- All supported dtypes must be tested. Dtype and shape coverage serve different purposes — do not cross them unless the combination triggers a distinct code path.

- Do not generate test fixtures from tileops/manifest/ workloads. Test parameters are a curated correctness subset.

- Before committing, review test cases added during development. Remove scaffolding tests that verified intermediate implementation steps but do not guard a final code path. Keep only tests that remain valuable after the implementation is stable.

- Run `scripts/test_node_delta.py` before submitting PRs that modify test files. No growth on existing files: nothing to report. Growth on existing files: include script output and one-line justification in PR description. New test files only: no delta report needed.

- Binary operator tests must cover broadcast semantics: bias-add `(B,S,D)+(1,1,D)`, row `(B,S,D)+(B,S,1)`, scalar `(M,N)+(1,1)`. Applies to arithmetic, comparison, logical, and bitwise binary ops.
