## Boundary

- **OWNS**: `tests/`, `workloads/` (test stage creates workload definitions first)
- **MUST NOT modify**: `tileops/ops/`, `tileops/kernels/`, `benchmarks/`, `tileops/ops_manifest.yaml`
- **MAY READ**: `workloads/`, `tileops/ops_manifest.yaml`

→ [trust-model.md §Test](../../docs/trust-model.md#test) | [testing.md §Writing a Test](../../docs/testing.md#writing-a-test)

______________________________________________________________________

- Every test case must trace back to a specific code path, dtype dispatch, or regression. Do not add cases for combinatorial confidence.
- All supported dtypes must be tested. Dtype and shape coverage serve different purposes — do not cross them unless the combination triggers a distinct code path.
- Do not generate test fixtures from ops_manifest.yaml workloads. Test parameters are a curated correctness subset.
- Before committing, review test cases added during development. Remove scaffolding tests that verified intermediate implementation steps but do not guard a final code path. Keep only tests that remain valuable after the implementation is stable.
- Run `scripts/test_node_delta.py` before submitting PRs that modify test files. No growth on existing files: nothing to report. Growth on existing files: include script output and one-line justification in PR description. New test files only: no delta report needed.
