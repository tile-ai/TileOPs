## Boundary

- **OWNS**: `tests/`, `workloads/` (test stage creates workload definitions first)
- **MUST NOT WRITE**: `tileops/ops/`, `tileops/kernels/`, `benchmarks/`, `tileops/manifest/`

→ [trust-model.md §Test](../../docs/design/trust-model.md#test) | [testing.md §Tests](../../docs/design/testing.md#tests)

______________________________________________________________________

- Every test case traces to a specific code path, dtype dispatch, or regression. No cases for combinatorial confidence.
- Test all supported dtypes. Don't cross dtype and shape coverage unless the combination triggers a distinct code path.
- Don't generate fixtures from `tileops/manifest/` workloads. Test parameters are a curated correctness subset.
- Before committing: drop scaffolding tests that guarded intermediate implementation steps but no final code path.
- Run `scripts/test_node_delta.py` on PRs touching test files. Growth on existing files → include the script output + a one-line justification in the PR body. New test files only → no delta report.
- Binary-op tests cover broadcast semantics: bias-add `(B,S,D)+(1,1,D)`, row `(B,S,D)+(B,S,1)`, scalar `(M,N)+(1,1)`. Applies to arithmetic, comparison, logical, bitwise.
- Skill development tests stay local — never commit anything under `.claude/skills/`.
