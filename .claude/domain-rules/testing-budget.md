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

- Binary operator tests must cover broadcast semantics: bias-add `(B,S,D)+(1,1,D)`, row `(B,S,D)+(B,S,1)`, scalar `(M,N)+(1,1)`. Applies to arithmetic, comparison, logical, and bitwise binary ops.

- When a PR intentionally degrades a test (xfail, skip, weakened assertion) due to a process constraint (e.g. trust model requiring separate manifest and code PRs), mark it with `FIXME(staged-rollout)` using this template:

  ```python
  # FIXME(staged-rollout): <one-line summary of what's degraded>
  #
  # Broken invariant: <what contract is currently violated>
  # Why: <which process constraint requires this temporary state>
  # Cleanup: <concrete condition that triggers removal of this marker>
  ```

  Do not use ad-hoc references like "PR-B" or "next PR" — state the cleanup condition in terms of the technical invariant being restored. These markers are machine-greppable: `grep -rn 'FIXME(staged-rollout)'`.
