- Input construction and the op's reference computation go in `workloads/<family>.py`; compose them (`class FooTest(FooWorkload, TestBase)`) instead of defining `gen_inputs` or `ref_program` inline. A workload describing only an input shape leaves `ref_program` to the test. Tolerances stay on the test class. `tests/test_workload_placement.py` catches a tolerance in a workload, not a reference left on a test.

→ [trust-model.md §Test](../../docs/design/trust-model.md#test) | [testing.md §Tests](../../docs/design/testing.md#tests)

- Every test case traces to a specific code path, dtype dispatch, or regression. No cases for combinatorial confidence.
- Test all supported dtypes. Don't cross dtype and shape coverage unless the combination triggers a distinct code path.
- Don't generate fixtures from `src/tileops/manifest/` workloads. Test parameters are a curated correctness subset.
- Before committing: drop scaffolding tests that guarded intermediate implementation steps and don't guard any final code path.
- Run `scripts/test_node_delta.py` on PRs touching test files. Growth on existing files → include the script output + a one-line justification in the PR body. New test files only → no delta report.
- Binary-op tests cover broadcast semantics: bias-add `(B,S,D)+(1,1,D)`, row `(B,S,D)+(B,S,1)`, scalar `(M,N)+(1,1)`. Applies to arithmetic, comparison, logical, bitwise.
