→ [testing.md §Test case policy](../../docs/design/testing.md#test-case-policy) states which purposes a case may serve, the dtype and shape coverage rules, and what `scripts/test_node_delta.py` reports. [trust-model.md §Test](../../docs/design/trust-model.md#test) states the layer boundaries. What follows is what those leave open.

- `tests/test_workload_placement.py` enforces the workload / test split one way only: it catches a tolerance that drifted into a workload, not a `ref_program` left on a test. The second case is on the author.
- Run `scripts/test_node_delta.py` on every PR touching a test file. What to report is in the linked section; the section does not say to run it unprompted.
- Before committing: drop scaffolding tests that guarded intermediate implementation steps and don't guard any final code path.
- Binary-op tests cover broadcast semantics: bias-add `(B,S,D)+(1,1,D)`, row `(B,S,D)+(B,S,1)`, scalar `(M,N)+(1,1)`. Applies to arithmetic, comparison, logical, bitwise.
