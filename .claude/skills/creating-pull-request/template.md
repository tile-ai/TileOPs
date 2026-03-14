Closes #<issue-number>

## Summary

- \<what was added/fixed/changed>
- \<what was removed/replaced>

## Test plan

- [x] pre-commit passed
- [x] pytest passed

<!-- OPTIONAL SECTIONS: Delete entirely if not applicable. Never leave empty headers. -->

## Structural Compliance

<!-- Required for new ops or kernel/op changes. Agent-generated — do not edit manually. -->

<details>
<summary>Kernel/Op convention checks (N/N passed)</summary>

| #   | Check                                      | Result |
| --- | ------------------------------------------ | ------ |
| 1   | `_<op>_kernel` closure exists              | PASS   |
| 2   | `@tilelang.jit(out_idx=[...])` wraps inner | PASS   |
| ... | ...                                        | ...    |

</details>

## Benchmark

<!-- Required when PR involves performance changes -->

<paste benchmark log or table here>

## Regression

<!-- Recommended when PR is bugfix or refactor -->

<paste regression test log here>

## Additional context

<!-- Design decisions, related issues, screenshots, etc. -->
