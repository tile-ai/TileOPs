# Reviewer Rule List

Cross-reference index for code review. Each rule is one sentence plus a citation
to the authoritative source. Read the citation before applying. Foundry
articulation consumes this file as Source B for source pair 3 (text vs rule);
agents cite the rule ID and follow the citation to ground a finding.

- R1: When implementation does not conform to spec, set `status: spec-only` in the manifest and fix the code in a follow-up PR — never modify the manifest to match code (see manifest-trust-model.md L3).
- R2: When the op name matches a PyTorch op, the manifest signature must match PyTorch's public API verbatim; do not invent parameters (see manifest-trust-model.md L2).
- R3: Do not remove `roofline.vars`, `shape_rules`, or `params` to silence validator errors (see manifest-trust-model.md L4).
- R4: The manifest layer owns signatures, dtype contracts, and roofline; tests and code must read these via `tileops.manifest`, not re-encode them (see trust-model.md §Manifest).
- R5: Tests verify the implementation against the manifest contract; a test that pins implementation details rather than the manifest contract is rejected (see trust-model.md §Test).
- R6: Benchmarks produce numbers and contain no correctness assertions; correctness lives in tests (see trust-model.md §Benchmark).
- R7: Per-op exceptions or fallbacks must not be promoted to base-class shared mechanisms (mixin / class attribute / shared method) within the same migration scope (see ops-design.md §Family-Base Refactoring).
- R8: Op-layer scaffolding follows the seven-step recipe — file header, class declaration, `__init__`, `default_kernel_map` + `forward`, `_infer_output_shapes` + `_validate_dtypes`, `eval_roofline`, package registration (see ops-design.md §Scaffolding an Op from a Manifest Entry).
- R9: Shipped source must not reference issue / PR numbers, AC labels, round numbers, reviewer names, or `Follow-up: #N` pointers — that context belongs in the commit message and PR body (see code-style.md L36).
- R10: Tests intentionally degraded (xfail, skip, weakened assertion) for a process constraint must carry a `FIXME(staged-rollout)` block whose Cleanup line states the invariant to restore (see code-style.md L11).
- R11: Use targeted inline `# noqa: XXXX` only when genuinely needed; do not use file-level lint suppressions like `# ruff: noqa` or `# flake8: noqa` (see code-style.md L5).
- R12: Public Python docstrings follow Google style with a one-sentence summary on the opening line, and abbreviations are expanded on first use (see code-style.md L27, L29).
