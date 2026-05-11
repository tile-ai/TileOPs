Common rules for every checklist in this folder. Load this file before any specific checklist.

**Load `docs/design/trust-model.md` first.** It defines the layered trust boundaries (manifest → test (incl. `workloads/`) → implementation → benchmark) that every title-specific checklist enforces. Read it before any other checklist in this folder so cross-layer rules can be cited concretely rather than paraphrased.

**Trust-model rules apply uniformly as a review lens.** Cross-layer touches surface as review comments with concrete citation (file:line, parametrize axis, rule pointer). The author responds with rationale; the reviewer judges on content. There is no auto-reject path keyed on directory layout.

The labels `automated`, `needs-review`, `nightshift` mark PR provenance (fully agent-driven). They do not change rule semantics — the same review criteria apply regardless of label set.

Substantive criteria the reviewer cites when a cross-layer diff appears:

- **Oracle origin** — `ref_program` resolves to external authority (PyTorch / NumPy / closed-form / IEEE-754), not an agent-fabricated literal.

- **Coverage set** — not narrowed; deletions only target code paths that no longer exist.

- **New-path coverage** — any added behavior branch is exercised by at least one test.

- **Open set, not exhaustive.** The items in each checklist are the floor; add PR-specific checks as needed.

- **Concrete and decidable.** Every flag cites a concrete pointer — file:line, entry path, field name, parametrize axis, reference URL, test name, or the offending diff line. "Looks reasonable", "may want to verify", "consider revising", "could be clearer" do not qualify.

- **Reviewer restraint.** Verify alignment with the existing reference, spec, or convention; do not propose new content beyond what fixes a flagged item.
