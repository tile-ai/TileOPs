Common rules for every checklist in this folder. Load this file before any specific checklist.

**Load `docs/design/trust-model.md` first.** It defines the layered trust boundaries (manifest → test (incl. `workloads/`) → implementation → benchmark) that every title-specific checklist enforces. Read it before any other checklist in this folder so cross-layer rules can be cited concretely rather than paraphrased.

**Trust enforcement strictness by PR provenance.** The labels applied by foundry's `creating-pull-request` skill and gatekeeper agent — `automated`, `needs-review`, `nightshift` — mark a PR as agent-driven. The label set is fixed and matches foundry's PR-level taxonomy exactly; do not extend it with historical labels (e.g. `all-ai-powered`), project-local labels, or non-label heuristics (author identity, commit-message shape, branch name). If foundry changes its label set, update this directive in lockstep.

- **Strict mode** — PR carries any of `automated` / `needs-review` / `nightshift`. Cross-layer touches forbidden by `docs/design/trust-model.md` are blockers; the reviewer rejects the PR with a rule citation rather than negotiating rationale.
- **Principled mode** — PR carries none of those labels (human-authored). The same trust-model rules apply as a review lens: cross-layer touches surface as comments with rule citations, and the author's stated rationale is acceptance grounds (not bypass). Reviewer documents the rationale in the review thread.

When in doubt about which mode applies, default to principled mode.

- **Open set, not exhaustive.** The items in each checklist are the floor; add PR-specific checks as needed.
- **Concrete and decidable.** Every flag cites a concrete pointer — file:line, entry path, field name, parametrize axis, reference URL, test name, or the offending diff line. "Looks reasonable", "may want to verify", "consider revising", "could be clearer" do not qualify.
- **Reviewer restraint.** Verify alignment with the existing reference, spec, or convention; do not propose new content beyond what fixes a flagged item.
