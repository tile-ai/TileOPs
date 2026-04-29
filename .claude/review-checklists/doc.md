# Review checklist: doc

For `[Doc]` and `[Design]`.

## How to apply

- **Open set, not exhaustive.** The agent may add checks for the PR's specifics; the items below are the floor.
- **Every check must be concrete and decidable.** A check names one of: a contradiction (cite file:line on both sides), a missing follow-up reference, a scope violation (quote the offending line), or a re-introduced removal (cite the prior commit). Anything else does not qualify.
- **Vague feedback is forbidden.** Phrases like "consider revising", "may want to clarify", "could be clearer" are not flags — either locate the violation or stay silent.
- **Stance is reviewer-restraint.** Do not push the author to add design-doc content beyond what fixes a flagged item.

## Design docs (`docs/design/*.md`)

TileOps is a design-first project. Design docs guide agent development; tight scope keeps agents on top-level decisions instead of mechanics. Also load `.claude/domain-rules/design-docs.md`.

- [ ] **Top-level only.** Content is a target convention, module boundary, or contract. Reject added codebase mechanics, file enumerations, or implementation snapshots.
- [ ] **No contradictions.** Cross-check against neighboring design docs and `tileops/manifest/`. Flag conflicting MUST / SHOULD with file:line on both sides.
- [ ] **Concise.** Flag content that does not constrain the decision — history narration, illustrative examples, rationale that doesn't explain a choice.
- [ ] **Drift-free.** If the doc states a target the code or manifest doesn't satisfy, the PR includes the change or links a follow-up issue.

## Other docs

Covers READMEs, `CLAUDE.md` family, agent-facing skill docs, and source comments. Lighter bar — only consistency and drift matter.

- [ ] **No contradictions** with current code, manifest, or design docs. Cite file:line on both sides.
- [ ] **Drift-free.** Implied code or manifest change is included in the PR or linked as a follow-up issue.
