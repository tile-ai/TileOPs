For `[Doc]` and `[Design]`.

A flag in this domain names one of: a contradiction (cite file:line on both sides), a missing follow-up reference, a scope violation (quote the offending line), or a re-introduced removal (cite the prior commit).

#### Design docs (`docs/design/*.md`)

TileOps is a design-first project. Design docs guide agent development; tight scope keeps agents on top-level decisions instead of mechanics. Also load `.claude/domain-rules/design-docs.md`.

- [ ] **Top-level only.** Content is a target convention, module boundary, or contract. Reject added codebase mechanics, file enumerations, or implementation snapshots.
- [ ] **No contradictions.** Cross-check against neighboring design docs and `tileops/manifest/`. Flag conflicting MUST / SHOULD with file:line on both sides.
- [ ] **Concise.** Flag content that does not constrain the decision — history narration, illustrative examples, rationale that doesn't explain a choice.
- [ ] **Drift-free.** If the doc states a target the code or manifest doesn't satisfy, the PR includes the change or links a follow-up issue.

#### Other docs

Covers READMEs, `CLAUDE.md` family, agent-facing skill docs, and source comments. Lighter bar — only consistency and drift matter.

- [ ] **No contradictions** with current code, manifest, or design docs. Cite file:line on both sides.
- [ ] **Drift-free.** Implied code or manifest change is included in the PR or linked as a follow-up issue.
