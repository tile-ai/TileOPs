# Review checklist: doc

For `[Doc]` and `[Design]`. The bar is highest for `docs/design/`: TileOps is a design-first project, and design docs guide agent development. For non-design paths (READMEs, agent-facing skill docs, comments), apply the same checks but the scope and concision items relax naturally.

When the change touches `docs/design/`, also load `.claude/domain-rules/design-docs.md`.

This checklist is **reviewer-restraint oriented**: its goal is to keep design docs at top-level decisions, not to push authors toward more detail or stricter wording. Do not propose additions; only flag the items below.

## Checklist

- [ ] [REQ] Content is a top-level design decision — target convention, module boundary, or contract. Reject additions of codebase-internal mechanics, file enumerations, or implementation snapshots; they belong in code or api-docs.
- [ ] [REQ] No contradiction with other design docs or `tileops/manifest/`. Cross-check the affected doc against its neighbors; flag any conflicting MUST / SHOULD.
- [ ] [REQ] Concise: no descriptive narration of history, no examples or rationale beyond what is needed to justify the decision.
- [ ] [REQ] Implied code or manifest change is called out — doc either includes the change or links a follow-up issue.
- [ ] [REC] No reintroduction of content already condensed or removed elsewhere.
