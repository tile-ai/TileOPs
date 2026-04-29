# Review rule: doc

Applies to PRs prefixed `[Doc]` or `[Design]` — any documentation change. Covers `docs/`, `CLAUDE.md` family files, agent-facing skill docs, READMEs, and prose-only edits inside source files.

When the change touches `docs/design/`, also load `.claude/domain-rules/design-docs.md`.

## Must check

- **Single source of truth**: the doc must not contradict another doc or `tileops/manifest/`. Where overlap exists, one side should defer (link) rather than duplicate.
- **Target convention, not history**: design docs state the target convention with MUST / SHOULD. Implementation gaps belong in follow-up issues, not as doc caveats.
- **Cross-references resolve**: links to other docs, manifest paths, code paths actually exist on this branch.
- **Audience match**:
  - Agent-facing docs (skills, `tileops-skills.md`) stay terse and decision-oriented.
  - Human design docs (`docs/design/*.md`) can be longer but must remain navigable (clear headings, no buried decisions).
- **No silent code/spec drift**: doc changes that imply a code or manifest change must call that out — either include the change or link a follow-up issue.

## Don't gate on

- Prose style preferences (sentence length, voice) unless the doc becomes hard to scan.
- Diagram polish unless the diagram is wrong or unrenderable.
- Whether a `[Doc]` PR also updates an index/TOC — nice to have, not a blocker.

## Hard rejects

- Doc claims a behavior that contradicts current code or manifest with no follow-up plan.
- Adds aspirational / "future" sections that aren't actionable — those belong in issues.
- Reintroduces content already condensed or removed from another doc (creates dual sources).
