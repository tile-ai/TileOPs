# Review rule: design-doc

Applies to PRs prefixed `[Doc]` that touch `docs/design/` or `CLAUDE.md` family files. Pure prose changes (READMEs, top-level guides) take a lighter version of these rules.

Load `.claude/domain-rules/design-docs.md` before reviewing.

## Must check

- **Single source of truth**: the doc must not contradict another design doc or `tileops/manifest/`. If overlap exists, one doc should defer (link) to the other rather than duplicate.
- **Target convention, not history**: design docs state the target convention with MUST/SHOULD. Implementation gaps go into follow-up issues, not into the doc as caveats. (See feedback memory `feedback_design_doc_style.md`.)
- **Cross-references resolve**: links to other docs, manifest paths, code paths actually exist on this branch.
- **Audience match**: agent-facing docs (skills, `tileops-skills.md`) stay terse and decision-oriented; human design docs (`docs/design/*.md`) can be longer but must remain navigable.
- **No code/spec drift**: doc changes that imply a code or manifest change must call that out explicitly and either (a) include the change or (b) link a follow-up issue.

## Don't gate on

- Prose style preferences (sentence length, voice) unless the doc becomes hard to scan.
- Diagram polish unless the diagram is wrong or unrenderable.

## Hard rejects

- Doc claims a behavior that contradicts current code or manifest with no follow-up plan.
- Adds aspirational/"future" sections that aren't actionable — those belong in issues.
- Reintroduces content already condensed/removed from another doc (creates dual sources).
