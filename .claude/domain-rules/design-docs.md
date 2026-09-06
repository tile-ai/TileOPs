## `docs/design/*.md`

- Records top-level design decisions only: a target convention, a module boundary, a contract. No file enumerations, line counts, or other implementation snapshots.
- Test a proposed addition against the next five implementations. Wording that would not survive them is a snapshot, not a decision.
- When a reviewer asks to add implementation detail ("list the current files"), push back unless the design itself changed.
- Content that does not constrain a decision — history narration, illustrative examples, rationale that explains nothing chosen — comes out.
- Cross-check every claim against neighboring design docs and `src/tileops/manifest/`. A conflicting MUST / SHOULD is a defect in one of the two; cite `file:line` on both sides.
- A doc stating a target the code or manifest does not satisfy needs the change in the same PR or a linked follow-up issue.

## Other docs

READMEs, the `CLAUDE.md` family, agent-facing rule and checklist files, source comments. Lighter bar — only consistency and drift:

- No contradiction with current code, manifest, or design docs. Cite `file:line` on both sides.
- An implied code or manifest change is in the PR or linked as a follow-up issue.
