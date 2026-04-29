# Review checklist: doc

For `[Doc]` and `[Design]`. When the change touches `docs/design/`, also load `.claude/domain-rules/design-docs.md`.

## Checklist

- [ ] [REQ] Doc states the target convention with MUST / SHOULD; implementation gaps go to follow-up issues, not into the doc as caveats
- [ ] [REQ] No dual sources: content already condensed or removed elsewhere is not reintroduced
- [ ] [REQ] Implied code or manifest change is called out — doc either includes the change or links a follow-up issue
- [ ] [REQ] Doc does not contradict current code or manifest without a follow-up plan
- [ ] [REC] No aspirational / "future" sections that aren't actionable — those belong in issues
