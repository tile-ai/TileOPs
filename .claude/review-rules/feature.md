# Review rule: feature

For `[Feat]` and `[Enhancement]`. If the PR adds an op/kernel, the structural axis is covered by `.foundry/mold/op-readiness-checklist.md` (filled by author into `## Structural Readiness`). This rule covers cross-cutting axes only.

## Checklist

- [ ] [REQ] Op/kernel public interface matches its `tileops/manifest/` entry. No entry → wrong PR order
- [ ] [REQ] Diff does not modify `tileops/manifest/` (`.claude/rules/manifest-trust-model.md`)
- [ ] [REQ] Spot-check at least one `[REQ]` item from `## Structural Readiness` against the diff; flag dishonest pass
- [ ] [REC] New module placement matches `docs/design/architecture.md` M1–M8 boundaries
- [ ] [REC] If feature is an op, follows `docs/design/ops-design.md`
