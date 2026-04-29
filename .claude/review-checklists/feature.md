# Review checklist: feature

For `[Feat]` and `[Enhancement]`. If the PR adds an op/kernel, the structural axis is covered by `.foundry/mold/op-readiness-checklist.md` (filled by author into `## Structural Readiness`). This checklist covers cross-cutting axes only.

## Checklist

- [ ] Op/kernel public interface matches its `tileops/manifest/` entry. No entry → wrong PR order
- [ ] Diff does not modify `tileops/manifest/` (`.claude/rules/manifest-trust-model.md`)
- [ ] Spot-check at least one `[REQ]` item from `## Structural Readiness` against the diff; flag dishonest pass
- [ ] If feature is an op, follows `docs/design/ops-design.md`
