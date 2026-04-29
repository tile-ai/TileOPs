# Review rule: feature

Applies to PRs prefixed `[Feat]` or `[Enhancement]` — adds new behavior or extends existing behavior (op, kernel, module, skill, tooling).

## Must check

- **Spec conformance**: if the PR adds an op/kernel, its public interface must match the corresponding `tileops/manifest/` entry. If no manifest entry exists, the PR is in the wrong order — manifest PR comes first.
- **Trust model**: feature PRs must NOT modify `tileops/manifest/` (see `.claude/rules/manifest-trust-model.md`). Flag any mixed diff.
- **Test coverage**: new public behavior needs tests. PyTorch is the ground truth for numerical ops.
- **Design fit**: cross-check against `docs/design/architecture.md` module boundaries (M1-M8) and `docs/design/ops-design.md` if the feature is an op.
- **Naming + style**: `.claude/rules/code-style.md` (PascalCase abbreviations, filename casing, docstring style).

## Don't gate on

- Optional polish (extra docstring sections, README updates) unless the change is user-facing API.
- Performance numbers unless the PR claims a perf goal.

## Hard rejects

- Adds an op/kernel without a matching manifest entry.
- Mixes feature code with manifest edits in one PR.
- Hardcoded secrets, `--no-verify`, or `# ruff: noqa` at file scope (`.claude/rules/security.md`, `.claude/rules/code-style.md`).
