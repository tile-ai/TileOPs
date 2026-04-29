# Review checklist: refactor

For `[Refactor]`.

## Checklist

- [ ] Any `xfail` / `skip` / loosened assertion has a `FIXME(staged-rollout)` block per `.claude/rules/code-style.md`
- [ ] If the op has a manifest entry, signature still matches it

## Sub-types

- `[Refactor][Manifest]` → load `manifest.md` instead. `tileops/ops/` and `tileops/kernels/` must NOT change in the same PR.
- `[Refactor][Ops]` / `[Refactor][<Family>]` that flips `status` → also load `manifest.md`.
