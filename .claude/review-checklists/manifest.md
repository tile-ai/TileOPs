# Review checklist: manifest

For `[Maintain]`, `[Refactor][Manifest]`, and any PR that flips a manifest entry's `status`. The manifest is the authoritative spec for op interfaces — every check below exists to keep that authority real.

Load `.claude/domain-rules/manifest-spec.md` before reviewing.

Two non-negotiable principles cut across every event:

- **Reference semantic alignment.** Any change to an op's spec (signature, shape rules, dtype combos, roofline vars) must map back to an authoritative reference — PyTorch public API for `torch.nn.*` / `torch.nn.functional.*` names; the paper or vendor docs otherwise. Reverse-engineering from current TileOps code is forbidden — spec is upstream of code.
- **`status` field truthfulness.** `status: implemented` is a hard claim that code conforms to the entry. The reviewer's job is to *disprove* it, not to take it on faith. Status flips that don't reflect actual conformance corrupt the trust model.

## Add-manifest (new entry)

PRs from the `add-manifest` skill, or any PR that adds a previously-absent op entry.

- [ ] **Reference cited and authoritative.** Entry's `source` field points to PyTorch / paper / vendor docs. Reviewer can derive shape and dtype rules from that link alone.
- [ ] **Signature matches reference exactly.** For `torch.nn.*` / `torch.nn.functional.*` names: parameter names, order, and defaults all match PyTorch's public API. No invented parameters.
- [ ] **Required fields present.** `signature`, `shape_rules`, `dtype_combos`, `roofline`; `kernel_map` and `static_dims` where the family requires them.
- [ ] **Lands as `spec-only`.** New entries never land as `implemented`, regardless of any existing code claiming to be ready.
- [ ] **Validator green.** `scripts/validate_manifest.py` passes with no checks disabled.
- [ ] **No code change.** Diff does not modify `tileops/ops/` or `tileops/kernels/`.

## Fix-manifest (patch existing entry)

PRs from the `fix-manifest` skill — patches one missing structural field (`kernel_map`, `static_dims`) on an existing entry.

- [ ] **Scope is one structural field.** Diff modifies exactly one missing field. Edits to `signature`, `shape_rules`, `dtype_combos`, or `roofline` belong to `add-manifest`, not `fix-manifest` — reject and split.
- [ ] **Reference still aligns.** Other reference-derivable fields (`signature`, `shape_rules`, `dtype_combos`) on the same entry have not silently drifted from the source URL. Spot-check at least one.
- [ ] **Validator green.**
- [ ] **No code change.** Diff does not modify `tileops/ops/` or `tileops/kernels/`.

## Status flip (`spec-only` ↔ `implemented`)

Often bundled with a `[Refactor][Ops]` op-migration PR.

- [ ] **Conformance verified, not asserted.** Reviewer diffs the op's `__init__` and `forward` against the manifest entry field-by-field — names, types, defaults, shapes.
- [ ] **Spec tests actually run.** No `pytest.skip`, `xfail`, or weakened assertion left from the `spec-only` era. Grep the test file before approving.
- [ ] **`FIXME(staged-rollout)` markers tied to this op removed.**
- [ ] **Flip back to `spec-only`** is legitimate only when implementation is removed or known-broken. Challenge any other rationale.
- [ ] **Pure flip.** This PR changes `status` only — no rewrite of `signature` / `shape_rules` / `roofline`. If the entry needs spec edits to match implementation, that is reverse-engineering from code; reject and require a separate `add-manifest`- or `fix-manifest`-style PR.
