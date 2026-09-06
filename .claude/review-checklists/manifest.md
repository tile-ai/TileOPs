For any PR that adds a `src/tileops/manifest/` entry, patches one, or flips one's `status`. The rules the entry must satisfy are in [manifest-spec.md](../domain-rules/manifest-spec.md); what follows is what a reviewer does with them.

The reviewer's job on `status: implemented` is to **disprove** it. It is a hard claim that the code conforms to the entry, and a flip that does not reflect conformance makes the spec unreliable for every reader downstream.

#### New entry

- [ ] **Reference cited and authoritative.** `ref_api` points at PyTorch, the paper, or vendor docs, and the shape and dtype rules are derivable from that link alone.
- [ ] **Required fields present.** `signature`, `shape_rules`, `dtype_combos`, `roofline`; `kernel_map` and `static_dims` where the op's family requires them.
- [ ] **Lands as `spec-only`.** A new entry never lands as `implemented`, whatever existing code claims.
- [ ] **Validator green.** `scripts/validate_manifest.py` passes with no check disabled.

#### Patch to an existing entry

- [ ] **Nothing spec-shaped rides along.** Filling `source.kernel_map` or `signature.static_dims` is derived from the code on disk. An edit to `signature`, `shape_rules`, `dtype_combos` or `roofline` in the same diff is a reference-derived spec change; reject and split.
- [ ] **The reference has not silently drifted.** Spot-check at least one reference-derivable field on the entry against `ref_api`.
- [ ] **Validator green.**

#### Status flip

Often bundled with an op-migration PR.

- [ ] **Conformance verified, not asserted.** Diff the op's `__init__` and `forward` against the entry field by field — names, types, defaults, shapes.
- [ ] **Spec tests actually run.** No `pytest.skip`, `xfail`, or weakened assertion left over from the `spec-only` era. Grep the test file.
- [ ] **`FIXME(staged-rollout)` markers tied to this op removed.**
- [ ] **Pure flip.** The PR changes `status` only. An entry needing spec edits to match the implementation is reverse-engineering from code; reject it and require a separate PR deriving those fields from the reference.
- [ ] **`source.kernel_map` present.** The validator only warns on a missing map; on a flipped entry the reviewer escalates that to a blocker.
- [ ] **`workloads` non-empty and tensor-input shapes complete.** At least 2 workloads (`test_every_op_has_at_least_two_workloads`), each declaring a shape for *every* tensor input in `signature.inputs` — `input_shape` alone is not enough when the op takes more (`mask_shape`, `value_shape: []`, `min_shape`, `max_shape`). Copy from the sibling variant or derive from the op's typical usage shapes.
- [ ] **Flip back to `spec-only`** is legitimate only when the implementation is removed or known-broken. Challenge any other rationale.
