For `[Feat]` and `[Enhancement]`. For op/kernel PRs the structural axis is governed by `docs/design/ops-design.md` (Slots S1–S21) and `docs/design/ops-design-reference.md`; this checklist covers cross-cutting axes only.

#### Checklist

- [ ] Op/kernel public interface matches its `tileops/manifest/` entry. No entry → wrong PR order
- [ ] Diff does not modify `tileops/manifest/` (`.claude/rules/manifest-trust-model.md`)
- [ ] If feature is an op, spot-check the diff against at least one of these `docs/design/ops-design.md` slots; flag any divergence:
  - S6 — class name ≡ manifest key
  - S12 — `__init__` kwargs derivable from manifest
  - S14 — `default_kernel_map` matches `source.kernel_map` verbatim
  - S16 — `forward` validation order
  - S18 — `_validate_dtypes` matches `dtype_combos`
  - S19 — `eval_roofline` plain-Python body
