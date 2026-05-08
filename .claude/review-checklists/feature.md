For `[Feat]` and `[Enhancement]`. For op/kernel PRs the structural axis is governed by `docs/design/ops-design.md` and `docs/design/ops-design-reference.md`; this checklist covers cross-cutting axes only.

#### Checklist

- [ ] Op/kernel public interface matches its `tileops/manifest/` entry. No entry → wrong PR order
- [ ] Diff does not modify `tileops/manifest/` (`.claude/rules/manifest-trust-model.md`)
- [ ] If feature is an op, spot-check the diff against at least one of the structural checks below; flag any divergence (full rules in `docs/design/ops-design-reference.md`):
  - Class name ≡ manifest entry key
  - `__init__` kwargs derivable from manifest (`static_dims` / `dtype` / `signature.params`)
  - `default_kernel_map` matches `source.kernel_map` verbatim
  - `forward` validation order: `_validate_dtypes` → shape rules → `static_dims` commitment → kernel call
  - `_validate_dtypes` matches manifest `dtype` / `dtype_combos`
  - `eval_roofline` is a plain-Python body (no class-level expression strings, no `ast.parse`)
