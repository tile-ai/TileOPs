# Manifest Kernel Dispatch — Schema Extension Proposal

Status: **proposal** — design-only. Implementation lands after reviewer agreement.

## Why

Today's `source.kernel_map` declares a flat `dispatch_key → KernelClassName` mapping. Reality is that several Ops select the kernel at runtime based on architecture or op state, so `kernel_map` records only one nominal class per key while the Op's `default_kernel_map` may return a different class on a given device or under a given op-state. The manifest can no longer answer "which kernel classes can run for this op?" without importing the module.

Goal: extend the schema so reviewers, validators, and agents can read the full set of kernels an Op may dispatch to and the condition that selects each, statically from YAML.

## Survey of `default_kernel_map` patterns

Scanned every `tileops/ops/**/*.py` site that defines `default_kernel_map`. Patterns observed:

| ID  | Pattern                                                      | Example sites                                                                                           |
| --- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| A   | Static single kernel: `return {"k": K}`                      | most norm/reduction/elementwise ops                                                                     |
| B   | Static multi-key: `return {"a": K1, "b": K2, "c": K3}`       | MHA bwd (preprocess + kernel + postprocess)                                                             |
| C   | Hopper boolean: `K_hopper if is_hopper() else K_fallback`    | `MHAFwd`, `MHABwd`, `MLADecode`, `GQASlidingWindowFwd`, `GQASlidingWindowVarlenFwd`                     |
| D   | Op-state conditional: `if self.fuse_rope: …`                 | `GqaFwd`, `GqaBwd`                                                                                      |
| E   | Predicate helper over op state + arch (best-of-N)            | `GqaFwd`, `GqaPrefillFwd`, `GqaPrefillVarlenFwd`, `GqaDecode` (all use a `_select_*_kernel_cls` helper) |
| F   | SM-version table: `if get_sm_version() in K.supported_archs` | `Gemm` (gemv/gemm split)                                                                                |
| G   | Lazy / shape-aware: `if self._last_m is None: …`             | `LayerNorm`                                                                                             |

Patterns A and B are already covered by the current flat `kernel_map`. C, D, E, F describe runtime conditionals on `arch` and/or `params`. G is a shape-conditioned reuse path that does not change correctness identity — out of scope for this proposal.

## Schema variants considered

Three sketches were evaluated against expressiveness, static validation, and reviewer ergonomics. Variant 2 is recommended; variants 1 and 3 are summarized for context.

### Variant 1 — extend `kernel_map` values per key

```yaml
source:
  kernel: tileops/kernels/attention/gqa_fwd.py
  kernel_map:
    qkv: QkvKernel                          # legacy str form still valid
    rope:
      dispatch:
        - kernel: HopperRopeKernel          # C
          when: {arch: {sm: ">=90"}}
        - kernel: FallbackRopeKernel
          when: otherwise
    attention:
      dispatch:
        - kernel: FusedRopeAttentionKernel  # D
          when: {op: {fuse_rope: true}}
        - kernel: PlainAttentionKernel
          when: otherwise
```

Trade-offs: best migration story (legacy form stays valid). Weak when several dispatch keys must switch together — they end up scattered across keys and reviewers can't see the joint decision in one place.

### Variant 2 — ordered `kernel_selection` table (recommended)

```yaml
source:
  kernel: tileops/kernels/attention/gqa_fwd.py
  op: tileops/ops/attention/gqa.py

  kernel_selection:
    - when: {all: [{arch: {sm: ">=90"}}, {op: {fuse_rope: true}}]}
      kernels:                              # B + correlated C/D as one row
        gqa_fwd_kernel: GQAFwdHopperFusedRopeKernel
    - when: {arch: {sm: ">=90"}}
      kernels:
        gqa_fwd_kernel: GQAFwdHopperKernel
    - when: {op: {fuse_rope: true}}
      kernels:
        gqa_fwd_kernel: GQAFwdFusedRopeKernel
    - when: otherwise
      kernels:
        gqa_fwd_kernel: GQAFwdKernel
```

Trade-offs: each row declares the full dispatch-key set selected jointly, so correlated multi-key decisions stay co-located. Verbose for ops with only one row, but Variant 2 keeps the legacy flat `kernel_map: str → str` form as sugar for the unconditional case so simple ops don't pay the cost.

Static validation is strongest here:

- exactly one `otherwise` row, last
- every row's `kernels` shares the same dispatch-key set
- every kernel class name resolves under `source.kernel`
- every condition references a known `signature.params` field or the closed `arch` vocabulary

### Variant 3 — candidate set with named predicates

```yaml
source:
  kernel_map:
    attention:
      candidates:
        - kernel: HopperFusedAttentionKernel
          when:
            predicate: supports_hopper_fused_rope
            uses: {arch: [sm], op: [fuse_rope, causal]}
        - kernel: Sm80AttentionKernel
          when: {all: [{arch: {sm: ">=80"}}, {op: {causal: true}}]}
        - kernel: PlainAttentionKernel
          when: otherwise
```

Trade-offs: best fit for pattern E (best-of-N helpers), since predicate names track code concepts. Cost: predicate semantics live outside the manifest unless we maintain a registry, so reviewers see "what may dispatch" but not always "why" without external docs.

### Comparison

| Variant                             | Covers A   | B       | C   | D   | E       | F             | Static validation                          | Reviewer ergonomics                                      |
| ----------------------------------- | ---------- | ------- | --- | --- | ------- | ------------- | ------------------------------------------ | -------------------------------------------------------- |
| 1. per-key `dispatch` lists         | ✓ (legacy) | ✓       | ✓   | ✓   | partial | partial       | strong                                     | best for independent keys, scattered when keys correlate |
| 2. ordered `kernel_selection` table | ✓ (legacy) | ✓       | ✓   | ✓   | ✓       | ✓ (sm ranges) | strongest (exhaustiveness, schema closure) | best when dispatch is a runtime decision tree            |
| 3. candidate + named predicates     | ✓          | partial | ✓   | ✓   | ✓       | ✓             | medium (predicate semantics opaque)        | depends on registry quality                              |

## Recommendation

Adopt **Variant 2 (`kernel_selection`)** as the canonical new form. Keep the existing flat `kernel_map: str → str` as legacy sugar — equivalent to a single `otherwise` row — so Pattern A/B sites do not need to migrate.

Rationale:

- correlated multi-key decisions stay readable as a row
- exhaustiveness invariant (`otherwise` last, exactly once) is enforceable
- the `when` DSL has a closed vocabulary (`arch.sm`, `op.<param>`, `all`/`any`/`not`, `otherwise`) — every term is statically resolvable against the manifest's own `signature.params` plus a small fixed `arch` namespace, so the validator can reject typos without importing the module
- pattern G (shape-conditioned lazy reuse) stays out of scope — it does not change correctness identity, so leaving it undeclared keeps the schema honest

## Validator enforcement decision

**Recommend phased enforcement** to match the manifest's existing posture for `kernel_map`:

| Stage            | Action                                                                                                                                      | Trigger                    |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| Parse            | Reject malformed `kernel_selection` (missing `when`, missing `otherwise`, duplicate keys per row)                                           | always                     |
| Reference        | Reject unknown kernel class names — must be declared in or imported by `source.kernel` files (AST scan, no module import)                   | always                     |
| Vocabulary       | Reject `when` terms outside the closed DSL (`arch.sm`, `op.<param>`, `all`/`any`/`not`, `otherwise`)                                        | always                     |
| Coverage         | Warn (not error) when an Op declares `kernel_selection` but the `default_kernel_map` body cannot be statically tied back to one of the rows | `status: implemented` only |
| Backwards-compat | Accept legacy flat `kernel_map: str → str` indefinitely (treat as single `otherwise` row)                                                   | always                     |

`Coverage` stays a warning rather than a hard error because pattern E sites use helper functions whose return value the validator cannot resolve without dataflow analysis. A warning is honest about that gap; a hard error would force false declarations.

## Out of scope (this proposal)

- Implementation of the validator changes, AST scan, or migration of existing entries
- Pattern G shape-conditioned lazy reuse
- Per-kernel autotune metadata
- Runtime dispatch-strategy declaration (this proposal documents *what may dispatch* and *under what condition*, not *how the dispatch is implemented*)

These belong in follow-up issues that reference this design.
