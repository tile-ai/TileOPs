# TileOPs Skills — Developer Decision Guide

Per-skill contracts live in each `SKILL.md`; this page maps intent → skill.

## Skills

Op-development skills follow `<verb>-<scope>` naming (`scope ∈ {op, family, manifest}`). Orchestrators are the day-to-day entry points; atomics are sub-skills, invoked standalone only for debugging. Workflow skills operate on PRs / sessions, not ops.

|               | Orchestrator                                              | Atomic                                                                                                                                                                                                                    |
| ------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| per op        | [`align-op`](../.claude/skills/align-op/SKILL.md)         | [`scaffold-op`](../.claude/skills/scaffold-op/SKILL.md) · [`test-op`](../.claude/skills/test-op/SKILL.md) · [`implement-op`](../.claude/skills/implement-op/SKILL.md) · [`bench-op`](../.claude/skills/bench-op/SKILL.md) |
| per op family | [`align-family`](../.claude/skills/align-family/SKILL.md) | [`audit-family`](../.claude/skills/audit-family/SKILL.md)                                                                                                                                                                 |
| manifest      | —                                                         | [`add-manifest`](../.claude/skills/add-manifest/SKILL.md) · [`fix-manifest`](../.claude/skills/fix-manifest/SKILL.md)                                                                                                     |
| workflow      | —                                                         | [`review-tileops`](../.claude/skills/review-tileops/SKILL.md) · [`resolve-tileops`](../.claude/skills/resolve-tileops/SKILL.md) · [`follow-up`](../.claude/skills/follow-up/SKILL.md)                                     |

## Intent → command

| Intent                                                             | Run                                               |
| ------------------------------------------------------------------ | ------------------------------------------------- |
| Align / add a single op to its manifest entry                      | `/align-op <op_name>`                             |
| Classify an op without touching anything                           | `/align-op <op_name> --classify-only`             |
| Migrate every spec-only op in a family                             | `/align-family <family>`                          |
| Read-only audit of a family's spec gaps                            | `/audit-family <family>`                          |
| Generate / re-align a manifest entry from a reference-API docs URL | `/add-manifest <op_name> <ref_url>`               |
| Patch `kernel_map` or `static_dims` on an existing manifest entry  | `/fix-manifest <op_name>`                         |
| Scaffold a fresh op file (bypass orchestrator)                     | `/scaffold-op <op_name>`                          |
| Debug one atomic phase                                             | `/test-op` · `/implement-op` · `/bench-op`        |
| Review a TileOPs PR (single-shot)                                  | `/review-tileops <PR>`                            |
| Review a TileOPs PR (autonomous loop until APPROVE)                | `bash .claude/skills/review-tileops/loop.sh <PR>` |
| Resolve reviewer feedback on a TileOPs PR (per-round driver)       | `/resolve-tileops <PR>`                           |
| Generate follow-up issues for a PR                                 | `/follow-up <PR>`                                 |

## Use-when notes (don't-use-when in `SKILL.md`)

| Skill                  | Use when                                                                                                                                                                         |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `align-op`             | Single op needs alignment after manifest / design change. Cases: `green` (calls `scaffold-op`), `redesign` (archive + rescaffold + port), `minor` (in-place via `implement-op`). |
| `scaffold-op`          | Called by `align-op` green path. Standalone is rare. Refuses if `source.op` exists. Does not emit family protocol vars or optional hooks.                                        |
| `implement-op`         | Called by orchestrators. Structural rewrites go through `align-op --mode=redesign`.                                                                                              |
| `test-op` / `bench-op` | Called by orchestrators.                                                                                                                                                         |
| `align-family`         | Whole family of spec-only ops needs migration. Single op → `align-op`.                                                                                                           |
| `audit-family`         | Read-only conformance check. Also called internally by `align-family`.                                                                                                           |
| `add-manifest`         | New op, or stale entry whose reference-derivable fields drifted. Gap in `kernel_map` / `static_dims` → `fix-manifest`.                                                           |
| `fix-manifest`         | Validator says `kernel_map` or `static_dims` is missing on an existing entry. Other gaps → `add-manifest`. Status flip → `align-op@FLIP_STATUS`.                                 |
| `review-tileops`       | Single-shot or autonomous loop reviewing a PR (separate GitHub identity).                                                                                                        |
| `resolve-tileops`      | Per-round driver for resolving reviewer feedback (`/loop` mode).                                                                                                                 |
| `follow-up`            | Generate up to 3 follow-up issues per invocation, using current session / PR as context.                                                                                         |

## Composition

How orchestrators delegate. Note that orchestrators may delegate to other orchestrators (e.g., `align-family` → `align-op`) as well as to atomic skills.

```text
align-family <family>                    ← per-op-family orchestrator
├─ audit-family
├─ per op: align-op <op_name>            ← full per-op pipeline delegated
└─ [orchestrator] CLEANUP_GATE + CLEANUP + CREATE_PR

align-op <op_name>                       ← per-op orchestrator
├─ [orchestrator] PRE_CHECK
├─ [orchestrator] CLASSIFY
├─ [orchestrator] DISPATCH
│   ├─ green:    scaffold-op
│   ├─ redesign: [orchestrator] ARCHIVE + CLEAR → scaffold-op → PORT → KERNEL_CHECK
│   └─ minor:    implement-op
└─ shared downstream:
    ├─ test-op
    ├─ implement-op                      ← conditional: green/redesign only; skipped on minor (already ran in DISPATCH) and on TEST DONE_SKIP
    ├─ bench-op
    ├─ [orchestrator] REVALIDATE
    ├─ [orchestrator] FLIP_STATUS        ← writes status field only (one of three manifest writers)
    ├─ [orchestrator] CLEANUP
    └─ [orchestrator] REPORT
```

`align-family` delegates each op to `align-op`; never calls `test-op` / `implement-op` / `bench-op` directly and never writes the manifest. `align-op@FLIP_STATUS` is the only writer of `status`. Manifest-trust enforcement: `.claude/rules/manifest-trust-model.md`.

## Trust model

| Resource             | Writer                                                                                                                                                                                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tileops/manifest/`  | `add-manifest` (reference-derivable: `signature.{inputs,outputs,params,shape_rules,dtype_combos}`, `roofline.{flops,bytes,vars}`); `fix-manifest` (on-disk-derivable: `source.kernel_map`, `signature.static_dims`, existing entries only); `align-op@FLIP_STATUS` (`status` only). Disjoint slices. |
| `tileops/ops/**`     | `scaffold-op` creates; `implement-op` edits                                                                                                                                                                                                                                                          |
| `tileops/kernels/**` | No op-dev skill writes kernels (`align-op --mode=redesign` surfaces mismatches via `kernel-check.json`); `resolve-tileops` / `follow-up` may commit kernel edits on a PR branch when triaging review feedback (reactive only)                                                                        |
| `tests/ops/**`       | `test-op`                                                                                                                                                                                                                                                                                            |
| `benchmarks/ops/**`  | `bench-op`                                                                                                                                                                                                                                                                                           |
| PR review state      | `review-tileops` posts reviews; `resolve-tileops` posts replies / resolves threads / commits fixes on non-manifest files                                                                                                                                                                             |
| Follow-up issues     | `follow-up` (≤3 per invocation; may push applied-fix commit on PR branch; never edits PR body)                                                                                                                                                                                                       |
