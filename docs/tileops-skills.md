# TileOPs Skills — Developer Decision Guide

Skills this repo provides for TileOPs op development: what each does, when to use it, when not to. Authoritative per-skill contracts live in each `SKILL.md`; this page is the human-facing map.

Naming is **verb-noun**. The verb is the action; the noun (`op` or `family`) is the scope.

## At a glance

|                | Orchestrator                                              | Atomic                                                                                                                                                                                                                    |
| -------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **per op**     | [`align-op`](../.claude/skills/align-op/SKILL.md)         | [`scaffold-op`](../.claude/skills/scaffold-op/SKILL.md) · [`test-op`](../.claude/skills/test-op/SKILL.md) · [`implement-op`](../.claude/skills/implement-op/SKILL.md) · [`bench-op`](../.claude/skills/bench-op/SKILL.md) |
| **per family** | [`align-family`](../.claude/skills/align-family/SKILL.md) | [`audit-family`](../.claude/skills/audit-family/SKILL.md)                                                                                                                                                                 |

Orchestrators are the day-to-day entry points. Atomics are their sub-skills — standalone invocation is for debugging.

## What do I want to do?

| Intent                                                            | Run                                        |
| ----------------------------------------------------------------- | ------------------------------------------ |
| Align / add a single op to its manifest entry (the common case)   | `/align-op <op_name>`                      |
| Find out which case an op is in, without touching anything        | `/align-op <op_name> --classify-only`      |
| Migrate every spec-only op in a whole family (historical backlog) | `/align-family <family>`                   |
| Read-only audit of a family's spec gaps                           | `/audit-family <family>`                   |
| Scaffold a fresh op file, bypassing the orchestrator              | `/scaffold-op <op_name>`                   |
| Debug one atomic phase by hand                                    | `/test-op` · `/implement-op` · `/bench-op` |

## Skills in detail

Each block names the skill, its one-line purpose, clear **use when** / **don't use when** guidance, and a link to the authoritative `SKILL.md`.

### `align-op`  ·  per-op orchestrator

Brings a single op into alignment with its manifest entry. Classifies into one of three cases and dispatches internally; runs the shared downstream (test → bench → validate → flip status → report).

- **Cases.** `green` (no code yet → calls `scaffold-op`), `redesign` (archive + rescaffold + port), `minor` (in-place edit via `implement-op`).
- **Use when.** You want to add or re-align a single op after a manifest or design-doc change.
- **Don't use when.** You need to batch-migrate a whole family — use `align-family` instead.
- **Contract:** [SKILL.md](../.claude/skills/align-op/SKILL.md)

### `align-family`  ·  per-family orchestrator

Drives the historical migration of an entire op family. Audits, delegates each per-op alignment to `align-op`, then handles family-scoped concerns: cross-op cleanup (dual-path removal) and PR creation. The family orchestrator never calls `test-op` / `implement-op` / `bench-op` directly and never writes `ops_manifest.yaml`.

- **Use when.** You have a whole family of spec-only ops to migrate.
- **Don't use when.** Only one op needs attention — use `align-op`.
- **Contract:** [SKILL.md](../.claude/skills/align-family/SKILL.md)

### `scaffold-op`  ·  per-op atomic

Writes a new T2 (L1-direct) op file from one manifest entry by following the 7-step playbook in `docs/ops-design.md`. Emits the 17 mechanical slots.

- **Use when.** Called by `align-op` on the green path; rarely needed standalone.
- **Don't use when.** `source.op` already exists — PRE_CHECK refuses. Use `align-op --mode=redesign`, which archives the old file first.
- **Don't expect.** Family protocol variables (`_op_kind`, `_kernel_key`, …) or optional hooks (`_pad_value`, `_validate_dim`, …). Those are op-specific business logic, outside the 17 mechanical slots.
- **Contract:** [SKILL.md](../.claude/skills/scaffold-op/SKILL.md)

### `implement-op`  ·  per-op atomic

Edits an existing op file to match the manifest-declared interface, making spec tests pass.

- **Use when.** Called by orchestrators.
- **Don't use when.** The change is a structural rewrite — `align-op --mode=redesign` archives the old file and regenerates cleanly before implementing.
- **Contract:** [SKILL.md](../.claude/skills/implement-op/SKILL.md)

### `test-op`  ·  per-op atomic

Writes tests for the target spec using PyTorch as ground truth; verifies they fail on current code (the TDD seed before `implement-op`).

- **Use when.** Called by orchestrators.
- **Contract:** [SKILL.md](../.claude/skills/test-op/SKILL.md)

### `bench-op`  ·  per-op atomic

Fixes the benchmark file to compile against the new op interface. Runs it, fixes errors, repeats until it produces numbers.

- **Use when.** Called by orchestrators.
- **Contract:** [SKILL.md](../.claude/skills/bench-op/SKILL.md)

### `audit-family`  ·  per-family atomic

Compares each op's code signature against its manifest spec, classifies gaps (`ready` / `semantic_gap` / `blocked`), writes `.foundry/migrations/<family>.json`.

- **Use when.** You want read-only inspection of a family's current conformance. Also called internally by `align-family`.
- **Contract:** [SKILL.md](../.claude/skills/audit-family/SKILL.md)

## Composition

How orchestrators delegate to atomic skills.

```text
align-family <family>                    ← per-family orchestrator
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
    ├─ [orchestrator] FLIP_STATUS        ← only manifest writer
    ├─ [orchestrator] CLEANUP
    └─ [orchestrator] REPORT
```

`align-family`'s per-op loop is a single `align-op` invocation — the family orchestrator does not call `test-op` / `implement-op` / `bench-op` directly, and it never writes the manifest; `align-op`'s FLIP_STATUS is the sole manifest-write site.

## Trust model  ·  who may write what

| Resource                          | Writer                                                                                                                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ops_manifest.yaml`               | Only `align-op` at `FLIP_STATUS`. No atomic skill writes the manifest; `align-family` delegates to `align-op` and never writes it directly.                        |
| `tileops/ops/**` op files         | `scaffold-op` creates; `implement-op` edits.                                                                                                                       |
| `tileops/kernels/**` kernel files | No TileOPs skill writes kernels. `align-op --mode=redesign` surfaces mismatches via `kernel-check.json`; a future `kernel-align` skill will own kernel-layer work. |
| `tests/ops/**`                    | `test-op`.                                                                                                                                                         |
| `benchmarks/ops/**`               | `bench-op`.                                                                                                                                                        |

## Maintenance

- **Per-skill blocks above** mirror each skill's `description` frontmatter. Edit the frontmatter first; update the matching block here to stay consistent.
- **At-a-glance matrix, intent table, use/don't-use rules, composition diagram, trust-model table**: hand-maintained. Add entries when a new skill lands; remove when one is retired.
- **Authoritative skill list**: `ls .claude/skills/` is the source of truth. Every directory there should appear in the at-a-glance matrix and have a block in "Skills in detail".
- **Lint automation**: none at 7-skill scale. Revisit if drift becomes observable or the count passes ~15.
