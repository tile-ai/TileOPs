# TileOPs Skills — Developer Decision Guide

Skills this repo provides for TileOPs op development and PR workflow: what each does, when to use it, when not to. Authoritative per-skill contracts live in each `SKILL.md`; this page is the human-facing map.

Naming for op-development skills follows a verb-noun convention. The verb is the action; the noun is the scope (`op`, `family`, or `manifest`). Workflow skills are named after the artifact they operate on.

## At a glance

|               | Orchestrator                                              | Atomic                                                                                                                                                                                                                    |
| ------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| per op        | [`align-op`](../.claude/skills/align-op/SKILL.md)         | [`scaffold-op`](../.claude/skills/scaffold-op/SKILL.md) · [`test-op`](../.claude/skills/test-op/SKILL.md) · [`implement-op`](../.claude/skills/implement-op/SKILL.md) · [`bench-op`](../.claude/skills/bench-op/SKILL.md) |
| per op family | [`align-family`](../.claude/skills/align-family/SKILL.md) | [`audit-family`](../.claude/skills/audit-family/SKILL.md)                                                                                                                                                                 |
| manifest      | —                                                         | [`add-manifest`](../.claude/skills/add-manifest/SKILL.md) · [`fix-manifest`](../.claude/skills/fix-manifest/SKILL.md)                                                                                                     |
| workflow      | —                                                         | [`review-tileops`](../.claude/skills/review-tileops/SKILL.md) · [`resolve-tileops`](../.claude/skills/resolve-tileops/SKILL.md) · [`follow-up`](../.claude/skills/follow-up/SKILL.md)                                     |

Orchestrators are the day-to-day entry points for op work. Atomics are their sub-skills — standalone invocation is for debugging. Manifest skills are standalone editors of `tileops/manifest/` and have no orchestrator: they precede op-layer work, not contain it. Workflow skills operate on PRs and sessions, not on ops; each is invoked directly with a PR number.

## What do I want to do?

| Intent                                                                | Run                                        |
| --------------------------------------------------------------------- | ------------------------------------------ |
| Align / add a single op to its manifest entry (the common case)       | `/align-op <op_name>`                      |
| Find out which case an op is in, without touching anything            | `/align-op <op_name> --classify-only`      |
| Migrate every spec-only op in a whole family (historical backlog)     | `/align-family <family>`                   |
| Read-only audit of a family's spec gaps                               | `/audit-family <family>`                   |
| Generate / re-align a manifest entry from a reference-API docs URL    | `/add-manifest <op_name> <ref_url>`        |
| Patch `kernel_map` or `static_dims` on an existing manifest entry     | `/fix-manifest <op_name>`                  |
| Scaffold a fresh op file, bypassing the orchestrator                  | `/scaffold-op <op_name>`                   |
| Debug one atomic phase by hand                                        | `/test-op` · `/implement-op` · `/bench-op` |
| Review a TileOPs PR (single-shot or autonomous loop)                  | `/review-tileops <PR>`                     |
| Resolve reviewer feedback on a TileOPs PR (per-round driver)          | `/resolve-tileops <PR>`                    |
| Generate follow-up issues for a PR (using current session as context) | `/follow-up <PR>`                          |

## Skills in detail

Each block names the skill, its one-line purpose, clear use-when / don't-use-when guidance, and a link to the authoritative `SKILL.md`. Op-, family-, and manifest-scoped skills get full per-block detail. Workflow skills are summarized in a single table at the end of this section; their per-block detail lives in their own `SKILL.md`.

### per-op

**align-op** — per-op orchestrator. Brings a single op into alignment with its manifest entry. Classifies into one of three cases and dispatches internally; runs the shared downstream (test → bench → validate → flip status → report).

- **Cases.** `green` (no code yet → calls `scaffold-op`), `redesign` (archive + rescaffold + port), `minor` (in-place edit via `implement-op`).
- **Use when.** You want to add or re-align a single op after a manifest or design-doc change.
- **Don't use when.** You need to batch-migrate a whole family — use `align-family` instead.
- **Contract:** [SKILL.md](../.claude/skills/align-op/SKILL.md)

**scaffold-op** — per-op atomic. Writes a new T2 (L1-direct) op file from one manifest entry by following the 7-step playbook in `docs/design/ops-design.md`. Emits the 17 mechanical slots.

- **Use when.** Called by `align-op` on the green path; rarely needed standalone.
- **Don't use when.** `source.op` already exists — PRE_CHECK refuses. Use `align-op --mode=redesign`, which archives the old file first.
- **Don't expect.** Family protocol variables (`_op_kind`, `_kernel_key`, …) or optional hooks (`_pad_value`, `_validate_dim`, …). Those are op-specific business logic, outside the 17 mechanical slots.
- **Contract:** [SKILL.md](../.claude/skills/scaffold-op/SKILL.md)

**implement-op** — per-op atomic. Edits an existing op file to match the manifest-declared interface, making spec tests pass.

- **Use when.** Called by orchestrators.
- **Don't use when.** The change is a structural rewrite — `align-op --mode=redesign` archives the old file and regenerates cleanly before implementing.
- **Contract:** [SKILL.md](../.claude/skills/implement-op/SKILL.md)

**test-op** — per-op atomic. Writes tests for the target spec using PyTorch as ground truth; verifies they fail on current code (the TDD seed before `implement-op`).

- **Use when.** Called by orchestrators.
- **Contract:** [SKILL.md](../.claude/skills/test-op/SKILL.md)

**bench-op** — per-op atomic. Fixes the benchmark file to compile against the new op interface. Runs it, fixes errors, repeats until it produces numbers.

- **Use when.** Called by orchestrators.
- **Contract:** [SKILL.md](../.claude/skills/bench-op/SKILL.md)

### per-op-family

**align-family** — per-op-family orchestrator. Drives the historical migration of an entire op family. Audits, delegates each per-op alignment to `align-op`, then handles family-scoped concerns: cross-op cleanup (dual-path removal) and PR creation. The family orchestrator never calls `test-op` / `implement-op` / `bench-op` directly and never writes `tileops/manifest/`.

- **Use when.** You have a whole family of spec-only ops to migrate.
- **Don't use when.** Only one op needs attention — use `align-op`.
- **Contract:** [SKILL.md](../.claude/skills/align-family/SKILL.md)

**audit-family** — per-op-family atomic. Compares each op's code signature against its manifest spec, classifies gaps (`ready` / `semantic_gap` / `blocked`), writes `.foundry/migrations/<family>.json`.

- **Use when.** You want read-only inspection of a family's current conformance. Also called internally by `align-family`.
- **Contract:** [SKILL.md](../.claude/skills/audit-family/SKILL.md)

### manifest

**add-manifest** — manifest atomic. Reads a reference-API docs URL (PyTorch / equivalent) and writes the auto-derivable fields of a manifest entry (`signature.{inputs,outputs,params,shape_rules,dtype_combos}`, `roofline` for well-known ops). Idempotent: human-curated fields (`workloads`, `parity_opt_out`, `source.*`, `status`, `family`, `ref_api`) are preserved verbatim if the entry already exists, defaulted otherwise. Same invocation works for greenfield and re-alignment.

- **Use when.** Adding a new op, or re-aligning a stale entry whose signature / shape rules / dtype combos / roofline have drifted from the reference.
- **Don't use when.** The gap is `kernel_map` or `static_dims` — those come from on-disk op / kernel code, not the reference; use `fix-manifest`.
- **Contract:** [SKILL.md](../.claude/skills/add-manifest/SKILL.md)

**fix-manifest** — manifest atomic. Surgical patch of an existing manifest entry for fields derived from on-disk op / kernel evidence — `source.kernel_map` and `signature.static_dims`. Diagnoses the missing field via the validator, reads the op file to infer the patch payload, writes the single-field change, opens a manifest PR.

- **Allowed fields.** `kernel_map`, `static_dims` only. Reference-derivable fields (`signature.*`, `shape_rules`, `dtype_combos`, `roofline`) belong to `add-manifest`.
- **Use when.** Validator says `kernel_map` or `static_dims` is missing on an existing entry. `kernel_map` is the most common case — it's required by `align-op`'s PRE_CHECK.
- **Don't use when.** The entry doesn't exist (`add-manifest`); the gap is in a reference-derivable field (`add-manifest` re-aligns the whole entry from the reference URL); you want to flip `status` (that is `align-op`'s `FLIP_STATUS`).
- **Contract:** [SKILL.md](../.claude/skills/fix-manifest/SKILL.md)

### workflow

Workflow skills operate on PRs and sessions, not on ops. They are concise on this page by design — full contracts and step-by-step procedures live in each skill's `SKILL.md`.

| Skill               | Purpose                                                                                        | Contract                                               |
| ------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| **review-tileops**  | Review a tile-ai/TileOPs PR (single-shot or autonomous multi-round loop until APPROVE).        | [SKILL.md](../.claude/skills/review-tileops/SKILL.md)  |
| **resolve-tileops** | Per-round driver for resolving reviewer feedback on a TileOPs PR (developer side, /loop mode). | [SKILL.md](../.claude/skills/resolve-tileops/SKILL.md) |
| **follow-up**       | Introspect a development session / PR and generate up to 3 follow-up issues for deferred work. | [SKILL.md](../.claude/skills/follow-up/SKILL.md)       |

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

`align-family`'s per-op loop is a single `align-op` invocation — the family orchestrator does not call `test-op` / `implement-op` / `bench-op` directly, and it never writes the manifest. Among the op- and family-scoped skills, `align-op`'s `FLIP_STATUS` is the only manifest writer (and writes only the `status` field). Manifest-scoped skills (`add-manifest`, `fix-manifest`) write disjoint slices — see the trust-model table below. Workflow skills never write `tileops/manifest/` regardless of what files the PR diff includes (manifest changes are reserved for the manifest skills above); `resolve-tileops` and `follow-up` may commit fixes to other files (op / kernel / tests / docs) on the PR branch when triaging review comments. Their primary surface remains PR review state, session retrospectives, and follow-up issues.

## Trust model — who may write what

| Resource                          | Writer                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tileops/manifest/`               | Three writers, disjoint slices. `add-manifest` writes reference-derivable fields (`signature.{inputs,outputs,params,shape_rules,dtype_combos}`, `roofline.{flops,bytes,vars}` for well-known ops); preserves `workloads`, `parity_opt_out`, `source.*`, `status`, `family`, `ref_api` verbatim if the entry exists, defaults otherwise. `fix-manifest` writes on-disk-derivable fields (`source.kernel_map`, `signature.static_dims`) on existing entries only. `align-op@FLIP_STATUS` writes only `status`. No other skill writes the manifest. |
| `tileops/ops/**` op files         | `scaffold-op` creates; `implement-op` edits.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `tileops/kernels/**` kernel files | No TileOPs skill writes kernels. `align-op --mode=redesign` surfaces mismatches via `kernel-check.json`; a future `kernel-align` skill will own kernel-layer work.                                                                                                                                                                                                                                                                                                                                                                               |
| `tests/ops/**`                    | `test-op`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `benchmarks/ops/**`               | `bench-op`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| PR review state (GitHub)          | `review-tileops` posts review comments and verdicts (does not write repo files); `resolve-tileops` posts replies, resolves threads, and commits fixes to non-manifest PR files when triaging accepted comments.                                                                                                                                                                                                                                                                                                                                  |
| Follow-up issues (GitHub)         | `follow-up` creates up to 3 issues per invocation and may push an applied-fix commit on the PR branch. Never edits the PR body.                                                                                                                                                                                                                                                                                                                                                                                                                  |

## Maintenance

- **Per-skill blocks above** mirror each skill's `description` frontmatter. Edit the frontmatter first; update the matching block (op / family / manifest categories) or table row (workflow category) here to stay consistent.
- **At-a-glance matrix, intent table, use/don't-use rules, composition diagram, trust-model table**: hand-maintained. Add entries when a new skill lands; remove when one is retired.
- **Authoritative skill list**: this guide covers every skill that lives under `.claude/skills/`. Op-, family-, and manifest-scoped skills get full per-block detail; workflow skills get a one-line table row pointing at their `SKILL.md`. Every directory under `.claude/skills/` must be represented exactly once.
- **Lint automation**: none at current scale. Revisit if drift becomes observable or the skill count grows substantially.
