# TileOPs Skills — Developer Decision Guide

Skills this repo provides for TileOPs op development, plus when to use each and when not to. For the authoritative per-skill contract (arguments, workflow, termination conditions), open the linked `SKILL.md`.

Naming convention: **verb-noun**. The noun (`op` or `family`) tells you the scope; the verb tells you the action. Orchestrator skills compose atomic skills — see [Composition](#composition) below.

## Which skill should I use?

"I want to…"

### … align or add a single op to its manifest entry

→ **[`/align-op <op_name>`](../.claude/skills/align-op/SKILL.md)**

This is the per-op entry for almost every day-to-day workflow. The skill internally classifies into three cases and dispatches:

- **green** — no op code yet (scaffolds from the manifest)
- **redesign** — op exists but the interface was restructured (archives old file → rescaffolds → ports business logic)
- **minor** — small manifest delta (in-place edit)

Omit `--mode` to auto-classify (green is auto-detected; redesign vs minor is prompted). Pass `--mode=green|redesign|minor` to force.

### … ask "which case is this op in?" without running anything

→ **`/align-op <op_name> --classify-only`**

Writes `.foundry/plan/<op_name>/mode.json` and exits. Zero side effects. Useful for triage before you commit to a workflow.

### … migrate a whole op family (historical backlog)

→ **[`/align-family <family>`](../.claude/skills/align-family/SKILL.md)**

Family-scoped orchestrator. Audits the family, pipelines every `spec-only` op through test → implement → bench → flip_status, handles cross-op cleanup (dual-path removal), creates the PR.

### … audit a family's spec-conformance gap without migrating anything

→ **[`/audit-family <family>`](../.claude/skills/audit-family/SKILL.md)**

Writes `.foundry/migrations/<family>.json` classifying every op as `ready` / `semantic_gap` / `blocked`. Read-only inspection.

### … run a single atomic phase by hand (debugging an orchestrator)

→ **[`/test-op`](../.claude/skills/test-op/SKILL.md)**, **[`/implement-op`](../.claude/skills/implement-op/SKILL.md)**, **[`/bench-op`](../.claude/skills/bench-op/SKILL.md)**

Normally these are invoked as sub-skills by `align-op` or `align-family`. Direct invocation is for debugging the orchestrator — rarely the right entry point.

### … scaffold a fresh op file directly (bypassing the orchestrator)

→ **[`/scaffold-op <op_name>`](../.claude/skills/scaffold-op/SKILL.md)**

`align-op` already calls this internally for the green path. Prefer `align-op` unless you are scripting around the orchestrator or debugging scaffold emission itself.

## When NOT to use a particular skill

### `scaffold-op` (atomic)

- **Don't** invoke directly when `source.op` already exists — PRE_CHECK refuses to overwrite. Use `align-op --mode=redesign`; the orchestrator archives the old file before rescaffolding.
- **Don't** expect it to emit family protocol variables (`_op_kind`, `_kernel_key`, …) or optional hooks (`_pad_value`, `_validate_dim`, …). Those are op-specific business logic that must be hand-ported — `align-op`'s redesign path owns that step.

### `align-family` (orchestrator)

- **Don't** use for single-op work — `align-op` is the per-op entry. `align-family` is the right choice only when you want to batch-migrate every `spec-only` op in a family.
- **Don't** use if the family has no `spec-only` ops — run `audit-family` first, or just check `ops_manifest.yaml`.

### `implement-op` / `test-op` / `bench-op` (atomic)

- **Don't** invoke standalone unless debugging — they are normally sub-skills inside `align-op` or `align-family`.
- **Don't** use `implement-op` for full rewrites — prefer `align-op --mode=redesign`, which archives the old file before regenerating.

### `audit-family` (atomic)

- Fine to invoke standalone for a read-only inspection. If you find yourself running audit + manually doing the per-op migration, just use `align-family` — it does the audit internally.

## Skill catalog

The purpose line comes from each skill's `SKILL.md` frontmatter `description` field — edit that and update this row to match.

| Skill                                                     | Scope      | Tier         | Purpose                                                                                                                                                                                                       |
| --------------------------------------------------------- | ---------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`align-op`](../.claude/skills/align-op/SKILL.md)         | per op     | orchestrator | Brings a single op into alignment with its manifest entry. Classifies into green / redesign / minor, dispatches, runs the shared downstream (test → bench → validate → flip status → report).                 |
| [`scaffold-op`](../.claude/skills/scaffold-op/SKILL.md)   | per op     | atomic       | Scaffolds a new T2 (L1-direct) Op file from one manifest entry by following the 7-step playbook in `docs/ops-design.md`. Emits the 17 mechanical slots; leaves hooks/protocol-vars/kernel impl to downstream. |
| [`test-op`](../.claude/skills/test-op/SKILL.md)           | per op     | atomic       | Writes tests for the target spec using PyTorch as ground truth; verifies they fail on current code.                                                                                                           |
| [`implement-op`](../.claude/skills/implement-op/SKILL.md) | per op     | atomic       | Modifies op code to match the manifest-declared interface, making spec tests pass.                                                                                                                            |
| [`bench-op`](../.claude/skills/bench-op/SKILL.md)         | per op     | atomic       | Fixes the benchmark file to work with the new Op interface. Runs benchmark, fixes errors, repeats until it produces numbers.                                                                                  |
| [`align-family`](../.claude/skills/align-family/SKILL.md) | per family | orchestrator | Drives the full migration for an op family — audit, test, implement, bench, flip status, create PR.                                                                                                           |
| [`audit-family`](../.claude/skills/audit-family/SKILL.md) | per family | atomic       | Compares each op's code signature against its manifest spec, classifies gaps, produces a structured report.                                                                                                   |

## Composition

How orchestrators delegate to atomic skills.

```
align-family <family>           ← family-scoped orchestrator
  ├─ audit-family               (internal call)
  ├─ per op:
  │   ├─ test-op
  │   ├─ implement-op
  │   ├─ bench-op
  │   └─ (orchestrator flips manifest status)
  └─ CLEANUP + CREATE_PR        (orchestrator)

align-op <op_name>              ← per-op orchestrator
  ├─ CLASSIFY                   (internal)
  ├─ GREEN path:
  │   └─ scaffold-op
  ├─ REDESIGN path:
  │   ├─ ARCHIVE + CLEAR        (internal, prepares clean target)
  │   ├─ scaffold-op
  │   ├─ PORT                   (internal, agent reads archive)
  │   └─ KERNEL_CHECK           (internal, writes kernel-check.json)
  ├─ MINOR path:
  │   └─ implement-op
  └─ shared downstream:
      ├─ test-op
      ├─ bench-op
      └─ FLIP_STATUS            (orchestrator; only manifest writer)
```

Future consolidation: `align-family`'s per-op inner loop may delegate to `align-op` to remove duplication. Tracked as follow-up.

## Trust model (who can write what)

| Concern                             | Owner                                                                                                                                                               |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ops_manifest.yaml` writes          | Only `align-op` (at FLIP_STATUS) or `align-family` (at its FLIP_STATUS step); no atomic skill may touch the manifest.                                               |
| `tileops/ops/**` op files           | `scaffold-op` creates; `implement-op` edits; `align-op` / `align-family` orchestrate.                                                                               |
| `tileops/kernels/**` kernel files   | No TileOPs skill modifies kernel files today. `align-op --mode=redesign` surfaces kernel mismatches as `kernel-check.json` entries for a future kernel-layer skill. |
| `tests/ops/**` test files           | Only `test-op`.                                                                                                                                                     |
| `benchmarks/ops/**` benchmark files | Only `bench-op`.                                                                                                                                                    |

## Maintenance

- **Purpose lines** in the Catalog table are the `description` YAML frontmatter in each skill's `SKILL.md`. Edit there first; update this table to match.
- **Decision tree, NOT-use rules, composition diagram** are hand-maintained. Add an entry when a new skill lands; remove when one is retired.
- **Source of skill list**: `ls .claude/skills/` is authoritative. Every directory there should appear in the Catalog.
- **No lint automation** at current scale (7 skills). Add a `scripts/lint_skill_index.py` only if drift becomes observable or the skill count grows past ~15.
