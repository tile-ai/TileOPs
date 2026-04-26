---
name: implement-op
description: Modify op code to match the manifest-declared interface, making spec tests pass.
---

## Arguments

`op_name`, `manifest_signature`, `source_op`, `source_test` — passed by align-family orchestrator.

## Contract

- **Input**: `op_name`, `manifest_signature`, `source_op`, `source_test`
- **Output**: modified op code + commit + `observations` list (returned to orchestrator)
- **Termination (success)**: `python scripts/validate_manifest.py --check-op <name>` all levels pass + new tests pass.
- **Termination (blocked)**: fix requires changes beyond Op layer. Return `blocked` with reason.
- **Constraint**: must NOT modify `ops_manifest.yaml`. Must NOT modify tests written by `test-op` in this align-op run (spec contract). MAY update pre-existing tests in `<source_test>` whose call-sites use the legacy API.
- **Behavioral compatibility**: default param values (from manifest) must produce identical results to the old implementation. The old API shape (e.g., `__init__(M, N)`) is NOT preserved — the manifest defines the target interface.

## Workflow

```mermaid
stateDiagram-v2
    [*] --> ANALYZE
    ANALYZE --> DIAGNOSE: semantic knowledge extracted
    DIAGNOSE --> VALIDATE: gap already resolved (base class fixed by previous op)
    DIAGNOSE --> IMPLEMENT: gap exists
    IMPLEMENT --> VALIDATE: sub-step completed
    VALIDATE --> IMPLEMENT: sub-step failed
    VALIDATE --> MARK_DONE: all pass
    VALIDATE --> BLOCKED: fix exceeds Op layer scope
    MARK_DONE --> COMMIT
    COMMIT --> [*]
    BLOCKED --> [*]: return blocked to orchestrator
```

## Dual-path policy

Before refactoring a base class, count its subclasses:

```bash
grep -lE "class [A-Z][A-Za-z0-9]*\(<BaseName>\)" tileops/ops/
```

- **One subclass** (the op being migrated): refactor the base in place. No dual-path.
- **Multiple subclasses**: keep the legacy `__init__` path alongside the new one so unmigrated siblings still pass. Cleanup gate removes the legacy path after all siblings migrate.

Do NOT preemptively migrate siblings to avoid dual-path — that violates per-op scope.

## Steps

### 1. ANALYZE

Read existing code (`source_op`) to extract semantic knowledge:

- What the computation does (the algorithm)
- Where constraints are hardcoded (e.g., `dim=-1`, `reshape(-1, N)`)
- What the generalization path is

Manifest = WHAT (target interface). Existing code = HOW (computation logic). The delta is the work.

This analysis is internal — not persisted.

### 2. DIAGNOSE

Check if the gap still exists. A previous op's migration may have fixed the shared base class.

- Gap resolved → skip to VALIDATE
- Gap exists → proceed to IMPLEMENT

### 3. IMPLEMENT

Decompose into independently verifiable sub-steps. Each sub-step either succeeds or fails with precise location (→ BLOCKED). No retry loops — if a sub-step fails with the same error twice, the task is beyond current scope.

Agent determines sub-steps based on the specific gap. Do not follow a fixed recipe.

### 4. VALIDATE

Run all checks:

```bash
python scripts/validate_manifest.py --check-op <op_name>
python -m pytest <source_test> -v
```

All must pass. If not, return to IMPLEMENT.

### 5. MARK_DONE

Record `observations` — design knowledge discovered during migration:

- Patterns (e.g., "scan ops need transpose, reduction ops reshape")
- Edge cases
- Abstraction opportunities

Do NOT modify manifest or design docs. Observations are returned to orchestrator and surfaced in PR for human review.

### 6. COMMIT

Commit code changes only. Do not commit manifest changes.
