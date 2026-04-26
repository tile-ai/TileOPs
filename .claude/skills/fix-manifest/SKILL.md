---
name: fix-manifest
description: Patch one missing structural field on an existing ops_manifest.yaml entry. Auto-detects the field via the validator or takes --field=. Refuses signature, status, and new entries.
---

## Arguments

| Argument         | Required | Description                                                                                                             |
| ---------------- | -------- | ----------------------------------------------------------------------------------------------------------------------- |
| `op_name`        | Yes      | Manifest key (e.g., `RMSNormFwdOp`)                                                                                     |
| `--field=<name>` | No       | One of `kernel_map`, `static_dims`, `shape_rules`, `roofline.vars`, `dtype_combos`. Omit to auto-detect from validator. |
| `--dry-run`      | No       | Print diff and exit; no write, no PR                                                                                    |

## Contract

- **MAY write** in `ops_manifest.yaml`: `kernel_map`, `static_dims`, `shape_rules`, `roofline.vars`, `dtype_combos`.
- **MUST NOT write**: `signature.*`, `status`, `family`, `ref_api`, `workloads`, `roofline.flops|bytes|func`, `source.*` (except inserting `kernel_map` under `source`).
- **MUST NOT** create new entries (use `add-manifest`).
- **MUST NOT** flip `status` (that is `align-op@FLIP_STATUS`).
- **MUST NOT** edit op / kernel / test / bench code.
- **One field per invocation.** To fix two fields, run twice.
- **Termination**: validator passes the level for the patched field, or BLOCKED.

## Workflow

```mermaid
stateDiagram-v2
    [*] --> PRE_CHECK
    PRE_CHECK --> [*]: entry missing → BLOCKED
    PRE_CHECK --> DIAGNOSE
    DIAGNOSE --> INFER: target in allowed list
    DIAGNOSE --> [*]: forbidden field → BLOCKED
    DIAGNOSE --> [*]: nothing to fix
    INFER --> PATCH
    INFER --> [*]: cannot infer → BLOCKED
    PATCH --> VALIDATE
    VALIDATE --> [*]: regression → revert and BLOCK
    VALIDATE --> CREATE_PR
    CREATE_PR --> [*]
```

## Steps

### 1. PRE_CHECK

Resolve `op_name` in `tileops/ops_manifest.yaml`. Missing → BLOCKED with message `op not in manifest; use add-manifest for greenfield`.

### 2. DIAGNOSE

Decide the target field:

- `--field=` provided → must be in the allowed list above; else BLOCKED.
- Else: run `python scripts/validate_manifest.py --check-op <op_name>`. Parse first error.
  - Field in allowed list → that field.
  - Field forbidden (e.g., `signature.params.dim`) → BLOCKED. Message must name the field, why it is out of scope, and the owning workflow (`add-manifest` for new entries; manifest-review issue for `signature.*`).
  - No errors → no-op; print `nothing to fix` and exit 0.

Write `.foundry/plan/<op_name>/fix-diagnosis.json`: `{op_name, target_field, validator_excerpt, action}`.

### 3. INFER

Build the patch payload from on-disk evidence. Source per field:

| Field           | Inference source                                                                                                                                                                                                                                                                                                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kernel_map`    | The op's runtime kernel dispatch. T2 (L1-direct): `default_kernel_map()` returns the dict verbatim. T1 (thin wrapper, see `docs/ops-design.md` §"Family-specific protocol variables"): combine the op's `_kernel_key` / `_kernel_cls` (or equivalent class-level protocol vars) with the family base's kernel-dispatch logic. Output: `{<key>: <fully-qualified Kernel class>}`. |
| `static_dims`   | `signature.inputs` shape names that the op binds at construction time (each entry in the op's `__init__` kwarg block, excluding `dtype` / `kernel_map` / `tune` / `signature.params` entries — see `docs/ops-design.md` §"Step 3"). Cross-check with `roofline.vars` if present.                                                                                                 |
| `shape_rules`   | `signature.inputs/outputs` shape relationships. PyTorch docs (`ref_api`) is tiebreaker.                                                                                                                                                                                                                                                                                          |
| `roofline.vars` | `static_dims` keys + any extra dims referenced in `roofline.flops` / `roofline.bytes`.                                                                                                                                                                                                                                                                                           |
| `dtype_combos`  | `source.test`: dtypes the tests parametrize over.                                                                                                                                                                                                                                                                                                                                |

If inference impossible, BLOCKED with an `evidence_needed` report listing what the human must decide. **Do not guess.**

### 4. PATCH

Apply the payload to the entry. Required formatting rules:

- Insert `kernel_map` directly under `source.kernel`.
- Insert `static_dims` directly under `signature.params` (or under `signature.outputs` if no `params`).
- Other fields: per `docs/manifest.md` canonical order.
- Preserve adjacent comments.
- Do not reorder unrelated keys.

### 5. VALIDATE

```bash
python scripts/validate_manifest.py --check-op <op_name>
```

Required passing level:

| Patched field   | Level |
| --------------- | ----- |
| `kernel_map`    | L0    |
| `static_dims`   | L1    |
| `shape_rules`   | L2    |
| `roofline.vars` | L3    |
| `dtype_combos`  | L1    |

If validator emits any error not present before the patch, revert the patch and BLOCKED. Patch must be monotonic.

### 6. CREATE_PR

If `--dry-run`, print the diff and exit 0. Otherwise invoke `foundry:creating-pull-request`:

| Element | Value                                                                                                         |
| ------- | ------------------------------------------------------------------------------------------------------------- |
| title   | `[Maintain][Manifest] fix <field> for <op_name>` (use `[Fix][Manifest]` if validator was actively rejecting)  |
| branch  | `maintain/manifest/fix-<op-slug>-<field>`                                                                     |
| body    | which field, evidence used to infer; validator output before vs. after; explicit list of what was NOT touched |

## Guardrails

- One field per invocation.
- Never widen scope to cover a forbidden field — emit BLOCKED instead.
- Never invent values. All payload data must trace to a file or to `ref_api`.
- Never flip `status`.
- Validator output ambiguous → STOP, ask user.
