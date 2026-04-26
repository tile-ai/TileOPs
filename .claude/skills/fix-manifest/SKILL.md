---
name: fix-manifest
description: Patch one missing structural field (kernel_map, static_dims, shape_rules, roofline.vars, dtype_combos) on an existing ops_manifest.yaml entry. Auto-detects the field via the validator or takes `--field=<name>`. Refuses signature.{inputs,outputs,params}, status, and new entries.
---

## Arguments

| Argument         | Required | Description                                                                                                             |
| ---------------- | -------- | ----------------------------------------------------------------------------------------------------------------------- |
| `op_name`        | Yes      | Manifest key (e.g., `RMSNormFwdOp`)                                                                                     |
| `--field=<name>` | No       | One of `kernel_map`, `static_dims`, `shape_rules`, `roofline.vars`, `dtype_combos`. Omit to auto-detect from validator. |
| `--dry-run`      | No       | Print diff and exit; no write, no PR                                                                                    |

## Contract

- **MAY write** in `ops_manifest.yaml`: `kernel_map`, `static_dims`, `shape_rules`, `roofline.vars`, `dtype_combos`.
- **MUST NOT write**: `signature.{inputs,outputs,params}`, `status`, `family`, `ref_api`, `workloads`, `roofline.flops|bytes|func`, `source.{kernel,op,test,bench,bench_manifest_driven}`. (Note: `signature.static_dims`, `signature.shape_rules`, `signature.dtype_combos`, and `source.kernel_map` ARE in the allowed write set above.)
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

Decide the target field. The validator does NOT flag every missing allowed field for spec-only entries (e.g., `source.kernel_map` is only warned when `status == implemented`). When `--field=` is omitted, run two checks in strict order — Check A first; only fall through to Check B if A finds nothing.

**Check A — `kernel_map` presence (single field only).** If `source.kernel_map` is missing or empty → target = `kernel_map`, jump to INFER. Otherwise fall through to Check B.

Do NOT extend Check A to `static_dims`, `shape_rules`, `dtype_combos`, or `roofline.vars`. `docs/manifest.md` R7 and R20 explicitly allow `static_dims` to be absent on fixed-rank ops; the other three fields are conditionally required. Patching them on absence alone would manufacture changes for valid entries (e.g., reduction-style ops).

**Check B — validator output.** Run `python scripts/validate_manifest.py --check-op <op_name>`. Parse the first error:

- Field in the allowed list above → target = that field, jump to INFER.
- Field forbidden (e.g., `signature.params.dim`) → BLOCKED. Message must name the field, why it is out of scope, and the owning workflow (`add-manifest` for new entries; manifest-review issue for `signature.{inputs,outputs,params}`).
- No errors and Check A also empty → no-op; print `nothing to fix` and exit 0.

When `--field=` IS provided: must be in the allowed list; else BLOCKED. Skip both checks.

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

Apply the payload to the entry. The target keys live at these exact YAML paths (see `docs/manifest.md`):

| Field           | YAML path                                                |
| --------------- | -------------------------------------------------------- |
| `kernel_map`    | `source.kernel_map` (sibling of `source.kernel`)         |
| `static_dims`   | `signature.static_dims` (sibling of `signature.params`)  |
| `shape_rules`   | `signature.shape_rules` (sibling of `signature.params`)  |
| `dtype_combos`  | `signature.dtype_combos` (sibling of `signature.params`) |
| `roofline.vars` | `roofline.vars` (sibling of `roofline.flops`)            |

Insertion rules:

- Insert each new key as a **sibling** of the existing keys in its parent block (NOT nested under another sibling).
- Order within parent block: per `docs/manifest.md` canonical order. When unsure, place the new key just before the first existing key (top of block).
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
| `roofline.vars` | L0    |
| `dtype_combos`  | L3    |

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
