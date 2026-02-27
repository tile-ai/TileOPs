---
name: migrating-new-op
description: Use when adding or migrating an operator to TileOPs with the required kernel->op->function->layer delivery path and validation checklist
---

## When to use

- Add a new op into TileOPs
- Migrate an op from `cuda`/`triton` into TileOPs with TileLang kernels
- Standardize implementation and tests to the project architecture order

______________________________________________________________________

## Level Confirmation (Ask First)

When this skill is invoked, ask the user first:

"What implementation level do you want for this operator?"

- **L1 (Kernel only)**
  - Deliverables: kernel implementation + kernel/functional correctness checks
  - Typical paths: `tileops/kernels/<op_name>/`, minimal verification script/tests
- **L2 (Kernel + Op)**
  - Deliverables: L1 + op interface + op unit tests + benchmark script
  - Typical paths: `tileops/ops/<op_name>.py`, `tests/ops/test_<op_name>.py`, `benchmarks/benchmark_<op_name>.py`
- **L3 (Kernel + Op + Function)**
  - Deliverables: L2 + functional API (`torch.autograd.Function` when needed) + function tests/grad checks
  - Typical paths: `tileops/functions/<op_name>.py`, `tests/functions/test_<op_name>_func.py`
- **L4 (Full stack: Kernel + Op + Function + Layer)**
  - Deliverables: L3 + `nn.Module` layer wrapper + layer tests + export synchronization
  - Typical paths: `tileops/layers/<op_name>.py`, `tests/layers/test_<op_name>_layer.py`

If user does not specify a level, default to **L2** and state this assumption explicitly.

After level confirmation, execute only the required layers for that level (do not over-deliver).

______________________________________________________________________

## Workflow

### Phase A: Requirement + Reference Alignment

1. Confirm target behavior, API, and expected test scope.
1. Read the external reference implementation and extract:
   - kernel stages
   - input/output semantics
   - accumulation and edge-case behavior

### Phase B: Delivery Design by Layer

1. Confirm target level (L1/L2/L3/L4) with the user before coding.
1. Plan implementation in strict order: `kernel -> op -> function -> layer`.
1. Define per-layer responsibilities and avoid cross-layer shortcuts.
1. List only level-required files/tests before coding (see checklist below).

### Phase C: Kernel Migration (TileLang)

1. Implement TileLang kernels by stage (same logical decomposition as reference).
1. Keep output semantics compatible with existing interfaces.
1. Handle core edge cases early (for example: empty paths / `num_topk == 0`).

### Phase D: Upper Layer Wiring

1. Add `op` wrapper for kernel invocation + runtime contract.
1. Add `function` API for reusable composition and shape/dtype validation.
1. Add `layer` abstraction only when module-style integration is needed.
1. Keep dependency direction one-way: `layer -> function -> op -> kernel`.

### Phase E: Tests and Validation

1. Ensure layer-matched tests exist:

- `tests/ops` for op behavior
- `tests/functions` for functional integration
- `tests/layers` when a layer is introduced

2. Add reference comparison for correctness checks.
1. Run incremental tests first, then the relevant chain.

### Phase F: Cleanup and Finalization

1. Remove obsolete wrappers/helpers/classes.
1. Update exports/imports (`__init__.py`) to avoid stale symbols.
1. Re-run the main regression chain and summarize results.

______________________________________________________________________

## Software Organization Guidelines

- **Kernel layer (`tileops/kernels`)**: compute logic and kernel configs only.
- **Op layer (`tileops/ops`)**: wraps kernel call contract and runtime dispatch.
- **Function layer (`tileops/functions`)**: reusable API composition and validation.
- **Layer (`tileops/layers`)**: module-level integration (`nn.Module` style).
- Keep dependency direction strict: `layer -> function -> op -> kernel`.
- After refactoring, always sync exports to prevent import-time failures.

______________________________________________________________________

## Minimum Deliverables Checklist

Before opening a PR, verify all required items are present:

1. **L1: Kernel (`tileops/kernels`)**

- [ ] New/updated kernel implementation exists
- [ ] Kernel handles documented edge cases

2. **L2: Op (`tileops/ops`)**

- [ ] Op API wraps kernel with stable argument contract
- [ ] Op-level tests added/updated in `tests/ops`
- [ ] Benchmark script added/updated in `benchmarks`

3. **L3: Function (`tileops/functions`)**

- [ ] Function API added/updated for composable usage
- [ ] Function tests added/updated in `tests/functions`
- [ ] `gradcheck` path is added when autograd is expected

4. **L4: Layer (`tileops/layers`, if needed)**

- [ ] Layer class added only when module abstraction is required
- [ ] Layer tests added/updated in `tests/layers`

5. **Project Hygiene**

- [ ] `__init__.py` exports are synchronized
- [ ] Relevant tests pass locally
- [ ] Migration notes / behavior deltas are documented

______________________________________________________________________

## Test Flow

### Recommended command pattern

```bash
PYTHONPATH="$PWD" python -m pytest -v tests/ops/test_xxx.py
PYTHONPATH="$PWD" python -m pytest -v tests/functions/test_xxx_func.py
PYTHONPATH="$PWD" python -m pytest -v tests/layers/test_xxx_layer.py
```

### Main-chain regression

```bash
PYTHONPATH="$PWD" python -m pytest -v \
  tests/ops/test_xxx.py \
  tests/functions/test_xxx_func.py \
  tests/layers/test_xxx_layer.py
```

______________________________________________________________________

## Done Criteria

Migration is considered complete when:

- kernel/op/function/layer (if needed) are delivered in correct order
- API behavior stays compatible
- stale legacy wrappers are removed
- relevant ops/functions/layers tests pass
- migration notes are documented for reuse
