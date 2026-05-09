# Per-element FLOP convention — before/after delta

Reproducible, formula-only evaluation of the FLOP/byte change introduced
by [`docs/design/roofline.md`](../design/roofline.md) §1.3 on three
representative workloads (one activation, one scalar 2-sided clamp, one
Tensor-bound 2-sided clamp). No GPU is required — the table is computed
purely from manifest YAML and the Python helper at
[`tileops/perf/formulas.py`](../../tileops/perf/formulas.py).

## Reproduce

```bash
python scripts/perf/flop_convention_delta.py \
  --out docs/perf/flop_convention_delta.csv
```

The CSV at [`flop_convention_delta.csv`](flop_convention_delta.csv) is
checked in. Re-running the command on the current checkout overwrites
it byte-identically.

## Result

| family     | op            | label                | shape     | dtype   | flops before | flops after | flops delta | bytes before | bytes after | bytes delta |
| ---------- | ------------- | -------------------- | --------- | ------- | -----------: | ----------: | ----------: | -----------: | ----------: | ----------: |
| activation | ReluFwdOp     | hidden-state-prefill | 2048×4096 | float16 |   16,777,216 |   8,388,608 |  -8,388,608 |   33,554,432 |  33,554,432 |           0 |
| clamp      | HardtanhFwdOp | hidden-state-prefill | 2048×4096 | float16 |   33,554,432 |   8,388,608 | -25,165,824 |   33,554,432 |  33,554,432 |           0 |
| min-max    | ClampFwdOp    | elementwise-16M      | 4096×4096 | float16 |   33,554,432 |  16,777,216 | -16,777,216 |  134,217,728 | 134,217,728 |           0 |

`flops before` columns reflect the coefficients that lived on each
entry on `upstream/testbed` immediately before the convention commit
(verifiable via `git diff upstream/testbed -- tileops/manifest/`).
`flops after` columns are evaluated from the manifest formulas (for
`ReluFwdOp` / `HardtanhFwdOp`) or from `clamp_fwd_roofline` (for the
Tensor-bound `ClampFwdOp`) on the current checkout. Byte counts are
unchanged by the convention and serve as a sanity column.

A GPU run was not performed; AC-5 explicitly accepts a
formula-evaluation table. Roofline efficiency depends on
`max(memory_time, compute_time)`; for these elementwise workloads
`memory_time` already dominates, so the FLOP-coefficient reduction
shifts each workload further into the memory-bound regime without
changing predicted achievable bandwidth.
