# Decode Benchmark `tune=True` Blockers

Date: 2026-03-12

## Scope

Blocked benchmark files while migrating to independent benchmark parameter sets with `tune=True`:

- `benchmarks/ops/bench_gqa_decode.py`
- `benchmarks/ops/bench_gqa_decode_paged.py`
- `benchmarks/ops/bench_mha_decode.py`
- `benchmarks/ops/bench_mha_decode_paged.py`

Validation was run locally in worktree `/tmp/TileOPs-nightly-benchmark-smoke-prune` with:

```bash
CUDA_VISIBLE_DEVICES=2 python -m pytest -q <bench file>
```

## Observed Failures

### 1. `gqa_decode` autotune fails for all configs

Symptom:

- `RuntimeError: Auto-tuning failed: No configuration successfully compiled and passed benchmarking/validation.`

Observed from:

- `benchmarks/ops/bench_gqa_decode.py::test_gqa_decode_bench[...]`

Notes:

- Failure reproduced on multiple shapes, not only one candidate benchmark case.
- This indicates a kernel/autotune issue on the `tune=True` path rather than a benchmark parameter problem.

### 2. `mha_decode` autotune fails for all configs

Symptom:

- `RuntimeError: Auto-tuning failed: No configuration successfully compiled and passed benchmarking/validation.`

Observed from:

- `benchmarks/ops/bench_mha_decode.py::test_mha_decode_bench[...]`

Notes:

- Failure reproduced immediately on the first tuned benchmark case.
- Like `gqa_decode`, this appears to be an operator/kernel issue on the tuned path.

### 3. `gqa_decode_paged` kernel generation fails on tuned path

Symptom:

- TileLang builder warning during autotune:
  `Immutable value 'k_global' is re-bound. If you want to modify its value, please use T.alloc_var to make it a variable!`

Observed from:

- Direct construction of `GroupQueryAttentionDecodePagedWithKVCacheOp(..., tune=True)`
- `benchmarks/ops/bench_gqa_decode_paged.py`

Notes:

- This is a kernel construction problem in `tileops/kernels/flash_decode/gqa_decode_paged.py`.
- Benchmark migration cannot fix this; the tuned kernel path needs code changes.

### 4. `mha_decode_paged` likely blocked by same tuned decode path class

Status:

- Benchmark migration updated the file to independent benchmark parameters.
- Full tuned benchmark verification was not completed after `gqa_decode`/`mha_decode` were shown to fail at autotune stage.

Inference:

- Given the same tuned decode kernel family behavior, `mha_decode_paged` should be treated as blocked for `tune=True` until explicitly revalidated after kernel fixes.

## Temporary Benchmark Policy

For benchmark migration continuity, these decode-family benchmark files should temporarily use `tune=False`:

- `benchmarks/ops/bench_gqa_decode.py`
- `benchmarks/ops/bench_gqa_decode_paged.py`
- `benchmarks/ops/bench_mha_decode.py`
- `benchmarks/ops/bench_mha_decode_paged.py`

This keeps the benchmark suite runnable while preserving a written record that tuned coverage is still missing.

## Follow-up Work

1. Fix decode autotune validation failures for `gqa_decode` and `mha_decode`.
1. Fix paged decode kernel generation issue around `k_global` mutation in `gqa_decode_paged`.
1. Re-enable `tune=True` in the four decode benchmark files and remeasure benchmark runtime.
