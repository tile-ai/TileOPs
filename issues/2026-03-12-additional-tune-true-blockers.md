# Additional Benchmark `tune=True` Blockers

Date: 2026-03-12

## Scope

Benchmarks migrated to independent parameter sets that still fail on the tuned path:

- `benchmarks/ops/bench_topk_selector.py`
- `benchmarks/ops/bench_fp8_lighting_indexer.py`
- `benchmarks/ops/bench_mhc_pre.py`

Validated in `/tmp/TileOPs-nightly-benchmark-smoke-prune` with:

```bash
CUDA_VISIBLE_DEVICES=2 python -m pytest -q <bench file>
```

## Observed Failures

### 1. `topk_selector` tuned path fails in TileLang builder/autotune

Symptoms:

- TileLang builder warning around rebinding immutable value `pos`
- Final autotune failure:
  `RuntimeError: Auto-tuning failed: No configuration successfully compiled and passed benchmarking/validation.`

Observed from:

- `benchmarks/ops/bench_topk_selector.py::test_topk_selector_bench[...]`

Inference:

- Kernel generation/autotune issue in `tileops/kernels/deepseek_mla/topk_selector.py`
- Not caused by benchmark parameter choice alone

### 2. `fp8_lighting_indexer` tuned path fails

Status:

- Benchmark run failed under `tune=True` in the same batch where `topk_selector` and `mhc_pre` failed.
- The benchmark file should be treated as tuned-path blocked until revalidated in isolation.

Observed from:

- `benchmarks/ops/bench_fp8_lighting_indexer.py`

Action:

- Temporarily use `tune=False` for benchmark continuity.

### 3. `mhc_pre` autotuner closure is not serializable

Symptom:

- `AssertionError: Cell contents <function _mhc_pre_kernel.<locals>.sigmoid ...> is not serializable`

Observed from:

- `benchmarks/ops/bench_mhc_pre.py::test_mhc_pre_bench[...]`

Inference:

- This is an autotuner integration problem in `tileops/kernels/mhc/mhc_pre.py`, not a benchmark design issue.

## Temporary Benchmark Policy

Until the tuned-path issues are fixed, these benchmarks should temporarily use `tune=False`:

- `benchmarks/ops/bench_topk_selector.py`
- `benchmarks/ops/bench_fp8_lighting_indexer.py`
- `benchmarks/ops/bench_mhc_pre.py`

## Verified Tuned Benchmarks In Same Batch

The following independent benchmark files did run successfully with `tune=True`:

- `benchmarks/ops/bench_fp8_quant.py`
- `benchmarks/ops/bench_mhc_post.py`
- `benchmarks/ops/bench_group_norm.py`
- `benchmarks/ops/bench_instance_norm.py`

## Follow-up Work

1. Fix TileLang/kernel builder issue in `topk_selector`.
1. Revalidate and fix `fp8_lighting_indexer` tuned path directly.
1. Fix autotuner closure serialization in `mhc_pre`.
1. Restore `tune=True` in the affected benchmark files after kernel-side fixes.
