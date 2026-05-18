For `[Bench]` PRs. Benchmarks live downstream of every other layer and must never silently substitute for tests or for the manifest.

Load `.claude/domain-rules/benchmark.md` before reviewing.

Two non-negotiable principles cut across every event:

- **Independent baselines.** Every benchmark must record at least one non-`"tileops"` baseline (PyTorch, vendor, third-party kernel). A benchmark that only times TileOps against TileOps is a regression harness, not a benchmark — reject it.
- **No correctness assertions.** Benchmarks measure performance; they do not gate behavior. `assert torch.allclose(...)` and equivalents belong in `tests/`. A correctness check inside `benchmarks/` is a trust-layer violation regardless of how convenient it is.

#### Checklist

- [ ] **Boundary respected.** Diff stays inside `benchmarks/`. No edits to `tileops/ops/`, `tileops/kernels/`, `tests/`, `workloads/`, or `tileops/manifest/` (`.claude/domain-rules/benchmark.md §Boundary`).
- [ ] **Independent baseline present.** Each new/edited benchmark records ≥1 non-`"tileops"` baseline. If the external baseline is conditional, a local torch fallback is registered.
- [ ] **No correctness gating.** No `assert`, `torch.allclose`, or equivalent inside `benchmarks/`. Numeric mismatches surface through report columns, not exceptions.
- [ ] **Realistic shapes.** Shape constants reflect real DNN workloads (LLaMA-family or equivalent), annotated with the model/scenario they represent. Arbitrary flat numbers (e.g., 262K, 1M, 4M) are rejected.
- [ ] **Tag and FLOPs/memory hygiene.** Tags are lowercase hyphen-separated; `"tileops"`-prefixed tags are TileOps entries, all others are baselines. `calculate_flops()` / `calculate_memory()` return numeric or `None` consistently.
- [ ] **`record()` arg style consistent.** Within one file, `BenchmarkReport.record()` uses Op object (preferred) or string name uniformly — no mid-file flips.
