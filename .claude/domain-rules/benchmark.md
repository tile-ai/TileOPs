→ [testing.md §Benchmarks](../../docs/design/testing.md#benchmarks) defines the file checklist, metrics and reporting rules.

## Cases and ownership

- Use `bench.cases(Op)` for manifest rows and `bench.Runner(op, case).compare(...)` for measurement. The runner records the op as `tileops`; implementation names are lowercase and hyphen-separated.
- Benchmarks select comparisons. Workloads own input data, references and numerical rules; tests own kernel branch coverage. Benchmark files use `Case` and workload-derived data without reaching into the manifest `Call`.
- Add a missing representative shape as a manifest row. Rows cover the signature's dtype domain and at least three shapes per op, including a non-power-of-2 shape where supported. A `label` names the scenario, as [manifest.md § Rows](../../docs/design/manifest.md#rows) states.
- Use real DNN workloads, LLaMA-family by default: hidden ∈ {4096, 5120, 8192}, intermediate ∈ {10240, 11008, 14336, 20480, 28672}, seq_len ∈ {2048, 4096}.

## Comparisons

- Check every comparable implementation against the case reference before timing. An adapter returns the workload's output structure, shape and dtype; do not add per-implementation tolerances or reference overrides. Benchmarks assert what they need to trust a measurement, not an op's behavioral contract.
- A known external semantic difference belongs to `Implementation.noncomparable_reason`. It retains timing without a correctness-backed ratio and cannot exempt the TileOps op.
- A selected library must be available; a preferred library may drop its tag when unavailable. Drop a tag when the library cannot express the case, and say why. A torch-compile row uses `compiled_reference`, which requires one graph with no break.

## Measurement

- Every implementation leaves shared case inputs unchanged. If it overwrites an argument, give it private arguments and restore them in `reset`. Each round runs `reset → L2 flush → timed call`.
- `count_copies` belongs to the case's registry entry and applies to every implementation in the comparison.
- A timed callable launches its own work. Drive gradients with `backward_of`; `Tensor.backward` launches kernels on another thread, outside the timer's attribution.
