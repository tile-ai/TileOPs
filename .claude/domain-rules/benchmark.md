→ [testing.md §Benchmarks](../../docs/design/testing.md#benchmarks) states the file checklist, the metrics and the reporting rules. What follows is what those leave open.

- A benchmark asserts only what it needs to trust its own numbers: that an implementation it is about to time matches the reference, or that a comparison which decides something came out the way the code assumes. It never becomes the place an op's behaviour is established.
- A library a row *selects* raises when it is missing — a degraded environment fails the row rather than reporting torch under a library's tag. One a row merely *prefers* keeps its guarded import and drops the tag.
- Where a library cannot express the case at all, drop its tag and say why.
- An implementation that overwrites an argument gets private copies in `Implementation.args` and restores them in `reset`, which every round runs before the L2 flush and outside the reading. A refill inside the timed call reads its inputs from L2. An op that overwrites its inputs registers a binder that does the same.
- `bench.Runner.compare` runs the reference once and checks every implementation against it with `case.verification`. An adapter must return the workload's output structure, shape and dtype; convert a foreign API at that boundary. Do not add per-implementation tolerances or reference overrides.
- A known external semantic difference is explicit in `Implementation(noncomparable_reason=...)`: timing is retained without a correctness-backed ratio. This cannot exempt the TileOPs op.
- `count_copies` belongs to the op's registry entry in `benchmarks/_cases/` and applies to every implementation of the case.
- A torch-compile row proves it compiled: `compiled_reference` fails the case unless dynamo built one graph with no break.
- A timed callable launches its own work. Gradients come from `backward_of`, never `Tensor.backward`: autograd's engine thread carries no iteration id, so the timer cannot attribute what it launches.
- A `label` names the scenario, not the parameters, as [manifest.md § Rows](../../docs/design/manifest.md#rows) states; the case id appends the dtype.
- Implementation names: lowercase, hyphen-separated. The runner records the op as `tileops`; no implementation passed to `compare` takes that name.
- Cases come from `bench.cases(Op)`, the entry's workload rows; a representative shape the benchmark lacks is a new row. Rows cover the signature's dtype domain and ≥3 shapes per op, including a non-power-of-2 where the op supports one.
- Shapes come from real DNN workloads, LLaMA-family by default: hidden ∈ {4096, 5120, 8192}, intermediate ∈ {10240, 11008, 14336, 20480, 28672}, seq_len ∈ {2048, 4096}. The row's `label` names the model or scenario.
