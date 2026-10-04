→ [testing.md §Benchmarks](../../docs/design/testing.md#benchmarks) states the file checklist, the metrics and the reporting rules. What follows is what those leave open.

- A benchmark asserts only what it needs to trust its own numbers: that an implementation it is about to time matches the reference, or that a comparison which decides something came out the way the code assumes. It never becomes the place an op's behaviour is established.
- A library a row *selects* raises when it is missing — a degraded environment fails the row rather than reporting torch under a library's tag. One a row merely *prefers* keeps its guarded import and drops the tag.
- Where a library cannot express the case at all, drop its tag and say why.
- A baseline that overwrites an input gets a private buffer for it, refilled inside the timed callable. Refilled outside, every iteration after the first reads the one before it.
- That refill is not the baseline's work: leave `count_copies` false. No other tag reads that buffer.
- `OpBenchmark.compare` reads `workload.verification(*inputs)` once for every tag and uses the shared verifier. A baseline adapter must return the workload's output structure, shape and dtype; convert a foreign API at that boundary. Do not add tag-specific tolerances or reference overrides.
- A known external semantic difference is explicit in `noncomparable={tag: reason}`: timing is retained without a correctness-backed ratio. This cannot exempt the TileOPs subject.
- A torch-compile row proves it compiled: `compiled_reference` fails the case unless dynamo built one graph with no break.
- A timed callable launches its own work. Gradients come from `backward_of`, never `Tensor.backward`: autograd's engine thread carries no iteration id, so the timer cannot attribute what it launches.
- A `label` names the scenario, not the parameters, as [manifest.md § Rows](../../docs/design/manifest.md#rows) states; the case id appends the dtype.
- Tag names: lowercase, hyphen-separated. A `tileops` prefix marks a TileOPs entry; everything else is a baseline. Exactly one `tileops`-prefixed entry per config — a variant tag like `tileops-lut` is that one entry, not an extra.
- Cases come from the entry's workload rows; a representative shape the benchmark lacks is a new row. Rows cover the signature's dtype domain and ≥3 shapes per op, including a non-power-of-2 where the op supports one.
- Shapes come from real DNN workloads, LLaMA-family by default: hidden ∈ {4096, 5120, 8192}, intermediate ∈ {10240, 11008, 14336, 20480, 28672}, seq_len ∈ {2048, 4096}. The row's `label` names the model or scenario.
