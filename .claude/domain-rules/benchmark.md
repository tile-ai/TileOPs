→ [testing.md §Benchmarks](../../docs/design/testing.md#benchmarks) states the file checklist, the metrics and the reporting rules. What follows is what those leave open.

- A benchmark asserts only what it needs to trust its own numbers: that an implementation it is about to time matches the reference, or that a comparison which decides something came out the way the code assumes. It never becomes the place an op's behaviour is established.
- A library a row *selects* raises when it is missing — a degraded environment fails the row rather than reporting torch under a library's tag. One a row merely *prefers* keeps its guarded import and drops the tag.
- Where a library cannot express the case at all, drop its tag and say why.
- A subclass that overrides `ref_program` because the baseline is deliberately not the reference says so in the subclass docstring, not only in the PR.
- A baseline that overwrites its inputs gets private copies. Sharing them silently feeds every later tag something the reference never read.
- A timed callable launches its own work. Gradients come from `backward_of`, never `Tensor.backward`: autograd's engine thread carries no iteration id, so the timer cannot attribute what it launches.
- Name the scenario (`serving-130m-4k`), not the parameters. A `label` omits the dtype; the case id appends it.
- Tag names: lowercase, hyphen-separated. A `tileops` prefix marks a TileOPs entry; everything else is a baseline. Exactly one `tileops`-prefixed entry per config — a variant tag like `tileops-lut` is that one entry, not an extra.
- Cover every dtype in `SUPPORTED_DTYPES`, and ≥3 shapes per op including a non-power-of-2 where the op supports one.
- Shapes come from real DNN workloads, LLaMA-family by default: hidden ∈ {4096, 5120, 8192}, intermediate ∈ {10240, 11008, 14336, 20480, 28672}, seq_len ∈ {2048, 4096}. Annotate every shape constant with the model or scenario it represents; never a bare flat number (262K, 1M, 4M).
