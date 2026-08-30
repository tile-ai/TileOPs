- **MUST NOT**: import from `tests/`, or author `gen_inputs` — take the op's workload from `workloads/`, adding it there if the op has none.
- Time the workload's `ref_program` for the torch baseline. Override it in a subclass only when the baseline is deliberately not the reference, and say so in its docstring.
- Another implementation of the same computation takes its own tag beside `ref_program` instead of overriding it, and is asserted against the reference before the case is timed. Time a library's kernel where it has one; where it cannot express the case, drop the tag and say why.
- A library a row selects raises when it is missing: a degraded environment must fail the row rather than report torch under a library's tag. One a row merely prefers keeps its guarded import and drops the tag.
- A baseline that overwrites its inputs gets private copies. Sharing them silently feeds every later tag something the reference never read.

→ [trust-model.md §Benchmark](../../docs/design/trust-model.md#benchmark) | [testing.md §Benchmarks](../../docs/design/testing.md#benchmarks)

______________________________________________________________________

- Every benchmark records ≥1 non-`tileops` baseline.
- A timed callable launches its own work. Gradients come from `backward_of`, never `Tensor.backward`: autograd's engine thread carries no iteration id, so the timer cannot attribute what it launches.
- Name the scenario (`serving-130m-4k`), not the parameters; a `label` omits the dtype, the case id appends it.
- Tag names: lowercase, hyphen-separated. A `tileops` prefix marks a TileOPs entry; everything else is a baseline.
- A benchmark takes its op at construction and publishes every row under it. What distinguishes one case from another belongs to the case, never to the row's name.
- The report tracks ops. A comparison whose subject is something else decides a question rather than tracking one, and states its conclusion as an assertion instead of publishing a row.
- A benchmark takes its roofline off the op. Writing the arithmetic by hand is what an op with no roofline of its own falls back to.
- Benchmark shapes reflect real DNN workloads (LLaMA-family by default). Annotate shape constants with the model/scenario; never arbitrary flat numbers (262K, 1M, 4M).
