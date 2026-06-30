# In-Kernel Timeline Trace

[`tileops.trace`](../api/trace.md) is an in-kernel timeline tracer for diagnosing kernel performance.
It records per-CTA timestamps from inside the kernel and renders a timeline you
can scroll through — showing execution, gaps, and producer/consumer overlap that
per-kernel profilers (e.g. `ncu`) do not surface. It is most useful for
warp-specialized kernels, where overlapping the producer (TMA) and consumer
(WGMMA) warpgroups is the whole point.

## How it works

1. You annotate the kernel body with markers (`trace.range`, `trace.group`, …).
1. Markers are **always emitted** as placeholders. At build time the kernel is
   either **lowered** (markers become real `clock64()`-recording code plus a
   trailing `slots` output) or **stripped** (markers become no-ops — the
   generated CUDA is identical to an un-instrumented build).
1. At runtime, `trace.run` executes the kernel, decodes the `slots` buffer, and
   writes a self-contained Plotly HTML timeline.
1. A process-local switch (`trace.enable()`) decides lowered-vs-stripped, so
   tracing is **zero cost when off** — you can leave markers in production code.

The timestamp source is `clock64()` (the per-SM cycle counter).

## Write a traced kernel

Below is a complete (illustrative, single-buffer) warp-specialized GEMM with
tracing wired in. The numbered markers `(1)`–`(7)` are the only trace-specific
additions; each is explained below, linked to its API doc. The production
multi-stage version lives in
[`tileops/kernels/gemm.py`](https://github.com/tile-ai/TileOPs/blob/main/tileops/kernels/gemm.py).

```{ .python .annotate }
import functools
import tilelang
import tilelang.language as T
from tileops.trace import trace  # (1)!


@functools.lru_cache(maxsize=32)
def build_gemm(m, n, k, dtype="float16", traced=False):
    @tilelang.jit(out_idx=trace.out_idx(1, traced))  # (2)!
    def factory(block_m=128, block_n=128, block_k=64):
        @T.prim_func
        def main(a: T.Tensor((m, k), dtype), b: T.Tensor((n, k), dtype),
                 c: T.Tensor((m, n), dtype)):
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m),
                          threads=256) as (bx, by):
                a_smem = T.alloc_shared((block_m, block_k), dtype)
                b_smem = T.alloc_shared((block_n, block_k), dtype)
                c_local = T.alloc_fragment((block_m, block_n), "float")
                full = T.alloc_barrier(128)
                tx = T.get_thread_binding()

                if tx < 128:
                    with trace.group("producer", lead=0):  # (3)!
                        for ki in T.serial(T.ceildiv(k, block_k)):
                            with trace.range("tma", lane="tma"):  # (4)!
                                T.tma_copy(a[by * block_m, ki * block_k], a_smem, barrier=full)
                                T.tma_copy(b[bx * block_n, ki * block_k], b_smem, barrier=full)
                            with trace.range("arrive", lane="barrier"):
                                T.barrier_arrive(full)
                else:
                    with trace.group("consumer", lead=128):
                        T.clear(c_local)
                        for ki in T.serial(T.ceildiv(k, block_k)):
                            with trace.range("wait", lane="barrier"):
                                T.barrier_wait(full, ki % 2)
                            with trace.range("mma", lane="wgmma"):  # (5)!
                                T.wgmma_gemm(a_smem, b_smem, c_local, transpose_B=True)
                        with trace.range("epilogue"):
                            T.copy(c_local, c[by * block_m, bx * block_n])

                trace.dag("arrive", "wait")  # (6)!
        return trace.finalize(main, traced=traced, max_events=1024)  # (7)!
    return factory
```

1. Import the trace namespace — every call below is a method on this single
   `trace` object (full [API reference](../api/trace.md)).
1. [`trace.out_idx(n_outputs, traced)`](../api/trace.md#tileops.trace.api._Trace.out_idx) — the
   `@tilelang.jit` `out_idx`. It grows by one (for the trailing `slots` output)
   only when `traced`, so the same builder works on or off.
1. [`trace.group(name, lead)`](../api/trace.md#tileops.trace.api._Trace.group) — declares which
   warpgroup records. `lead` is the elected writer thread (`tx == lead`);
   compute still runs on all threads, only the timestamps are written by `lead`.
1. [`trace.range(name, lane)`](../api/trace.md#tileops.trace.api._Trace.range) — a `with` block
   timed from enter to exit, drawn as a bar on sub-lane `lane`. For control flow
   that does not fit a `with`, use
   [`trace.range_start`](../api/trace.md#tileops.trace.api._Trace.range_start) /
   [`trace.range_end`](../api/trace.md#tileops.trace.api._Trace.range_end); for a zero-width mark
   use [`trace.record`](../api/trace.md#tileops.trace.api._Trace.record).
1. Lane names (`"tma"`, `"barrier"`, `"wgmma"`, the default `"main"`) become the
   rows in the timeline.
1. [`trace.dag(src, dst)`](../api/trace.md#tileops.trace.api._Trace.dag) — declares a dependency
   arrow from one named range to another (`arrive` → `wait`), drawn once per
   occurrence.
1. [`trace.finalize(func, traced, max_events)`](../api/trace.md#tileops.trace.api._Trace.finalize)
   — lowers the markers (adding the `slots` output) when `traced`, else strips
   them to zero cost. `traced` **must be part of the builder's cache key** so a
   traced and an untraced build for the same shape do not collide.

## Running a traced kernel

Call the builder with `traced=trace.enabled` and hand the compiled kernel to
`trace.run`. The same `forward` works in both modes: with tracing off it returns
the outputs unchanged; with tracing on it dumps the timeline and returns the real
outputs only — so you never branch on the switch yourself.

```{ .python .annotate }
def forward(self, a, b):
    compiled = build_gemm(self.m, self.n, self.k, self.dtype_str,
                          traced=trace.enabled)(**self.config)  # (1)!
    return trace.run(compiled, (a, b), stem="gemm_128x256x512")  # (2)!
```

1. Build the kernel matching the switch — [`trace.enabled`](../api/trace.md#tileops.trace.api._Trace.enabled)
   picks the traced or stripped variant (and is part of the cache key from
   marker `(7)`).
1. [`trace.run(compiled, inputs, stem=...)`](../api/trace.md#tileops.trace.api._Trace.run) — runs
   the kernel; when traced, splits the trailing `slots` off, decodes it, and
   writes `debug/<stem>.html` (a fresh, non-colliding file each call). Under the
   hood it is [`decode`](../api/trace.md#tileops.trace.api._Trace.decode) +
   [`dump`](../api/trace.md#tileops.trace.api._Trace.dump), which you can also call directly.

## Enabling tracing

Tracing is off by default. Flip the process-local switch once at startup, then
run as usual:

```{ .python .annotate }
from tileops.trace import trace

trace.enable()  # (1)!
c = op.forward(a, b)  # (2)!
```

1. [`trace.enable(output="debug")`](../api/trace.md#tileops.trace.api._Trace.enable) — turn
   tracing on and choose the dump directory (default `debug/`, gitignored). The
   switch is process-local — no environment variable, and `tilelang` is not
   monkeypatched. See also [`trace.disable()`](../api/trace.md#tileops.trace.api._Trace.disable),
   [`trace.enabled`](../api/trace.md#tileops.trace.api._Trace.enabled),
   [`trace.output`](../api/trace.md#tileops.trace.api._Trace.output).
1. Any traced kernel that now runs writes `debug/<stem>.html`.

From a pytest run, `--trace-kernel` calls `trace.enable()` from `pytest_configure`
before any kernel is built:

```bash
pytest tests/ops/test_gemm.py --trace-kernel
```

## Reading the timeline

<iframe src="../gemm-trace.html" title="GEMM timeline" width="100%" height="540"
        style="border:1px solid var(--md-default-fg-color--lightest);border-radius:4px;"></iframe>

- **CTA tabs** at the top — one timeline per CTA (block).
- **Lanes** (rows) come from your `group` and `lane` names — e.g. `producer / tma`,
  `consumer / wgmma`. Each `range` is a bar; hover for its name and cycle span.
- **X axis** is raw SM cycles (`clock64()`), zeroed per CTA.
- **Arrows** are your `dag` edges (e.g. producer `arrive` → consumer `wait`), one
  per occurrence — so you can read the handoff latency and whether the consumer
  is starved.
- Zoom and pan are horizontal-only.

What to look for: gaps on the `wgmma` lane (consumer stalled waiting on a TMA
load), `dag` arrows that do not overlap across iterations (no pipelining), or one
lane dominating the others (imbalance).
