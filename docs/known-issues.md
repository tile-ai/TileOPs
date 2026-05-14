# Known Issues

Tracks issues where the spec / manifest is correct but the current
implementation has a non-trivial gap (performance, precision, missing
specialization). Each entry should name the affected op(s), the gap, the
root cause, and a pointer to where a fix would land.

Entries here are not bugs against the spec — those go to GitHub issues
with the `bug` label. Entries here are *acknowledged* deviations that
the project chooses not to fix immediately.

______________________________________________________________________

## Logical reductions — input dtype pre-conversion overhead

**Affected ops**: `AllFwdOp`, `AnyFwdOp`, `CountNonzeroFwdOp`
(`tileops/manifest/reduction.yaml`, `tileops/kernels/reduction/logical_reduce.py`)

**Spec status**: aligned with PyTorch as of #1451 — signatures accept
`bool | float16 | bfloat16 | float32 | int32 | int64 | complex64 | complex128`.

**Gap**: the TileLang kernel only supports `float16 | bfloat16 | float32`
inputs. The op layer bridges the gap by calling
`to_logical_float32(x)` (`logical_reduce.py:51`) before kernel dispatch
for any input whose dtype is in `_UNSUPPORTED_STORAGE_DTYPES = {bool, int32, int64, complex64, complex128}`. This produces a full-sized
`float32` intermediate tensor and reads the source tensor end-to-end in
PyTorch before the kernel ever runs.

**Concrete costs**:

| Input dtype  | Pre-conversion cost                                                        | Notes                                                                                 |
| ------------ | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `bool`       | 1 B → 4 B write amplification (4× DRAM traffic on the materialized tensor) | logical reduce on bool is bandwidth-bound; the predicate dominates the actual reduce. |
| `int32`      | one extra full-tensor read + float32 write                                 | Bandwidth-neutral on the *write* side, but the materialize is not fused.              |
| `int64`      | 8 B → 4 B; cast loses precision for \|x\| > 2²⁴                            | Semantically safe (we only need `!= 0`), but the lost bits are wasted bandwidth.      |
| `complex64`  | 8 B input read twice (`.real`, `.imag`), then 4 B float32 write            | At least 1.25× DRAM traffic vs. a fused kernel.                                       |
| `complex128` | 16 B input read twice, then 4 B float32 write                              | Worst case; biggest opportunity if specialized.                                       |

Additionally:

- The kernel itself contains a redundant `T.if_then_else(x_f32 != 0.0, 1.0, 0.0)` (`logical_reduce.py:131`, `:210`). For tensors that came
  through `to_logical_float32` the input is already `{0.0, 1.0}` and
  this predicate is a no-op; the kernel cannot distinguish the two
  paths so the instruction stays.
- `source.kernel_map` in `reduction.yaml` is not dtype-specialized.
  Adding integer/complex kernels later will require expanding both the
  kernel side and the manifest's `kernel_map` shape.

**Why not fixed now**: PR #1451 was scoped to manifest alignment per the
trust-model carve-out (manifest-only PR). The fix needs new TileLang
kernels that take `int{32,64}`, `bool`, and `complex{64,128}` directly
and do the `!= 0` predicate on the first load — separate work.

**Pointer for a fix**:

- Add dtype-specialized entries to
  `tileops/kernels/reduction/logical_reduce.py` so the kernel ingests
  the source dtype and writes 0/1 into shared memory at first load.
- Drop the op-layer `to_logical_float32` call for dtypes the kernel can
  now consume directly.
- Expand `reduction.yaml`'s `source.kernel_map` accordingly.
- Remove the redundant `!= 0` predicate on the post-conversion path
  (or split the kernel into pre-converted vs. raw entry points).
