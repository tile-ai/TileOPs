# Known Issues

Acknowledged gaps where the spec is correct but the implementation has a
non-trivial perf / precision / specialization deficit and we're choosing
not to fix it now. Bugs go to GitHub issues; this file is the register.

Each entry: affected ops, the gap, costs, and where a fix lands.

______________________________________________________________________

## Logical reductions — input dtype pre-conversion overhead

**Ops**: `AllFwdOp`, `AnyFwdOp`, `CountNonzeroFwdOp`
(`tileops/manifest/reduction.yaml`, `tileops/kernels/reduction/logical_reduce.py`)

**Gap**: PyTorch's `torch.all` / `torch.any` / `torch.count_nonzero`
accept `bool | int32 | int64 | complex64 | complex128` in addition to
the floating dtypes; the manifest currently lists only
`float16 | bfloat16 | float32 | bool` (`AllFwdOp` / `AnyFwdOp`) and
`float16 | bfloat16 | float32` (`CountNonzeroFwdOp`). Once the
manifest is expanded to match the PyTorch surface (tracked in #1451),
the int / complex inputs will be admissible but the TileLang kernel
still only ingests `float16 | bfloat16 | float32`. The op layer
bridges via `to_logical_float32` (`logical_reduce.py:51`),
materializing a full-sized `float32` intermediate before dispatch.

**Per-dtype cost of the materialization**:

| Input dtype  | Cost                                                       |
| ------------ | ---------------------------------------------------------- |
| `bool`       | 1 B → 4 B, 4× DRAM traffic; reduce was bandwidth-bound     |
| `int32`      | extra full-tensor read + write, not fused                  |
| `int64`      | 8 B → 4 B; bits wasted (only `!= 0` is needed)             |
| `complex64`  | 8 B read twice (`.real`, `.imag`) → 4 B write; ≥1.25× DRAM |
| `complex128` | 16 B read twice → 4 B write; worst case                    |

**Also**:

- Kernel has a redundant `T.if_then_else(x_f32 != 0.0, ...)`
  (`logical_reduce.py:131`, `:210`) — no-op on the pre-converted path,
  needed on the raw float path; kernel can't tell them apart.
- `source.kernel_map` is not dtype-specialized; future specialization
  must extend both kernel and manifest map shape.

**Fix sketch**: dtype-specialized kernels that consume the source dtype
directly and predicate on first load; drop `to_logical_float32` for
those dtypes; extend `kernel_map`; split or DCE the redundant predicate.
