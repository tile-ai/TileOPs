# Kernel and Op Conventions

Reference for TileOps structural requirements. All kernel and op code must conform to these
conventions before opening a PR. See `check-kernel-format` skill for the validation workflow.

---

## 1. `T.prim_func` Structure

A TileOps kernel follows a three-level nesting pattern:

```
_<op_name>_kernel(static_params)          ← closure function (Python)
  └─ @tilelang.jit(out_idx=[...])         ← JIT compiler entry point
       └─ inner(config_keys...)            ← tunable config parameters
            ├─ @T.macro                   ← GPU kernel body (TileLang IR)
            │    └─ with T.Kernel(...)
            └─ @T.prim_func               ← top-level function (calls macros only)
```

Rules:

- A standalone `_<op_name>_kernel(static_params) -> Callable` function must exist; it
  captures shapes and dtypes as closure variables.
- Inside it, `@tilelang.jit(out_idx=[...])` wraps a config-parameterised function whose
  arguments are the tunable config keys.
- `with T.Kernel()` blocks **must** be inside `@T.macro` functions, **not** directly in
  `@T.prim_func`. The `@T.prim_func` should only call macros and contain minimal control
  flow.
- Use TileLang primitives only inside macros: `T.Kernel`, `T.Parallel`, `T.serial`,
  `T.Pipelined`, `T.alloc_shared`, `T.alloc_fragment`, `T.copy`, `T.gemm`, `T.clear`,
  `T.cast`, `T.cos`, `T.sin`, etc.
- Do **not** apply Python builtins (`float()`, `int()`, `math.cos()`) to TileLang symbolic
  IR nodes — use `T.cast`, `T.cos`, `T.sin`, `T.min`, etc. instead.
- Tile-ops (`T.clear`, `T.copy`, `T.gemm`) are **not** placed inside `T.Parallel` — they
  must be at `T.Kernel` scope; per-thread data is accessed by indexing shared/fragment
  buffers with the thread index.

### Correct vs. incorrect T.macro usage

```python
# WRONG — T.Kernel directly in @T.prim_func:
@tilelang.jit(out_idx=[2, 3])
def _kernel_func(block_size: int, threads: int):
    @T.prim_func
    def _main(x: T.Tensor, y: T.Tensor):
        with T.Kernel(n_blocks, threads=threads) as bx:  # ← WRONG
            for i in T.Parallel(block_size):
                y[i] = x[i] * 2

# CORRECT — T.Kernel inside @T.macro, called from @T.prim_func:
@tilelang.jit(out_idx=[2, 3])
def _kernel_func(block_size: int, threads: int):
    @T.macro
    def compute_kernel(x: T.Tensor, y: T.Tensor):
        with T.Kernel(n_blocks, threads=threads) as bx:  # ← CORRECT
            for i in T.Parallel(block_size):
                y[i] = x[i] * 2

    @T.prim_func
    def _main(x: T.Tensor, y: T.Tensor):
        compute_kernel(x, y)  # ← just call the macro
```

Reference implementation: `tileops/kernels/flash_decode/gqa_decode.py`.

---

## 2. `Kernel` Class Structure

The `Kernel` class is a thin wrapper that manages the JIT-compiled function. It must **not**
contain any PyTorch data processing logic.

Requirements:

- `Kernel.forward` takes **only** the GPU tensor arguments it passes directly to the
  TileLang kernel — no format conversion, no batching loop, no dtype casting inside
  `forward`.
- `Kernel.forward` body is essentially `return self.kernel(config...)(tensors)` and nothing
  else.
- `default_config` property is defined and returns a `dict` with all tunable keys.
- `autotune_configs` property is defined and returns a list of config dicts covering
  hardware-aligned values.
- `supported_archs` class attribute is set (e.g. `[75, 80, 86, 89, 90]`).
- `accum_dtype` is **hardcoded** inside the kernel (e.g. `"float32"` for `float16` inputs)
  — never a property, never a config key, never a parameter.
- The `@torch.library.custom_op` + `.register_fake` wrapper exists for `torch.compile` /
  autograd compatibility, even if `forward` does not call it directly.

---

## 3. `Op` Class Structure

The `Op` class owns all user-facing pre/post-processing. Everything that is not raw GPU
tensor arithmetic belongs here.

Requirements:

- `Op.forward` owns **all** pre/post-processing: format conversions (e.g. complex → real +
  imag split and recombine), dtype casting, padding, batching loops, and reshaping.
- `Op.forward` delegates the core GPU computation entirely to `self.kernel(...)` (which
  calls `Kernel.__call__` → `Kernel.forward`).
- `accum_dtype` is **not** stored on the `Op` class and is **not** a parameter of
  `Op.__init__`.
- `kernel_map: Optional[Dict[str, Kernel]] = None` is the **last** `__init__` parameter,
  after `tune`.
- `dispatch_kernel(kernel_map)` is called **before** any kernel is instantiated.

### Correct vs. incorrect Kernel/Op split

```python
# WRONG — format conversion inside Kernel.forward:
class MyKernel(Kernel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real = x.real.contiguous()          # ← PyTorch in Kernel!
        x_imag = x.imag.contiguous()
        y_r, y_i = self.kernel(config)(x_real, x_imag)
        return torch.complex(y_r, y_i)        # ← PyTorch in Kernel!

# CORRECT — Kernel.forward is pure TileLang invocation:
class MyKernel(Kernel):
    def forward(self, x_real: Tensor, x_imag: Tensor) -> tuple[Tensor, Tensor]:
        return self.kernel(self.config["block_size"], self.config["threads"])(x_real, x_imag)

class MyOp(Op):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real = x.real.contiguous()          # ← format conversion in Op ✓
        x_imag = x.imag.contiguous()
        y_r, y_i = self.kernel(x_real, x_imag)
        return torch.complex(y_r, y_i)        # ← recombine in Op ✓
```

---

## 4. Tuning Requirements

> Tuning workflow is handled by the `tune` skill (`tune/skill.md`). The items below
> are the deliverables that must be present before a PR is opened.

- All known optimizations from `tune/skill.md` have been applied recursively until no
  further improvements are possible, or performance is within 5–10% of the reference.
- If the kernel contains matrix multiplication (GEMV, GEMM, batch matmul, or a
  dot-product reduction loop), consult `tune-multiplication/skill.md` for applicable
  patterns and document which were applied or ruled out.
- All serial/sequential computation has been evaluated for restructuring with `T.Parallel`,
  `T.copy`, or `T.Pipelined`; the decision is recorded in the PR description.
- Autotune covers all hardware-aligned config values; the search space is documented in
  the PR description or inline comments.
- Benchmark results (latency, BW/TFLOPs vs. baseline) are recorded in the PR description.
- Each optimization iteration is documented: what was tried, results before/after, and why
  optimization stopped.

---

## References

- [DEVELOPMENT.md](DEVELOPMENT.md) — architecture overview (2-layer stack), coding
  standards, testing strategy, and PR process
- `tileops/kernels/flash_decode/gqa_decode.py` — reference implementation
- `.claude/skills/migrating-new-op/SKILL.md` — op/kernel delivery workflow
- `.claude/skills/tune/skill.md` — benchmarking methodology and nsys analysis
- `.claude/skills/tune-multiplication/skill.md` — matmul/GEMV tuning patterns
- `.claude/skills/kernel-debug/skill.md` — TileLang debugging techniques
