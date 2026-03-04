---
name: check-kernel-format
description: Checklist for verifying that a TileOps Kernel and Op conform to all format requirements. Run this after creating a new op AND after tuning.
---

# Kernel Format Checklist

> **When to run**: after `create-new-op` completes **and** again after `tune` completes.
> This checklist consolidates all structural requirements from `create-new-op`, `tune`, and lessons learned in sessions. If any item fails, fix it before opening a PR.

______________________________________________________________________

## 1. `T.prim_func` Structure

- [ ] A standalone `_<op_name>_kernel(static_params) -> Callable` function exists that captures shapes and dtypes as closure variables
- [ ] Inside it, `@tilelang.jit(out_idx=[...])` wraps a config-parameterised function whose arguments are the tunable config keys
- [ ] **`with T.Kernel()` blocks MUST be inside `@T.macro` functions, NOT directly in `@T.prim_func`** — the `@T.prim_func` should only call macros and contain minimal control flow
- [ ] Inside the jit function, `@T.macro` functions define GPU kernel launches using `with T.Kernel()` blocks
- [ ] Inside the macro functions, use TileLang primitives only: `T.Kernel`, `T.Parallel`, `T.serial`, `T.Pipelined`, `T.alloc_shared`, `T.alloc_fragment`, `T.copy`, `T.gemm`, `T.clear`, `T.cast`, `T.cos`, `T.sin`, etc.
- [ ] No Python builtins (`float()`, `int()`, `math.cos()`) applied to TileLang symbolic IR nodes — use `T.cast`, `T.cos`, `T.sin`, `T.min`, etc. instead
- [ ] Tile-ops (`T.clear`, `T.copy`, `T.gemm`) are NOT placed inside `T.Parallel` — they must be at `T.Kernel` scope; per-thread data is accessed by indexing shared/fragment buffers with the thread index

______________________________________________________________________

## 2. `Kernel` Class Structure

- [ ] `Kernel.forward` takes only the GPU tensor arguments it passes directly to the TileLang kernel — **no format conversion, no batching loop, no dtype casting** inside `forward`
- [ ] `Kernel.forward` body is essentially `return self.kernel(config...)(tensors)` — nothing else
- [ ] `default_config` property is defined and returns a dict with all tunable keys
- [ ] `autotune_configs` property is defined and returns a list of config dicts covering hardware-aligned values
- [ ] `supported_archs` class attribute is set (e.g. `[75, 80, 86, 89, 90]`)
- [ ] `accum_dtype` is **hardcoded** inside the kernel (e.g. `"float64"` for float32 inputs) — never a property, never a config key, never a parameter
- [ ] The `@torch.library.custom_op` + `.register_fake` wrapper exists for `torch.compile` / autograd compatibility (even if `forward` does not call it directly)

______________________________________________________________________

## 3. `Op` Class Structure

- [ ] `Op.forward` owns **all** pre/post-processing: format conversions (e.g. complex → real + imag split, and recombine), dtype casting, padding, batching loops, reshaping
- [ ] `Op.forward` delegates the core GPU computation entirely to `self.kernel(...)` (which calls `Kernel.__call__` → `Kernel.forward`)
- [ ] `accum_dtype` is **not** stored on the Op class and is **not** a parameter of `Op.__init__`
- [ ] `kernel_map: Optional[Dict[str, Kernel]] = None` is the **last** `__init__` parameter, after `tune`
- [ ] `dispatch_kernel(kernel_map)` is called **before** any kernel is instantiated

______________________________________________________________________

## 4. Tuning Requirements

- [ ] **Iterative optimization protocol followed**: All known optimizations from `.claude/skills/tune/skill.md` §6 have been applied recursively until no further improvements are possible or performance is within 5-10% of reference implementation
- [ ] If the kernel algorithm contains any matrix-multiplication (GEMV, GEMM, batch matmul, or a dot-product reduction loop), consult `.claude/skills/tune-multiplication/skill.md` for applicable patterns (coalescing, warp reduction, pipeline stages, tile sizes, autotune search space) and document which patterns were applied or ruled out
- [ ] All serial / sequential computation has been evaluated for restructuring with `T.Parallel`, `T.copy`, or `T.Pipelined`; the decision (apply or rule out with justification) is documented in the op-specific tune skill
- [ ] Autotune covers all hardware-aligned config values; search space is documented in the op-specific `tune-<op>/skill.md`
- [ ] Benchmark results (latency, BW/TFLOPs vs baseline) are recorded in the op-specific tune skill
- [ ] Each optimization iteration is documented: what was tried, benchmark results before/after, why optimization stopped

______________________________________________________________________

## 5. Quick Reference: Correct Kernel/Op Split

```
WRONG — format conversion inside Kernel.forward:

class MyKernel(Kernel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_real = x.real.contiguous()          # ← PyTorch in Kernel!
        x_imag = x.imag.contiguous()
        y_r, y_i = self.kernel(config)(x_real, x_imag)
        return torch.complex(y_r, y_i)        # ← PyTorch in Kernel!

CORRECT — Kernel.forward is pure TileLang invocation:

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

______________________________________________________________________

## 6. Quick Reference: Correct @T.macro and @T.prim_func Structure

```
WRONG — T.Kernel directly in @T.prim_func:

@tilelang.jit(out_idx=[2, 3])
def _kernel_func(block_size: int, threads: int):
    @T.prim_func
    def _main(x: T.Tensor, y: T.Tensor):
        with T.Kernel(n_blocks, threads=threads) as bx:  # ← WRONG!
            for i in T.Parallel(block_size):
                y[i] = x[i] * 2

CORRECT — T.Kernel inside @T.macro, called from @T.prim_func:

@tilelang.jit(out_idx=[2, 3])
def _kernel_func(block_size: int, threads: int):
    @T.macro
    def compute_kernel(x: T.Tensor, y: T.Tensor):
        with T.Kernel(n_blocks, threads=threads) as bx:  # ← CORRECT!
            for i in T.Parallel(block_size):
                y[i] = x[i] * 2

    @T.prim_func
    def _main(x: T.Tensor, y: T.Tensor):
        compute_kernel(x, y)  # ← Just call the macro
```

See `tileops/kernels/flash_decode/gqa_decode.py` for a complete reference implementation.

______________________________________________________________________

## References

- `.claude/skills/create-new-op/skill.md` — Op/Kernel architecture and registration
- `.claude/skills/tune/skill.md` — Benchmarking methodology and pre-benchmark checklist
- `.claude/skills/tune-multiplication/skill.md` — Matmul/GEMV tuning patterns
- `.claude/skills/kernel-debug/skill.md` — TileLang debugging techniques (T.alloc_var, tile-op restrictions, precision issues)
