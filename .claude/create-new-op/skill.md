# Skill: Creating a New TileOps Op

Reference implementations: `tileops/ops/mha.py`, `tileops/ops/gqa_decode.py`, `tileops/ops/deepseek_nsa.py`
Base class: `tileops/ops/op.py`

______________________________________________________________________

## Overview

An `Op` is a thin orchestration layer that:

1. Holds kernel instances (one or more `Kernel` subclasses)
1. Dispatches to the correct kernel based on hardware via `dispatch_kernel`
1. Exposes a `forward` method that calls the kernel(s) and handles any pre/post-processing

An `Op` does **not** implement GPU computation itself — that lives in the `Kernel`.

______________________________________________________________________

## File Location and Naming

```
tileops/ops/<op_name>.py
```

- File name: `<op_name>.py`, all lowercase with underscores (e.g. `gqa_decode.py`, `deepseek_nsa.py`)
- Class name: CamelCase + `Op` suffix (e.g. `GroupQueryAttentionDecodeWithKVCacheOp`)
- Group multiple related ops in one file when they share the same kernel set (e.g. `mha.py` contains both `MultiHeadAttentionFwdOp` and `MultiHeadAttentionBwdOp`)
- After creating the file, register all new classes in `tileops/ops/__init__.py`

______________________________________________________________________

## File Structure

```
tileops/ops/<op_name>.py
```

A single op file contains:

1. Imports
1. `__all__` declaration
1. One or more `Op` subclasses

After creating the file, register the new class in `tileops/ops/__init__.py`.

______________________________________________________________________

## Part 1: Imports

```python
from typing import Dict, Optional

import torch

from tileops.kernels.<module> import <KernelClass>
from tileops.kernels.kernel import Kernel

from .op import Op
```

Only import `is_hopper` from `tileops.utils` if the op needs to dispatch different kernels per architecture.

______________________________________________________________________

## Part 2: Op Class

### Naming

- CamelCase, descriptive, ending in `Op`
- Examples: `MultiHeadAttentionFwdOp`, `NSATopkVarlenOp`, `GroupQueryAttentionDecodeWithKVCacheOp`

### `__init__` — two patterns

**Pattern A: simple (one kernel, few params)**

Assign each param explicitly, then call `dispatch_kernel` and instantiate the kernel:

```python
class <OpName>Op(Op):

    def __init__(self,
                 param1: int,
                 param2: int,
                 dtype: torch.dtype = torch.float16,
                 tune: bool = False,
                 kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        self.param1 = param1
        self.param2 = param2
        self.dtype = dtype

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["<kernel_key>"](
            param1, param2, dtype, tune=tune)
```

**Pattern B: many params (use locals() shortcut)**

When there are many parameters, use `locals()` to avoid repetition:

```python
    def __init__(self,
                 param1: int,
                 ...,
                 dtype: torch.dtype = torch.float16,
                 tune: bool = False,
                 kernel_map: Optional[Dict[str, Kernel]] = None,
    ) -> None:
        params = {k: v for k, v in locals().items() if k not in ('self', 'kernel_map')}
        for key, value in params.items():
            setattr(self, key, value)

        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map["<kernel_key>"](**params)
```

> `kernel_map` is always the last parameter, after `tune`. When using `locals()`, exclude both `self` and `kernel_map` from `params`; `tune` is included and passed through to the kernel. `accum_dtype` is never a parameter of the Op — it is hardcoded inside the Kernel.

### `default_kernel_map`

Maps string keys to `Kernel` classes. This is what `dispatch_kernel` uses to resolve the actual kernel type:

```python
    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"<kernel_key>": <KernelClass>}
```

For ops that need different kernels per architecture:

```python
    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"<kernel_key>": <HopperKernelClass> if is_hopper() else <DefaultKernelClass>}
```

For ops with multiple kernels (e.g. fwd + pre/post-process):

```python
    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {
            "preprocess_kernel": PreprocessKernel,
            "main_kernel": MainWgmmaPipelinedKernel if is_hopper() else MainKernel,
            "postprocess_kernel": PostprocessKernel if not is_hopper() else None,
        }
```

### `forward`

Calls `self.kernel(...)` directly (the `Kernel.__call__` delegates to `Kernel.forward`).

```python
    def forward(self, input1: torch.Tensor, ...) -> torch.Tensor:
        return self.kernel(input1, ...)
```

For ops with pre/post-processing:

```python
    def forward(self, ...) -> tuple[torch.Tensor, ...]:
        intermediate = self.prep_kernel(...)
        result = self.kernel(..., intermediate)
        return self.post_kernel(result)
```

For ops that need input validation or padding before calling the kernel (see `gqa_decode.py`):

```python
    def forward(self, q: torch.Tensor, k: torch.Tensor, ...) -> torch.Tensor:
        # e.g. pad k/v to match declared seqlen
        if k.shape[1] < self.seqlen_kv:
            k = F.pad(k, ...)
        return self.kernel(q, k, ...)
```

______________________________________________________________________

## Part 3: Register in `__init__.py`

Add the import and the class name to `__all__` in `tileops/ops/__init__.py`:

```python
# import
from .<op_module> import <OpName>Op

# __all__
__all__ = [
    ...
    "<OpName>Op",
]
```

______________________________________________________________________

## Key Points

- `dispatch_kernel(kernel_map)` must be called before instantiating any kernel; it validates arch compatibility via `Kernel.supported_archs`
- `self.kernel` is the primary kernel; additional kernels (pre/post) are stored as `self.prep_kernel`, `self.post_kernel`, etc.
- The `Op` does not store `accum_dtype` — that is internal to the `Kernel`
- `dtype` defaults to `torch.float16` unless the op has a specific reason to require explicit dtype
- `kernel_map` parameter allows callers to inject alternative kernel implementations (for testing or custom dispatch); always default to `None`
- `tune=False` is always before `kernel_map`; `kernel_map` is always the last `__init__` parameter
- `accum_dtype` is never a parameter of the Op — it is hardcoded inside the Kernel

______________________________________________________________________

## Unit Test Requirement

Every new op **must** have a passing unit test in `tests/ops/test_<op_name>.py` before it is considered complete. See `.claude/create-new-op-test/skill.md` for how to write the test.

### Correctness debugging protocol

When the unit test fails due to numerical mismatch, follow this process in order:

1. **Do not loosen tolerances.** A large numerical error indicates a real bug, not a precision issue. The PyTorch reference is the ground truth.

1. **Check the algorithm first.** Compare the tilelang implementation step-by-step against the official reference URL in the kernel's docstring — scale factors, layout (BSHD vs BHSD), masking logic, softmax normalization, etc.

1. **Check memory layout and indexing.** Verify that tensor shapes, strides, and index expressions in `T.copy` / `T.gemm` match the intended layout.

1. **Isolate the stage.** If the kernel has multiple stages (e.g. QK gemm → softmax → PV gemm), add intermediate checks to identify which stage produces the wrong result.

1. **Replace tilelang primitives with scalar equivalents for debugging.** If the above steps do not reveal the bug, temporarily replace tilelang primitives with simpler equivalents to rule out compiler/primitive bugs:

   - Replace `T.Pipelined` with `T.serial` (removes software pipelining)
   - Replace `T.copy(src, dst)` with a `T.Parallel` loop and element-wise assignment:
     ```python
     # DEBUG: replaced T.copy with explicit parallel assignment
     for i, j in T.Parallel(M, N):
         dst[i, j] = src[i, j]
     ```
   - Replace `T.gemm(A, B, C)` with explicit `T.Serial` + `T.Parallel` accumulation:
     ```python
     # DEBUG: replaced T.gemm with explicit serial-parallel matmul
     for k in T.serial(K):
         for i, j in T.Parallel(M, N):
             C[i, j] += A[i, k] * B[j, k]  # adjust indices for transpose_B
     ```
   - Mark all such changes with a `# DEBUG:` comment so they are easy to find and revert.

1. **Revert debug changes** once the bug is found and fixed. Do not leave `T.serial` / explicit loops in production code.

______________________________________________________________________

- [ ] Class name is CamelCase and ends with `Op`
- [ ] File placed at `tileops/ops/<op_name>.py` following lowercase underscore naming
- [ ] `dispatch_kernel(kernel_map)` is called before kernel instantiation
- [ ] `default_kernel_map` is implemented and returns at least one entry
- [ ] `forward` calls `self.kernel(...)`, not the raw kernel function
- [ ] Unit test at `tests/ops/test_<op_name>.py` exists and passes
- [ ] Op is imported and added to `__all__` in `tileops/ops/__init__.py`
- [ ] `accum_dtype` is not stored on the Op class
- [ ] `kernel_map: Optional[Dict[str, Kernel]] = None` is the last `__init__` parameter, after `tune`
- [ ] `accum_dtype` is not a parameter of the Op
