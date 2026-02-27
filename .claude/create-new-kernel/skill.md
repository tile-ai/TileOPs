---
name: create-new-kernel
description: Create a new TileOps GPU kernel — file layout, tilelang kernel function, wrapper registration, and Kernel subclass. For attention-specific conventions (variants, causal masking, split-K decode, paged KV), also use the create-new-op and create-new-op-attention skills. Auto-invoke when the user asks to create, implement, or add a new kernel in TileOps.
---

# Skill: Creating a New TileOps Kernel

Reference implementation: `tileops/kernels/deepseek_nsa/nsa_topk.py`
Base class: `tileops/kernels/kernel.py`

## Development Environment

| Item     | Version      |
| -------- | ------------ |
| GPU      | H200 (sm90a) |
| CUDA     | 12.9         |
| TileLang | 0.1.7.post1  |

Target architecture is `sm90a`. All kernels are developed and validated on this environment. Do not assume compatibility with older architectures unless explicitly tested.

______________________________________________________________________

## File Location and Naming

```
tileops/kernels/<feature_name>/<kernel_name>.py
```

- Group related kernels under a feature subdirectory (e.g. `tileops/kernels/deepseek_nsa/`)
- File name: `<kernel_name>.py`, all lowercase with underscores (e.g. `nsa_topk.py`)
- Kernel function: `_<kernel_name>_kernel` (e.g. `_nsa_topk_varlen_kernel`)
- Wrapper function: `_<kernel_name>_wrapped_kernel`
- Class name: CamelCase + `Kernel` suffix (e.g. `NSATopkVarlenKernel`)
- Export the class from `tileops/kernels/<feature_name>/__init__.py`

______________________________________________________________________

## File Structure

A kernel file contains five parts in order:

1. Imports
1. Kernel function (tilelang implementation)
1. Wrapper function (`_<kernel_name>_wrapped_kernel`)
1. `register_fake` for the wrapper
1. `Kernel` subclass

______________________________________________________________________

## Part 1: Imports

```python
import torch
from typing import Optional, Any, Callable

import tilelang
from tilelang import language as T

from tileops.kernels.kernel import Kernel
```

Tilelang kernel implementations generally need no additional imports beyond these.

______________________________________________________________________

## Part 2: Kernel Function

The kernel is implemented as a two-level closure:

```python
def _<kernel_name>_kernel(
    # Fixed kernel parameters: shapes, dtypes, algorithm constants
    param1: int,
    param2: int,
    dtype: str,
    accum_dtype: str,
) -> Callable:
    # Precompute constants and define tensor shapes here
    # e.g. q_shape = [seq_len, heads, dim]

    @tilelang.jit(
        out_idx=[-1],  # index of output tensor in @T.prim_func's parameter list
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        })
    def _<kernel_name>_func(threads: int):  # auto-tune parameters go here

        @T.prim_func
        def _main(
            input1: T.Tensor(shape1, dtype),
            ...,
            output: T.Tensor(output_shape, accum_dtype),
        ):
            with T.Kernel(grid_x, grid_y, threads=threads) as (bx, by):
                # kernel body

        return _main

    return _<kernel_name>_func
```

### Dtype conventions

- Default compute dtype: `float16` (`"float16"`); exposed as `dtype: str` in the outermost kernel function so callers can override.
- Accumulator dtype: always `float32`, hardcoded internally as `accum_dtype = "float32"`. Do not expose it as a parameter.

### Memory hierarchy

- Always move data from global memory to **shared memory** (`T.alloc_shared`) before computation.
- For reductions, move data from shared memory to **fragment** (`T.alloc_fragment`) first, then call `T.reduce_sum` / `T.reduce_max` on the fragment.
- Never reduce directly on shared memory.

Typical pattern:

```
global → T.copy → shared → T.gemm / T.copy → fragment → T.reduce_* → fragment
```

### Code organisation

- Keep `@T.prim_func` minimal — it should read like a high-level algorithm outline.
- Extract non-trivial logic (sorting, masking, multi-step computations) into `@T.macro` helpers defined inside `_<kernel_name>_func`, and call them from `_main`.

### Tilelang coding guidelines

| Concern        | Guidance                                                                                                                                                                           |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Exponentiation | Use `T.exp2(x * LOG2_E)` instead of `T.exp(x)` for performance (`LOG2_E = 1.44269504`)                                                                                             |
| Data movement  | Use `T.copy()` for copies; `T.gemm()` for matrix multiply                                                                                                                          |
| Reduction      | Use `T.reduce_sum()` / `T.reduce_max()` on fragments (not shared memory)                                                                                                           |
| Parallelism    | Prefer `T.Parallel` over `T.Serial`; only use `T.Serial` when sequential dependency is unavoidable                                                                                 |
| Loop structure | Prefer `T.Pipelined` for outer loops to enable software pipelining                                                                                                                 |
| Bug workaround | Tilelang may have bugs with nested `T.Parallel` / `T.Serial` / `T.Pipelined`. If compilation or runtime errors occur, try reordering the nesting levels as a first debugging step. |

______________________________________________________________________

## Part 3: Wrapper Function

The wrapper registers the kernel with `torch.library` so it is visible to `torch.compile` and `torch.export`. The `Kernel` class calls this wrapper in `forward`.

### 3a. custom_op wrapper

```python
@torch.library.custom_op("top::<kernel_name>_wrapped_kernel", mutates_args=())
def _<kernel_name>_wrapped_kernel(
    # All scalar params first (int, float, str for dtype names)
    param1: int,
    dtype: str,
    threads: int,
    # Tensor inputs last
    input1: torch.Tensor,
    ...
) -> torch.Tensor:
    return _<kernel_name>_kernel(param1, dtype)(threads)(input1, ...)
```

The call chain is: `_kernel(fixed_params)(autotune_params)(tensor_inputs)`.

Note: `accum_dtype` is hardcoded inside `_<kernel_name>_kernel` and is not a parameter of the wrapper.

### 3b. register_fake

Provides a shape/dtype inference rule for tracing (no actual computation):

```python
@_ < kernel_name > _wrapped_kernel.register_fake
def _(
    param1: int,
    dtype: str,
    threads: int,
    *inputs: tuple[Any],
) -> torch.Tensor:
    _ = (param1, dtype, threads)  # suppress unused warnings
    # Return an empty tensor with the correct output shape and dtype.
    # Shape must match what the real kernel produces.
    return torch.empty(
        [output_dim0, output_dim1, ...],
        dtype=inputs[0].dtype,
        device=inputs[0].device,
    )
```

> The output shape here must exactly match the real kernel's output. Derive it from the scalar parameters (e.g. `c_seq_len`, `heads`, `selected_block_num`), not from input tensor shapes.

______________________________________________________________________

## Part 4: Kernel Class

- Naming: CamelCase, descriptive, ending in `Kernel` (e.g. `NSATopkVarlenKernel`)
- Inherit from `Kernel` (see `tileops/kernels/kernel.py`)
- Declare `supported_archs` for GPU arch gating; current target is H200 (sm90a), so use `[90]`

The class **must** include a docstring with three sections:

1. Input tensor shapes — list each tensor parameter with its layout (e.g. `[batch, seqlen, heads, dim]`)
1. Computation logic — a brief description of what the kernel computes
1. Reference — URL to the official PyTorch / Triton / paper implementation this is based on

Example:

```python
class <KernelName>Kernel(Kernel):
    """<One-line description of what this kernel computes>.

    Args:
        q: Query tensor, shape [batch, seqlen_q, heads, dim]
        k: Key tensor, shape [batch, seqlen_k, heads_kv, dim]
        v: Value tensor, shape [batch, seqlen_k, heads_kv, dim_v]
        offsets: Sequence boundary offsets, shape [seq_num + 1], dtype int32
        ...

    Computation:
        <Brief description of the algorithm, e.g.:
        Computes multi-head attention scores via Q @ K^T * scale, applies
        softmax, then outputs softmax(QK^T) @ V.>

    Reference:
        <URL to official implementation, e.g.:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py>
    """
    supported_archs: list[int] = [90]

    def __init__(self,
                 param1: int,
                 ...,
                 dtype: torch.dtype = torch.float16,
                 config: Optional[dict] = None,
                 tune: bool = False) -> None:
        super().__init__()
        self.param1 = param1
        ...
        # Store dtype as string for tilelang; accum_dtype is hardcoded in the kernel
        self.dtype_name = str(dtype).split('.')[-1]
        self.init_config(config, tune)  # must be called last

    @property
    def default_config(self) -> dict:
        return {"threads": 32}

    @property
    def autotune_configs(self) -> list[dict]:
        return [{"threads": t} for t in [32, 64, 128]]

    def forward(self, input1: torch.Tensor, ...) -> torch.Tensor:
        return _<kernel_name>_wrapped_kernel(
            self.param1,
            ...,
            self.dtype_name,
            self.config["threads"],
            input1.to(getattr(torch, self.dtype_name)),
            ...,
        )
```

Key points:

- `forward` calls `_<kernel_name>_wrapped_kernel` (the registered wrapper), not the raw kernel function directly
- `init_config(config, tune)` must be the last call in `__init__`; it reads `default_config` and `autotune_configs`
- `accum_dtype` is not stored on the class; it is hardcoded inside the kernel function
- Only cast index/offset tensors (e.g. `offsets`, `token_indices`) to `torch.int32` in `forward`; do NOT cast floating-point tensors

- [ ] `out_idx` in `@tilelang.jit` points to the correct output tensor position
- [ ] `register_fake` output shape matches the real kernel output
- [ ] `custom_op` name is unique: `"top::<kernel_name>_wrapped_kernel"`
- [ ] File placed under `tileops/kernels/<feature_name>/<kernel_name>.py` and exported from its `__init__.py`
- [ ] Class name is CamelCase and ends with `Kernel`
- [ ] Class has docstring with: input tensor shapes, computation logic, reference URL
- [ ] `supported_archs` is set appropriately
- [ ] `init_config` is called at the end of `__init__`
- [ ] `forward` calls the wrapper (`_<kernel_name>_wrapped_kernel`), not the raw kernel directly
- [ ] Index/offset tensors (e.g. `offsets`, `token_indices`) are cast to `torch.int32` in `forward`; do NOT cast floating-point tensors
- [ ] Default dtype is `float16` (exposed); accumulator is `float32` (hardcoded internally)
- [ ] Global memory is copied to shared memory before computation
- [ ] Reductions operate on fragments, not shared memory
- [ ] `@T.prim_func` is kept minimal; complex logic is in `@T.macro` helpers
