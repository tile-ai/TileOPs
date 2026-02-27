---
name: create-new-op-attention
description: Attention-specific conventions for creating a new TileOps attention kernel — variants (fwd/decode/bwd), causal masking, split-K decode design, and paged KV cache. Use together with create-new-kernel (kernel structure) and create-new-op (op registration). Auto-invoke when the user asks to create or implement an attention kernel (MHA, GQA, MLA, flash attention, decode attention, etc.) in TileOps.
---

# Skill: Attention Kernel Conventions

Attention kernels have additional conventions beyond the general rules in `create-new-kernel` and `create-new-op`.

______________________________________________________________________

## Kernel variants

Every attention kernel must be classified into one of three variants, which determines its file name, class name, and internal structure:

| Variant  | File suffix  | Class suffix   | Description                                   |
| -------- | ------------ | -------------- | --------------------------------------------- |
| Forward  | `_fwd.py`    | `FwdKernel`    | Full-sequence prefill / training forward pass |
| Decode   | `_decode.py` | `DecodeKernel` | Single-token decode with KV cache             |
| Backward | `_bwd.py`    | `BwdKernel`    | Training backward pass                        |

Examples: `mha_fwd.py` → `MhaFwdKernel`, `mha_decode.py` → `MhaDecodeKernel`

______________________________________________________________________

## `causal` parameter

All attention kernels **must** expose `is_causal: bool` as a parameter of the outermost kernel function (the two-level closure). It controls the causal masking logic inside the `@T.prim_func`.

```python
def _<attn_name>_kernel(
    batch: int,
    heads: int,
    seqlen_q: int,
    seqlen_kv: int,
    dim: int,
    is_causal: bool,   # required for all attention kernels
    dtype: str,
) -> Callable:
    ...
```

______________________________________________________________________

## Decode kernel: split-K design

Decode kernels must support both a **no-split** and a **split-K** execution path, selected at runtime by `num_split`:

- `num_split = 1` → use the no-split `@T.prim_func` (single pass over KV)
- `num_split > 1` → use the split `@T.prim_func` (parallel over KV chunks, then combine)

Both paths are implemented as `@T.macro` functions inside `_<kernel_name>_func`, and the outer `@T.prim_func` simply calls the appropriate macro. The wrapper function (`_<kernel_name>_wrapped_kernel`) computes `split_length` and dispatches to the correct path.

`num_split` is a tunable parameter and must appear in `default_config` and `autotune_configs`.

Structure inside `_<kernel_name>_func`:

```python
def _<attn_name>_decode_func(block_M, block_N, num_split, num_stages, threads):

    @T.macro
    def _<attn_name>_no_split(Q, K, V, real_seqlen_kv, Output):
        # single-pass attention over full KV
        ...

    @T.macro
    def _<attn_name>_split(Q, K, V, real_seqlen_kv, glse, Output_partial, split_length):
        # attention over one KV chunk; writes partial output + log-sum-exp
        ...

    @T.macro
    def combine(glse, Output_partial, Output):
        # merge partial outputs using LSE rescaling
        ...

    @T.prim_func
    def <attn_name>_decode_no_split(Q, K, V, real_seqlen_kv, Output):
        _<attn_name>_no_split(Q, K, V, real_seqlen_kv, Output)

    @T.prim_func
    def <attn_name>_decode_split(Q, K, V, real_seqlen_kv, glse, Output_partial, split_length, Output):
        _<attn_name>_split(Q, K, V, real_seqlen_kv, glse, Output_partial, split_length)
        combine(glse, Output_partial, Output)

    if num_split > 1:
        return <attn_name>_decode_split
    else:
        return <attn_name>_decode_no_split
```

The wrapper allocates `glse` and `Output_partial` buffers and computes `split_length` before dispatching:

```python
@torch.library.custom_op("top::<attn_name>_decode_wrapped_kernel", mutates_args=())
def _<attn_name>_decode_wrapped_kernel(
    ..., num_split: int,
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    glse: torch.Tensor, Output_partial: torch.Tensor,
) -> torch.Tensor:
    split_length = ...  # compute per-split chunk sizes
    if split_length[0] == 0:
        num_split = 1
    if num_split == 1:
        return _<attn_name>_decode_kernel(...)(block_M, block_N, 1, num_stages, threads)(Q, K, V, real_seqlen_kv)
    return _<attn_name>_decode_kernel(...)(block_M, block_N, num_split, num_stages, threads)(
        Q, K, V, real_seqlen_kv, glse, Output_partial, split_length)
```

The `Kernel.forward` allocates `glse` and `Output_partial` as temporary buffers before calling the wrapper:

```python
def forward(self, Q, K, V, real_seqlen_kv):
    glse = torch.empty((..., self.config["num_split"], ...), dtype=..., device=Q.device)
    Output_partial = torch.empty(
        (..., self.config["num_split"], ...), dtype=..., device=Q.device
    )
    return (
        _
        < attn_name
        > _decode_wrapped_kernel(
            ..., self.config["num_split"], Q, K, V, glse, Output_partial
        )
    )
```

______________________________________________________________________

## Decode kernel: paged KV variant

For paged KV cache support, create a separate file `<attn_name>_decode_paged.py`. The paged variant:

- Adds `page_size: int` to the outermost kernel function
- Replaces the flat KV shape `[batch, seqlen_kv, heads, dim]` with a paged pool shape `[total_pages * page_size, heads, dim]`
- Adds a `block_table: T.Tensor([batch, max_pages], T.int32)` parameter to index into the page pool
- `real_seqlen_kv` becomes a per-batch tensor `T.Tensor([batch], T.int32)` instead of a scalar

Reference: `tileops/kernels/flash_decode/mha_decode_paged.py`

______________________________________________________________________

## Checklist

- [ ] File/class name includes variant suffix: `_fwd` / `_decode` / `_bwd`
- [ ] `is_causal: bool` is a parameter of the outermost kernel function
- [ ] (Decode) Both no-split and split-K `@T.prim_func` are implemented; `num_split` is in `default_config` and `autotune_configs`
- [ ] (Decode) Wrapper computes `split_length` and dispatches to the correct path
- [ ] (Decode) `Kernel.forward` allocates `glse` and `Output_partial` before calling the wrapper
- [ ] (Decode paged) Separate `_decode_paged.py` file; uses `page_size`, `block_table`, and per-batch `real_seqlen_kv`
