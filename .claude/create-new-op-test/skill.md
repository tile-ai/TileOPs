# Skill: Writing Op Correctness Unit Tests

Reference tests: `tests/ops/`

---

## Overview

Each op has a pytest file at `tests/ops/test_<op_name>.py`. The test:

1. Instantiates the `Op` with given parameters
2. Generates random input tensors
3. Runs the op and a PyTorch reference implementation
4. Asserts numerical closeness with `torch.allclose`

No benchmark infrastructure is needed — correctness only.

---

## File Location and Naming

```
tests/ops/test_<op_name>.py
```

File name mirrors the op module name (e.g. `tileops/ops/gqa_decode.py` → `tests/ops/test_gqa_decode.py`).

---

## Test File Structure

```python
import pytest
import torch

from tileops.ops import <OpName>Op


def ref_<op_name>(*inputs, **params) -> torch.Tensor:
    # Pure PyTorch reference implementation
    # Use torch.nn.functional or F.scaled_dot_product_attention
    ...


@pytest.mark.parametrize("<param_names>", [
    (...),  # case 1
    (...),  # case 2
    # ... at least 5 cases
])
def test_<op_name>(param1: int, ..., dtype: torch.dtype, tune: bool) -> None:
    torch.manual_seed(42)

    op = <OpName>Op(param1, ..., dtype=dtype, tune=tune)

    # generate inputs
    x = torch.randn(..., device='cuda', dtype=dtype)
    ...

    # run op
    with torch.no_grad():
        out = op(x, ...)

    # run reference
    ref = ref_<op_name>(x, ..., param1=param1, ...)

    assert torch.allclose(out, ref, atol=<atol>, rtol=<rtol>), \
        f"max err: {(out - ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
```

---

## Test Case Requirements

### Minimum coverage

- At least **5 parametrized test cases** per test function
- Vary structural parameters across cases: batch, seq_len, heads, dim, etc.
- Include at least one `tune=True` case if the op supports autotuning

### dtype coverage

| Op type | Required dtypes |
|---|---|
| General ops | `torch.float16`, `torch.bfloat16` |
| Quantization ops (fp8, fp4, etc.) | Include the target quantized dtype (e.g. `torch.float8_e4m3fn`) |
| Mixed-precision ops | Cover all relevant input/output dtype combinations |

### Random seed

Fix `torch.manual_seed(42)` at the top of every test function, before any tensor creation.

### Tolerance guidelines

| Op type | `atol` | `rtol` |
|---|---|---|
| Attention fwd (fp16/bf16) | `5e-3` | `1e-5` |
| Attention decode | `1e-2` | `1e-2` |
| GEMM / linear | `1e-3` | `1e-3` |
| Elementwise / quantization | `1e-1` or custom | — |

### Reference implementation

- If `torch.nn` or `torch.nn.functional` provides the operation directly, use it.
- If not, implement the reference manually using basic PyTorch ops (`torch.matmul`, `torch.softmax`, elementwise ops, etc.), following the algorithm described in the kernel's docstring and the official reference URL.
- For attention fwd: use `SDPBackend.FLASH_ATTENTION`
- For attention decode: use `SDPBackend.MATH` (flash attention does not support single-token decode)
- Never use another TileOps op as the reference
- If the op has an official PyTorch or Triton reference implementation (e.g. in the kernel's docstring `Reference:` URL), consult it when writing `ref_program` to ensure the reference matches the intended algorithm exactly — including scale factors, layout conventions, and masking logic.

### Fallback: cosine similarity test

If numerical error persists after exhausting all debugging steps (see `.claude/create-new-op/skill.md` debugging protocol), the `torch.allclose` assertion may be replaced with a cosine similarity check. The threshold is **0.999 and must not be relaxed**:

```python
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()

# in the test:
sim = cosine_sim(out, ref)
assert sim >= 0.999, f"cosine similarity {sim:.6f} < 0.999"
```

Add a comment explaining why `torch.allclose` was replaced and what debugging was attempted.

---

## Example: attention fwd op

```python
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tileops.ops import MultiHeadAttentionFwdOp


def ref_mha_fwd(q, k, v, is_causal):
    # input layout: [batch, seqlen, heads, dim] → transpose to [batch, heads, seqlen, dim]
    q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=is_causal)
    return out.transpose(1, 2).contiguous()


@pytest.mark.parametrize("batch, seq_len, heads, dim, is_causal, dtype, tune", [
    (1,  1024,  8,  64, False, torch.float16,  False),
    (4,  2048, 16, 128, False, torch.bfloat16, False),
    (8,  4096, 16, 128, True,  torch.float16,  False),
    (2,  1024, 32,  64, True,  torch.bfloat16, False),
    (4,  2048, 16, 128, False, torch.bfloat16, True),
])
def test_mha_fwd(batch, seq_len, heads, dim, is_causal, dtype, tune):
    torch.manual_seed(42)
    op = MultiHeadAttentionFwdOp(batch, heads, seq_len, dim, is_causal, dtype, tune=tune)

    q = torch.randn(batch, seq_len, heads, dim, device='cuda', dtype=dtype)
    k = torch.randn(batch, seq_len, heads, dim, device='cuda', dtype=dtype)
    v = torch.randn(batch, seq_len, heads, dim, device='cuda', dtype=dtype)

    with torch.no_grad():
        out = op(q, k, v)
    ref = ref_mha_fwd(q, k, v, is_causal)

    assert torch.allclose(out, ref, atol=5e-3, rtol=1e-5), \
        f"max err: {(out - ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
```

---

## Checklist

- [ ] File placed at `tests/ops/test_<op_name>.py`
- [ ] `torch.manual_seed(42)` at the top of every test function
- [ ] At least 5 parametrized test cases
- [ ] Both `torch.float16` and `torch.bfloat16` covered (unless op is dtype-specific)
- [ ] Quantization ops include the target quantized dtype
- [ ] Structural parameters varied across cases
- [ ] At least one `tune=True` case if op supports autotuning
- [ ] `atol` / `rtol` appropriate for the op type
- [ ] Reference uses PyTorch built-ins only
- [ ] Test file ends with `if __name__ == "__main__": pytest.main([__file__, "-vvs"])`
