# -*- coding: utf-8 -*-
"""Detect whether FlashAttention-3 is installed/used, vs FlashAttention-2.

This snippet checks:
  1) Which Python entrypoint you're importing (FA3 uses `flash_attn_interface`).
  2) Whether the FA3 CUDA extension `flash_attn_3_cuda` is present & importable.
  3) Whether the FA2 CUDA extension `flash_attn_2_cuda` is present.
  4) Device capability (Hopper is SM >= 90), required by FA-3.

Style: Google Python Style.
"""
from __future__ import annotations

import importlib.util
import sys
import torch

report = {}

# 1) Import path check
fa3_entry = importlib.util.find_spec("flash_attn_interface") is not None
fa2_entry = importlib.util.find_spec("flash_attn") is not None
report["has_flash_attn_interface"] = fa3_entry
report["has_flash_attn_pkg"] = fa2_entry  # FA2-style package

# 2) C-extensions presence
fa3_ext = importlib.util.find_spec("flash_attn_3_cuda")
fa2_ext = importlib.util.find_spec("flash_attn_2_cuda")
report["flash_attn_3_cuda"] = fa3_ext.origin if fa3_ext else None
report["flash_attn_2_cuda"] = fa2_ext.origin if fa2_ext else None

# 3) Device capability
report["cuda_available"] = torch.cuda.is_available()
report["sm"] = torch.cuda.get_device_capability() if torch.cuda.is_available() else None

# 4) Try a smoke run via FA3 entrypoint if available
used_path = None
ok_smoke = None
err = None
try:
    if fa3_entry:
        import flash_attn_interface as fai  # FA3 entrypoint
        used_path = getattr(fai, "__file__", "unknown")
        # Tiny shape to ensure the call path works.
        q = torch.randn(1, 16, 1, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = fai.flash_attn_func(q, k, v, softmax_scale=None, causal=False)
        ok_smoke = tuple(out.shape)
    else:
        used_path = "flash_attn_interface NOT found"
except Exception as e:  # pragma: no cover
    err = f"{type(e).__name__}: {e}"

print("=== FA runtime check ===")
for k, v in report.items():
    print(f"{k:28s}: {v}")
print(f"used_entry_file              : {used_path}")
print(f"smoke_forward_out_shape      : {ok_smoke}")
print(f"errors                       : {err}")
