# Third-Party Notices

This repository includes or adapts code from the following MIT-licensed
projects. The top-level `LICENSE` contains the MIT license text used for this
repository distribution; the original copyright notices below are retained in
the relevant source files.

## Qwen FlashQLA

Files under `tileops/kernels/gated_deltanet/gdn_prefill/` implement a TileOps
version of the CP-split Gated DeltaNet prefill schedule. The schedule-level
reference for the h-state / corrected-segment-start part of this implementation
comes from Qwen FlashQLA:

- `__init__.py`
- `cp_fwd.py`
- `fused_fwd.py`
- `prepare_h.py`
- `tilelang_compat.py`

Original notice:

```text
Copyright (c) 2026 The Qwen team, Alibaba Group.
Licensed under the MIT License.
```

The TileOps versions are not direct wrappers around the FlashQLA kernels. They
adapt the CP-split scheduling idea into the TileOps operator API, BTHD dispatch,
TileLang compatibility layer, benchmarking, tests, and local replay/output
implementation.

## Flash Linear Attention

`tileops/kernels/gated_deltanet/gdn_prefill/utils.py` adapts utility helpers
from Flash Linear Attention.

Original notice:

```text
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
Licensed under the MIT License.
```

The TileOps version is modified for the local Gated DeltaNet prefill integration.
