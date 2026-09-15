"""Facts used to select a GLA forward implementation."""

import dataclasses
from typing import Optional

import torch

from ...call_spec import CallSpec

__all__ = ["GLAFwdCall"]


@dataclasses.dataclass(frozen=True)
class GLAFwdCall(CallSpec):
    """One GLA forward call after the Op reads its inputs."""

    batch: int = 0
    seq_len: int = 0
    heads: int = 0
    dim_k: int = 0
    dim_v: int = 0
    chunk_size: int = 0
    dtype: Optional[torch.dtype] = None
    tune: bool = False
    has_initial_state: bool = False
