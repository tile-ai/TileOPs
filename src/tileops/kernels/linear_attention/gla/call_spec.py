"""Facts used to select an inference GLA prefill implementation."""

import dataclasses
from typing import Optional

import torch

from ...call_spec import CallSpec

__all__ = ["GLAPrefillCall"]


@dataclasses.dataclass(frozen=True)
class GLAPrefillCall(CallSpec):
    """One zero-state GLA prefill call after the Op reads its inputs."""

    batch: int = 0
    seq_len: int = 0
    heads: int = 0
    dim_k: int = 0
    dim_v: int = 0
    chunk_size: int = 0
    dtype: Optional[torch.dtype] = None
    tune: bool = False
