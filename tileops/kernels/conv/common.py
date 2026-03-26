import torch

__all__ = ["HOPPER_SHARED_MEMORY_LIMIT_BYTES", "conv_shared_memory_bytes"]

HOPPER_SHARED_MEMORY_LIMIT_BYTES = 227 * 1024


def conv_shared_memory_bytes(
    block_m: int,
    block_n: int,
    block_k: int,
    num_stages: int,
    dtype: torch.dtype,
) -> int:
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    per_stage_bytes = (block_m * block_k + block_k * block_n) * dtype_bytes
    return per_stage_bytes * max(1, num_stages)
