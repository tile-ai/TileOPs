import torch

__all__ = ["conv_shared_memory_bytes", "get_shared_memory_limit_bytes"]


def get_shared_memory_limit_bytes() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).shared_memory_per_block_optin


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
