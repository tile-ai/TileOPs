import torch

from tileops.kernels.pool.common import validate_channels_last_input

__all__ = [
    "conv_shared_memory_bytes",
    "get_shared_memory_limit_bytes",
    "resolve_conv1d_padding",
    "validate_conv1d_input",
    "validate_conv1d_params",
    "validate_conv1d_tensors",
]


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


def validate_conv1d_params(
    *,
    stride: int,
    dilation: int,
    groups: int,
    c_in: int,
    c_out: int,
) -> None:
    if stride <= 0:
        raise ValueError("stride must be positive")
    if dilation <= 0:
        raise ValueError("dilation must be positive")
    if groups <= 0:
        raise ValueError("groups must be positive")
    if c_in % groups != 0:
        raise ValueError("c_in must be divisible by groups")
    if c_out % groups != 0:
        raise ValueError("c_out must be divisible by groups")


def resolve_conv1d_padding(
    padding: int | str,
    kernel_size: int,
    stride: int,
    dilation: int,
) -> tuple[int, int]:
    if isinstance(padding, str):
        if padding == "valid":
            return 0, 0
        if padding != "same":
            raise ValueError(f"Unsupported padding mode: {padding}")
        if stride != 1:
            raise ValueError("padding='same' is only supported for stride=1")
        total_padding = dilation * (kernel_size - 1)
        pad_left = total_padding // 2
        pad_right = total_padding - pad_left
        return pad_left, pad_right
    if padding < 0:
        raise ValueError("padding must be non-negative")
    return padding, padding


def validate_conv1d_input(
    *,
    op_name: str,
    input_shape: tuple[int, ...],
    expected_shape: tuple[int, ...],
) -> None:
    validate_channels_last_input(
        op_name=op_name,
        x_shape=input_shape,
        expected_shape=expected_shape,
        layout="NLC",
        ambiguous_layout_shape=(expected_shape[0], expected_shape[2], expected_shape[1]),
    )


def validate_conv1d_tensors(
    *,
    op_name: str,
    input_shape: tuple[int, ...],
    expected_input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    expected_weight_shape: tuple[int, ...],
    bias_shape: tuple[int, ...] | None,
    expected_bias_shape: tuple[int, ...],
) -> None:
    validate_conv1d_input(
        op_name=op_name,
        input_shape=input_shape,
        expected_shape=expected_input_shape,
    )
    if weight_shape != expected_weight_shape:
        raise ValueError(
            f"{op_name} expects weight shape {expected_weight_shape}, but got {weight_shape}"
        )
    if bias_shape is not None and bias_shape != expected_bias_shape:
        raise ValueError(
            f"{op_name} expects bias shape {expected_bias_shape}, but got {bias_shape}"
        )
