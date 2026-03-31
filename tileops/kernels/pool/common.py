from typing import Tuple


def normalize_1d(value: int | Tuple[int]) -> int:
    if isinstance(value, tuple):
        return value[0]
    return value


def normalize_2d(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def normalize_3d(value: int | Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value, value)


def pool_output_dim(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    ceil_mode: bool,
) -> int:
    if ceil_mode:
        out = (input_size + 2 * padding - kernel_size + stride - 1) // stride + 1
    else:
        out = (input_size + 2 * padding - kernel_size) // stride + 1

    if ceil_mode and out > 0 and (out - 1) * stride >= input_size + padding:
        out -= 1

    return max(out, 0)
