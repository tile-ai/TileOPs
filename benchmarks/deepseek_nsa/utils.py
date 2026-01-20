import torch
import tilelang
import functools
from typing import Any, Callable, Dict, Literal, Optional, Tuple

def tensor_cache(
    fn: Callable[..., torch.Tensor]
) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent result of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    If the function is called again with the same input tensors, it will return the cached result.


    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """
    last_args: Optional[Tuple] = None
    last_kwargs: Optional[Dict] = None
    last_result: Any = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal last_args, last_kwargs, last_result

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and \
                        all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper



@tensor_cache
def prepare_lens(offsets: torch.LongTensor) -> torch.LongTensor:
    return offsets[1:] - offsets[:-1]


@tensor_cache
def prepare_position_ids(offsets: torch.LongTensor) -> torch.LongTensor:
    return torch.cat([torch.arange(n) for n in prepare_lens(offsets).tolist()]).to(offsets.device)


@tensor_cache
def prepare_sequence_ids(position_ids: torch.LongTensor) -> torch.LongTensor:
    return position_ids.eq(0).cumsum(0) - 1


@tensor_cache
def prepare_token_indices(offsets: torch.LongTensor) -> torch.LongTensor:
    position_ids = prepare_position_ids(offsets)
    return torch.stack([prepare_sequence_ids(position_ids), position_ids], 1).to(offsets)


@tensor_cache
def prepare_chunk_offsets(
    offsets: torch.Tensor,
    chunk_size: int
) -> torch.LongTensor:
    return torch.cat([offsets.new_tensor([0]), tilelang.cdiv(prepare_lens(offsets), chunk_size)]).cumsum(-1)


@tensor_cache
def prepare_chunk_indices(
    offsets: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in tilelang.cdiv(prepare_lens(offsets), chunk_size).tolist()])
    return torch.stack([prepare_sequence_ids(indices), indices], 1).to(offsets)
