from typing import Callable, List
import torch
from tilelang.profiler import do_bench
from functools import partial
from dataclasses import dataclass

# A mapping from string dtype names to torch dtypes
str2dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}

# A mapping from torch dtypes to string names
dtype2str = {v: k for k, v in str2dtype.items()}


@dataclass
class Performance:
    time: float # ms
    tflops: float
    io_bandwidth: float
    baseline_time: List[float]
    baseline_tflops: List[float] # TFLOPS
    baseline_io_bandwidth: List[float] # TB/s



@torch.compile
def reduce_on_dim0(x: torch.Tensor) -> torch.Tensor:
    """Reduce a tensor on dimension 0.
    Arguments:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Reduced tensor."""
    return x[0] if x.size(0) == 1 else x.sum(dim=0)


@torch.compile
def zero_pad(x: torch.Tensor, pad_size: int, dim: int) -> torch.Tensor:
    """Pad a tensor with 0 to a be divisible by `pad_size` along a specified dimension.
    Arguments:
        x (torch.Tensor): Input tensor.
        pad_size (int): The size to pad to be divisible by.
        dim (int): The dimension to pad.
    Returns:
        torch.Tensor: Padded tensor."""
    if x.size(dim) % pad_size == 0:
        return x
    pad_len = (pad_size - x.size(dim) % pad_size)
    assert 0 < pad_len < pad_size

    zero_shape = list(x.shape)
    zero_shape[dim] = pad_len
    zero_shape = tuple(zero_shape)
    zeros = torch.zeros(zero_shape, dtype=x.dtype, device=x.device)
    return torch.cat((x, zeros), dim=dim)


def ensure_contiguous(func: callable) -> callable:
    """Decorator to ensure that all tensor arguments are contiguous before calling the function.
    Arguments:
        func (callable): The function to decorate.
    Returns:
        callable: The decorated function."""

    def wrapper(*args, **kwargs):
        args = [arg.contiguous() if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {
            k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapper

def partity(module, *args, **kwargs):
    output = module.forward(*args, **kwargs)

    ref_program = module.ref_program

    ref_output = ref_program(*args, **kwargs)

    torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=1e-2)


def performance(module, baseline_list: List[Callable], *args, **kwargs):
    tilelang_time = do_bench(partial(module.forward, *args, **kwargs))

    baseline_time_list = []
    for baseline_func in baseline_list:
        baseline_time_list.append(do_bench(partial(baseline_func, *args, **kwargs)))
    

    flops = module.get_flops(*args, **kwargs)

    memory_footprint = module.get_memory_footprint(*args, **kwargs)


    tilelang_tflops = flops / tilelang_time * 1e-9

    tilelang_io_bandwidth = memory_footprint / tilelang_time * 1e-9 # TB/s


    baseline_tflops_list = []
    baseline_io_bandwidth_list = []
    for baseline_time in baseline_time_list:
        baseline_tflops = flops / baseline_time * 1e-9
        baseline_io_bandwidth = memory_footprint / baseline_time * 1e-9
        baseline_tflops_list.append(baseline_tflops)
        baseline_io_bandwidth_list.append(baseline_io_bandwidth)

    return Performance(tilelang_time, tilelang_tflops, tilelang_io_bandwidth, baseline_time_list, baseline_tflops_list, baseline_io_bandwidth_list)