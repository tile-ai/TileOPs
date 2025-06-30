import torch
from torch.nn import functional as F

# A mapping from string dtype names to torch dtypes
str2dtype = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32
}

# A mapping from torch dtypes to string names
dtype2str = {v: k for k, v in str2dtype.items()}


def reduce_on_dim0(x: torch.Tensor) -> torch.Tensor:
    """Reduce a tensor on dimension 0.
    Arguments:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Reduced tensor."""
    return x[0] if x.size(0) == 1 else x.sum(dim=0)


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
        kwargs = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper