from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch

from top.kernels.kernel import Kernel
from top.utils import get_sm_version


class Op(ABC):
    """Base class for TileOPs operations.

    A Op represents a computational operation with:
    - Hardware-aware kernel dispatch
    - Correctness testing via reference implementation
    - Performance profiling
    - Autotuning interface

    Examples:
        >>> from top.ops import MultiHeadAttentionFwdOp
        >>> MultiHeadAttentionFwdOp is a subclass of Op
        >>> op = MultiHeadAttentionFwdOp(batch=1, heads=8, seq_len=512, dim=64, is_causal=True)
        >>> Q, K, V = op.gen_inputs()
        >>> output = op(Q, K, V)
        >>> op.check()  # Verify correctness
        >>> latency = op.profile()  # Benchmark performance

    Attributes:
        kernel: top.Kernel instance (e.g. mha_fwd_kernel)
        dtype: Data type for computation (e.g., torch.float16)
        device: Device for computation (e.g., 'cuda')
        input_shapes: Expected input tensor shapes

    Properties:
        total_flops (optional): Total flops for the op.
            If specified, will be used to calculate TFlops in profile().
        total_memory (optional): Total memory for the op.
            If specified, will be used to calculate Bandwidth in profile().
    """

    kernel: Kernel
    kernel_map: Optional[Dict[str, Kernel]] = {}
    dtype: Optional[torch.dtype] = None
    device: Optional[Union[torch.device, str]] = 'cuda'
    input_shapes: Optional[list[tuple]] = None

    @property
    @abstractmethod
    def default_kernel_map(self) -> Dict[str, Kernel]:
        raise NotImplementedError("Op must implement default_kernel_map")

    def dispatch_kernel(self, kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        assert self.default_kernel_map is not None and len(self.default_kernel_map) > 0
        for name, default_kernel in self.default_kernel_map.items():
            if kernel_map is not None and name in kernel_map:
                kernel_type = kernel_map[name]
            else:
                kernel_type = default_kernel
            current_arch = get_sm_version()
            if kernel_type is not None:
                assert current_arch in kernel_type.supported_archs, \
                    f'{kernel_type.__name__} is not supported on architecture {current_arch}'
            self.kernel_map[name] = kernel_type

    def autotune(self) -> None:
        """Autotune all kernels of the op"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Kernel):
                attr.autotune()

    @abstractmethod
    def forward(
            self,
            *args: object,  # noqa: U100
            **kwargs: object) -> Union[torch.Tensor, Tuple]:  # noqa: U100
        raise NotImplementedError("forward method is not implemented")

    def __call__(self, *args: object, **kwargs: object) -> Union[torch.Tensor, Tuple]:
        """Make the op callable - delegates to forward()"""
        return self.forward(*args, **kwargs)
