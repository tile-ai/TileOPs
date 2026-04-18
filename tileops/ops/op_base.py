import warnings
from abc import ABC, abstractmethod
from typing import Dict, FrozenSet, Hashable, Optional, Tuple, Union

import torch

from tileops.kernels.kernel_base import Kernel
from tileops.utils import get_sm_version

# Module-level dedup for empty-static_dims warnings; keyed by Op subclass.
_EMPTY_STATIC_DIMS_WARNED: set = set()


class Op(ABC):
    """Base class for TileOPs operations.

    A Op represents a computational operation with:
    - Hardware-aware kernel dispatch
    - Correctness testing via reference implementation
    - Performance profiling
    - Autotuning interface

    Examples:
        >>> from tileops.ops import MultiHeadAttentionFwdOp
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
    kernel_map: Optional[Dict[str, Kernel]] = None
    dtype: Optional[torch.dtype] = None
    device: Optional[Union[torch.device, str]] = 'cuda'
    input_shapes: Optional[list[tuple]] = None

    # Set of (input_index, axis) pairs identifying static (ctor-committed) axes.
    # `input_index` is the position in *input_shapes; `axis` is a non-negative
    # axis index within that shape. Subclasses set this to reflect their
    # manifest `static_dims`. Default empty = no committed axes.
    _static_axes: FrozenSet[Tuple[int, int]] = frozenset()

    @property
    @abstractmethod
    def default_kernel_map(self) -> Dict[str, Kernel]:
        raise NotImplementedError("Op must implement default_kernel_map")

    def dispatch_kernel(self, kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        if self.default_kernel_map is None or len(self.default_kernel_map) == 0:
            raise ValueError("default_kernel_map must be non-empty")
        self.kernel_map = {}
        for name, default_kernel in self.default_kernel_map.items():
            if kernel_map is not None and name in kernel_map:
                kernel_type = kernel_map[name]
            else:
                kernel_type = default_kernel
            current_arch = get_sm_version()
            if kernel_type is not None and current_arch not in kernel_type.supported_archs:
                raise ValueError(
                    f'{kernel_type.__name__} is not supported on architecture {current_arch}')
            self.kernel_map[name] = kernel_type

    def autotune(self) -> None:
        """Autotune all kernels of the op"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Kernel):
                attr.autotune()

    @abstractmethod
    def forward(self, *args: object, **kwargs: object) -> Union[torch.Tensor, Tuple]:
        raise NotImplementedError("forward method is not implemented")

    def __call__(self, *args: object, **kwargs: object) -> Union[torch.Tensor, Tuple]:
        """Make the op callable - delegates to forward()"""
        return self.forward(*args, **kwargs)

    def _cache_key(self, *input_shapes: Tuple[int, ...]) -> Hashable:
        """Return a cache key for kernel dispatch given forward-time input shapes.

        Default implementation returns the tuple of non-static-axis sizes across
        all input shapes, using ``self._static_axes`` to decide which axes are
        committed at ctor. This is always correct for any Op, but may
        over-fragment the kernel cache when ``_static_axes`` is empty (one
        compile per distinct input shape).

        Override in subclasses to project the shape onto whatever the kernel
        actually depends on — for example, flattening leading dims to a single
        product when the kernel treats input as 2D.

        When ``_static_axes`` is empty AND the subclass does not override
        ``_cache_key``, a ``UserWarning`` is emitted once per subclass type to
        surface the missing override.
        """
        if not self._static_axes and type(self)._cache_key is Op._cache_key:
            cls = type(self)
            if cls not in _EMPTY_STATIC_DIMS_WARNED:
                _EMPTY_STATIC_DIMS_WARNED.add(cls)
                warnings.warn(
                    f"{cls.__name__}: Op._cache_key() called with empty "
                    f"_static_axes and no subclass override. The default "
                    f"keys the kernel cache by the full input shape, which "
                    f"produces one compile per distinct shape under dynamic "
                    f"inputs. Override _cache_key to project onto whatever "
                    f"the kernel math actually depends on.",
                    UserWarning,
                    stacklevel=2,
                )
        return tuple(
            s
            for i, shape in enumerate(input_shapes)
            for axis, s in enumerate(shape)
            if (i, axis) not in self._static_axes
        )
