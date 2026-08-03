"""Abstract base class for operations.

An Op represents a computational operation with:
  - Kernel dispatch (lazy compilation cache)
  - Input validation
  - Roofline evaluation for performance metrics

Subclasses implement:
  - ``default_kernel_map``: maps kernel name -> Kernel class
  - ``forward()``: execution entry point
  - ``eval_roofline()``: returns (flops, bytes) for the bound instance
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, Optional, Union

import torch

from kernels.kernel_base import Kernel
from utils import device_str


class Op(ABC):
    """Base class for all operations."""

    kernel: Optional[Kernel] = None
    kernel_map: Optional[Dict[str, Kernel]] = None

    def __init__(self) -> None:
        self._kernel_cache: Dict[Hashable, Kernel] = {}

    @property
    @abstractmethod
    def default_kernel_map(self) -> Dict[str, Kernel]:
        raise NotImplementedError

    def dispatch_kernel(self, kernel_map: Optional[Dict[str, Kernel]] = None) -> None:
        """Resolve and install the kernel map."""
        default = self.default_kernel_map
        resolved: Dict[str, Kernel] = {}
        for name, default_kernel in default.items():
            if kernel_map is not None and name in kernel_map:
                resolved[name] = kernel_map[name]
            else:
                resolved[name] = default_kernel
        self.kernel_map = resolved

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Union[torch.Tensor, tuple]:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Union[torch.Tensor, tuple]:
        return self.forward(*args, **kwargs)

    def eval_roofline(self) -> tuple[int, int]:
        """Return (flops, bytes) for this op instance.

        Override in subclasses to call the appropriate roofline formula.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement eval_roofline()")

    def autotune(self) -> None:
        if self.kernel is not None:
            self.kernel.autotune()
