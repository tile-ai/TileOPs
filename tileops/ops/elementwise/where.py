"""Where op: out = condition ? input : other (with broadcasting)."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import WhereFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _OP_REGISTRY


class WhereFwdOp(Op):
    """Where: out = condition ? input : other (with full PyTorch broadcasting).

    Conforms to ``torch.where(condition, input, other)``: ``condition`` is a
    bool tensor and ``input`` / ``other`` may broadcast with each other and
    with ``condition`` to produce the output. The Op layer expands all
    three inputs to the broadcast shape and dispatches the existing flat
    where kernel on ``N_total = product(broadcast_shape)`` elements.

    Args:
        condition: Shape of the condition tensor (any shape broadcastable
            with ``input`` / ``other``).
        input: Shape of the value-when-true tensor.
        other: Shape of the value-when-false tensor.
        dtype: Torch dtype for ``input`` / ``other``.
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
    """

    _op_name = "where"
    _wrapped = None

    # Manifest declares ``input`` / ``other`` dtype as
    # ``float16 | bfloat16 | float32``. fp8 dtypes are not in the contract;
    # reject them at the op-layer signature so the impl matches the manifest.
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        condition: tuple,
        input: tuple,  # noqa: A002 — manifest-aligned PyTorch param name
        other: tuple,
        dtype: torch.dtype,
        *,
        kernel_map: Optional[Dict[str, Kernel]] = None,
    ):
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"WhereFwdOp does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )
        self.condition_shape = tuple(condition)
        self.input_shape = tuple(input)
        self.other_shape = tuple(other)
        self.dtype = dtype
        self.out_shape = tuple(
            torch.broadcast_shapes(self.condition_shape, self.input_shape, self.other_shape)
        )
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](self.N_total, dtype)
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self):
        return {"where": WhereFwdKernel}

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Expand ``t`` to ``target_shape`` and return a contiguous flat view."""
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def _eager_forward(
        self, condition: torch.Tensor, input: torch.Tensor, other: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        out_shape = self.out_shape if self.out_shape else (1,)
        cond_b = condition if condition.dtype == torch.bool else condition.bool()
        cond_flat = self._expand_flat(cond_b, out_shape).view(torch.uint8)
        x_flat = self._expand_flat(input, out_shape)
        y_flat = self._expand_flat(other, out_shape)
        result = self.kernel(cond_flat, x_flat, y_flat).view(out_shape if self.out_shape else ())
        return result

    def forward(
        self, condition: torch.Tensor, input: torch.Tensor, other: torch.Tensor,  # noqa: A002
    ) -> torch.Tensor:
        if not (condition.is_cuda and input.is_cuda and other.is_cuda):
            raise ValueError("Inputs must be CUDA tensors")
        if condition.dtype != torch.bool:
            raise ValueError(
                f"Expected condition.dtype torch.bool, got {condition.dtype}"
            )
        if input.dtype != self.dtype:
            raise ValueError(f"Expected input.dtype {self.dtype}, got {input.dtype}")
        if other.dtype != self.dtype:
            raise ValueError(f"Expected other.dtype {self.dtype}, got {other.dtype}")
        if tuple(condition.shape) != self.condition_shape:
            raise ValueError(
                f"Expected condition.shape {self.condition_shape}, got {tuple(condition.shape)}"
            )
        if tuple(input.shape) != self.input_shape:
            raise ValueError(
                f"Expected input.shape {self.input_shape}, got {tuple(input.shape)}"
            )
        if tuple(other.shape) != self.other_shape:
            raise ValueError(
                f"Expected other.shape {self.other_shape}, got {tuple(other.shape)}"
            )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(condition, input, other, self._instance_key)
        return self._eager_forward(condition, input, other)
