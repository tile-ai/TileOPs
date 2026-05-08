"""Tensor-weight lerp op: out = input + weight * (end - input)."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.kernels.elementwise import LerpTensorFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _OP_REGISTRY


class LerpTensorFwdOp(Op):
    """Tensor-weight lerp: out = input + weight * (end - input).

    Conforms to the Tensor-weight overload of ``torch.lerp`` —
    ``torch.lerp(input, end, weight: Tensor)`` where ``weight`` is a
    Tensor that broadcasts together with ``input`` and ``end`` to the
    output shape. The Op layer expands the three inputs to the broadcast
    shape and dispatches the flat ``LerpTensorFwdKernel`` on
    ``N_total = product(broadcast_shape)`` elements. The scalar-weight
    overload is handled separately by ``LerpFwdOp``.

    Args:
        input: Shape of the start tensor.
        end: Shape of the end tensor.
        weight: Shape of the per-element weight tensor.
        dtype: Torch dtype for all three operands.
    """

    _op_name = "lerp_tensor"
    _wrapped = None

    # Manifest declares all three operands as ``float16 | bfloat16 | float32``;
    # fp8 dtypes are rejected at the op-layer signature so the impl matches
    # the manifest contract (the kernel also rejects fp8 independently).
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        *,
        input: tuple,  # noqa: A002 — manifest-aligned PyTorch param name
        end: tuple,
        weight: tuple,
        dtype: torch.dtype,
        strategy: Optional[str] = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(
                f"LerpTensorFwdOp does not support dtype {dtype}. "
                f"Supported: [{names}]"
            )
        self.input_shape = tuple(input)
        self.end_shape = tuple(end)
        self.weight_shape = tuple(weight)
        self.dtype = dtype
        self.strategy = strategy
        self.out_shape = tuple(
            torch.broadcast_shapes(
                self.input_shape, self.end_shape, self.weight_shape,
            )
        )
        self.N_total = prod(self.out_shape) if self.out_shape else 1
        self.dispatch_kernel(kernel_map)
        self.kernel = self.kernel_map[self._op_name](
            self.N_total, dtype, tune=tune,
        )
        self._instance_key = id(self)
        _OP_REGISTRY[self._instance_key] = self

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"lerp_tensor": LerpTensorFwdKernel}

    @staticmethod
    def _expand_flat(t: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Expand ``t`` to ``target_shape`` and return a contiguous flat view."""
        if tuple(t.shape) != tuple(target_shape):
            t = t.expand(target_shape)
        return t.contiguous().view(-1)

    def _eager_forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        end: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        out_shape = self.out_shape if self.out_shape else (1,)
        a_flat = self._expand_flat(input, out_shape)
        b_flat = self._expand_flat(end, out_shape)
        w_flat = self._expand_flat(weight, out_shape)
        result = self.kernel(a_flat, b_flat, w_flat)
        return result.view(self.out_shape if self.out_shape else ())

    def forward(
        self,
        input: torch.Tensor,  # noqa: A002 — manifest-aligned PyTorch param name
        end: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        if not (input.is_cuda and end.is_cuda and weight.is_cuda):
            raise ValueError("Inputs must be CUDA tensors")
        for name, t, expected in [
            ("input", input, self.input_shape),
            ("end", end, self.end_shape),
            ("weight", weight, self.weight_shape),
        ]:
            if t.dtype != self.dtype:
                raise ValueError(
                    f"Expected {name}.dtype {self.dtype}, got {t.dtype}"
                )
            if tuple(t.shape) != expected:
                raise ValueError(
                    f"Expected {name}.shape {expected}, got {tuple(t.shape)}"
                )
        wrapped = type(self)._wrapped
        if wrapped is not None:
            return wrapped(input, end, weight, self._instance_key)
        return self._eager_forward(input, end, weight)
