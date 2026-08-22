"""Where op: out = condition ? input : other (with broadcasting)."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import WhereFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..compile_boundary import get_instance
from ..op_base import Op
from ._base import (
    _PerDtypeKernels,
    _require_one_device,
    _require_shape_inference,
    broadcast_or_raise,
    resolve_output_dtype,
)


class WhereFwdOp(_PerDtypeKernels, Op):
    """Where: out = condition ? input : other (with full PyTorch broadcasting).

    Conforms to ``torch.where(condition, input, other)``: ``condition`` is a
    bool tensor and ``input`` / ``other`` may broadcast with each other and
    with ``condition`` to produce the output. The three operand shapes arrive with
    the tensors; broadcasting them is the kernel's business.

    Args:
        target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
            the in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
        tune: Whether to autotune.
    """

    _op_name = "where"
    _wrapped = None

    # Manifest declares ``input`` / ``other`` dtype as
    # ``float16 | bfloat16 | float32``. fp8 dtypes are not in the contract;
    # reject them at the op-layer signature so the impl matches the manifest.
    _SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.target = target
        self.tune = tune
        self.condition_shape: Optional[tuple] = None
        self.input_shape: Optional[tuple] = None
        self.other_shape: Optional[tuple] = None
        self.dispatch_kernel(kernel_map)

    def _build(self, dtype: torch.dtype, n_total: int):
        if dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(f"WhereFwdOp does not support dtype {dtype}. Supported: [{names}]")
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return impl(n_total, ctor_dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"where": WhereFwdKernel}

    def _infer_output_shapes(
        self,
        condition_shape: tuple,
        input_shape: tuple,
        other_shape: tuple,
    ) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == broadcast_shapes(...)``."""
        return {
            "output": broadcast_or_raise(
                "WhereFwdOp",
                condition=condition_shape,
                input=input_shape,
                other=other_shape,
            )
        }

    @property
    def out_shape(self) -> tuple:
        """Broadcast output shape of the most recent forward."""
        if self.input_shape is None:
            raise RuntimeError(
                "WhereFwdOp needs a prior forward() call: the operand shapes arrive "
                "with the tensors"
            )
        return self._infer_output_shapes(self.condition_shape, self.input_shape, self.other_shape)[
            "output"
        ]

    @property
    def N_total(self) -> int:
        """Output element count of the most recent forward."""
        return prod(self.out_shape)

    def _validate_dtypes(
        self,
        condition: torch.Tensor,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> None:
        if condition.dtype != torch.bool:
            raise ValueError(f"Expected condition.dtype torch.bool, got {condition.dtype}")
        if input.dtype not in self._SUPPORTED_DTYPES:
            names = ", ".join(str(dt) for dt in self._SUPPORTED_DTYPES)
            raise ValueError(f"Expected input.dtype in [{names}], got {input.dtype}")
        if other.dtype != input.dtype:
            raise ValueError(
                f"Expected other.dtype == input.dtype ({input.dtype}), got {other.dtype}"
            )

    def _eager_forward(
        self,
        condition: torch.Tensor,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        _require_one_device("WhereFwdOp", condition=condition, input=input, other=other)
        self._validate_dtypes(condition, input, other)
        shapes = dict(
            condition_shape=tuple(condition.shape),
            input_shape=tuple(input.shape),
            other_shape=tuple(other.shape),
        )
        n_total = prod(self._infer_output_shapes(*shapes.values())["output"])
        condition = condition.contiguous()
        input = input.contiguous()
        other = other.contiguous()
        result = self._kernel((condition, input, other), input.dtype, n_total)(
            condition, input, other
        )
        self._note_call(input.dtype, **shapes)
        return result

    def forward(
        self,
        condition: torch.Tensor,
        input: torch.Tensor,
        other: torch.Tensor,
    ) -> torch.Tensor:
        return type(self)._wrapped(condition, input, other, self._instance_key)


# The compile boundary: one operator for this op, registered at import time. The op's
# key crosses it, and the body trades the key back for the instance — see
# src/tileops/ops/compile_boundary.py.

_require_shape_inference(WhereFwdOp)


@torch.library.custom_op("top::elementwise_where", mutates_args=())
def _where_fwd(
    condition: torch.Tensor,
    input: torch.Tensor,
    other: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(condition, input, other)


@_where_fwd.register_fake
def _where_fwd_fake(
    condition: torch.Tensor,
    input: torch.Tensor,
    other: torch.Tensor,
    instance_key: str,
) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(condition.shape), tuple(input.shape), tuple(other.shape))
    return input.new_empty(
        shapes["output"], dtype=resolve_output_dtype(WhereFwdOp.__name__, input.dtype)
    )


WhereFwdOp._wrapped = _where_fwd
WhereFwdOp.compile_op_names = ("top::elementwise_where",)
