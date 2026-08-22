"""PReLU op: y = x if x > 0 else weight[channel] * x."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import PreluFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..compile_boundary import get_instance
from ..op_base import Op
from ._base import (
    _PerDtypeKernels,
    _require_one_device,
    _require_shape_inference,
    resolve_output_dtype,
)


class PreluFwdOp(_PerDtypeKernels, Op):
    """PReLU: y = x if x > 0 else weight[channel] * x.

    Channel dimension follows PyTorch convention: dimension 1 for inputs
    with ndim >= 2, dimension 0 for 1-D inputs. Both the shape and the channel
    count arrive with the tensors.

    Args:
        target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
            the in-tree kernels, or ``None`` to decide from the input device.
        kernel_map: Optional dispatch override mapping kernel keys to
            ``Kernel`` subclasses. Falls back to ``default_kernel_map``.
        tune: Whether to autotune.
    """

    _op_name = "prelu"
    _wrapped = None

    def __init__(
        self,
        *,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        self.target = target
        self.tune = tune
        # The synthesized eval_roofline resolves each signature.inputs entry as
        # self.<name>_shape (docs/design/roofline.md §4.4.3); the first forward binds them.
        self.input_shape: Optional[tuple] = None
        self.weight_shape: Optional[tuple] = None
        self.dispatch_kernel(kernel_map)

    @staticmethod
    def _inner_size(shape: tuple) -> int:
        """Elements per channel per row: PyTorch puts the channel at dim 1."""
        return (prod(shape[2:]) if len(shape) > 2 else 1) if len(shape) >= 2 else 1

    def _build(self, dtype: torch.dtype, n_total: int, num_channels: int, inner_size: int):
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return impl(n_total, num_channels, inner_size, ctor_dtype)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"prelu": PreluFwdKernel}

    def _infer_output_shapes(self, input_shape: tuple, weight_shape: tuple) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == input.shape``."""
        return {"output": tuple(input_shape)}

    @property
    def num_channels(self) -> int:
        """Weight length of the most recent forward."""
        if self.weight_shape is None:
            raise RuntimeError(
                "PreluFwdOp needs a prior forward() call: the channel count arrives with the weight"
            )
        return prod(self.weight_shape)

    @property
    def N_total(self) -> int:
        """Element count of the most recent forward."""
        if self.input_shape is None:
            raise RuntimeError(
                "PreluFwdOp needs a prior forward() call: the element count arrives with the tensor"
            )
        return prod(self.input_shape)

    def _eager_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        _require_one_device("PreluFwdOp", input=input, weight=weight)
        self._validate_dtypes(input, weight)
        # ``weight`` is part of the manifest contract: one entry per channel of the
        # axis PyTorch designates, so its length is what names the channel count.
        if weight.dtype != input.dtype:
            raise ValueError(f"Expected weight.dtype {input.dtype}, got {weight.dtype}")
        # Mirrors the manifest shape rule: a 0-dim or length-1 weight is the shared
        # slope, and a longer one has to match the channel axis PyTorch designates.
        if weight.ndim > 1:
            raise ValueError(f"Expected weight to be 0-D or 1-D, got {weight.ndim}D")
        shared = weight.ndim == 0 or weight.shape[0] == 1
        if not shared and (input.ndim < 2 or weight.shape[0] != input.shape[1]):
            raise ValueError(
                f"Expected weight of length 1 or input.shape[1], got "
                f"{weight.shape[0]} for input shape {tuple(input.shape)}"
            )
        input = input.contiguous()
        weight = weight.contiguous()
        shapes = dict(input_shape=tuple(input.shape), weight_shape=tuple(weight.shape))
        kernel = self._kernel(
            (input, weight),
            input.dtype,
            prod(shapes["input_shape"]),
            prod(shapes["weight_shape"]),
            self._inner_size(shapes["input_shape"]),
        )
        result = kernel(input, weight)
        self._note_call(input.dtype, **shapes)
        return result

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return type(self)._wrapped(input, weight, self._instance_key)


# The compile boundary: one operator for this op, registered at import time. The op's
# key crosses it, and the body trades the key back for the instance — see
# src/tileops/ops/compile_boundary.py.

_require_shape_inference(PreluFwdOp)


@torch.library.custom_op("top::elementwise_prelu", mutates_args=())
def _prelu_fwd(x: torch.Tensor, weight: torch.Tensor, instance_key: str) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(x, weight)


@_prelu_fwd.register_fake
def _prelu_fwd_fake(x: torch.Tensor, weight: torch.Tensor, instance_key: str) -> torch.Tensor:
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(x.shape), tuple(weight.shape))
    return x.new_empty(shapes["output"], dtype=resolve_output_dtype(PreluFwdOp.__name__, x.dtype))


PreluFwdOp._wrapped = _prelu_fwd
PreluFwdOp.compile_op_names = ("top::elementwise_prelu",)
