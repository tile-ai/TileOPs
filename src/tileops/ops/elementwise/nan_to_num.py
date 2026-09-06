"""NanToNum op: replace NaN, +Inf, -Inf with specified values."""

from math import prod
from typing import Dict, Optional

import torch

from tileops.backend import Target
from tileops.kernels.elementwise import NanToNumFwdKernel
from tileops.kernels.kernel_base import Kernel

from ..op_base import Op
from ._base import _PerDtypeKernels, _validate_scalar_param_repr


class NanToNumFwdOp(_PerDtypeKernels, Op):
    """NanToNum: replace NaN, +Inf, -Inf with specified values."""

    _op_name = "nan_to_num"
    _wrapped = None

    def __init__(
        self,
        *,
        nan: float = 0.0,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        target: Target = None,
        kernel_map: Optional[Dict[str, Kernel]] = None,
        tune: bool = False,
    ):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            nan: Replacement for NaN (default 0.0).
            posinf: Replacement for +Inf. Manifest default ``None`` resolves
                to the largest finite value representable in the element type of the
                call (matches ``torch.nan_to_num``). An explicit value outside
                that element type's finite range is rejected rather than stored
                as Inf.
            neginf: Replacement for -Inf. Manifest default ``None`` resolves
                to the smallest (most negative) finite value representable
                in the element type of the call.
            target: Which set of kernels serves this op — a target name, ``BUILTIN`` for
                the in-tree kernels, or ``None`` to decide from the input device.
            kernel_map: Optional kernel dispatch override.
            tune: Whether to autotune the kernel.
        """
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf
        self.target = target
        self.tune = tune
        # Manifest input binding for the synthesized eval_roofline
        # (docs/design/roofline.md §4.4.3); bound by the first forward.
        self.input_shape: Optional[tuple] = None
        self.dispatch_kernel(kernel_map)

    def _build(self, dtype: torch.dtype, n_total: int):
        """Resolve the replacement values against *dtype*, then build.

        A ``None`` bound means "this dtype's largest finite value", so it
        cannot be resolved before the element type is known. Picking
        ``finfo(dtype).max`` matches ``torch.nan_to_num``; forwarding ``+inf``
        would write back the infinity the op was called to replace.
        """
        _validate_scalar_param_repr("nan", self.nan, dtype, self._op_name)
        if self.posinf is None:
            posinf = torch.finfo(dtype).max
        else:
            _validate_scalar_param_repr("posinf", self.posinf, dtype, self._op_name)
            posinf = self.posinf
        if self.neginf is None:
            neginf = torch.finfo(dtype).min
        else:
            _validate_scalar_param_repr("neginf", self.neginf, dtype, self._op_name)
            neginf = self.neginf
        # Replacement values are positional; the kernel constructor's
        # parameter naming is encapsulated below the Op layer.
        impl, ctor_dtype = self._selected_kernel_cls().specialize(dtype)
        return impl(n_total, ctor_dtype, self.nan, posinf, neginf, tune=self.tune)

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"nan_to_num": NanToNumFwdKernel}

    def _infer_output_shapes(self, input_shape: tuple) -> Dict[str, tuple]:
        """Manifest ``shape_rules``: ``output.shape == input.shape``."""
        return {"output": tuple(input_shape)}

    @property
    def N_total(self) -> int:
        """Element count of the most recent forward."""
        if self.input_shape is None:
            raise RuntimeError(
                "NanToNumFwdOp needs a prior forward() call: the element count arrives "
                "with the tensor"
            )
        return prod(self.input_shape)

    def _eager_forward(self, input: torch.Tensor) -> torch.Tensor:
        self._validate_dtypes(input)
        input = input.contiguous()
        result = self._kernel((input,), input.dtype, input.numel())(input)
        self._note_call(input.dtype, input_shape=tuple(input.shape))
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the op on the inputs the manifest declares.

        Args:
            input: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``output``, as the manifest declares. Shape rules: ``output.shape == input.shape``.
        """
        return type(self)._wrapped(input, self._instance_key)
