from typing import Dict, Optional, Tuple

import torch

from tileops.kernels.fp8_quant import FP8QuantKernel
from tileops.kernels.kernel_base import Kernel

from .op_base import Op

__all__ = ["FP8QuantFwdOp"]


class FP8QuantFwdOp(Op):
    """Quantize each row of an index tensor to ``float8_e4m3fn`` against its own maximum.

    ``scale_tensor`` is the row's absolute maximum, floored at ``1e-4``, over 448, to
    within one float32 ulp. ``output_tensor`` is the row multiplied by the reciprocal of
    that scale rather than divided by it, which for finite rows puts every element within
    one ``float8_e4m3fn`` code of ``clamp(input / scale, -448, 448)``: the two products
    differ by a few float32 ulp, too little to cross more than one rounding boundary of a
    format whose codes are an eighth apart.

    Neither output propagates a non-finite input. A row holding an infinity or a NaN is
    quantized against the maximum of its finite values, and an element that division would
    make NaN saturates instead. Callers needing NaN to survive must check for it.
    """

    def __init__(self, kernel_map: Optional[Dict[str, Kernel]] = None, tune: bool = False):
        """Build the op. Shapes and dtype are taken from the first call.

        Args:
            kernel_map: Optional kernel override dict.
            tune: Whether to autotune, applied when a kernel is first built.
        """
        self.batch = None
        self.seq_len_kv = None
        self.kv_group = None
        self.index_dim = None
        self.in_dtype = None
        self.tune = tune
        self.dispatch_kernel(kernel_map)
        self.kernel = None

    @property
    def default_kernel_map(self) -> Dict[str, Kernel]:
        return {"fp8_quant_kernel": FP8QuantKernel}

    def _get_kernel(
        self,
        inputs: "tuple[torch.Tensor | None, ...]",
        batch: int,
        seq_len_kv: int,
        kv_group: int,
        index_dim: int,
        in_dtype: torch.dtype,
        device_index: int | None,
    ) -> Kernel:
        key = (batch, seq_len_kv, kv_group, index_dim, in_dtype, device_index, self.tune)
        return self.get_or_build_kernel(
            "fp8_quant_kernel",
            inputs,
            key=key,
            build=lambda: self.kernel_map["fp8_quant_kernel"](
                batch, seq_len_kv, kv_group, index_dim, in_dtype, tune=self.tune
            ),
        )

    def _infer_output_shapes(
        self,
        input_tensor_shape: tuple[int, ...],
    ) -> dict[str, tuple[int, ...]]:
        """Manifest ``outputs``: one scale per row, and the quantized tensor itself."""
        return {
            "scale_tensor": tuple(input_tensor_shape[:-1]),
            "output_tensor": tuple(input_tensor_shape),
        }

    def forward(self, input_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the op on the inputs the manifest declares.

        Args:
            input_tensor: Input tensor, dtype ``float16 | bfloat16 | float32``.

        Returns:
            ``scale_tensor``, ``output_tensor``, as the manifest declares.
        """
        if not input_tensor.is_cuda:
            raise ValueError("FP8QuantFwdOp expects a CUDA input tensor")
        if input_tensor.ndim != 4:
            raise ValueError("FP8QuantFwdOp expects input_tensor shape [B, S, G, D]")
        if input_tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"FP8QuantFwdOp does not support dtype {input_tensor.dtype}")

        batch, seq_len_kv, kv_group, index_dim = input_tensor.shape
        if min(batch, seq_len_kv, kv_group, index_dim) <= 0:
            raise ValueError("FP8QuantFwdOp input dimensions must be positive")

        self.batch = batch
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.index_dim = index_dim
        self.in_dtype = input_tensor.dtype
        self.kernel = self._get_kernel(
            (input_tensor),
            batch,
            seq_len_kv,
            kv_group,
            index_dim,
            input_tensor.dtype,
            input_tensor.device.index,
        )
        return self.kernel(input_tensor)
