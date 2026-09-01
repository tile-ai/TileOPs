"""The FP8 quantization ops, imported from ``tileops.fp8_quant``.

Implemented under ``tileops.ops.fp8_quant``; this module is the public path.
"""

from .ops.fp8_quant import (
    FP8QuantFwdOp,
)

__all__ = [
    "FP8QuantFwdOp",
]
