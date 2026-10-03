"""The pooling ops, at the public path ``tileops.pool``."""

from tileops.ops.pool.adaptive_pool import (
    AdaptiveAvgPool2dFwdOp,
    AdaptiveMaxPool2dFwdOp,
    AdaptiveMaxPool2dIndicesFwdOp,
)
from tileops.ops.pool.avg_pool import AvgPool1dFwdOp, AvgPool2dFwdOp, AvgPool3dFwdOp
from tileops.ops.pool.max_pool import (
    MaxPool1dFwdOp,
    MaxPool1dIndicesFwdOp,
    MaxPool2dFwdOp,
    MaxPool2dIndicesFwdOp,
    MaxPool3dFwdOp,
    MaxPool3dIndicesFwdOp,
)
from tileops.ops.pool.mean_pooling import MeanPoolingFwdOp

__all__ = [
    "AvgPool1dFwdOp",
    "AvgPool2dFwdOp",
    "AvgPool3dFwdOp",
    "MaxPool1dFwdOp",
    "MaxPool1dIndicesFwdOp",
    "MaxPool2dFwdOp",
    "MaxPool2dIndicesFwdOp",
    "MaxPool3dFwdOp",
    "MaxPool3dIndicesFwdOp",
    "AdaptiveAvgPool2dFwdOp",
    "AdaptiveMaxPool2dFwdOp",
    "AdaptiveMaxPool2dIndicesFwdOp",
    "MeanPoolingFwdOp",
]
