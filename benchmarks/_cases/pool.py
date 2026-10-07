"""Case factories of the pool family."""

import functools

from benchmarks._cases import Entry
from workloads.pool import (
    AdaptiveAvgPool2dCall,
    AdaptiveMaxPool2dCall,
    AvgPoolCall,
    MaxPoolCall,
    MeanPoolingCallWorkload,
)

_MAX_POOL_INDICES = Entry(functools.partial(MaxPoolCall, return_indices=True))

ENTRIES = {
    "AvgPool1dFwdOp": Entry(AvgPoolCall),
    "AvgPool2dFwdOp": Entry(AvgPoolCall),
    "AvgPool3dFwdOp": Entry(AvgPoolCall),
    "MaxPool1dFwdOp": Entry(MaxPoolCall),
    "MaxPool1dIndicesFwdOp": _MAX_POOL_INDICES,
    "MaxPool2dFwdOp": Entry(MaxPoolCall),
    "MaxPool2dIndicesFwdOp": _MAX_POOL_INDICES,
    "MaxPool3dFwdOp": Entry(MaxPoolCall),
    "MaxPool3dIndicesFwdOp": _MAX_POOL_INDICES,
    "AdaptiveAvgPool2dFwdOp": Entry(AdaptiveAvgPool2dCall),
    "AdaptiveMaxPool2dFwdOp": Entry(AdaptiveMaxPool2dCall),
    "AdaptiveMaxPool2dIndicesFwdOp": Entry(
        functools.partial(AdaptiveMaxPool2dCall, return_indices=True)
    ),
    "MeanPoolingFwdOp": Entry(MeanPoolingCallWorkload, count_copies=True),
}
