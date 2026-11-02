"""Call workloads of the pool family."""

import functools

from workloads.pool import (
    AdaptiveAvgPool2dCall,
    AdaptiveMaxPool2dCall,
    AvgPoolCall,
    MaxPoolCall,
    MeanPoolingCallWorkload,
)

_MAX_POOL_INDICES = functools.partial(MaxPoolCall, return_indices=True)

WORKLOADS = {
    "AvgPool1dFwdOp": AvgPoolCall,
    "AvgPool2dFwdOp": AvgPoolCall,
    "AvgPool3dFwdOp": AvgPoolCall,
    "MaxPool1dFwdOp": MaxPoolCall,
    "MaxPool1dIndicesFwdOp": _MAX_POOL_INDICES,
    "MaxPool2dFwdOp": MaxPoolCall,
    "MaxPool2dIndicesFwdOp": _MAX_POOL_INDICES,
    "MaxPool3dFwdOp": MaxPoolCall,
    "MaxPool3dIndicesFwdOp": _MAX_POOL_INDICES,
    "AdaptiveAvgPool2dFwdOp": AdaptiveAvgPool2dCall,
    "AdaptiveMaxPool2dFwdOp": AdaptiveMaxPool2dCall,
    "AdaptiveMaxPool2dIndicesFwdOp": functools.partial(AdaptiveMaxPool2dCall, return_indices=True),
    "MeanPoolingFwdOp": MeanPoolingCallWorkload,
}
