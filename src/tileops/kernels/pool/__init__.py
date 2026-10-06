from tileops.kernels.pool.adaptive_avg_pool2d import AdaptiveAvgPool2dKernel
from tileops.kernels.pool.adaptive_max_pool2d import (
    AdaptiveMaxPool2dKernel,
    AdaptiveMaxPool2dWithIndicesKernel,
)
from tileops.kernels.pool.avg_pool1d import AvgPool1dKernel
from tileops.kernels.pool.avg_pool2d import AvgPool2dKernel
from tileops.kernels.pool.avg_pool2d_register import AvgPool2dRegisterKernel
from tileops.kernels.pool.avg_pool3d import AvgPool3dKernel
from tileops.kernels.pool.call_spec import (
    AdaptiveAvgPool2dFwdInterface,
    AdaptiveMaxPool2dFwdInterface,
    AdaptiveMaxPool2dIndicesFwdInterface,
    AdaptivePool2dCall,
    AvgPool1dFwdInterface,
    AvgPool2dFwdInterface,
    AvgPool3dFwdInterface,
    AvgPoolCall,
    MaxPool1dFwdInterface,
    MaxPool1dIndicesFwdInterface,
    MaxPool2dFwdInterface,
    MaxPool2dIndicesFwdInterface,
    MaxPool3dFwdInterface,
    MaxPool3dIndicesFwdInterface,
    MaxPoolCall,
    MeanPoolingCall,
    MeanPoolingFwdInterface,
)
from tileops.kernels.pool.max_pool1d import MaxPool1dKernel, MaxPool1dWithIndicesKernel
from tileops.kernels.pool.max_pool2d import MaxPool2dKernel, MaxPool2dWithIndicesKernel
from tileops.kernels.pool.max_pool3d import MaxPool3dKernel, MaxPool3dWithIndicesKernel
from tileops.kernels.pool.mean_pooling import MeanPoolingFwdKernel

__all__ = [
    "AdaptiveAvgPool2dFwdInterface",
    "AdaptiveAvgPool2dKernel",
    "AdaptiveMaxPool2dFwdInterface",
    "AdaptiveMaxPool2dIndicesFwdInterface",
    "AdaptiveMaxPool2dKernel",
    "AdaptiveMaxPool2dWithIndicesKernel",
    "AdaptivePool2dCall",
    "AvgPool1dFwdInterface",
    "AvgPool1dKernel",
    "AvgPool2dFwdInterface",
    "AvgPool2dKernel",
    "AvgPool2dRegisterKernel",
    "AvgPool3dFwdInterface",
    "AvgPool3dKernel",
    "AvgPoolCall",
    "MaxPool1dFwdInterface",
    "MaxPool1dIndicesFwdInterface",
    "MaxPool1dKernel",
    "MaxPool1dWithIndicesKernel",
    "MaxPool2dFwdInterface",
    "MaxPool2dIndicesFwdInterface",
    "MaxPool2dKernel",
    "MaxPool2dWithIndicesKernel",
    "MaxPool3dFwdInterface",
    "MaxPool3dIndicesFwdInterface",
    "MaxPool3dKernel",
    "MaxPool3dWithIndicesKernel",
    "MaxPoolCall",
    "MeanPoolingCall",
    "MeanPoolingFwdInterface",
    "MeanPoolingFwdKernel",
]
