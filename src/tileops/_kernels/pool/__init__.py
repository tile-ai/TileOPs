from tileops._kernels.pool.adaptive_avg_pool2d import AdaptiveAvgPool2dKernel
from tileops._kernels.pool.adaptive_max_pool2d import (
    AdaptiveMaxPool2dKernel,
    AdaptiveMaxPool2dRegisterKernel,
    AdaptiveMaxPool2dWithIndicesKernel,
)
from tileops._kernels.pool.avg_pool1d import AvgPool1dKernel
from tileops._kernels.pool.avg_pool1d_register import AvgPool1dRegisterKernel
from tileops._kernels.pool.avg_pool2d import AvgPool2dKernel
from tileops._kernels.pool.avg_pool2d_register import AvgPool2dRegisterKernel
from tileops._kernels.pool.avg_pool3d import AvgPool3dKernel
from tileops._kernels.pool.call_spec import (
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
from tileops._kernels.pool.max_pool1d import MaxPool1dKernel, MaxPool1dWithIndicesKernel
from tileops._kernels.pool.max_pool2d import MaxPool2dKernel, MaxPool2dWithIndicesKernel
from tileops._kernels.pool.max_pool2d_register import (
    MaxPool2dIndicesRegisterKernel,
    MaxPool2dRegisterKernel,
)
from tileops._kernels.pool.max_pool3d import MaxPool3dKernel, MaxPool3dWithIndicesKernel
from tileops._kernels.pool.mean_pooling import MeanPoolingFwdKernel

__all__ = [
    "AdaptiveAvgPool2dFwdInterface",
    "AdaptiveAvgPool2dKernel",
    "AdaptiveMaxPool2dFwdInterface",
    "AdaptiveMaxPool2dIndicesFwdInterface",
    "AdaptiveMaxPool2dKernel",
    "AdaptiveMaxPool2dRegisterKernel",
    "AdaptiveMaxPool2dWithIndicesKernel",
    "AdaptivePool2dCall",
    "AvgPool1dFwdInterface",
    "AvgPool1dKernel",
    "AvgPool1dRegisterKernel",
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
    "MaxPool2dIndicesRegisterKernel",
    "MaxPool2dKernel",
    "MaxPool2dRegisterKernel",
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
