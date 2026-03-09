import itertools
from typing import Callable, Optional, Tuple

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel import Kernel
from tileops.kernels.online_softmax import make_log2e_scale, make_online_softmax, make_rescale

__all__ = [
    'GqaSlidingWindowFwdKernel',
    'GqaSlidingWindowFwdWgmmaPipelinedKernel',
]
