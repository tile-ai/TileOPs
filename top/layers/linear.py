import math
import torch
from torch import nn
from top.functions import matmul


class Linear(nn.Module):

    def __init__(
            self,
            M: int, 
            N: int,
            K: int,
            device='cuda',
            dtype=torch.float16,
            tune=False,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((K, N), **factory_kwargs)
        )
        self.fn = matmul(
            M,
            N,
            K,
            dtype=self.weight.dtype,
            tune=tune,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.fn(input, self.weight)