import torch
from torch import nn
from abc import abstractmethod, ABC
from typing import Callable

class KernelBase(nn.Module, ABC):

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass
    
    @property
    @abstractmethod
    def ref_program(self) -> Callable:
        pass
    
    @abstractmethod
    def get_flops(self, *args, **kwargs) -> float:
        pass
    
    @abstractmethod
    def get_memory_footprint(self, *args, **kwargs) -> float:
        pass
    