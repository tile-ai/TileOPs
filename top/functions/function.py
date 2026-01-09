from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch


class Function(ABC):

    @abstractmethod
    def forward(
            self,
            *args: object,  # noqa: U100
            **kwargs: object) -> Union[torch.Tensor, Tuple]:  # noqa: U100
        raise NotImplementedError

    def __call__(self, *args: object, **kwargs: object) -> Union[torch.Tensor, Tuple]:
        return self.forward(*args, **kwargs)
