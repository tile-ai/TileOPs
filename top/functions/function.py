from abc import ABC, abstractmethod
from typing import Any


class Function(ABC):

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
