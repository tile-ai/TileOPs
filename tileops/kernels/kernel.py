from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import torch
from tilelang.autotuner import autotune


class Kernel(ABC):
    dtype: Optional[torch.dtype] = None
    config: Dict[str, Any]
    autotune_configs: Optional[list[dict]] = None
    supported_archs: Optional[list[int]] = None
    kernel: Callable[[dict], Callable]

    def __init__(self, *args, **kwargs) -> None:
        self.config = {}

    def init_config(self, config: Optional[Dict[str, Any]] = None, tune: bool = False) -> None:
        if tune:
            if config is not None:
                import warnings
                warnings.warn(  # noqa: B028
                    "Both 'config' and 'tune' are set. "
                    "'config' will be ignored in favor of autotuning.")
            self.autotune()
        else:
            if config is not None:
                for k, v in self.default_config.items():
                    self.config[k] = config[k] if config.get(k) is not None else v
            else:
                self.config = self.default_config

        print(f"{self.__class__.__name__} initialized with config: {self.config}")

    @property
    def dtype_str(self) -> str:
        """Convert dtype to str for tl kernels"""
        return str(self.dtype).split('.')[-1]

    @property
    def default_config(self) -> Dict[str, Any]:
        """Return the default config for the kernel"""
        return {}

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Run the kernel"""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self.forward(*args, **kwargs)

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        if self.autotune_configs is None:
            return  # kernel doesn't support autotuning
        print(f'Start autotuning {self.__class__.__name__}...')

        # Apply autotune decorator to the kernel function
        autotuned_kernel_fn = autotune(
            configs=self.autotune_configs, warmup=warmup, rep=rep)(
                self.kernel)

        # Call without config parameters to trigger autotuning, returns the tuned kernel
        tuned_kernel = autotuned_kernel_fn()

        # Extract and store the best config
        self.config = tuned_kernel.config
        print(f'Best config: {self.config}')
