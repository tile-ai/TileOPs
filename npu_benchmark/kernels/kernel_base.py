"""Abstract base class for kernels.

A Kernel wraps a TileLang JIT-compiled function and manages configuration
and optional autotuning.  Subclasses provide:
  - ``default_config``: dict of config params
  - ``autotune_configs``: list of candidate configs (optional)
  - ``forward()``: execution entry point
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import torch


class Kernel(ABC):
    config: Dict[str, Any]
    autotune_configs: Optional[list[dict]] = None
    supported_archs: Optional[list[int]] = None
    kernel: Callable

    def __init__(self, *args, **kwargs) -> None:
        self.config = {}

    def init_config(self, config: Optional[Dict[str, Any]] = None,
                    tune: bool = False) -> None:
        if tune and self.autotune_configs is None:
            warnings.warn(f"{self.__class__.__name__} has no autotune_configs; "
                          "falling back to default_config.")
            tune = False

        if tune:
            if config is not None:
                warnings.warn("Both 'config' and 'tune' set; 'config' ignored.")
            self.autotune()
        elif config is not None:
            self.config = {k: config.get(k, v) for k, v in self.default_config.items()}
        else:
            self.config = dict(self.default_config)

        print(f"{self.__class__.__name__} config: {self.config}")

    @property
    def default_config(self) -> Dict[str, Any]:
        return {}

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @property
    def autotune_supply_prog(self) -> Optional[Callable]:
        return None

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        if self.autotune_configs is None:
            return
        if not hasattr(self, 'kernel') or self.kernel is None:
            raise AttributeError(
                f"Cannot autotune {self.__class__.__name__}: 'self.kernel' not set.")

        from tilelang.autotuner import autotune

        print(f"Start autotuning {self.__class__.__name__}...")
        tunable_params = list(self.default_config.keys())
        kwargs: Dict[str, Any] = dict(
            configs=self.autotune_configs, warmup=warmup, rep=rep)
        if tunable_params:
            kwargs["do_not_specialize"] = tunable_params
        if self.autotune_supply_prog is not None:
            kwargs["supply_prog"] = self.autotune_supply_prog

        autotuned_fn = autotune(**kwargs)(self.kernel)
        tuned = autotuned_fn(**self.default_config)
        self.config = tuned.config
        print(f"Best config: {self.config}")

    @staticmethod
    def dtype_to_str(dtype: torch.dtype) -> str:
        return str(dtype).split('.')[-1]
