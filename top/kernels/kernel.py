from typing import Callable

class Kernel:
    # dict类型成员变量config，需要子类初始化
    config: dict
    mod: Callable

    def __init__(self, *args, **kwargs):
        self.config = {}

    @property
    def default_config(self) -> dict:
        """Return the default config for the kernel"""
        return {}

    def forward(self, *args, **kwargs):
        """Run the kernel"""
        return self.mod(*args, **kwargs)

    __call__ = forward

    def autotune(self):
        if hasattr(self, 'get_best_config'):
            self.config = self.get_best_config()
        