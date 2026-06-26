import inspect
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
    # JIT parameter names that differ from their autotune config key.
    _AUTOTUNE_PARAM_ALIASES = {
        "threads_arg": "threads",
        "npt_arg": "num_per_thread",
        "num_per_thread_arg": "num_per_thread",
    }

    def __init__(self, *args, **kwargs) -> None:
        self.config = {}

    def init_config(self, config: Optional[Dict[str, Any]] = None, tune: bool = False) -> None:
        if tune and self.autotune_configs is None:
            import warnings

            warnings.warn(  # noqa: B028
                f"{self.__class__.__name__} does not define autotune_configs; "
                "falling back to the provided config or default_config.")
            tune = False

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
        return self.dtype_to_str(self.dtype)

    @staticmethod
    def dtype_to_str(dtype: torch.dtype) -> str:
        """Convert a torch dtype to the TileLang dtype string."""
        return str(dtype).split('.')[-1]

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

    @property
    def autotune_supply_prog(self) -> Optional[Callable]:
        """Return a supply_prog callback for autotuning input generation.

        Override in subclasses whose kernels have scalar (T.int32, etc.) parameters
        that the default tensor-only auto-generation cannot handle.

        The callback signature is: (params: list[KernelParam]) -> list[Tensor | int | ...]
        """
        return None

    @staticmethod
    def _seed_autotune_binder(kernel: Optional[Callable], source: Optional[Dict[str, Any]]) -> None:
        """Make a kernel's tunable JIT params bindable from ``source`` defaults.

        TileLang 0.1.11 binds the JIT signature before autotuning, so a tunable
        parameter with no default and no supplied value raises at bind time;
        but supplying the value on the call makes the autotuner skip tuning
        entirely (returning ``config=None``). To get neither, seed the JIT
        argument binder's per-parameter defaults from a single source of truth
        (the kernel's ``default_config``, or the per-sub-kernel config a custom
        autotune override passes), so binding succeeds with no args while the
        autotuner still overrides every tunable parameter per candidate.

        ``source`` values are bind-time placeholders only; they are never the
        executed config (the search overrides them, and ``forward`` always
        passes the chosen config explicitly).
        """
        if not source:
            return
        binder = getattr(getattr(kernel, "func", None), "_argument_binder", None)
        if binder is None:
            return
        names = getattr(binder, "param_names", None)
        if not names:
            return
        sig = getattr(kernel, "signature", None)
        params = sig.parameters if sig is not None else {}
        defaults: list[Any] = []
        required: list[int] = []
        for i, name in enumerate(names):
            key = name if name in source else Kernel._AUTOTUNE_PARAM_ALIASES.get(name)
            if key in source:
                defaults.append(source[key])
                continue
            existing = params.get(name)
            if existing is not None and existing.default is not inspect.Parameter.empty:
                defaults.append(existing.default)  # keep a pre-existing signature default
            else:
                defaults.append(None)  # genuinely required (not a tunable config key)
                required.append(i)
        binder.defaults = tuple(defaults)
        binder.required_indices = tuple(required)

    def _call_autotuned_kernel(
        self,
        autotuned_kernel_fn: Callable,
        kernel: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Run the TileLang autotuner search and return its tuned kernel.

        The autotuned wrapper is invoked with no tunable arguments (supplying
        them makes TileLang 0.1.11 skip tuning and return ``config=None``).
        Binding without args is made possible by seeding the JIT binder defaults
        from the kernel's ``default_config`` (or the ``config`` a custom override
        passes) -- see ``_seed_autotune_binder``.
        """
        self._seed_autotune_binder(kernel, config if config is not None else self.default_config)
        return autotuned_kernel_fn()

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        if self.autotune_configs is None:
            return  # kernel doesn't support autotuning
        if not hasattr(self, 'kernel') or self.kernel is None:
            raise AttributeError(
                f"Cannot autotune {self.__class__.__name__}: 'self.kernel' is not set. "
                "Set 'self.kernel' in __init__ before calling init_config with tune=True.")
        print(f'Start autotuning {self.__class__.__name__}...')

        # Apply autotune decorator to the kernel function
        autotune_kwargs: Dict[str, Any] = dict(
            configs=self.autotune_configs, warmup=warmup, rep=rep)
        if self.autotune_supply_prog is not None:
            autotune_kwargs["supply_prog"] = self.autotune_supply_prog
        autotuned_kernel_fn = autotune(**autotune_kwargs)(self.kernel)

        # Run the autotuner search (no tunable args passed; see
        # _call_autotuned_kernel). Tunable JIT parameters are bindable via
        # defaults on the kernel @tilelang.jit signatures.
        tuned_kernel = self._call_autotuned_kernel(autotuned_kernel_fn, self.kernel)

        # Extract and store the best config
        self.config = tuned_kernel.config
        print(f'Best config: {self.config}')
