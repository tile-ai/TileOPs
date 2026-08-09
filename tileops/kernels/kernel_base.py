from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union

import torch
from tilelang.autotuner import autotune

#: Sentinel for ``tune_jit_kernel(supply_prog=...)``: inherit the whole-kernel
#: supplier. Distinct from ``None``, which means "no supplier".
_INHERIT_SUPPLY_PROG = object()


class Kernel(ABC):
    dtype: Optional[torch.dtype] = None
    config: Dict[str, Any]
    autotune_configs: Optional[list[dict]] = None
    supported_archs: Optional[list[int]] = None
    kernel: Callable[[dict], Callable]
    _AUTOTUNE_PARAM_ALIASES = {
        "threads_arg": "threads",
        "npt_arg": "num_per_thread",
        "num_per_thread_arg": "num_per_thread",
    }

    def __init__(self, *args, **kwargs) -> None:
        self._check_arch()
        self.config = {}

    #: Whether this implementation is the one behind the specialised ones.
    #: A dispatch key may have at most one general implementation applying to a
    #: call; it runs when no specialised implementation of that key serves it.
    #: Stating it here rather than as an exclusion in every specialised sibling
    #: is what lets a specialisation appear, or be replaced, without the general
    #: implementation naming it.
    general: bool = False

    @classmethod
    def applies(cls, call: Any) -> bool:
        """Whether this implementation serves the call *call* describes.

        States a region positively — what this class serves, never what a
        sibling serves. Architecture is not part of it: ``supported_archs``
        already answers that, and a specialised implementation the device
        cannot run simply does not apply, leaving the call to the general one.

        Answered by the class that would run, so a ``kernel_map`` override is
        asked about its own region rather than the region of the class it
        replaced.

        The default serves anything the architecture allows. A specialised
        implementation that leaves it unset therefore claims every call, which
        collides with its siblings and is reported rather than silently
        preferred.
        """
        return True

    @classmethod
    def refusal(cls, call: Any) -> Optional[str]:
        """Why this class cannot serve *call*, or ``None`` when it can.

        The class that declines says why. A caller told only that nothing served
        the call cannot tell an architecture it does not have from a shape the
        implementation was never written for, and whoever is selecting has no
        way to find out without reading the class it just rejected.
        """
        archs = cls.supported_archs
        if archs is not None and call.arch not in archs:
            return f"built for architectures {sorted(archs)}, device reports {call.arch}"
        if not cls.applies(call):
            return "does not serve this call"
        return None

    @classmethod
    def _check_arch(cls) -> None:
        """Reject construction on a device this kernel is not built for.

        The device is probed only by a class that constrains the architecture,
        and only here — kernel construction is deferred to the first forward,
        so a device is present by the time this runs. The op layer performs no
        architecture check of its own; a role served by several kernels filters
        candidates during selection instead.

        Raises:
            ValueError: When the current device architecture is not among
                ``supported_archs``. Selection raises the same class when no
                candidate for a dispatch key can serve a call, so a caller
                catches one exception type whether the key has one
                implementation or several.
        """
        if cls.supported_archs is None:
            return
        from tileops.utils import get_sm_version
        arch = get_sm_version()
        if arch not in cls.supported_archs:
            raise ValueError(
                f"{cls.__name__} is built for architectures "
                f"{sorted(cls.supported_archs)}, but the current device reports "
                f"{arch}")

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

    #: Implementation serving bool operands, as ``(class, construction dtype)``.
    #: Left unset by a backend whose general implementation handles bool.
    BOOL_IMPL: Optional[tuple] = None

    @classmethod
    def specialize(cls, dtype: "torch.dtype") -> tuple:
        """Return the class to build for *dtype*, and the dtype to build it with.

        A backend may serve a semantic dtype with an implementation other than
        its general one, or compute it in a different storage type — bool
        operands on a uint8 kernel, say. Answering here keeps that choice inside
        the backend: the caller passes the semantic dtype and never names an
        implementation or a storage type. The default is the identity.
        """
        if dtype == torch.bool and cls.BOOL_IMPL is not None:
            return cls.BOOL_IMPL
        return cls, dtype

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the kernel"""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    @property
    def autotune_supply_prog(self) -> Optional[Callable]:
        """Return a supply_prog callback for autotuning input generation.

        Override in subclasses whose kernels have scalar (T.int32, etc.) parameters
        that the default tensor-only auto-generation cannot handle.

        The callback signature is: (params: list[KernelParam]) -> list[Tensor | int | ...]
        """
        return None

    def _autotune_initial_kwargs(
        self,
        kernel: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return initial JIT kwargs for TileLang autotuner binding.

        TileLang 0.1.11 validates/binds the JIT signature before candidate
        configs are applied. Passing the kernel's default config keeps required
        tunable parameters bindable while the autotuner still overrides them
        with each candidate config during benchmarking.
        """
        source = self.default_config if config is None else config
        if not source:
            return {}

        jit_kernel = getattr(self, "kernel", None) if kernel is None else kernel
        signature = getattr(jit_kernel, "signature", None)
        parameters = getattr(signature, "parameters", None)
        if parameters is None:
            return dict(source)

        kwargs = {}
        for name in parameters:
            if name in source:
                kwargs[name] = source[name]
                continue
            alias = self._AUTOTUNE_PARAM_ALIASES.get(name)
            if alias is not None and alias in source:
                kwargs[name] = source[alias]
        return kwargs

    def _call_autotuned_kernel(
        self,
        autotuned_kernel_fn: Callable,
        kernel: Optional[Callable] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return autotuned_kernel_fn(
            **self._autotune_initial_kwargs(kernel=kernel, config=config)
        )

    def tune_jit_kernel(
        self,
        jit_kernel: Callable,
        configs: list[dict],
        warmup: int = 25,
        rep: int = 50,
        seed_config: Optional[Dict[str, Any]] = None,
        supply_prog: Union[Callable, None, object] = _INHERIT_SUPPLY_PROG,
    ) -> Any:
        """Benchmark *configs* against one JIT kernel and return the winner.

        A kernel built from several sub-kernels tunes each one on its own
        candidate list, so the JIT object and the candidates are arguments here
        rather than being read from ``self.kernel`` / ``self.autotune_configs``.

        Args:
            jit_kernel: The ``@tilelang.jit``-decorated builder to tune.
            configs: Candidate configs handed to TileLang's autotuner.
            warmup: Warmup iterations per candidate.
            rep: Timed iterations per candidate.
            seed_config: Config supplying the seeded JIT parameter values;
                ``default_config`` when omitted.
            supply_prog: Input supplier for the candidates. Defaults to
                ``autotune_supply_prog``, which is written against
                ``self.kernel``; a sub-kernel taking different inputs must pass
                its own, since the whole-kernel supplier would feed it the
                wrong ones. Pass ``None`` for no supplier.

        Returns:
            The tuned kernel, carrying the winning ``config`` and its measured
            ``latency``.
        """
        # Pass do_not_specialize so TileLang excludes the seeded JIT parameters
        # from the cache key; without this, the seed values read as "already
        # tuned" and the benchmarking sweep is skipped (returning config=None).
        seeds = self._autotune_initial_kwargs(jit_kernel, seed_config)
        autotune_kwargs: Dict[str, Any] = dict(configs=configs, warmup=warmup, rep=rep)
        if seeds:
            autotune_kwargs["do_not_specialize"] = list(seeds)
        if supply_prog is _INHERIT_SUPPLY_PROG:
            supply_prog = self.autotune_supply_prog
        if supply_prog is not None:
            autotune_kwargs["supply_prog"] = supply_prog
        autotuned_kernel_fn = autotune(**autotune_kwargs)(jit_kernel)

        return self._call_autotuned_kernel(autotuned_kernel_fn, jit_kernel, seed_config)

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        if self.autotune_configs is None:
            return  # kernel doesn't support autotuning
        if not hasattr(self, 'kernel') or self.kernel is None:
            raise AttributeError(
                f"Cannot autotune {self.__class__.__name__}: 'self.kernel' is not set. "
                "Set 'self.kernel' in __init__ before calling init_config with tune=True.")
        print(f'Start autotuning {self.__class__.__name__}...')

        tuned_kernel = self.tune_jit_kernel(
            self.kernel, self.autotune_configs, warmup=warmup, rep=rep)

        self.config = tuned_kernel.config
        print(f'Best config: {self.config}')
