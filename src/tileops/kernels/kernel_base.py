import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Hashable, Optional, Union

import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES

__all__ = ["Entry", "Kernel", "KernelInterface", "vector_aligned"]

# What ``Op.kernel_for`` stores for one specialization: the identity two
# builds share to be the same entry, and the thunk that produces it.
Entry = tuple[Hashable, Callable[[], object]]

# Sentinel for ``tune_jit_kernel(supply_prog=...)``: inherit the whole-kernel
# supplier. Distinct from ``None``, which means "no supplier".
_INHERIT_SUPPLY_PROG = object()


def vector_aligned(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """*t*, or a copy of it when its storage does not start on a 16-byte vector boundary.

    For an input a kernel reads in 16-byte vectors; never for a tensor it writes.
    """
    return t.clone() if t is not None and t.data_ptr() % VECTOR_ACCESS_BYTES else t


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

    @staticmethod
    def _int_tensor_input_names(jit_kernel: Any, seeds: Dict[str, Any]) -> list[str]:
        """Integer tensor parameters *jit_kernel* takes as inputs, outputs excluded.

        Raises:
            ValueError: When the parameters cannot be read, which the caller treats
                as unproven rather than safe.
        """
        get_tir = getattr(jit_kernel, "get_tir", None)
        if not callable(get_tir):
            raise ValueError(
                f"{jit_kernel!r} does not expose get_tir, so its parameters cannot be read"
            )
        try:
            prim_func = get_tir(**seeds)
        except Exception as exc:
            raise ValueError(f"the parameters of {jit_kernel!r} cannot be read: {exc}") from exc

        out_idx = getattr(jit_kernel, "out_idx", None) or []
        if isinstance(out_idx, int):
            out_idx = [out_idx]
        count = len(prim_func.params)
        outputs = {i + count if i < 0 else i for i in out_idx}

        names = []
        for i, param in enumerate(prim_func.params):
            if i in outputs:
                continue
            buffer = prim_func.buffer_map.get(param)
            if buffer is not None and "int" in str(buffer.dtype):
                names.append(str(param.name))
        return names

    def __init__(self, *args, device_index: "int | None" = None, **kwargs) -> None:
        self.device_index = device_index
        self._check_arch()
        self.config = {}

    # Set True when this kernel's integer tensor inputs are data or masks, so
    # autotuning may generate them from ``randint(-2, 3)``. A kernel whose
    # integer values decide how much work runs overrides
    # ``autotune_supply_prog`` instead; left False, autotuning refuses.
    autotune_accepts_random_int_inputs: bool = False

    # The device types this implementation runs on. With ``supported_archs`` it states where
    # the implementation is available; selection filters on it before asking ``refusal``.
    devices: ClassVar[frozenset[str]] = frozenset({"cuda"})

    # Whether this implementation is below every other implementation of its interface.
    # An interface has at most one; it runs where no other implementation serves the call.
    general: bool = False

    # The keys of implementations of the same interface this one wins over where both are
    # available and apply. Transitive.
    preferred_over: ClassVar[frozenset[str]] = frozenset()

    # Set when tuning was requested before the program existed; the next launch tunes it.
    _tune_pending: bool = False
    # Whether this kernel has been put in tuned mode, so a second request changes nothing.
    _tune_requested: bool = False

    @classmethod
    def refusal(cls, call: Any) -> Optional[str]:
        """Why this implementation does not serve *call*, or ``None`` when it does.

        States the calls it serves, never what a sibling serves; every refusal names the
        limit it hits, so a caller told that nothing served the call learns why each
        implementation declined. Where two non-general implementations both serve a call,
        ``preferred_over`` says which one wins. Where it is available is ``devices`` and
        ``supported_archs``, not this.

        The default serves every call. An override that narrows it ends with
        ``super().refusal(call)``, so the limits a base class states still apply.
        """
        return None

    @classmethod
    def unavailable(cls, call: Any) -> Optional[str]:
        """Why this class cannot run on *call*'s device, or ``None`` when it can.

        From ``devices`` and ``supported_archs``. Selection asks this before ``refusal``.
        """
        device = getattr(call, "device", None)
        if device is not None and device.type not in cls.devices:
            return f"runs on {sorted(cls.devices)}, not {device.type}"
        archs = cls.supported_archs
        if archs is not None and call.arch not in archs:
            return f"built for architectures {sorted(archs)}, device reports {call.arch}"
        return None

    @classmethod
    def entry_for(cls, call: Any) -> Entry:
        """How to build this class for *call*, and what makes two builds one entry.

        The identity is the construction arguments other than ``tune``, so two calls
        that would compile the same kernel share an entry and none reuses one compiled
        for different arguments. The thunk runs only on a cache miss.

        The default is the identity mapping: this class is constructed from the call
        record itself. A class with a narrower constructor overrides it and states
        which of the call's facts it is built from.

        The device index is in the identity wherever this class could build a
        different object on another device, whether it takes one as an argument or
        reads it while compiling. A class that only validates the architecture it
        was handed does not put it there.
        """
        return call, lambda: cls(call)

    def _check_arch(self) -> None:
        """Reject construction on a device this kernel is not built for.

        Read from ``device_index``, the device the op handed over — not from whichever
        device happens to be current, which need not be the one the input lives on. The op
        layer performs no architecture check of its own; selection filters an interface's
        implementations by availability first.

        ``device_index`` ``None`` reads the current device. An op builds every kernel with
        the call's device current (``Op.kernel_for``), so that is the call's device there.

        Raises:
            ValueError: The device's architecture is not among ``supported_archs``.
                Selection raises the same class when no implementation for a dispatch key can
                serve a call, so a caller catches one exception type whether the key has
                one implementation or several.
        """
        cls = type(self)
        if cls.supported_archs is None:
            return
        from tileops.utils import get_sm_version

        arch = get_sm_version(self.device_index)
        if arch not in cls.supported_archs:
            where = (
                "the current device" if self.device_index is None else f"cuda:{self.device_index}"
            )
            raise ValueError(
                f"{cls.__name__} is built for architectures "
                f"{sorted(cls.supported_archs)}, but {where} reports {arch}"
            )

    def _require_cuda(self, **tensors: Optional[torch.Tensor]) -> None:
        """Raise unless every named tensor is on a CUDA device.

        The op layer checks that a call's tensors agree on a device; which devices a set of
        kernels runs on is the kernel's own statement. An ``optional: true`` input the call
        did not pass arrives as ``None`` and is skipped.
        """
        for name, tensor in tensors.items():
            if tensor is not None and not tensor.is_cuda:
                raise ValueError(
                    f"{type(self).__name__} is a CUDA kernel; got {name} on {tensor.device}. "
                    "Another target's backend serves other devices."
                )

    def init_config(self, config: Optional[Dict[str, Any]] = None, tune: bool = False) -> None:
        if tune and self.autotune_configs is None:
            import warnings

            warnings.warn(
                f"{self.__class__.__name__} does not define autotune_configs; "
                "falling back to the provided config or default_config.",
                stacklevel=2,
            )
            tune = False

        if tune:
            if config is not None:
                import warnings

                warnings.warn(
                    "Both 'config' and 'tune' are set. "
                    "'config' will be ignored in favor of autotuning.",
                    stacklevel=2,
                )
            self._tune_requested = True
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
        return str(dtype).split(".")[-1]

    @property
    def default_config(self) -> Dict[str, Any]:
        """Return the default config for the kernel"""
        return {}

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the kernel"""
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.forward(*args, **kwargs)
        if self._tune_pending:
            # Tuning was requested before the program existed; the launch has built it.
            self._tune_pending = False
            if getattr(self, "kernel", None) is None:
                warnings.warn(
                    f"{type(self).__name__} exposes no program as `self.kernel`, so the "
                    "tuning requested of it cannot run.",
                    stacklevel=2,
                )
            else:
                self.autotune()
        return result

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
        return autotuned_kernel_fn(**self._autotune_initial_kwargs(kernel=kernel, config=config))

    def _refuse_random_int_inputs(self, jit_kernel: Callable, seeds: Dict[str, Any]) -> None:
        """Refuse to tune *jit_kernel* on integer inputs TileLang would randomise.

        Raises:
            ValueError: When the kernel takes integer tensor inputs, supplies
                none of them, and has not declared random values safe.
        """
        if self.autotune_accepts_random_int_inputs:
            return
        names = Kernel._int_tensor_input_names(jit_kernel, seeds)
        if not names:
            return
        raise ValueError(
            f"{type(self).__name__} autotunes with integer tensor inputs "
            f"{', '.join(names)} and no supply_prog, so TileLang would generate them "
            f"from randint(-2, 3). Override autotune_supply_prog to build them, or set "
            f"autotune_accepts_random_int_inputs = True where random values cannot "
            f"change what the candidates measure."
        )

    def _param_name_aliases(self, jit_kernel: Callable) -> Dict[str, str]:
        """Map config key -> the builder parameter that carries it, for this builder.

        TileLang binds a candidate config by parameter name and refuses a key naming
        no parameter, so a builder taking ``threads_arg`` never sees ``threads``.
        """
        signature = getattr(jit_kernel, "signature", None)
        parameters = getattr(signature, "parameters", None)
        if not parameters:
            return {}
        return {
            key: name
            for name, key in self._AUTOTUNE_PARAM_ALIASES.items()
            if name in parameters and key not in parameters
        }

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
        rename = self._param_name_aliases(jit_kernel)
        if rename:
            configs = [{rename.get(k, k): v for k, v in cfg.items()} for cfg in configs]
        autotune_kwargs: Dict[str, Any] = dict(configs=configs, warmup=warmup, rep=rep)
        if seeds:
            autotune_kwargs["do_not_specialize"] = list(seeds)
        if supply_prog is _INHERIT_SUPPLY_PROG:
            supply_prog = self.autotune_supply_prog
        if supply_prog is not None:
            autotune_kwargs["supply_prog"] = supply_prog
        else:
            self._refuse_random_int_inputs(jit_kernel, seeds)
        from tilelang.autotuner import autotune

        autotuned_kernel_fn = autotune(**autotune_kwargs)(jit_kernel)

        return self._call_autotuned_kernel(autotuned_kernel_fn, jit_kernel, seed_config)

    def request_tune(self) -> None:
        """Put this kernel in tuned mode, once.

        What :meth:`autotune` does with it: tune now, or at the launch that builds the program.
        """
        if self._tune_requested:
            return
        self._tune_requested = True
        self.autotune()

    def autotune(self, warmup: int = 25, rep: int = 50) -> None:
        if self.autotune_configs is None:
            return  # kernel doesn't support autotuning
        if getattr(self, "kernel", None) is None:
            # The program is built at launch, and the launch tunes it (``__call__``).
            self._tune_pending = True
            return
        print(f"Start autotuning {self.__class__.__name__}...")

        tuned_kernel = self.tune_jit_kernel(
            self.kernel, self.autotune_configs, warmup=warmup, rep=rep
        )

        self.config = {
            self._AUTOTUNE_PARAM_ALIASES.get(key, key): value
            for key, value in tuned_kernel.config.items()
        }
        print(f"Best config: {self.config}")


class KernelInterface(ABC):
    """The call contract of one kernel interface, which each of its implementations inherits.

    ``request`` names the ``CallSpec`` subclass the op passes as the call spec. ``forward``
    states the call the op makes on the built entry: each tensor's shape, dtype, layout and
    device, which ones it writes in place or may alias, and what it returns. An
    implementation is a ``Kernel`` that inherits the interface, with a classmethod
    ``entry_for`` and a constructor of its own.
    """

    request: ClassVar[type]

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """The call the op makes on an implementation's entry."""
        raise NotImplementedError
