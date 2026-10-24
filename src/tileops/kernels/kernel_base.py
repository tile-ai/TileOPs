import contextlib
import copy
import threading
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar, Dict, Hashable, Iterator, Optional, Sequence, Union

import torch

from tileops.kernels.constants import VECTOR_ACCESS_BYTES

__all__ = ["CompiledProgram", "Entry", "Kernel", "KernelInterface", "vector_aligned"]

# What ``Op.kernel_for`` stores for one specialization: the identity two
# builds share to be the same entry, and the thunk that produces it.
Entry = tuple[Hashable, Callable[[], object]]

# Sentinel for ``tune_jit_kernel(supply_prog=...)``: inherit the whole-kernel
# supplier. Distinct from ``None``, which means "no supplier".
_INHERIT_SUPPLY_PROG = object()


class CompiledProgram:
    """A compiled TileLang program, launched through its executable with the output
    descriptors read once when it was built.

    TileLang's call path reads, on every launch, which parameters are outputs and their
    shapes and dtypes from the program's parameter list; a program whose outputs have static
    shapes knows them when it is built. A launch here allocates the outputs from those
    descriptors on the device TileLang would use, the first tensor input's or else the
    current one, and calls the executable TileLang would call. A program with an output whose
    shape a launch decides, or without a TVM FFI executable, is launched through TileLang.
    Every other attribute is the TileLang program's.
    """

    __slots__ = ("jit", "adapter", "inputs", "outputs", "single")

    def __init__(self, jit: Any) -> None:
        """Read *jit*'s output descriptors, or leave its launches to TileLang."""
        self.jit, self.adapter, self.outputs, self.single = jit, None, (), None
        adapter = getattr(jit, "adapter", None)
        from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

        if type(adapter) is not TVMFFIKernelAdapter:
            return
        results = set(adapter.result_idx)
        outputs = []
        for i, param in enumerate(adapter.params):
            if i not in results:
                continue
            shape = tuple(param.shape)
            dtype = param.dtype
            if not shape or not all(isinstance(d, int) for d in shape) or dtype.bits < 8:
                return
            outputs.append((i, shape, param.torch_dtype()))
        self.adapter = adapter
        self.inputs = len(adapter.params) - len(outputs)
        self.outputs = tuple(outputs)
        if len(outputs) == 1 and outputs[0][0] == len(adapter.params) - 1:
            self.single = outputs[0][1:]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.jit, name)

    def __call__(self, *inputs: Any) -> Any:
        adapter = self.adapter
        if adapter is None:
            return self.jit(*inputs)
        executable = adapter.executable
        if executable is None:
            executable = adapter._executables_by_device.get(torch._C._cuda_getDevice())
            if executable is None:
                # TileLang builds the executable for the current device on this launch.
                return self.jit(*inputs)
        if len(inputs) != self.inputs:
            raise ValueError(
                f"Kernel expected {self.inputs} inputs, but {len(inputs)} are provided."
            )
        device = None
        for t in inputs:
            if isinstance(t, torch.Tensor):
                device = t.device
                break
        else:
            device = torch.device("cuda", torch._C._cuda_getDevice())
        single = self.single
        if single is not None:
            out = torch.empty(single[0], dtype=single[1], device=device)
            executable(*inputs, out)
            return out
        args = list(inputs)
        outputs = []
        for i, shape, dtype in self.outputs:
            out = torch.empty(shape, dtype=dtype, device=device)
            args.insert(i, out)
            outputs.append(out)
        executable(*args)
        return outputs[0] if len(outputs) == 1 else outputs


class _Config(dict):
    """A published configuration: a dict that refuses to change.

    A generation's programs were compiled from it, so a change in place would pair them with
    a configuration they were not compiled from; a new configuration is published instead.
    Its values are frozen copies: a nested dict as another `_Config`, a list or tuple as a
    tuple, a set as a frozenset, anything else deep-copied, so neither the dict it was
    published from nor a value read from it changes it. A copy is an ordinary dict.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__((k, _frozen(v)) for k, v in config.items())

    def _refuse(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "a published kernel config does not change in place; assign a new dict to "
            "`config` to publish it"
        )

    __setitem__ = __delitem__ = update = pop = popitem = setdefault = clear = _refuse
    __ior__ = _refuse

    def __reduce__(self) -> tuple:
        return dict, (dict(self),)


def _frozen(value: Any) -> Any:
    """*value* as a `_Config` value: a copy that refuses to change."""
    if isinstance(value, dict):
        return _Config(value)
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_frozen(v) for v in value)
    return copy.deepcopy(value)


class _Generation:
    """One published configuration of a kernel, and the programs compiled from it.

    Publishing a configuration starts a generation with no program: a launch compiles each
    program it needs from this generation's configuration, and a later generation never
    launches one compiled from an earlier configuration.
    """

    __slots__ = ("config", "programs", "building")

    def __init__(self, config: Dict[str, Any]) -> None:
        """Start a generation of a copy of *config*, with no program compiled yet."""
        self.config = None if config is None else _Config(config)
        # Each program by what selects it: the code that builds it and the launch facts.
        self.programs: dict = {}
        # Held while a program is built, so launches missing together build it once. A build
        # may ask for another program of the same generation.
        self.building = threading.RLock()


def vector_aligned(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """*t*, or a copy of it when its storage does not start on a 16-byte vector boundary.

    For an input a kernel reads in 16-byte vectors; never for a tensor it writes.
    """
    return t.clone() if t is not None and t.data_ptr() % VECTOR_ACCESS_BYTES else t


class Kernel(ABC):
    dtype: Optional[torch.dtype] = None
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

    # The generation each thread runs this kernel's launch, or tries a configuration, under,
    # by thread id: a launch reads one generation throughout, whatever is published
    # meanwhile. Each instance holds its own once it launches.
    _pinned: "dict[int, _Generation]" = {}

    def _current(self) -> _Generation:
        """The generation this thread's launch runs under: its pinned one, else the published."""
        pinned = self._pinned
        if pinned:
            generation = pinned.get(threading.get_ident())
            if generation is not None:
                return generation
        return self._generation

    @property
    def config(self) -> Dict[str, Any]:
        """The configuration of the generation this launch runs, which refuses to change in
        place: the published one outside a launch.

        Assigning one publishes a new generation, whose programs are compiled from it on
        their first launch, so a launch never runs a program with another configuration than
        the one it was compiled from. A launch already running keeps its own generation.
        """
        return self._current().config

    @config.setter
    def config(self, config: Dict[str, Any]) -> None:
        self._generation = _Generation(config)

    @contextlib.contextmanager
    def _pinning(self, generation: _Generation) -> Iterator[None]:
        """Run this thread's launches inside the block under *generation*."""
        pinned = self.__dict__.get("_pinned")
        if pinned is None:
            # Threads pinning for the first time together keep one table.
            pinned = self.__dict__.setdefault("_pinned", {})
        ident = threading.get_ident()
        outer = pinned.get(ident)
        pinned[ident] = generation
        try:
            yield
        finally:
            if outer is None:
                del pinned[ident]
            else:
                pinned[ident] = outer

    def _publish(self, config: Dict[str, Any], build: Callable[..., Any], *facts: Hashable) -> None:
        """Publish *config* with the program *build* compiles for *facts* already in it.

        A build that raises publishes nothing: the published generation stays.
        """
        generation = _Generation(config)
        with self._pinning(generation):
            self.program(build, *facts)
        self._generation = generation

    def _trying(self, config: Dict[str, Any]) -> contextlib.AbstractContextManager:
        """Run this thread's launches inside the block under *config*, unpublished.

        What a tuning sweep times a candidate with: every other thread keeps launching the
        published generation, and the candidate is published only if the sweep assigns it
        to ``config`` afterwards.
        """
        return self._pinning(_Generation(config))

    def program(self, build: Callable[..., Any], *facts: Hashable) -> Any:
        """The program of the generation this launch runs that *build* compiles for *facts*.

        ``build(config, *facts)`` compiles it on the generation's first launch with these
        facts: *config* is the generation's configuration, and *facts* are what the launch
        decides beyond it, such as a row count the kernel was not constructed for. What
        *build* reads is its arguments and what the kernel was constructed with, so the
        program is held by the code of *build* and *facts*: a function defined per launch
        shares its holder's programs. A build that raises leaves the generation as it was.

        Returns:
            What *build* returned: the compiled program, or the programs of one launch.
        """
        pinned = self._pinned
        generation = (pinned and pinned.get(threading.get_ident())) or self._generation
        programs = generation.programs
        # By the identity of the code: hashing a code object reads its whole body, and the
        # code of a function lives as long as the function or module defining it.
        key = (id(build.__code__), *facts)
        program = programs.get(key)
        if program is None:
            with generation.building:
                program = programs.get(key)
                if program is None:
                    program = build(generation.config, *facts)
                    from tilelang.jit.kernel import JITKernel

                    if isinstance(program, JITKernel):
                        program = CompiledProgram(program)
                    # A kernel launched with facts that keep changing, such as the row count
                    # of a growing batch, compiles a program each; the oldest goes once this
                    # many are held.
                    held = 64
                    if len(programs) >= held:
                        programs.pop(next(iter(programs)))
                    programs[key] = program
        return program

    # Set True when this kernel's integer tensor inputs are data or masks, so
    # autotuning may generate them from ``randint(-2, 3)``. A kernel whose
    # integer values decide how much work runs overrides
    # ``autotune_supply_prog`` instead; left False, autotuning refuses.
    autotune_accepts_random_int_inputs: bool = False

    # The device types this implementation runs on. With ``supported_archs`` it states where
    # the implementation is available; selection filters on it before asking ``refusal``.
    # An implementation declaring ``supported_archs`` names one type: an architecture number
    # compares only within one device type.
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
        A call that names no device runs on the accelerator's current one; without an
        accelerator its device is unknown, and no implementation serves it.
        """
        from tileops.utils.device import current_device

        device = getattr(call, "device", None) or current_device()
        if device is None:
            return "the call names no device, and this process has no accelerator"
        if device.type not in cls.devices:
            return f"runs on {sorted(cls.devices)}, not {device.type}"
        archs = cls.supported_archs
        if archs is not None and call.arch not in archs:
            return f"built for architectures {sorted(archs)}, device reports {call.arch}"
        return None

    @classmethod
    def entry_for(cls, call: Any) -> Entry:
        """How to build this class for *call*, and what makes two builds one entry.

        The identity is the construction arguments, so two calls that would compile the
        same kernel share an entry and none reuses one compiled for different arguments.
        The thunk runs only on a cache miss.

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

        Read from ``device_index``, the device of this class's one device type the op
        handed over — not from whichever device happens to be current, which need not be the
        one the input lives on. The op layer performs no architecture check of its own;
        selection filters an interface's implementations by availability first.

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
        from tileops.backend.dispatch import device_facts_of

        (kind,) = cls.devices
        arch = device_facts_of(torch.device(kind, self.device_index)).arch
        if arch not in cls.supported_archs:
            where = (
                "the current device" if self.device_index is None else f"{kind}:{self.device_index}"
            )
            raise ValueError(
                f"{cls.__name__} is built for architectures "
                f"{sorted(cls.supported_archs)}, but {where} reports {arch}"
            )

    def _require_devices(self, **tensors: Optional[torch.Tensor]) -> None:
        """Raise unless every named tensor is on a device type this class's ``devices`` names.

        Selection already keeps an op's calls on those devices; this guards a kernel called
        directly. An ``optional: true`` input the call did not pass arrives as ``None`` and is
        skipped.
        """
        devices = type(self).devices
        for name, tensor in tensors.items():
            if tensor is not None and tensor.device.type not in devices:
                raise ValueError(
                    f"{type(self).__name__} runs on {sorted(devices)}; got {name} on "
                    f"{tensor.device}. Another target's backend serves other devices."
                )

    def init_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Set ``self.config``: *config*'s value for each key it states, the default for the rest.

        Tuning is not a construction step: ``request_tune()`` puts the kernel in tuned mode
        once it exists.
        """
        if config is not None:
            self.config = {
                k: config[k] if config.get(k) is not None else v
                for k, v in self.default_config.items()
            }
        else:
            self.config = self.default_config

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
        pinned = self.__dict__.get("_pinned")
        if pinned is None:
            # Threads pinning for the first time together keep one table.
            pinned = self.__dict__.setdefault("_pinned", {})
        ident = threading.get_ident()
        if ident in pinned:
            result = self.forward(*args, **kwargs)
        else:
            # A kernel that never set a configuration launches with no generation.
            pinned[ident] = self.__dict__.get("_generation")
            try:
                result = self.forward(*args, **kwargs)
            finally:
                del pinned[ident]
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

    def tuning_candidates(
        self, configs: Sequence[Dict[str, Any]], untuned: Optional[Dict[str, Any]] = None
    ) -> list[dict]:
        """Return *configs* with the untuned config added when none of them is it.

        Tuning keeps the fastest config it times, so a search space that leaves out the
        config a kernel runs untuned can make the tuned kernel slower than the untuned
        one. Every tuning loop times what this returns.

        Args:
            configs: The candidates. Each names the same keys.
            untuned: The config the kernel runs untuned, cut to the candidates' keys;
                ``default_config`` when omitted.

        Returns:
            A new list: *configs*, then the untuned config if it is not among them.

        Raises:
            ValueError: The candidates name different keys, or *untuned* lacks one.
        """
        candidates = [dict(config) for config in configs]
        if not candidates:
            return candidates
        keys = list(candidates[0])
        if any(set(config) != set(keys) for config in candidates):
            raise ValueError(
                f"{type(self).__name__} tunes candidates naming different keys: "
                f"{sorted({tuple(sorted(config)) for config in candidates})}"
            )
        untuned = self.default_config if untuned is None else untuned
        missing = [key for key in keys if key not in untuned]
        if missing:
            raise ValueError(
                f"{type(self).__name__}'s untuned config {untuned} lacks the tuned keys {missing}"
            )
        default = {key: untuned[key] for key in keys}
        if default not in candidates:
            candidates.append(default)
        return candidates

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
            configs: Candidate configs handed to TileLang's autotuner. The seed
                config joins them when none of them is it.
            warmup: Warmup iterations per candidate.
            rep: Timed iterations per candidate.
            seed_config: The config *jit_kernel* runs untuned, which supplies the
                seeded JIT parameter values; ``default_config`` when omitted.
            supply_prog: Input supplier for the candidates. Defaults to
                ``autotune_supply_prog``, which is written against
                ``self.kernel``; a sub-kernel taking different inputs must pass
                its own, since the whole-kernel supplier would feed it the
                wrong ones. Pass ``None`` for no supplier. Its fingerprint
                (``tileops.kernels.autotune.supply_fingerprint``) is part of the
                key of the stored result, so a changed supplier is tuned again.

        Returns:
            The tuned kernel, carrying the winning ``config`` and its measured
            ``latency``.
        """
        # Pass do_not_specialize so TileLang excludes the seeded JIT parameters
        # from the cache key; without this, the seed values read as "already
        # tuned" and the benchmarking sweep is skipped (returning config=None).
        seeds = self._autotune_initial_kwargs(jit_kernel, seed_config)
        configs = self.tuning_candidates(configs, seed_config)
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
        from tileops.kernels.autotune import autotune

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

        tuned_kernel = self.tune_jit_kernel(
            self.kernel, self.autotune_configs, warmup=warmup, rep=rep
        )
        self.config = self._tuned_config(
            {
                self._AUTOTUNE_PARAM_ALIASES.get(key, key): value
                for key, value in tuned_kernel.config.items()
            }
        )

    def _tuned_config(self, tuned: Dict[str, Any]) -> Dict[str, Any]:
        """The configuration tuning publishes, from *tuned*, the values of the axes it swept.

        *tuned* itself by default; a kernel whose program reads more than the swept axes
        completes it here, so tuning publishes one whole configuration.
        """
        return tuned


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
