"""TileLang's autotuner, with the input supplier in the key of every result it persists.

TileLang keys a tuning result by the builder's source and arguments, the candidates and the
profiling settings; ``supply_prog`` is not among them. A result tuned on one supplier's inputs
is therefore reused after the supplier changes, from the in-process cache and from
``TILELANG_CACHE_DIR`` alike. :func:`autotune` adds :func:`supply_fingerprint` to that key.
Every TileOPs tuning goes through it.
"""

import dataclasses
import enum
import functools
import hashlib
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from tilelang.autotuner import autotune as tilelang_autotune
from tilelang.autotuner.tuner import AutoTuneImpl, AutoTuner

__all__ = ["autotune", "supply_fingerprint"]


@functools.cache
def _first_party(module_name: str) -> bool:
    """Whether the module named *module_name* is TileOPs' or the caller's own code.

    TileOPs is first-party wherever it is installed. Any other module is first-party unless it
    is built in or lives under the standard library or an installed-packages directory.
    """
    if module_name == "__main__" or module_name.partition(".")[0] == "tileops":
        return True
    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None)
    if path is None:
        return False
    resolved = Path(path).resolve()
    if {"site-packages", "dist-packages"} & set(resolved.parts):
        return False
    stdlib = Path(inspect.getfile(inspect)).resolve().parent
    return not resolved.is_relative_to(stdlib)


def _code_names(code: types.CodeType) -> set[str]:
    """The global and attribute names *code* and the functions nested in it read."""
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _code_names(const)
    return names


class _Fingerprint:
    """A digest of the code and data a walk from one supplier reaches.

    The walk visits values in a fixed order and records names and values, never an ``id()``
    or a ``hash()``, so the same supplier gives the same digest in every process.
    """

    # Attributes ``Kernel`` sets for the device a kernel runs on, the generation tuning
    # publishes, its config and programs, and the generation each running launch reads; none
    # decides the inputs, and the config is what tuning writes.
    _UNKEYED_ATTRIBUTES = frozenset(
        {"device_index", "_generation", "_pinned", "_tune_pending", "_tune_requested"}
    )

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._modules: set[str] = set()
        self._module_attributes: set[tuple[str, str]] = set()
        # Retain visited objects so temporary containers cannot reuse their identities.
        self._walked: dict[int, object] = {}

    def hexdigest(self) -> str:
        return self._digest.hexdigest()

    def _add(self, *tokens: object) -> None:
        for token in tokens:
            self._digest.update(str(token).encode())
            self._digest.update(b"\0")

    def _first_visit(self, obj: object) -> bool:
        if id(obj) in self._walked:
            return False
        self._walked[id(obj)] = obj
        return True

    def _digest_of(self, value: Any) -> str:
        """*value*'s own digest, for ordering it among the items of a set or dict."""
        child = _Fingerprint()
        child._modules = set(self._modules)
        child._module_attributes = set(self._module_attributes)
        child._walked = dict(self._walked)
        child.value(value)
        return child.hexdigest()

    def value(self, value: Any) -> None:
        """Record *value*: its contents for data, its code for a function or class."""
        if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
            self._add(type(value).__name__, repr(value))
        elif isinstance(value, enum.Enum):
            self._add("enum", type(value).__module__, type(value).__qualname__, value.name)
        elif isinstance(value, torch.dtype):
            self._add("dtype", value)
        elif isinstance(value, torch.device):
            # Which device of a type the inputs land on does not change what is timed.
            self._add("device", value.type)
        elif isinstance(value, torch.Tensor):
            self._add("tensor", value.dtype, tuple(value.shape), value.stride(), value.device.type)
            self._add(_tensor_contents(value))
        elif isinstance(value, (tuple, list, set, frozenset, dict)):
            self._add(type(value).__name__, len(value))
            if not self._first_visit(value):
                self._add("walked")  # contents recorded where it was first reached
            elif isinstance(value, (tuple, list)):
                for item in value:
                    self.value(item)
            elif isinstance(value, dict):
                # By key digest: insertion order can follow a set's per-process order.
                keys = sorted(((self._digest_of(k), k) for k in value), key=lambda pair: pair[0])
                for _, key in keys:
                    self.value(key)
                    self.value(value[key])
            else:
                # By digest: a set iterates in the per-process string-hash order.
                for digest in sorted(self._digest_of(item) for item in value):
                    self._add(digest)
        elif isinstance(value, functools.partial):
            self._add("partial")
            if self._first_visit(value):
                self.value(value.func)
                self.value(value.args)
                self.value(value.keywords)
        elif isinstance(value, types.MethodType):
            self._add("method")
            self.value(value.__func__)
            self.value(value.__self__)
        elif isinstance(value, types.FunctionType):
            self.function(value)
        elif isinstance(value, types.CodeType):
            self._add(
                "code",
                value.co_code,
                value.co_argcount,
                value.co_posonlyargcount,
                value.co_kwonlyargcount,
                value.co_flags,
            )
            self.value(
                (
                    value.co_consts,
                    value.co_names,
                    value.co_varnames,
                    value.co_freevars,
                    value.co_cellvars,
                )
            )
        elif isinstance(value, type):
            self.cls(value)
        elif isinstance(value, types.ModuleType):
            self._add("module", value.__name__)
            if _first_party(value.__name__):
                self.module(value.__name__)
        elif callable(value) and not hasattr(value, "__dict__"):
            # A builtin or C function: its name is all there is to read.
            self._add("callable", getattr(value, "__module__", None), _qualname(value))
        else:
            self.instance(value)

    def module(self, name: str) -> None:
        """Record the source of first-party module *name*, once."""
        if name in self._modules:
            return
        self._modules.add(name)
        try:
            source = inspect.getsource(sys.modules[name])
        except (KeyError, OSError, TypeError):
            source = ""
        self._add("source", name, hashlib.sha256(source.encode()).hexdigest())

    def function(self, fn: types.FunctionType) -> None:
        """Record *fn*'s name; for first-party code, its module, defaults, closure and globals."""
        self._add("function", fn.__module__, fn.__qualname__)
        if not self._first_visit(fn) or not _first_party(fn.__module__):
            return
        self.module(fn.__module__)
        self.value(fn.__code__)
        self.value(fn.__defaults__)
        self.value(fn.__kwdefaults__)
        names = _code_names(fn.__code__)
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or (), strict=True):
            self._add("free", name)
            try:
                self.dependency(cell.cell_contents, names)
            except ValueError:  # a cell not yet bound
                self._add("unbound")
        # What the function calls and reads: each global it names. A helper reached as a
        # module attribute is covered by that module's source.
        for name in sorted(names):
            if name in fn.__globals__:
                self._add("global", name)
                self.dependency(fn.__globals__[name], names)

    def dependency(self, value: Any, names: set[str]) -> None:
        """Include runtime attributes accessed through first-party module imports."""
        self.value(value)
        if isinstance(value, types.ModuleType) and _first_party(value.__name__):
            for name in sorted(names & vars(value).keys()):
                key = (value.__name__, name)
                if key in self._module_attributes:
                    continue
                self._module_attributes.add(key)
                self._add("module attribute", name)
                self.dependency(vars(value)[name], names)

    def cls(self, cls: type) -> None:
        """Record *cls*'s name; for first-party code, its module, bases, methods and other
        attributes."""
        self._add("class", cls.__module__, cls.__qualname__)
        if not self._first_visit(cls) or not _first_party(cls.__module__):
            return
        self.module(cls.__module__)
        for base in cls.__mro__[1:]:
            self.cls(base)
        for name in sorted(vars(cls)):
            attr = vars(cls)[name]
            if isinstance(attr, (staticmethod, classmethod)):
                attr = attr.__func__
            elif isinstance(attr, property):
                attr = attr.fget
            elif isinstance(attr, functools.cached_property):
                attr = attr.func
            if isinstance(attr, types.FunctionType):
                self.function(attr)
            else:
                self._add("class attribute", name)
                self.value(attr)

    def instance(self, obj: object) -> None:
        """Record *obj*'s class; for a first-party class, its attributes too."""
        from tileops.kernels.kernel_base import Kernel

        cls = type(obj)
        self.cls(cls)
        if not self._first_visit(obj) or not _first_party(cls.__module__):
            return
        if dataclasses.is_dataclass(obj):
            for field in dataclasses.fields(obj):
                self._add("field", field.name)
                self.value(getattr(obj, field.name))
            return
        attributes = dict(getattr(obj, "__dict__", {}))
        for klass in cls.__mro__:
            slots = vars(klass).get("__slots__", ())
            for name in (slots,) if isinstance(slots, str) else slots:
                if hasattr(obj, name):
                    attributes.setdefault(name, getattr(obj, name))
        for name in sorted(attributes):
            if isinstance(obj, Kernel) and name in self._UNKEYED_ATTRIBUTES:
                continue
            self._add("attribute", name)
            self.value(attributes[name])


def _tensor_contents(tensor: torch.Tensor) -> str:
    """A digest of an integer or boolean *tensor*'s values, or a marker in their place.

    Integer values (lengths, offsets, page tables) decide how much work a candidate runs;
    floating values do not change what is timed, and a kernel's float placeholders are often
    ``torch.empty``, whose values differ in each process.
    """
    if tensor.is_floating_point() or tensor.is_complex():
        return "floating values"
    try:
        data = tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
    except (RuntimeError, NotImplementedError):  # a meta tensor, or one without storage
        return "no contents"
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def _qualname(value: object) -> str:
    return getattr(value, "__qualname__", None) or type(value).__qualname__


def supply_fingerprint(supply_prog: Optional[Callable]) -> str:
    """Fingerprint supplier code and captured state for persistent autotune results.

    First-party function bytecode, module source and referenced runtime attributes
    participate. Containers have deterministic ordering; tensors include metadata and
    integer or boolean contents. Floating tensor contents are excluded because tuning
    placeholders may be uninitialized. Third-party objects participate by name.

    Kernel device indices, chosen configs and tuning flags do not participate.
    Object identities are used only for cycle detection, never as digest values.
    """

    fingerprint = _Fingerprint()
    fingerprint.value(supply_prog)
    return fingerprint.hexdigest()


@dataclasses.dataclass
class _SupplyKeyedAutoTune(AutoTuneImpl):
    """TileLang's tuner, whose cache key also holds the supplier's fingerprint."""

    supply_key: str = ""

    def get_tunner(self) -> AutoTuner:
        tuner = super().get_tunner()
        tilelang_key = tuner.generate_cache_key

        def generate_cache_key(parameters: dict, extra_parameters: dict) -> str:
            key = tilelang_key(parameters, extra_parameters)
            return hashlib.sha256(f"{key}:{self.supply_key}".encode()).hexdigest()

        # An instance attribute, as TileLang's own ``get_tunner`` overrides ``run``.
        tuner.generate_cache_key = generate_cache_key
        return tuner


def autotune(*, supply_prog: Optional[Callable] = None, **kwargs: Any) -> Callable:
    """``tilelang.autotuner.autotune``, keying each result by the fingerprint of *supply_prog*.

    Takes the arguments TileLang's ``autotune`` takes. The key is TileLang's key hashed with
    :func:`supply_fingerprint`, so the in-process and the on-disk cache both miss once the
    supplier changes, and both hit while it does not.
    """
    supply_key = supply_fingerprint(supply_prog)
    tilelang_tune = tilelang_autotune(supply_prog=supply_prog, **kwargs)

    def decorator(jit_kernel: Callable) -> _SupplyKeyedAutoTune:
        impl = tilelang_tune(jit_kernel)
        fields = {f.name: getattr(impl, f.name) for f in dataclasses.fields(AutoTuneImpl)}
        return _SupplyKeyedAutoTune(**fields, supply_key=supply_key)

    return decorator
