"""Elementwise kernel base classes."""

import math

import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from ._broadcast import (
    BroadcastPlan,
    _flat,
    _is_contiguous_same_shape,
    coalesce_broadcast_dims,
    register_broadcast_plan,
)
from ._builders import (
    _make_binary_direct,
    _make_binary_explicit,
    _make_binary_register_copy,
    _make_fused_gated_direct,
    _make_fused_gated_explicit,
    _make_unary_direct,
    _make_unary_explicit,
    _make_unary_regcopy,
)
from ._dtype import _BITWISE_DTYPES, _FLOAT_DTYPES, _LOGICAL_DTYPES, _is_fp8
from ._op_body import (
    _store_binary_bool_as_int8,
    _store_unary_bool_as_int8,
    _wrap_fp8_accumulation,
    register_op_func,
)
from ._policy import (
    _DEFAULT_THREADS,
    _get_fp8_output_dtypes,
    choose_binary_strategy,
    choose_unary_strategy,
    default_launch_config,
    elementwise_autotune_configs,
    elementwise_output_plan,
)

__all__ = [
    "BinaryKernel",
    "FloatPredicateKernel",
    "FloatUnaryKernel",
    "FusedGatedKernel",
    "LogicalUnaryKernel",
    "ParametricUnaryKernel",
    "UnaryKernel",
]


class _ElementwiseKernel(Kernel):
    """What every elementwise family shares: which dtypes it takes, and what it returns."""

    #: Input dtypes admitted; ``None`` admits every dtype the builder handles.
    SUPPORTED_DTYPES = None
    #: Whether bool results are stored through an int8 buffer.
    _bool_via_int8: bool = False
    #: Dtype the result is cast back to when the kernel writes a wider one.
    _fp8_output_dtype = None

    def _validate_supported_dtype(self, dtype) -> None:
        if self.SUPPORTED_DTYPES is None or dtype in self.SUPPORTED_DTYPES:
            return
        supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
        raise ValueError(f"{type(self).__name__} only supports dtypes [{supported}], got {dtype}")

    def _restore_output_dtype(self, result):
        if self._bool_via_int8:
            result = result.view(torch.bool)
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result


class _StrategyKernel(_ElementwiseKernel):
    """An elementwise family whose kernel body is picked by a named strategy."""

    @property
    def default_config(self) -> dict:
        return default_launch_config(
            strategy=self.strategy,
            input_dtype=self.dtype,
            output_dtype=self.output_dtype,
            n_total=self.N_total,
            stores_bool=not self._bool_via_int8,
        )

    @property
    def autotune_configs(self) -> list[dict]:
        return elementwise_autotune_configs(self.dtype, self.strategy)

    def init_config(self, config=None, tune=False) -> None:
        Kernel.init_config(self, config, tune)
        # Tuning returns only the axes it swept; the rest keeps its default.
        self.config = {**self.default_config, **self.config}
        self.config["strategy"] = self.strategy
        cfg = self.config
        if self.strategy == "direct":
            self._compiled_fn = self.kernel(cfg["threads"])
        else:
            self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])


class UnaryKernel(_StrategyKernel):
    """Template base class for unary elementwise kernels.

    Subclass must override ``op_func`` with a static method implementing
    the pointwise operation (e.g., relu, sigmoid).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype for input.
        config: Optional dict with "strategy", "threads" and "num_per_thread".
            "strategy" is one of "direct", "explicit_parallel",
            "register_copy"; it selects the kernel body at build time.
        tune: Whether to autotune (sweeps "threads" / "num_per_thread"
            within the resolved strategy).
    """

    #: Integer tensors are data; uint8 cond/mask tensors only select results.
    autotune_accepts_random_int_inputs: bool = True

    supported_archs: list[int] = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel", "register_copy"]
    #: Fragment copy is the measured default for float unaries.
    DEFAULT_STRATEGY = "register_copy"
    OUTPUT_DTYPE = None
    SUPPORTED_DTYPES = None

    @staticmethod
    def op_func(x):
        """Pointwise operation. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(self, N_total, dtype, config=None, tune=False):
        super().__init__()
        self._validate_supported_dtype(dtype)
        self.N_total = N_total
        self.dtype = dtype
        requested = (config or {}).get("strategy")
        self.strategy = choose_unary_strategy(
            requested=requested,
            strategies=self.STRATEGIES,
            default_strategy=self.DEFAULT_STRATEGY,
            input_dtype=dtype,
            declared_output_dtype=self.OUTPUT_DTYPE,
        )
        self.output_plan = elementwise_output_plan(
            dtype,
            self.OUTPUT_DTYPE,
            strategy=self.strategy,
            bool_storage=True,
        )
        self.output_dtype = self.output_plan.logical_dtype
        self._fp8_output_dtype = self.output_plan.post_cast_dtype
        self._bool_via_int8 = self.output_plan.bool_via_int8
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _get_effective_op_func(self):
        """The op body this kernel builds with, and the name that identifies it."""
        name = self._op_func_name()
        if self._bool_via_int8:
            return name, _store_unary_bool_as_int8(self.op_func)
        if self.OUTPUT_DTYPE is not None:
            return name, self.op_func
        return name, _wrap_fp8_accumulation(self.op_func, self.dtype, self.dtype_str, arity=1)

    def _op_func_name(self) -> str:
        """Name every input that changes the op body: see ``register_op_func``."""
        return f"{type(self).__qualname__}|{self.dtype_str}|{self.output_dtype_str}|{self.strategy}"

    def _build_kernel(self, strategy):
        cfg = self.default_config
        effective_op = register_op_func(*self._get_effective_op_func())
        if strategy == "direct":
            return _make_unary_direct(
                self.N_total,
                self.dtype_str,
                effective_op,
                output_dtype=self.output_dtype_str,
                threads=cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_unary_explicit(
                self.N_total,
                self.dtype_str,
                effective_op,
                output_dtype=self.output_dtype_str,
                threads=cfg["threads"],
                num_per_thread=cfg["num_per_thread"],
            )
        elif strategy == "register_copy":
            return _make_unary_regcopy(
                self.N_total,
                self.dtype_str,
                effective_op,
                output_dtype=self.output_dtype_str,
                threads=cfg["threads"],
                num_per_thread=cfg["num_per_thread"],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def output_dtype_str(self) -> str:
        return self.output_plan.kernel_output_dtype

    def forward(self, x):
        self._require_cuda(x=x)
        result = self._compiled_fn(_flat(x))
        return self._restore_output_dtype(result).reshape(x.shape)


class BinaryKernel(_StrategyKernel):
    """Template base class for binary elementwise kernels with N-dim broadcast.

    Subclass must override ``op_func`` with a static method implementing
    the pointwise operation (e.g., add, mul).

    Args:
        a_shape: Shape of input a.
        b_shape: Shape of input b. Broadcasts against *a_shape* under the
            PyTorch broadcasting rules.
        dtype: Torch dtype for input.
        config: Optional dict with "strategy", "threads" and "num_per_thread".
            "strategy" is one of "direct", "explicit_parallel",
            "register_copy". If "register_copy" is requested but inputs
            require broadcast, silently downgrades to "explicit_parallel".
        tune: Whether to autotune (sweeps "threads" / "num_per_thread"
            within the resolved strategy).

    Attributes:
        out_shape: Broadcast output shape.
        N_total: Total output elements.
    """

    #: Integer tensors are data; uint8 cond/mask tensors only select results.
    autotune_accepts_random_int_inputs: bool = True

    supported_archs: list[int] = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel", "register_copy"]
    DEFAULT_STRATEGY = "explicit_parallel"
    OUTPUT_DTYPE = None  # Subclass override for output dtype (e.g., torch.int8)
    SUPPORTED_DTYPES = None  # Subclass override to restrict input dtypes

    @staticmethod
    def op_func(a, b):
        """Pointwise operation. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(self, a_shape, b_shape, dtype, config=None, tune=False):
        super().__init__()
        self._validate_supported_dtype(dtype)
        self.a_shape = tuple(a_shape)
        self.b_shape = tuple(b_shape)
        out_shape, coalesced_shape, a_strides, b_strides = coalesce_broadcast_dims(
            self.a_shape,
            self.b_shape,
        )
        self.out_shape = out_shape
        self.result_shape = tuple(torch.broadcast_shapes(self.a_shape, self.b_shape))
        self.N_total = math.prod(out_shape)
        self.a_numel = math.prod(self.a_shape)
        self.b_numel = math.prod(self.b_shape)
        self.dtype = dtype
        self.coalesced_shape = coalesced_shape
        self.a_strides = a_strides
        self.b_strides = b_strides
        self._same_shape = _is_contiguous_same_shape(
            coalesced_shape,
            a_strides,
            b_strides,
        )
        requested = (config or {}).get("strategy")
        self.strategy = choose_binary_strategy(
            requested=requested,
            strategies=self.STRATEGIES,
            default_strategy=self.DEFAULT_STRATEGY,
            input_dtype=dtype,
            declared_output_dtype=self.OUTPUT_DTYPE,
            same_shape=self._same_shape,
        )
        self.output_plan = elementwise_output_plan(
            dtype,
            self.OUTPUT_DTYPE,
            strategy=self.strategy,
            bool_storage=True,
        )
        self.output_dtype = self.output_plan.logical_dtype
        self._fp8_output_dtype = self.output_plan.post_cast_dtype
        self._bool_via_int8 = self.output_plan.bool_via_int8
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _get_effective_op_func(self):
        """The op body this kernel builds with, and the name that identifies it."""
        name = self._op_func_name()
        if self._bool_via_int8:
            return name, _store_binary_bool_as_int8(self.op_func)
        if self.OUTPUT_DTYPE is not None:
            return name, self.op_func
        return name, _wrap_fp8_accumulation(self.op_func, self.dtype, self.dtype_str, arity=2)

    def _op_func_name(self) -> str:
        cls = type(self)
        out = self.output_plan.kernel_output_dtype
        return f"{cls.__module__}.{cls.__qualname__}|{self.dtype_str}|{out}|{self.strategy}"

    def _build_kernel(self, strategy):
        cfg = self.default_config
        effective_op = register_op_func(*self._get_effective_op_func())
        plan = register_broadcast_plan(
            BroadcastPlan(self.coalesced_shape, self.a_strides, self.b_strides)
        )
        kernel_output_dtype = (
            self.output_plan.kernel_output_dtype
            if self.OUTPUT_DTYPE is not None or self._bool_via_int8 or self._fp8_output_dtype
            else None
        )
        if strategy == "direct":
            return _make_binary_direct(
                self.N_total,
                self.dtype_str,
                effective_op,
                plan,
                self.a_numel,
                self.b_numel,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"],
            )
        elif strategy == "explicit_parallel":
            return _make_binary_explicit(
                self.N_total,
                self.dtype_str,
                effective_op,
                plan,
                self.a_numel,
                self.b_numel,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"],
                num_per_thread=cfg["num_per_thread"],
            )
        elif strategy == "register_copy":
            return _make_binary_register_copy(
                self.N_total,
                self.dtype_str,
                effective_op,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"],
                num_per_thread=cfg["num_per_thread"],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def forward(self, a, b):
        self._require_cuda(a=a, b=b)
        result = self._compiled_fn(_flat(a), _flat(b))
        return self._restore_output_dtype(result).reshape(self.result_shape)


class FusedGatedKernel(_StrategyKernel):
    """Template base class for fused gated elementwise kernels.

    Input layout: x has shape (M, 2*N) where x[:, :N] is the gate
    and x[:, N:] is the value. Output: y = activation(gate) * value.

    Subclass must override ``activation_func`` with a static method.

    Args:
        M: Number of rows.
        N: Half the column dimension (output width).
        dtype: Torch dtype.
        config: Optional dict with "strategy", "threads" and "num_per_thread".
            "strategy" is one of "direct", "explicit_parallel"; it selects
            the kernel body at build time.
        tune: Whether to autotune (sweeps "threads" / "num_per_thread"
            within the resolved strategy).
    """

    #: Integer tensors are data; uint8 cond/mask tensors only select results.
    autotune_accepts_random_int_inputs: bool = True

    supported_archs: list[int] = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel"]
    DEFAULT_STRATEGY = "explicit_parallel"
    SUPPORTED_DTYPES = None  # Subclass override to restrict input dtypes

    @staticmethod
    def activation_func(x):
        """Activation function. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(self, M, N, dtype, config=None, tune=False):
        super().__init__()
        self._validate_supported_dtype(dtype)
        self.M = M
        self.N = N
        self.dtype = dtype
        self.strategy = (config or {}).get("strategy") or self.DEFAULT_STRATEGY
        if self.strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{self.strategy}', expected one of {self.STRATEGIES}"
            )
        self.output_plan = elementwise_output_plan(dtype)
        self.output_dtype = self.output_plan.logical_dtype
        self._fp8_output_dtype = self.output_plan.post_cast_dtype
        self._kernel_output_dtype = (
            self.output_plan.kernel_output_dtype if self.output_plan.post_cast_dtype else None
        )
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _get_effective_op_func(self):
        """The op body this kernel builds with, and the name that identifies it."""
        act = self.activation_func

        def fused_op(gate, value):
            return act(gate) * value

        cls = type(self)
        name = f"{cls.__module__}.{cls.__qualname__}|{self.dtype_str}|{self.strategy}"
        return name, _wrap_fp8_accumulation(fused_op, self.dtype, self.dtype_str, arity=2)

    def _build_kernel(self, strategy):
        cfg = self.default_config
        effective_op = register_op_func(*self._get_effective_op_func())
        if strategy == "direct":
            return _make_fused_gated_direct(
                self.M,
                self.N,
                self.dtype_str,
                effective_op,
                threads=cfg["threads"],
                output_dtype=self._kernel_output_dtype,
            )
        elif strategy == "explicit_parallel":
            return _make_fused_gated_explicit(
                self.M,
                self.N,
                self.dtype_str,
                effective_op,
                cfg["threads"],
                cfg["num_per_thread"],
                output_dtype=self._kernel_output_dtype,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def default_config(self) -> dict:
        if self.strategy == "explicit_parallel":
            if self.dtype in (torch.float16, torch.bfloat16):
                return {"strategy": self.strategy, "threads": 128, "num_per_thread": 8}
            if self.dtype == torch.float32:
                return {"strategy": self.strategy, "threads": 256, "num_per_thread": 4}
        return default_launch_config(
            strategy=self.strategy,
            input_dtype=self.dtype,
            output_dtype=self.output_dtype,
            n_total=self.M * self.N,
        )

    def forward(self, x):
        self._require_cuda(x=x)
        result = self._compiled_fn(x)
        return self._restore_output_dtype(result)


class FloatUnaryKernel(UnaryKernel):
    """Unary kernel base for float-only elementwise ops."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES


class FloatPredicateKernel(FloatUnaryKernel):
    """Unary kernel base for float predicates with bool output."""

    DEFAULT_STRATEGY = "register_copy"
    OUTPUT_DTYPE = torch.bool


class LogicalUnaryKernel(UnaryKernel):
    """Unary kernel base for logical predicates with bool output."""

    DEFAULT_STRATEGY = "register_copy"
    SUPPORTED_DTYPES = _LOGICAL_DTYPES
    OUTPUT_DTYPE = torch.bool


class _Uint8StorageUnaryKernel(UnaryKernel):
    """Unary kernel that computes on uint8 but accepts and returns bool."""

    DEFAULT_STRATEGY = "register_copy"
    SUPPORTED_DTYPES = (torch.uint8,)

    @property
    def default_config(self) -> dict:
        return {"strategy": self.strategy, "threads": _DEFAULT_THREADS, "num_per_thread": 16}

    def forward(self, x):
        as_bool = x.dtype == torch.bool
        if as_bool:
            x = x.view(torch.uint8)
        result = super().forward(x)
        return result.view(torch.bool) if as_bool else result


class _Uint8StorageBinaryKernel(BinaryKernel):
    """Binary kernel that computes on uint8 but accepts and returns bool."""

    DEFAULT_STRATEGY = "explicit_parallel"
    SUPPORTED_DTYPES = (torch.uint8,)

    @property
    def default_config(self) -> dict:
        return {"strategy": self.strategy, "threads": _DEFAULT_THREADS, "num_per_thread": 16}

    def forward(self, a, b):
        as_bool = a.dtype == torch.bool
        if as_bool:
            a = a.view(torch.uint8)
        if b.dtype == torch.bool:
            b = b.view(torch.uint8)
        result = super().forward(a, b)
        return result.view(torch.bool) if as_bool else result


class _AlphaScaledBinaryKernel(BinaryKernel):
    """Shared base for ``y = a (op) alpha * b`` kernels.

    Subclasses set ``_combine`` to either addition or subtraction. ``alpha``
    is baked in at kernel construction time (one specialization per distinct
    ``alpha`` value, matching the lru_cache key shape used by the binary
    builders) so the kernel surface stays scalar-free. It is keyword-only so
    the positional ``(dtype, config, tune)`` tail stays uniform.
    """

    @staticmethod
    def _combine(a_scaled, b_scaled):
        raise NotImplementedError

    @staticmethod
    def op_func(a, b):
        raise NotImplementedError(
            "_AlphaScaledBinaryKernel uses a per-instance op_func built from "
            "alpha; use the kernel via __init__ instead of calling op_func."
        )

    def __init__(self, a_shape, b_shape, dtype, config=None, tune=False, *, alpha=1):
        # PyTorch rejects a floating alpha on an integral input; mirror that so
        # the kernel cannot silently truncate alpha through an fp32 cast.
        # Out-of-range integer alphas are NOT rejected — PyTorch wraps them via
        # the input dtype (uint8 alpha=-1 → 255), which T.cast reproduces.
        if dtype in _BITWISE_DTYPES and float(alpha) != float(int(alpha)):
            raise ValueError("alpha must be an integer when input dtype is integral")
        self._alpha = alpha
        super().__init__(a_shape, b_shape, dtype, config=config, tune=tune)

    def _alpha_op_func(self):
        """Build a binary op_func with ``alpha`` baked in.

        Floating inputs route the scalar multiply through fp32 to dodge
        narrow-type literal issues for fp16 / bf16; integer/bool inputs
        keep native integer arithmetic. Following PyTorch, the integral
        alpha is coerced via the input dtype, so out-of-range values
        wrap silently (uint8 alpha=-1 -> 255; bool alpha=2 -> low-bit).
        """
        alpha = self._alpha
        combine = type(self)._combine

        if alpha == 1:
            # Identity multiplier: skip the scalar multiply so the kernel
            # stays byte-identical to the pre-alpha fast path.
            def op_func(a, b):
                return combine(a, b)

            return op_func

        if self.dtype in _BITWISE_DTYPES:
            # Native integer arithmetic. Coerce alpha into the input dtype's
            # representable range in Python before T.cast: TVM rejects a
            # negative literal cast to an unsigned dtype, so reproduce
            # PyTorch's "scalar wraps via the input dtype" semantics here.
            if self.dtype is torch.bool:
                int_alpha = int(bool(alpha))
            else:
                info = torch.iinfo(self.dtype)
                width = info.max - info.min + 1
                int_alpha = int(alpha)
                if int_alpha < info.min or int_alpha > info.max:
                    int_alpha = ((int_alpha - info.min) % width) + info.min

            def op_func(a, b):
                scaled_b = T.cast(int_alpha, a.dtype) * b
                return combine(a, scaled_b)

            return op_func

        def op_func(a, b):
            scaled_b = T.cast(T.cast(alpha, "float32") * T.cast(b, "float32"), a.dtype)
            return combine(a, scaled_b)

        return op_func

    def _get_effective_op_func(self):
        """Inject the alpha-baked op_func into the parent build pipeline."""
        op_func = self._alpha_op_func()
        name = f"{self._op_func_name()}|alpha={self._alpha!r}"
        if self.OUTPUT_DTYPE is not None:
            return name, op_func
        return name, _wrap_fp8_accumulation(op_func, self.dtype, self.dtype_str, arity=2)


class ParametricUnaryKernel(_ElementwiseKernel):
    """Shared base for independent parametric elementwise kernels.

    Subclasses must define:
    - ``_builder_fn``: a ``@staticmethod`` returning the ``@lru_cache``-d
      builder function (e.g. ``_make_leaky_relu_kernel``).
    - ``_builder_args(self) -> tuple``: positional args for the builder
      *between* ``N_total`` and the common ``output_dtype, is_fp8, threads,
      npt`` suffix.

    Optional overrides:
    - ``_DEFAULT_THREADS``: class-level default thread count (default 256).
    - ``_NPT_FP8``: npt when dtype is fp8 but not fp32 (default 16).
    - ``_NPT_NON_FP32``: npt for non-fp32, non-fp8 (default 8).
    - ``_skip_fp8_output``: set to ``True`` if the kernel should *not*
      use ``_get_fp8_output_dtypes`` (e.g. Where, which is a pure selection
      op). When True, ``_fp8_output_dtype`` is ``None``.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = _FLOAT_DTYPES

    _DEFAULT_THREADS: int = 256
    _NPT_FP8: int = 16
    _NPT_NON_FP32: int = 8
    _skip_fp8_output: bool = False

    def __init__(self, N_total, dtype, config=None, tune=False):
        super().__init__()
        self._validate_supported_dtype(dtype)
        self.N_total = N_total
        self.dtype = dtype
        if self._skip_fp8_output:
            self._fp8_output_dtype = None
        else:
            self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        self._post_init_params()
        cfg = self.default_config
        builder_kwargs = {
            "is_fp8": _is_fp8(dtype),
            "threads": cfg["threads"],
            "npt": cfg["num_per_thread"],
        }
        if not self._skip_fp8_output:
            builder_kwargs["output_dtype"] = self.dtype_to_str(self.output_dtype)
        self.kernel = self._builder_fn()(
            *self._builder_positional_args(),
            **builder_kwargs,
        )
        self.init_config(config, tune)

    @staticmethod
    def _builder_fn():
        """Return the @lru_cache builder function for this kernel."""
        raise NotImplementedError

    def _builder_positional_args(self) -> tuple:
        """Return all positional args for the builder function.

        Default: ``(N_total, dtype_str, *_builder_args())``.
        Override if the builder has a different parameter order (e.g. PReLU).
        """
        return (self.N_total, self.dtype_str, *self._builder_args())

    def _builder_args(self) -> tuple:
        """Return op-specific positional args (after N_total, dtype_str)."""
        return ()

    def _post_init_params(self):
        """Hook called after fp8 output dtypes are set, before kernel build."""

    @property
    def default_config(self):
        if self.dtype == torch.float32:
            npt = 4
        elif _is_fp8(self.dtype):
            npt = self._NPT_FP8
        else:
            npt = self._NPT_NON_FP32
        return {"threads": self._DEFAULT_THREADS, "num_per_thread": npt}

    @property
    def autotune_configs(self) -> list[dict]:
        return elementwise_autotune_configs(self.dtype)

    def init_config(self, config=None, tune=False):
        Kernel.init_config(self, config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        self._require_cuda(x=x)
        result = self._compiled_fn(_flat(x))
        return self._restore_output_dtype(result).reshape(x.shape)
