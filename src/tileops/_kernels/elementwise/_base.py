"""Elementwise kernel base classes."""

import functools
import math

import tilelang.language as T
import torch

from tileops._kernels.elementwise._broadcast import (
    BroadcastPlan,
    broadcast_target,
    coalesce_broadcast_dims,
    coalesce_broadcast_operands,
    expand_flat,
    flat,
    is_contiguous_same_shape,
    register_broadcast_plan,
    row_broadcast_split,
)
from tileops._kernels.elementwise._builders import (
    make_binary_direct,
    make_binary_explicit,
    make_binary_register_copy,
    make_fused_gated_explicit,
    make_strided_multi_input,
    make_unary_direct,
    make_unary_regcopy,
)
from tileops._kernels.elementwise._dtype import (
    BITWISE_DTYPES,
    FLOAT_DTYPES,
    LOGICAL_DTYPES,
    torch_dtype_nbytes,
)
from tileops._kernels.elementwise._op_body import (
    GuardedOpFunc,
    LaneMaskedOpFunc,
    is_simple_unary,
    register_op_func,
    store_binary_bool_as_int8,
    store_unary_bool_as_int8,
)
from tileops._kernels.elementwise._policy import (
    TARGET_BLOCKS,
    choose_binary_strategy,
    choose_unary_strategy,
    default_launch_config,
    elementwise_autotune_configs,
    elementwise_output_plan,
)
from tileops._kernels.elementwise.call_spec import (
    AlphaScaledBinaryFwdInterface,
    AlphaScaledCall,
    BroadcastCall,
    ElementwiseCall,
    FusedGatedCall,
    FusedGatedFwdInterface,
)
from tileops._kernels.kernel_base import Entry, Kernel

__all__ = [
    "BinaryKernel",
    "FloatPredicateKernel",
    "FloatUnaryKernel",
    "FusedGatedKernel",
    "LogicalUnaryKernel",
    "MultiInputElementwiseKernel",
    "ScalarParamUnaryKernel",
    "TorchFallbackKernel",
    "UnaryKernel",
]


class _ElementwiseKernel(Kernel):
    """What every elementwise family shares: which dtypes it takes, and what it returns."""

    # Bytes each thread carries, which sets the default ``num_per_thread``. 16
    # is one vector load. A body the memory pipe waits on sets 32, keeping a
    # second load in flight while the first element's arithmetic runs.
    BYTES_PER_THREAD: int = 16
    # The floor of the grid-filling shrink. Four keeps a staged copy wide; a body
    # costing more than its own access lowers it, for the blocks that buys.
    MIN_NUM_PER_THREAD: int = 4
    # Input dtypes admitted; ``None`` admits every dtype the builder handles.
    SUPPORTED_DTYPES = None
    # Elements a ``register_copy`` thread carries at least, up to two 16-byte vectors,
    # before the grid-filling shrink; the dtype's one vector decides above it.
    REGISTER_COPY_NUM_PER_THREAD: int = 4
    # Whether bool results are stored through an int8 buffer.
    _bool_via_int8: bool = False

    # Extent of the row a broadcast block walks, or ``None`` where blocks are not
    # cut from rows. Only the binary families lay a grid out that way.
    row_broadcast_inner: int | None = None

    @classmethod
    def refusal(cls, call) -> "str | None":
        """Refuse an element type this family's programs do not serve.

        A caller told that nothing served the call learns which element types each class
        does serve. A class
        that computes a semantic dtype in another storage type states its own.
        """
        if cls.SUPPORTED_DTYPES is None or call.dtype in cls.SUPPORTED_DTYPES:
            return None
        supported = ", ".join(str(dt) for dt in cls.SUPPORTED_DTYPES)
        return f"serves dtypes [{supported}], not {call.dtype}"

    def _restore_output_dtype(self, result):
        return result.view(torch.bool) if self._bool_via_int8 else result

    def _resident_threads(self) -> int:
        """Threads the kernel's device holds at once."""
        target = self.build_target
        return target.sm_count * target.threads_per_sm

    _one_wave_in_l2 = False


class _StrategyKernel(_ElementwiseKernel):
    """An elementwise family whose kernel body is picked by a named strategy."""

    # Integer tensors are data; uint8 cond/mask tensors only select results.
    autotune_accepts_random_int_inputs: bool = True

    @property
    def default_config(self) -> dict:
        return default_launch_config(
            strategy=self.strategy,
            input_dtype=self.dtype,
            output_dtype=self.output_dtype,
            n_total=self.N_total,
            stores_bool=not self._bool_via_int8,
            bytes_per_thread=self.BYTES_PER_THREAD,
            min_num_per_thread=self.MIN_NUM_PER_THREAD,
            row_broadcast_inner=self.row_broadcast_inner,
            register_copy_num_per_thread=self.REGISTER_COPY_NUM_PER_THREAD,
            resident_threads=self._resident_threads(),
            one_wave_in_l2=self._one_wave_in_l2,
        )

    @property
    def autotune_configs(self) -> list[dict]:
        return elementwise_autotune_configs(
            self.dtype,
            self.strategy,
            self.BYTES_PER_THREAD,
            self.MIN_NUM_PER_THREAD,
            self.REGISTER_COPY_NUM_PER_THREAD,
        )

    def init_config(self, config=None) -> None:
        Kernel.init_config(self, config)
        self.config = self._tuned_config(self.config)
        self.program(self._compile)

    def _tuned_config(self, tuned: dict) -> dict:
        # Tuning sweeps some axes; the rest keeps its default.
        return {**self.default_config, **tuned, "strategy": self.strategy}

    def _compile(self, cfg: dict):
        """The program ``forward`` launches, compiled from *cfg*."""
        if self.strategy == "direct":
            return self.kernel(cfg["threads"])
        return self.kernel(cfg["threads"], cfg["num_per_thread"])


class UnaryKernel(_StrategyKernel):
    """Template base class for unary elementwise kernels.

    Subclass must override ``op_func`` with a static method implementing
    the pointwise operation (e.g., relu, sigmoid).

    Args:
        N_total: Total number of elements (flattened).
        dtype: Torch dtype for input.
        config: Optional dict with "threads" and "num_per_thread".
            "strategy" is chosen from the dtype: "direct" for bool, else
            "register_copy".
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    # Fragment copy is the default for float unaries.
    DEFAULT_STRATEGY = "register_copy"
    OUTPUT_DTYPE = None
    SUPPORTED_DTYPES = None
    # Headers under ``tileops/csrc`` whose functions ``op_func`` calls.
    CSRC_HEADERS: tuple[str, ...] = ()

    @staticmethod
    def op_func(x):
        """Pointwise operation. Must be overridden by subclass."""
        raise NotImplementedError

    # A cheaper form of ``op_func``, or ``None``: see ``GuardedOpFunc``.
    fast_func = None

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, N_total, dtype, config=None):
        super().__init__()
        self.N_total = N_total
        self.dtype = dtype
        self.strategy = choose_unary_strategy(
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
        self._bool_via_int8 = self.output_plan.bool_via_int8
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config)

    def _get_effective_op_func(self):
        """The op body this kernel builds with, and the name that identifies it."""
        name = self._op_func_name()
        if self._bool_via_int8:
            return name, store_unary_bool_as_int8(self.op_func)
        if self.fast_func is not None:
            return name, GuardedOpFunc(self.op_func, self.fast_func)
        return name, self.op_func

    @functools.cached_property
    def _one_wave_in_l2(self) -> bool:
        input_bytes = torch_dtype_nbytes(self.dtype)
        output_bytes = torch_dtype_nbytes(self.output_dtype)
        if input_bytes != output_bytes:
            return False
        if not is_simple_unary(self._get_effective_op_func()[1], self.dtype_str):
            return False
        return self.fits_l2_budget(
            self.N_total, input_bytes + output_bytes, self.build_target.l2_bytes
        )

    @staticmethod
    def fits_l2_budget(n_total: int, bytes_per_element: int, l2_bytes: int) -> bool:
        """Whether *n_total* elements of *bytes_per_element* fit three quarters of L2; the
        other quarter is left to other cache traffic."""
        return 4 * n_total * bytes_per_element <= 3 * l2_bytes

    def _op_func_name(self) -> str:
        """Name every input that changes the op body: see ``register_op_func``.

        Module included: ``_OP_FUNCS`` is one dict for the whole package, and two
        classes sharing a qualname would otherwise overwrite each other's body.
        """
        cls = type(self)
        return (
            f"{cls.__module__}.{cls.__qualname__}"
            f"|{self.dtype_str}|{self.output_dtype_str}|{self.strategy}"
        )

    def _build_kernel(self, strategy):
        effective_op = register_op_func(*self._get_effective_op_func())
        if strategy == "direct":
            return make_unary_direct(
                self.N_total, self.dtype_str, effective_op, output_dtype=self.output_dtype_str
            )
        elif strategy == "register_copy":
            return make_unary_regcopy(
                self.N_total,
                self.dtype_str,
                effective_op,
                output_dtype=self.output_dtype_str,
                headers=self.CSRC_HEADERS,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def output_dtype_str(self) -> str:
        return self.output_plan.kernel_output_dtype

    def forward(self, x):
        self._require_devices(x=x)
        result = self.program(self._compile)(flat(x))
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
        config: Optional dict with "threads" and "num_per_thread".
            "strategy" is chosen from the operands: "direct" for bool,
            "register_copy" for same-shape operands, else "explicit_parallel".

    Attributes:
        out_shape: Broadcast output shape.
        N_total: Total output elements.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    DEFAULT_STRATEGY = "explicit_parallel"
    OUTPUT_DTYPE = None  # Subclass override for output dtype (e.g., torch.int8)
    SUPPORTED_DTYPES = None  # Subclass override to restrict input dtypes
    # Sixteen elements a thread, capped at two 16-byte vectors, keep two loads of a 2-,
    # 4- or 8-byte operand in flight; a single load streams slower, except for an 8-byte
    # operand on a long launch (see ``default_launch_config``). A 1-byte operand's
    # sixteen are one vector. A body whose arithmetic outlasts that second 2-byte load
    # sets 8; one whose arithmetic outlasts any load sets 4.
    REGISTER_COPY_NUM_PER_THREAD = 16

    @staticmethod
    def op_func(a, b):
        """Pointwise operation. Must be overridden by subclass."""
        raise NotImplementedError

    # A cheaper form of ``op_func``, as a static method: see ``GuardedOpFunc``.
    fast_func = None
    # A bool body's form over two 16-bit lanes, by dtype (``lane_compare``): see
    # ``LaneMaskedOpFunc``.
    lane_mask_func = None
    # Headers under ``tileops/csrc`` whose functions ``op_func`` calls.
    CSRC_HEADERS: tuple[str, ...] = ()

    @classmethod
    def entry_for(cls, call: BroadcastCall) -> Entry:
        return call, lambda: cls(call.a_shape, call.b_shape, call.dtype)

    def __init__(self, a_shape, b_shape, dtype, config=None):
        super().__init__()
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
        self._same_shape = is_contiguous_same_shape(
            coalesced_shape,
            a_strides,
            b_strides,
        )
        split = row_broadcast_split(coalesced_shape, a_strides, b_strides)
        self.row_broadcast_inner = None if self._same_shape or split is None else split[1]
        self.strategy = choose_binary_strategy(
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
        self._bool_via_int8 = self.output_plan.bool_via_int8
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config)

    @property
    def stage_broadcast(self) -> bool:
        """Whether a broadcast block reads and writes through fragments.

        A body TileLang cannot vectorise -- a predicate, whose ``boolx<N>`` has no
        CUDA type -- drags the loads and stores to its own width when they share
        its loop. Staging keeps the copies wide, at the cost of a round trip
        through registers. A body that scalarises for another reason overrides
        this.
        """
        return self.output_dtype == torch.bool

    @property
    def bias_rows(self) -> "int | None":
        """Rows sharing a thread's bias vector, or ``None`` to disable reuse.

        Register-heavy bodies can opt out to preserve occupancy.
        """
        return 4

    def _get_effective_op_func(self):
        """The op body this kernel builds with, and the name that identifies it."""
        name = self._op_func_name()
        if self._bool_via_int8:
            body = store_binary_bool_as_int8(self.op_func)
            lanes = self.lane_mask_func(self.dtype_str) if self.lane_mask_func else None
            return name, body if lanes is None else LaneMaskedOpFunc(body, lanes)
        if self.fast_func is not None:
            return name, GuardedOpFunc(self.op_func, self.fast_func)
        return name, self.op_func

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
            if self.OUTPUT_DTYPE is not None or self._bool_via_int8
            else None
        )
        if strategy == "direct":
            return make_binary_direct(
                self.N_total,
                self.dtype_str,
                effective_op,
                plan,
                self.a_numel,
                self.b_numel,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"],
                headers=self.CSRC_HEADERS,
            )
        elif strategy == "explicit_parallel":
            return make_binary_explicit(
                self.N_total,
                self.dtype_str,
                effective_op,
                plan,
                self.a_numel,
                self.b_numel,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"],
                headers=self.CSRC_HEADERS,
                num_per_thread=cfg["num_per_thread"],
                stage=self.stage_broadcast,
                bias_rows=self.bias_rows,
                max_resident_threads=self._resident_threads(),
            )
        elif strategy == "register_copy":
            return make_binary_register_copy(
                self.N_total,
                self.dtype_str,
                effective_op,
                output_dtype=kernel_output_dtype,
                threads=cfg["threads"],
                headers=self.CSRC_HEADERS,
                num_per_thread=cfg["num_per_thread"],
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def forward(self, a, b):
        self._require_devices(a=a, b=b)
        result = self.program(self._compile)(flat(a), flat(b))
        return self._restore_output_dtype(result).reshape(self.result_shape)


class FusedGatedKernel(_StrategyKernel, FusedGatedFwdInterface):
    """Template base class for fused gated elementwise kernels.

    Input layout: x has shape (M, 2*N) where x[:, :N] is the gate
    and x[:, N:] is the value. Output: y = activation(gate) * value.

    Subclass must override ``activation_func`` with a static method.

    Args:
        M: Number of rows.
        N: Half the column dimension (output width).
        dtype: Torch dtype.
        config: Optional dict with "threads" and "num_per_thread".
            "strategy" is "explicit_parallel", the one body these kernels build.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    DEFAULT_STRATEGY = "explicit_parallel"
    SUPPORTED_DTYPES = None  # Subclass override to restrict input dtypes

    @staticmethod
    def activation_func(x):
        """Activation function. Must be overridden by subclass."""
        raise NotImplementedError

    @classmethod
    def entry_for(cls, call: FusedGatedCall) -> Entry:
        return call, lambda: cls(call.m, call.n, call.dtype)

    def __init__(self, M, N, dtype, config=None):
        super().__init__()
        self.M = M
        self.N = N
        self.dtype = dtype
        self.strategy = self.DEFAULT_STRATEGY
        self.output_dtype = dtype
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config)

    def _get_effective_op_func(self):
        """The op body this kernel builds with, and the name that identifies it."""
        act = self.activation_func

        def fused_op(gate, value):
            return act(gate) * value

        cls = type(self)
        name = f"{cls.__module__}.{cls.__qualname__}|{self.dtype_str}|{self.strategy}"
        return name, fused_op

    def _build_kernel(self, strategy):
        effective_op = register_op_func(*self._get_effective_op_func())
        if strategy == "explicit_parallel":
            return make_fused_gated_explicit(self.M, self.N, self.dtype_str, effective_op)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @property
    def default_config(self) -> dict:
        cfg = default_launch_config(
            strategy=self.strategy,
            input_dtype=self.dtype,
            output_dtype=self.output_dtype,
            n_total=self.M * self.N,
            bytes_per_thread=self.BYTES_PER_THREAD,
            min_num_per_thread=self.MIN_NUM_PER_THREAD,
            resident_threads=self._resident_threads(),
        )
        # A grid under TARGET_BLOCKS waits on latency, and each block it launches adds to
        # that wait, so it doubles its threads per block where that cuts the blocks a row
        # takes. 16: below it, each block's share of the row, not the block count, sets
        # the latency, and doubling that share costs more than the blocks it saves.
        min_blocks = 16
        width = cfg["threads"] * cfg["num_per_thread"]
        row_blocks = math.ceil(self.N / width)
        blocks = self.M * row_blocks
        if min_blocks <= blocks < TARGET_BLOCKS and math.ceil(self.N / (2 * width)) < row_blocks:
            cfg["threads"] *= 2
        return cfg

    def forward(self, x):
        self._require_devices(x=x)
        result = self.program(self._compile)(x)
        return self._restore_output_dtype(result)


class FloatUnaryKernel(UnaryKernel):
    """Unary kernel base for float-only elementwise ops."""

    SUPPORTED_DTYPES = FLOAT_DTYPES


class FloatPredicateKernel(FloatUnaryKernel):
    """Unary kernel base for float predicates with bool output."""

    DEFAULT_STRATEGY = "register_copy"
    OUTPUT_DTYPE = torch.bool


class LogicalUnaryKernel(UnaryKernel):
    """Unary kernel base for logical predicates with bool output."""

    DEFAULT_STRATEGY = "register_copy"
    SUPPORTED_DTYPES = LOGICAL_DTYPES
    OUTPUT_DTYPE = torch.bool


class Uint8StorageUnaryKernel(UnaryKernel):
    """Unary kernel that computes on uint8 but accepts and returns bool.

    It serves the bool calls of the key it is preferred over, which is where bool storage
    being one byte is this backend's fact rather than the op's.
    """

    DEFAULT_STRATEGY = "register_copy"
    SUPPORTED_DTYPES = (torch.uint8,)

    @classmethod
    def refusal(cls, call) -> "str | None":
        return None if call.dtype == torch.bool else f"serves bool operands, not {call.dtype}"

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, torch.uint8)

    def forward(self, x):
        as_bool = x.dtype == torch.bool
        if as_bool:
            x = x.view(torch.uint8)
        result = super().forward(x)
        return result.view(torch.bool) if as_bool else result


class Uint8StorageBinaryKernel(BinaryKernel):
    """Binary kernel that computes on uint8 but accepts and returns bool.

    It serves the bool calls of the key it is preferred over, which is where bool storage
    being one byte is this backend's fact rather than the op's.
    """

    DEFAULT_STRATEGY = "explicit_parallel"
    SUPPORTED_DTYPES = (torch.uint8,)

    @classmethod
    def refusal(cls, call) -> "str | None":
        return None if call.dtype == torch.bool else f"serves bool operands, not {call.dtype}"

    @classmethod
    def entry_for(cls, call: BroadcastCall) -> Entry:
        return call, lambda: cls(call.a_shape, call.b_shape, torch.uint8)

    def forward(self, a, b):
        as_bool = a.dtype == torch.bool
        if as_bool:
            a = a.view(torch.uint8)
        if b.dtype == torch.bool:
            b = b.view(torch.uint8)
        result = super().forward(a, b)
        return result.view(torch.bool) if as_bool else result


class AlphaScaledBinaryKernel(BinaryKernel, AlphaScaledBinaryFwdInterface):
    """Shared base for ``y = a (op) alpha * b`` kernels.

    Subclasses set ``_combine`` to either addition or subtraction. ``alpha``
    is baked in at kernel construction time (one specialization per distinct
    ``alpha`` value, matching the target_cache key shape used by the binary
    builders) so the kernel surface stays scalar-free. It is keyword-only so
    the positional ``(dtype, config)`` tail stays uniform.
    """

    @staticmethod
    def _combine(a_scaled, b_scaled):
        raise NotImplementedError

    @staticmethod
    def op_func(a, b):
        raise NotImplementedError(
            "AlphaScaledBinaryKernel uses a per-instance op_func built from "
            "alpha; use the kernel via __init__ instead of calling op_func."
        )

    @classmethod
    def entry_for(cls, call: AlphaScaledCall) -> Entry:
        return call, lambda: cls(call.a_shape, call.b_shape, call.dtype, alpha=call.alpha)

    def __init__(self, a_shape, b_shape, dtype, config=None, *, alpha=1):
        # The op's signature admits an integral input only an integral alpha the dtype
        # represents; for uint8 that includes the negative values PyTorch wraps
        # (alpha=-1 -> 255), which the integer path below reproduces.
        self._alpha = alpha
        super().__init__(a_shape, b_shape, dtype, config=config)

    def _alpha_op_func(self):
        """Build a binary op_func with ``alpha`` baked in.

        Floating inputs route the scalar multiply through fp32 to dodge
        narrow-type literal issues for fp16 / bf16; integer/bool inputs
        keep native integer arithmetic. Following PyTorch, the integral
        alpha is coerced via the input dtype (uint8 alpha=-1 -> 255; bool
        alpha=2 -> True).
        """
        alpha = self._alpha
        combine = type(self)._combine

        if alpha == 1:
            # Identity multiplier: skip the scalar multiply so the kernel
            # stays byte-identical to the pre-alpha fast path.
            def op_func(a, b):
                return combine(a, b)

            return op_func

        if self.dtype in BITWISE_DTYPES:
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
        return f"{self._op_func_name()}|alpha={self._alpha!r}", self._alpha_op_func()


class ScalarParamUnaryKernel(UnaryKernel):
    """A unary kernel whose op body is built from construction-time scalars.

    The scalars are bound into the body rather than passed to the PrimFunc, so
    every distinct value is its own specialization. Subclasses implement
    ``_make_op_func`` and ``_param_key``; ``op_func`` is never called.

    The body is always ``register_copy``.
    """

    SUPPORTED_DTYPES = FLOAT_DTYPES
    DEFAULT_STRATEGY = "register_copy"

    def _make_op_func(self):
        """Return the op body for this specialization's scalars."""
        raise NotImplementedError

    def _param_key(self) -> str:
        """Return the scalar values, spelled out for the op body's name."""
        raise NotImplementedError

    @staticmethod
    def op_func(x):
        raise NotImplementedError(
            "ScalarParamUnaryKernel builds its body from construction parameters; "
            "use the kernel through __init__ instead of calling op_func."
        )

    def _get_effective_op_func(self):
        return f"{self._op_func_name()}|{self._param_key()}", self._make_op_func()


class MultiInputElementwiseKernel(_ElementwiseKernel):
    """Several inputs broadcast to one shape, one output.

    Subclasses define ``_builder_fn`` (the ``@target_cache`` builder of the flat
    program), ``_builder_args`` (its positional arguments after ``N_total`` and the
    dtype string) and ``_element`` (the per-element body of the in-place program), and
    describe their forward arguments in ``INPUTS`` as ``(name, kind)`` pairs in
    PrimFunc parameter order:

    =========== =============================================================
    ``tile``    flattened; holds the output's element count on the flat
                program, its own on the in-place one
    ``mask``    the same, reinterpreted as uint8 -- this backend has no
                vectorised bool
    ``value``   a 0-dim scalar reshaped to a length-one buffer
    =========== =============================================================

    An argument the call leaves as ``None`` names no PrimFunc parameter.
    """

    supported_archs: list[int] = [80, 86, 89, 90]
    SUPPORTED_DTYPES = FLOAT_DTYPES
    INPUTS: tuple = ()
    # The only integer inputs are uint8 cond/mask tensors, which select results.
    autotune_accepts_random_int_inputs: bool = True

    @classmethod
    def entry_for(cls, call: ElementwiseCall) -> Entry:
        return call, lambda: cls(call.n_total, call.dtype)

    def __init__(self, N_total, dtype, config=None):
        super().__init__()
        self.N_total = N_total
        self.dtype = dtype
        self.output_dtype = dtype
        self.kernel = self._builder_fn()(self.N_total, self.dtype_str, *self._builder_args())
        self.init_config(config)

    @staticmethod
    def _builder_fn():
        """Return the ``@target_cache`` builder function for this kernel."""
        raise NotImplementedError

    def _builder_args(self) -> tuple:
        """Return the builder's positional args after ``N_total`` and the dtype."""
        return ()

    @property
    def default_config(self):
        # The body stages its per-element inputs into fragments and writes the result
        # back out of one, which is what `register_copy` names elsewhere in the family.
        cfg = default_launch_config(
            strategy="register_copy",
            input_dtype=self.dtype,
            output_dtype=self.output_dtype,
            n_total=self.N_total,
            bytes_per_thread=self.BYTES_PER_THREAD,
            min_num_per_thread=self.MIN_NUM_PER_THREAD,
            register_copy_num_per_thread=self.REGISTER_COPY_NUM_PER_THREAD,
            resident_threads=self._resident_threads(),
        )
        return {"threads": cfg["threads"], "num_per_thread": cfg["num_per_thread"]}

    @property
    def autotune_configs(self) -> list[dict]:
        return elementwise_autotune_configs(
            self.dtype,
            "register_copy",
            self.BYTES_PER_THREAD,
            self.MIN_NUM_PER_THREAD,
            self.REGISTER_COPY_NUM_PER_THREAD,
        )

    def init_config(self, config=None):
        Kernel.init_config(self, config)
        self.program(self._compile)

    def _compile(self, cfg: dict):
        """The program ``_run`` launches, compiled from *cfg*."""
        return self.kernel(cfg["threads"], cfg["num_per_thread"])

    def _element(self, *values):
        """The output element from one element of each operand the call passes, in ``INPUTS``
        order: what the in-place program computes per element."""
        raise NotImplementedError

    def _run(self, **tensors):
        """Run on the named inputs and restore the call's shape.

        The first ``tile`` input carries the semantic dtype: a bool one is
        computed in uint8 storage, and the result is viewed back to bool. When every
        operand holds the output's element count, the flat program runs on them; otherwise
        each operand is read in place at its own strides, 0 along a broadcast dimension,
        rather than expanded first.
        """
        self._require_devices(**tensors)
        present = [(name, kind) for name, kind in self.INPUTS if tensors.get(name) is not None]
        out_shape = broadcast_target(*(tensors[n] for n, kind in present if kind != "value"))
        as_bool = tensors[next(n for n, kind in present if kind == "tile")].dtype == torch.bool
        in_place = any(
            kind != "value" and tensors[name].numel() != self.N_total for name, kind in present
        )

        args = []
        for name, kind in present:
            tensor = tensors[name]
            if kind == "value":
                tensor = tensor.contiguous().reshape(1)
                args.append(tensor.view(torch.uint8) if as_bool else tensor)
                continue
            if kind == "mask" or as_bool:
                tensor = tensor.view(torch.uint8)
            args.append(flat(tensor.contiguous()) if in_place else expand_flat(tensor, out_shape))

        if in_place:
            shapes = [() if kind == "value" else tuple(tensors[n].shape) for n, kind in present]
            _, coalesced_shape, strides = coalesce_broadcast_operands(*shapes)
            numels = tuple(arg.numel() for arg in args)
            dtypes = tuple(self.dtype_to_str(arg.dtype) for arg in args)
            program = self.program(self._compile_strided, coalesced_shape, strides, numels, dtypes)
        else:
            program = self.program(self._compile)
        result = program(*args).reshape(out_shape)
        return result.view(torch.bool) if as_bool else result

    def _compile_strided(self, cfg: dict, coalesced_shape, strides, numels, dtypes):
        """The program for one in-place layout of the operands, compiled from *cfg*."""
        return make_strided_multi_input(
            self.N_total,
            self.dtype_to_str(torch.uint8 if self.dtype == torch.bool else self.output_dtype),
            dtypes,
            coalesced_shape,
            strides,
            numels,
            self._element,
            cfg["threads"],
            cfg["num_per_thread"],
        )


class TorchFallbackKernel(Kernel):
    """The answer for element types this backend compiles no program for.

    Torch computes it. The class takes no construction argument, so every call shares one
    entry, and it exposes no program, so tuning walks past it.
    """

    # The element types this class answers for; a subclass states its own.
    SUPPORTED_DTYPES: tuple = ()

    @staticmethod
    def handler(input: "torch.Tensor") -> "torch.Tensor":
        """The answer for one input. Must be overridden by subclass."""
        raise NotImplementedError

    @classmethod
    def refusal(cls, call) -> "str | None":
        if call.dtype in cls.SUPPORTED_DTYPES:
            return None
        served = ", ".join(str(dt) for dt in cls.SUPPORTED_DTYPES)
        return f"serves dtypes [{served}], not {call.dtype}"

    @classmethod
    def entry_for(cls, call) -> Entry:
        return (), lambda: cls()

    def forward(self, input: "torch.Tensor") -> "torch.Tensor":
        # Contiguous like the compiled path: the layout must not depend on the element type.
        return type(self).handler(input).contiguous()
