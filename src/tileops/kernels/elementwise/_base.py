"""Elementwise kernel templates and strategy factories.

Template base classes, one per input shape:
- UnaryKernel: 1-input → 1-output (relu, sigmoid, abs, ...)
- BinaryKernel: 2-input → 1-output with N-dim broadcast (add, mul, ...)
- FusedGatedKernel: fused gate+activation (silu_and_mul, gelu_and_mul, ...)
- ParametricUnaryKernel: 1-input plus values baked in at build time, and the
  multi-input kernels built on it (where, clamp, masked_fill, prelu, lerp)

An op hands over the shapes its manifest entry declares; flattening, broadcasting and
restoring the output shape are each kernel's ``forward``. No kernel here uses shared
memory: Global → Register → Compute → Register → Global.

Three strategies pick the loop body at build time:
- direct: 1 element per thread
- explicit_parallel: N elements per thread via T.Parallel(threads, npt)
- register_copy: fragment load → compute → fragment store

Boundary checks are TileLang's, via LegalizeSafeMemoryAccess.

fp8 (e4m3fn, e5m2) accumulates in fp16 — direct fp8 arithmetic loses too much
precision for sigmoid/exp and friends. Defaults: num_per_thread=16 (128-bit
alignment) and explicit_parallel (register_copy is unreliable for fp8). Saturation
follows from the format: e4m3fn has no Inf, so a saturating ``T.Cast`` clamping to
±448.0 is what it can represent; e5m2 does have Inf, so the PrimFunc emits fp16 and
``forward`` casts to e5m2 without saturating, which keeps Inf and NaN."""

import math

import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

from ._broadcast import _broadcast_target as _broadcast_target
from ._broadcast import _expand_flat as _expand_flat
from ._broadcast import _flat, _is_contiguous_same_shape, coalesce_broadcast_dims
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
from ._dtype import _BINARY_FULL_DTYPES as _BINARY_FULL_DTYPES
from ._dtype import _BINARY_NO_BOOL_DTYPES as _BINARY_NO_BOOL_DTYPES
from ._dtype import _BITWISE_DTYPES, _FLOAT_DTYPES, _LOGICAL_DTYPES, _is_fp8
from ._dtype import _FP8_DTYPES as _FP8_DTYPES
from ._dtype import _clamp_to_dtype_range as _clamp_to_dtype_range
from ._dtype import _fp8_accum_dtype_str as _fp8_accum_dtype_str
from ._launch import _DEFAULT_THREADS, _LAUNCH_POLICY
from ._output import (
    ElementwiseOutputPlan,
    _get_fp8_output_dtypes,
    _store_binary_bool_as_int8,
    _store_unary_bool_as_int8,
    _wrap_fp8_accumulation,
)
from ._strategy import BinaryStrategyPolicy, UnaryStrategyPolicy

__all__ = [
    "BinaryKernel",
    "FloatPredicateKernel",
    "FloatUnaryKernel",
    "FusedGatedKernel",
    "LogicalUnaryKernel",
    "ParametricUnaryKernel",
    "UnaryKernel",
    "coalesce_broadcast_dims",
]


def _log_for_output_precision(value, wide):
    """Return ``log(wide)`` computed to the precision *value*'s dtype can keep.

    A narrow result cannot keep the precision difference, so it takes the fast log.
    """
    return T.log(wide) if value.dtype == "float32" else T.__log(wide)


class UnaryKernel(Kernel):
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

    #: Elementwise work is sized by the tensor shape: an integer operand is
    #: data, and a ``uint8`` ``cond``/``mask`` only selects a result.
    autotune_accepts_random_int_inputs: bool = True

    supported_archs: list[int] = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel", "register_copy"]
    #: The fragment strategy. Measured on H200 at 16M and 256M fp16 over fourteen float
    #: unaries, no body reads faster on the looping one.
    DEFAULT_STRATEGY = "register_copy"
    OUTPUT_DTYPE = None
    SUPPORTED_DTYPES = None

    @staticmethod
    def op_func(x):
        """Pointwise operation. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(self, N_total, dtype, config=None, tune=False):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        requested = (config or {}).get("strategy")
        self.strategy = UnaryStrategyPolicy(self.STRATEGIES, self.DEFAULT_STRATEGY).choose(
            requested=requested,
            input_dtype=dtype,
            declared_output_dtype=self.OUTPUT_DTYPE,
        )
        # A bool result goes through int8 wherever a fragment carries it: the store is
        # then an ordinary byte store, so nothing caps ``num_per_thread``.
        self.output_plan = ElementwiseOutputPlan.for_unary(dtype, self.OUTPUT_DTYPE, self.strategy)
        self.output_dtype = self.output_plan.logical_dtype
        self._fp8_output_dtype = self.output_plan.post_cast_dtype
        self._bool_via_int8 = self.output_plan.bool_via_int8
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _get_effective_op_func(self):
        """Return op_func wrapped for the output the kernel declares.

        A bool result is wrapped to land as ``_BOOL_STORAGE_DTYPE``; otherwise fp8
        accumulation is the only wrapping, via ``_wrap_fp8_accumulation``. When
        ``OUTPUT_DTYPE`` is set the kernel already outputs a non-fp8 type.
        """
        if self._bool_via_int8:
            return _store_unary_bool_as_int8(self.op_func)
        if self.OUTPUT_DTYPE is not None:
            return self.op_func
        return _wrap_fp8_accumulation(self.op_func, self.dtype, self.dtype_str, arity=1)

    def _build_kernel(self, strategy):
        cfg = self.default_config
        effective_op = self._get_effective_op_func()
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

    @property
    def default_config(self) -> dict:
        return _LAUNCH_POLICY.default_config(
            strategy=self.strategy,
            input_dtype=self.dtype,
            output_dtype=self.output_dtype,
            n_total=self.N_total,
            stores_bool=not self._bool_via_int8,
        ).as_dict()

    @property
    def autotune_configs(self) -> list[dict]:
        """Search space: threads in {128, 256, 512} x num_per_thread in {2, 4, 8}.

        Covers a range of occupancy/register-pressure tradeoffs for
        bandwidth-bound unary elementwise kernels. "strategy" is a
        build-time config key (it selects the kernel body, not a JIT
        parameter), so it is excluded from the sweep.
        """
        if _is_fp8(self.dtype):
            # fp8 needs 128-bit alignment: npt >= 16 for 1-byte elements
            threads_opts = [128, 256, 512]
            npt_opts = [16, 32]
        else:
            # fp16 / bf16 / fp32
            threads_opts = [128, 256, 512]
            npt_opts = [2, 4, 8]
        return [{"threads": t, "num_per_thread": n} for t in threads_opts for n in npt_opts]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Override to handle serialization failures in the TileLang autotuner.

        UnaryKernel JIT functions capture op_func closures that the autotuner
        subprocess cannot serialize.  Catch the error and fall back to the
        default config so that ``tune=True`` never crashes.
        """
        import warnings

        try:
            super().autotune(warmup=warmup, rep=rep)
        except (AssertionError, Exception) as exc:
            if "not serializable" in str(exc) or "pickle" in str(exc).lower():
                warnings.warn(
                    f"{self.__class__.__name__} autotuning failed "
                    f"(op_func is not serializable); falling back to "
                    f"default_config.",
                    stacklevel=2,
                )
                self.config = dict(self.default_config)
            else:
                raise

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        # Record the resolved strategy so ``self.config`` is the single source of
        # truth for it, whether the request was coerced, downgraded or autotuned.
        self.config["strategy"] = self.strategy
        # Pre-compile and cache the kernel function for the chosen config
        # to avoid JIT lookup overhead on every forward() call.
        cfg = self.config
        if self.strategy == "direct":
            self._compiled_fn = self.kernel(cfg["threads"])
        else:
            self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        self._require_cuda(x=x)
        result = self._compiled_fn(_flat(x))
        if self._bool_via_int8:
            # 0 and 1 in int8 are bool's own byte patterns, so this copies nothing.
            result = result.view(torch.bool)
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result.reshape(x.shape)


class BinaryKernel(Kernel):
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

    #: Elementwise work is sized by the tensor shape: an integer operand is
    #: data, and a ``uint8`` ``cond``/``mask`` only selects a result.
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
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.a_shape = tuple(a_shape)
        self.b_shape = tuple(b_shape)
        # Index representation is the kernel's own choice: collapse the
        # broadcast into the fewest dims the divmod chain has to walk.
        out_shape, coalesced_shape, a_strides, b_strides = coalesce_broadcast_dims(
            self.a_shape,
            self.b_shape,
        )
        self.out_shape = out_shape
        # What a caller gets back. ``out_shape`` is the coalesced index space, which
        # normalizes a 0-dim operand to one axis; two 0-dim operands broadcast to ``()``.
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
        self.strategy = BinaryStrategyPolicy(self.STRATEGIES, self.DEFAULT_STRATEGY).choose(
            requested=requested,
            input_dtype=dtype,
            declared_output_dtype=self.OUTPUT_DTYPE,
            same_shape=self._same_shape,
        )
        # As in UnaryKernel: a bool result rides in an int8 fragment where there is one.
        self.output_plan = ElementwiseOutputPlan.for_binary(dtype, self.OUTPUT_DTYPE, self.strategy)
        self.output_dtype = self.output_plan.logical_dtype
        self._fp8_output_dtype = self.output_plan.post_cast_dtype
        self._bool_via_int8 = self.output_plan.bool_via_int8
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _get_effective_op_func(self):
        """Return op_func wrapped for the output the kernel declares.

        A bool result is wrapped to land as ``_BOOL_STORAGE_DTYPE``; otherwise fp8
        accumulation is the only wrapping, via ``_wrap_fp8_accumulation`` (arity=2). When
        ``OUTPUT_DTYPE`` is set the kernel already outputs a non-fp8 type.
        """
        if self._bool_via_int8:
            return _store_binary_bool_as_int8(self.op_func)
        if self.OUTPUT_DTYPE is not None:
            return self.op_func
        return _wrap_fp8_accumulation(self.op_func, self.dtype, self.dtype_str, arity=2)

    def _build_kernel(self, strategy):
        cfg = self.default_config
        effective_op = self._get_effective_op_func()
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
                self.coalesced_shape,
                self.a_strides,
                self.b_strides,
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
                self.coalesced_shape,
                self.a_strides,
                self.b_strides,
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

    @property
    def default_config(self) -> dict:
        return _LAUNCH_POLICY.default_config(
            strategy=self.strategy,
            input_dtype=self.dtype,
            output_dtype=self.output_dtype,
            n_total=self.N_total,
            stores_bool=not self._bool_via_int8,
        ).as_dict()

    @property
    def autotune_configs(self) -> list[dict]:
        """Search space: threads in {128, 256, 512} x num_per_thread in {2, 4, 8}.

        Covers a range of occupancy/register-pressure tradeoffs for
        bandwidth-bound binary elementwise kernels.
        """
        if _is_fp8(self.dtype):
            # fp8 needs 128-bit alignment: npt >= 16 for 1-byte elements
            threads_opts = [128, 256, 512]
            npt_opts = [16, 32]
        else:
            # fp16 / bf16 / fp32
            threads_opts = [128, 256, 512]
            npt_opts = [2, 4, 8]
        return [{"threads": t, "num_per_thread": n} for t in threads_opts for n in npt_opts]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Override to handle known TileLang autotuner fallback failures.

        BinaryKernel JIT functions capture op_func closures that the autotuner
        subprocess cannot serialize.  Newer TileLang binders can also reject
        the autotune wrapper signature.  Catch these errors and fall back to
        the default config so that ``tune=True`` never crashes.
        """
        import warnings

        try:
            super().autotune(warmup=warmup, rep=rep)
        except (AssertionError, Exception) as exc:
            message = str(exc)
            if (
                "not serializable" in message
                or "pickle" in message.lower()
                or "missing a required argument" in message
            ):
                warnings.warn(  # noqa: B028
                    f"{self.__class__.__name__} autotuning failed "
                    f"({message}); falling back to default_config."
                )
                self.config = dict(self.default_config)
            else:
                raise

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        self.config["strategy"] = self.strategy
        # Pre-compile and cache the kernel function for the chosen config
        # to avoid JIT lookup overhead on every forward() call.
        cfg = self.config
        if self.strategy == "direct":
            self._compiled_fn = self.kernel(cfg["threads"])
        else:
            self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, a, b):
        self._require_cuda(a=a, b=b)
        result = self._compiled_fn(_flat(a), _flat(b))
        if self._bool_via_int8:
            # 0 and 1 in int8 are bool's own byte patterns, so this copies nothing.
            result = result.view(torch.bool)
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result.reshape(self.result_shape)


class FusedGatedKernel(Kernel):
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

    #: Elementwise work is sized by the tensor shape: an integer operand is
    #: data, and a ``uint8`` ``cond``/``mask`` only selects a result.
    autotune_accepts_random_int_inputs: bool = True

    supported_archs: list[int] = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel"]
    # Benchmark (H200, 4096x4096 fp16): explicit_parallel ~2x faster than direct
    #   silu_and_mul:       3.04 TB/s explicit vs 1.50 TB/s direct
    #   gelu_and_mul:       2.72 TB/s explicit vs 1.47 TB/s direct
    #   gelu_tanh_and_mul:  3.38 TB/s explicit vs 1.51 TB/s direct
    DEFAULT_STRATEGY = "explicit_parallel"
    SUPPORTED_DTYPES = None  # Subclass override to restrict input dtypes

    @staticmethod
    def activation_func(x):
        """Activation function. Must be overridden by subclass."""
        raise NotImplementedError

    def __init__(self, M, N, dtype, config=None, tune=False):
        super().__init__()
        if self.SUPPORTED_DTYPES is not None and dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.M = M
        self.N = N
        self.dtype = dtype
        self.strategy = (config or {}).get("strategy") or self.DEFAULT_STRATEGY
        if self.strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{self.strategy}', expected one of {self.STRATEGIES}"
            )
        self.output_plan = ElementwiseOutputPlan.for_fused_gated(dtype)
        self.output_dtype = self.output_plan.logical_dtype
        self._fp8_output_dtype = self.output_plan.post_cast_dtype
        self._kernel_output_dtype = (
            self.output_plan.kernel_output_dtype if self.output_plan.post_cast_dtype else None
        )
        self.kernel = self._build_kernel(self.strategy)
        self.init_config(config, tune)

    def _get_effective_op_func(self):
        """Return compound op ``(gate, value) -> activation(gate) * value``.

        Delegates to the shared ``_wrap_fp8_accumulation`` helper (arity=2)
        so that fp8 cast-in / cast-out logic is centralised.
        """
        act = self.activation_func

        def fused_op(gate, value):
            return act(gate) * value

        return _wrap_fp8_accumulation(fused_op, self.dtype, self.dtype_str, arity=2)

    def _build_kernel(self, strategy):
        cfg = self.default_config
        effective_op = self._get_effective_op_func()
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
        if _is_fp8(self.dtype):
            return _LAUNCH_POLICY.default_config(
                strategy=self.strategy,
                input_dtype=self.dtype,
                output_dtype=self.output_dtype,
                n_total=self.M * self.N,
            ).as_dict()
        if self.strategy == "explicit_parallel":
            # 128x8 keeps block_N=1024 but widens loads to 128-bit and lifts occupancy.
            # Only fp16/bf16 gain the width: fp32 npt=4 already saturates LDG.128, and
            # measured on H200 over 4096x4096 and 2048x14336 fp32 the two thread counts
            # tie, so it keeps the wider block.
            if self.dtype in (torch.float16, torch.bfloat16):
                return {"strategy": self.strategy, "threads": 128, "num_per_thread": 8}
            if self.dtype == torch.float32:
                return {"strategy": self.strategy, "threads": 256, "num_per_thread": 4}
        return _LAUNCH_POLICY.default_config(
            strategy=self.strategy,
            input_dtype=self.dtype,
            output_dtype=self.output_dtype,
            n_total=self.M * self.N,
        ).as_dict()

    @property
    def autotune_configs(self) -> list[dict]:
        """Search space: threads in {128, 256, 512} x num_per_thread in {2, 4, 8}.

        Covers a range of occupancy/register-pressure tradeoffs for
        bandwidth-bound fused gated elementwise kernels.
        """
        if _is_fp8(self.dtype):
            # fp8 needs 128-bit alignment: npt >= 16 for 1-byte elements
            threads_opts = [128, 256, 512]
            npt_opts = [16, 32]
        else:
            # fp16 / bf16 / fp32
            threads_opts = [128, 256, 512]
            npt_opts = [2, 4, 8]
        return [{"threads": t, "num_per_thread": n} for t in threads_opts for n in npt_opts]

    def autotune(self, warmup: int = 10, rep: int = 10) -> None:
        """Override to handle serialization failures in the TileLang autotuner.

        FusedGatedKernel JIT functions capture activation_func closures that
        the autotuner subprocess cannot serialize.  Catch the error and fall
        back to the default config so that ``tune=True`` never crashes.
        """
        import warnings

        try:
            super().autotune(warmup=warmup, rep=rep)
        except (AssertionError, Exception) as exc:
            if "not serializable" in str(exc) or "pickle" in str(exc).lower():
                warnings.warn(
                    f"{self.__class__.__name__} autotuning failed "
                    f"(activation_func is not serializable); falling back to "
                    f"default_config.",
                    stacklevel=2,
                )
                self.config = dict(self.default_config)
            else:
                raise

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        self.config["strategy"] = self.strategy
        # Pre-compile and cache the kernel function for the chosen config
        # to avoid JIT lookup overhead on every forward() call.
        cfg = self.config
        if self.strategy == "direct":
            self._compiled_fn = self.kernel(cfg["threads"])
        else:
            self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        self._require_cuda(x=x)
        result = self._compiled_fn(x)
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result


class FloatUnaryKernel(UnaryKernel):
    """Unary kernel base for float-only elementwise ops."""

    SUPPORTED_DTYPES = _FLOAT_DTYPES


class FloatPredicateKernel(FloatUnaryKernel):
    """Unary kernel base for float predicates with bool output.

    Takes the fragment strategy, which is what lets the bool result ride in an int8
    fragment instead of being stored as bool.
    """

    DEFAULT_STRATEGY = "register_copy"
    OUTPUT_DTYPE = torch.bool


class LogicalUnaryKernel(UnaryKernel):
    """Unary kernel base for logical predicates with bool output.

    As ``FloatPredicateKernel``: the fragment strategy carries the result as int8. A
    dtype the vectorized load cannot lower is still forced to the scalar path.
    """

    DEFAULT_STRATEGY = "register_copy"
    SUPPORTED_DTYPES = _LOGICAL_DTYPES
    OUTPUT_DTYPE = torch.bool


class _Uint8StorageUnaryKernel(UnaryKernel):
    """Unary kernel that computes on uint8 but accepts and returns bool.

    Reinterpreting bool storage as uint8 is this backend's requirement, not part
    of the op's semantics, so the reinterpretation happens here.
    """

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
    """Binary kernel that computes on uint8 but accepts and returns bool.

    Reinterpreting bool storage as uint8 is this backend's requirement, not part
    of the op's semantics, so the reinterpretation happens here. Callers pass
    bool tensors and get a bool result; a caller that already holds uint8 is
    passed through unchanged.
    """

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
        if self.OUTPUT_DTYPE is not None:
            return op_func
        return _wrap_fp8_accumulation(op_func, self.dtype, self.dtype_str, arity=2)


class ParametricUnaryKernel(Kernel):
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
        if dtype not in self.SUPPORTED_DTYPES:
            supported = ", ".join(str(dt) for dt in self.SUPPORTED_DTYPES)
            raise ValueError(
                f"{self.__class__.__name__} only supports dtypes [{supported}], got {dtype}"
            )
        self.N_total = N_total
        self.dtype = dtype
        # fp8 output handling
        if self._skip_fp8_output:
            self._fp8_output_dtype = None
        else:
            self._fp8_output_dtype, self.output_dtype = _get_fp8_output_dtypes(dtype)
        # Post-fp8 parameter processing (e.g. clamping scalars to output dtype range)
        self._post_init_params()
        # Build the kernel via the subclass-provided builder
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
        """Hook called after fp8 output dtypes are set, before kernel build.

        Override to clamp scalar parameters to the output dtype range (e.g.
        MaskedFill, NanToNum).
        """

    @property
    def default_config(self):
        if self.dtype == torch.float32:
            npt = 4
        elif _is_fp8(self.dtype):
            npt = self._NPT_FP8
        else:
            npt = self._NPT_NON_FP32
        return {"threads": self._DEFAULT_THREADS, "num_per_thread": npt}

    def init_config(self, config=None, tune=False):
        """Override to cache the compiled kernel function after config is set."""
        super().init_config(config, tune)
        cfg = self.config
        self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])

    def forward(self, x):
        self._require_cuda(x=x)
        result = self._compiled_fn(_flat(x))
        if self._fp8_output_dtype is not None:
            result = result.to(self._fp8_output_dtype)
        return result.reshape(x.shape)
