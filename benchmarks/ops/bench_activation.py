"""Benchmarks for unary activation ops covering risk points R2-R7.

Risk points covered:
- R2: Divmod overhead on small tensors (relu x fp16 x 4K)
- R3: JIT compilation cost (relu x 10 different N, cold vs warm)
- R4: DEFAULT_STRATEGY confirmation (relu x 3 strategies x 3 dtypes x 3 sizes)
- R5: Boundary auto-guard tail vectorization (relu x aligned/unaligned sizes)
- R6: threads=256 vs 128 for complex ops (relu/erf/mish x thread configs)
- R7: dtype-aware num_per_thread (relu x fp32/fp16 x npt=4/npt=8)

Profiles all 3 strategies (direct, explicit_parallel, register_copy) and
compares against PyTorch baseline to determine optimal DEFAULT_STRATEGY.
"""

from math import prod
from typing import Callable, Optional, Protocol

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, ManifestBenchmark
from tileops.kernels.elementwise import (
    ErfFwdKernel,
    MishFwdKernel,
    ReluFwdKernel,
    _make_unary_explicit,
)
from tileops.manifest import load_workloads
from tileops.ops.elementwise import (
    EluFwdOp,
    ErfFwdOp,
    GeluFwdOp,
    HardsigmoidFwdOp,
    HardswishFwdOp,
    HardtanhFwdOp,
    LeakyReluFwdOp,
    MishFwdOp,
    ReluFwdOp,
    SeluFwdOp,
    SiluFwdOp,
    SoftplusFwdOp,
)
from workloads.activation import ReluTest
from workloads.workload_base import FixtureBase

# ---------------------------------------------------------------------------
# LLM-realistic shapes (LLaMA-family defaults)
# ---------------------------------------------------------------------------

_SHAPES_2D = [
    (1, 4096),         # 4K  -- single-token small
    (1024, 4096),      # 4M  -- small transformer hidden dim
    (1024, 11008),     # ~11M -- non-pow2 LLaMA-7B intermediate
]
_SIZE_LABELS = ("4K", "4M", "11M")
_SHAPE_BY_LABEL = dict(zip(_SIZE_LABELS, _SHAPES_2D, strict=True))

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_UNARY_STRATEGIES = ("direct", "explicit_parallel", "register_copy")


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


class _UnaryWorkload(Protocol):
    """Structural type for unary benchmark workloads."""

    shape: tuple[int, ...]
    n_total: int
    dtype: torch.dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]: ...


class UnaryBenchCase:
    """Minimal test harness for unary benchmarks.

    Accepts either a shape tuple or a scalar element count. The tuple form
    is preferred so the original input geometry survives into the report.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...],
        dtype: torch.dtype,
    ):
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.n_total = prod(shape)
        self.dtype = dtype
        self.output_dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(*self.shape, device="cuda", dtype=self.dtype),)


class UnaryBenchmark(BenchmarkBase[_UnaryWorkload]):
    """Bandwidth-oriented benchmark for unary elementwise ops."""

    def calculate_flops(self) -> Optional[float]:
        return self.workload.n_total

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        out_dtype = getattr(t, "output_dtype", t.dtype)
        return t.n_total * (t.dtype.itemsize + out_dtype.itemsize)


# ---------------------------------------------------------------------------
# R2: Divmod overhead on small tensors
# ---------------------------------------------------------------------------


class R2SmallTensorFixture(FixtureBase):
    PARAMS = [
        ("shape, dtype", [
            pytest.param((1, 4096), torch.float16, marks=pytest.mark.smoke),
        ]),
    ]


@R2SmallTensorFixture
def test_r2_small_tensor_unary(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """R2: Benchmark divmod overhead on small tensors (unary relu, 4K)."""
    test = UnaryBenchCase(shape, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    n_total = prod(shape)
    op = ReluFwdOp(N_total=n_total, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.relu(x)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ---------------------------------------------------------------------------
# R3: JIT compilation cost
# ---------------------------------------------------------------------------


# R3 uses 1D shapes whose total element count varies per case; the goal is
# to measure JIT compile cost as a function of N, so we keep a 1D layout
# but record the shape tuple so the report stays consistent.
_R3_SHAPES = [
    (1_000,), (2_000,), (4_000,), (8_000,), (16_000,),
    (32_000,), (64_000,), (128_000,), (256_000,), (512_000,),
]


class R3JitFixture(FixtureBase):
    PARAMS = [
        ("shape", [
            pytest.param(s, marks=pytest.mark.full) for s in _R3_SHAPES
        ]),
    ]


@R3JitFixture
def test_r3_jit_compilation_cost(shape: tuple[int, ...]) -> None:
    """R3: Benchmark JIT compilation cost — relu with 10 different N values.

    Each test case creates a new kernel (different N -> different codegen),
    measuring both first-call (cold JIT) and subsequent (warm) latency.
    """
    import time

    dtype = torch.float16
    n_total = prod(shape)
    x = torch.randn(*shape, device="cuda", dtype=dtype)

    # Cold: time the first call including JIT compilation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    op = ReluFwdOp(N_total=n_total, dtype=dtype)
    _ = op(x)
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - t0) * 1000.0

    # Warm: profile subsequent calls
    test = UnaryBenchCase(shape, dtype)
    bm = UnaryBenchmark(test)
    warm_result = bm.profile(op, x)

    BenchmarkReport.record(
        "r3_jit_cost",
        {"shape": shape, "cold_ms": round(cold_ms, 2)},
        warm_result,
        tag="relu_jit",
    )


# ---------------------------------------------------------------------------
# R4: DEFAULT_STRATEGY confirmation (full matrix)
# ---------------------------------------------------------------------------


_R4_PARAMS = []
for size_label, _shape in _SHAPE_BY_LABEL.items():
    for dt in _DTYPES:
        for strategy in _UNARY_STRATEGIES:
            mark = pytest.mark.smoke if (
                size_label == "4M" and dt == torch.float16
                and strategy == "register_copy"
            ) else pytest.mark.full
            _R4_PARAMS.append(
                pytest.param(
                    _shape, size_label, dt, strategy,
                    id=f"{size_label}-{dt}-{strategy}",
                    marks=mark,
                )
            )


class R4StrategyFixture(FixtureBase):
    PARAMS = [
        ("shape, size_label, dtype, strategy", _R4_PARAMS),
    ]


@R4StrategyFixture
def test_r4_default_strategy_unary(
    shape: tuple[int, ...],
    size_label: str,
    dtype: torch.dtype,
    strategy: str,
) -> None:
    """R4: Benchmark all 3 unary strategies to confirm DEFAULT_STRATEGY."""
    test = UnaryBenchCase(shape, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    n_total = prod(shape)
    op = ReluFwdOp(N_total=n_total, dtype=dtype, strategy=strategy)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "r4_strategy_unary",
        {"shape": shape, "size_label": size_label,
         "dtype": dtype, "strategy": strategy},
        result,
        tag=f"relu_{strategy}",
    )


# Also benchmark gelu to verify strategy choice holds for transcendental ops
_R4_GELU_PARAMS = []
for size_label, _shape in _SHAPE_BY_LABEL.items():
    for dt in _DTYPES:
        for strategy in _UNARY_STRATEGIES:
            _R4_GELU_PARAMS.append(
                pytest.param(
                    _shape, size_label, dt, strategy,
                    id=f"gelu-{size_label}-{dt}-{strategy}",
                    marks=pytest.mark.full,
                )
            )


class R4GeluStrategyFixture(FixtureBase):
    PARAMS = [
        ("shape, size_label, dtype, strategy", _R4_GELU_PARAMS),
    ]


@R4GeluStrategyFixture
def test_r4_default_strategy_gelu(
    shape: tuple[int, ...],
    size_label: str,
    dtype: torch.dtype,
    strategy: str,
) -> None:
    """R4: Benchmark gelu strategies (transcendental op) to confirm DEFAULT_STRATEGY."""
    test = UnaryBenchCase(shape, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    n_total = prod(shape)
    op = GeluFwdOp(N_total=n_total, dtype=dtype, strategy=strategy)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "r4_strategy_gelu",
        {"shape": shape, "size_label": size_label,
         "dtype": dtype, "strategy": strategy},
        result,
        tag=f"gelu_{strategy}",
    )


# ---------------------------------------------------------------------------
# R5: Boundary auto-guard tail vectorization
# ---------------------------------------------------------------------------


# npt=8, threads=256 -> block_size=2048
_BLOCK_SIZE = 256 * 8
_R5_SHAPES = [
    ((_BLOCK_SIZE * 1000,), "aligned"),                   # perfectly aligned
    ((_BLOCK_SIZE * 1000 + 1,), "unaligned_plus_1"),      # minimal tail
    ((_BLOCK_SIZE * 1000 + 127,), "unaligned_plus_127"),  # large partial tail
]


class R5BoundaryFixture(FixtureBase):
    PARAMS = [
        ("shape, align_label", [
            pytest.param(s, label, marks=pytest.mark.full)
            for s, label in _R5_SHAPES
        ]),
    ]


@R5BoundaryFixture
def test_r5_boundary_guard(shape: tuple[int, ...], align_label: str) -> None:
    """R5: Benchmark boundary auto-guard tail vectorization.

    Compares aligned vs unaligned sizes under explicit_parallel strategy
    to detect performance cliff from boundary guard overhead.
    """
    dtype = torch.float16
    test = UnaryBenchCase(shape, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    n_total = prod(shape)
    op = ReluFwdOp(N_total=n_total, dtype=dtype, strategy="explicit_parallel")
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "r5_boundary",
        {"shape": shape, "align_label": align_label},
        result,
        tag=f"relu_{align_label}",
    )


# ---------------------------------------------------------------------------
# R6: threads=256 vs 128 for complex ops
# ---------------------------------------------------------------------------


_R6_KERNEL_OPS = [
    ("relu", ReluFwdKernel),
    ("erf", ErfFwdKernel),
    ("mish", MishFwdKernel),
]

_R6_THREADS = [128, 256]

_R6_PARAMS = []
for size_label, _shape in _SHAPE_BY_LABEL.items():
    for op_name, _ in _R6_KERNEL_OPS:
        for threads in _R6_THREADS:
            mark = pytest.mark.smoke if (
                size_label == "4M" and op_name == "relu" and threads == 256
            ) else pytest.mark.full
            _R6_PARAMS.append(
                pytest.param(
                    _shape, size_label, op_name, threads,
                    id=f"{op_name}-{size_label}-t{threads}",
                    marks=mark,
                )
            )


class R6ThreadsFixture(FixtureBase):
    PARAMS = [
        ("shape, size_label, op_name, threads", _R6_PARAMS),
    ]


_R6_KERNEL_MAP = {name: cls for name, cls in _R6_KERNEL_OPS}


@R6ThreadsFixture
def test_r6_threads_comparison(
    shape: tuple[int, ...],
    size_label: str,
    op_name: str,
    threads: int,
) -> None:
    """R6: Benchmark threads=256 vs 128 for simple and complex ops.

    Complex ops (erf, mish) may benefit from fewer threads (128) due to
    higher register pressure. Simple ops (relu) should prefer 256.

    Builds kernels directly via _make_unary_explicit to ensure block_size
    is baked with the requested threads/npt at build time, not overridden
    after construction.
    """
    dtype = torch.float16
    dtype_str = "float16"
    test = UnaryBenchCase(shape, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    n_total = prod(shape)
    npt = 8  # default for fp16
    kernel_cls = _R6_KERNEL_MAP[op_name]
    # Build kernel directly with the desired threads/npt so block_size is correct
    kernel_fn = _make_unary_explicit(
        n_total, dtype_str, kernel_cls.op_func, threads=threads, num_per_thread=npt,
    )
    # The explicit-parallel kernel expects a 1D contiguous tensor, so flatten
    # the (possibly multi-dim) input here. The shape tuple is still recorded
    # via ``BenchmarkReport.record(...)`` so the report carries the original
    # input geometry.
    flat_inputs = tuple(t.reshape(-1) for t in inputs)
    # Profile: call the JIT kernel with matching runtime args
    compiled = kernel_fn(threads, npt)
    result = bm.profile(compiled, *flat_inputs)
    BenchmarkReport.record(
        "r6_threads",
        {"shape": shape, "size_label": size_label,
         "op_name": op_name, "threads": threads},
        result,
        tag=f"{op_name}_t{threads}",
    )


# ---------------------------------------------------------------------------
# R7: dtype-aware num_per_thread
# ---------------------------------------------------------------------------


_R7_PARAMS = []
for dt, dt_label in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
    for npt in [4, 8]:
        _R7_PARAMS.append(
            pytest.param(
                dt, dt_label, npt,
                id=f"{dt_label}-npt{npt}",
                marks=pytest.mark.full,
            )
        )


class R7NptFixture(FixtureBase):
    PARAMS = [
        ("dtype, dtype_label, num_per_thread", _R7_PARAMS),
    ]


@R7NptFixture
def test_r7_dtype_npt(
    dtype: torch.dtype,
    dtype_label: str,
    num_per_thread: int,
) -> None:
    """R7: Benchmark dtype-aware num_per_thread.

    Compares npt=4 vs npt=8 for fp32 and fp16 to validate whether the
    current default (fp32->4, fp16->8) is optimal. If no difference,
    simplify to fixed npt=8.

    Builds kernels directly via _make_unary_explicit to ensure block_size
    is baked with the requested npt at build time.
    """
    shape = (1_000_000,)
    n_total = prod(shape)
    threads = 256
    dtype_str = "float32" if dtype == torch.float32 else "float16"
    test = UnaryBenchCase(shape, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    # Build kernel directly with the desired threads/npt so block_size is correct
    kernel_fn = _make_unary_explicit(
        n_total, dtype_str, ReluFwdKernel.op_func,
        threads=threads, num_per_thread=num_per_thread,
    )
    compiled = kernel_fn(threads, num_per_thread)
    result = bm.profile(compiled, *inputs)
    BenchmarkReport.record(
        "r7_dtype_npt",
        {"shape": shape, "dtype_label": dtype_label,
         "num_per_thread": num_per_thread},
        result,
        tag=f"relu_{dtype_label}_npt{num_per_thread}",
    )


# ---------------------------------------------------------------------------
# Baseline throughput benchmarks (existing, refined with LLaMA shapes)
# ---------------------------------------------------------------------------


_RELU_BENCH_PARAMS = [
    pytest.param(_SHAPES_2D[1], torch.float16, id="throughput-fp16"),
    pytest.param(_SHAPES_2D[1], torch.bfloat16, id="throughput-bf16"),
    pytest.param(_SHAPES_2D[1], torch.float32, id="baseline-fp32"),
]


@pytest.mark.parametrize("shape, dtype", _RELU_BENCH_PARAMS)
def test_relu_bench(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    n_total = prod(shape)
    # ``ReluTest`` (workloads) accepts a flat element count; the bench
    # harness still records the original shape tuple via ``record(...)``.
    test = ReluTest(n_total, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    op = ReluFwdOp(N_total=n_total, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.relu(x)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===========================================================================
# Manifest-driven per-op benchmarks for the elementwise_unary_activation
# family. Each op has its own ``test_*_bench`` function so the validator
# (``scripts/validate_manifest.py`` → ``check_l4_benchmark``) can match each
# ``load_workloads("<OpName>FwdOp")`` /
# ``ManifestBenchmark("<OpName>FwdOp", ...)`` call one-to-one. A shared
# ``_activation_params_from_manifest`` helper expands the manifest's
# ``input_shape`` / ``dtypes`` workload entries to pytest params; a shared
# ``_profile_and_record`` helper handles the profile + record pair.
# ===========================================================================


class _ActivationWorkload:
    """Minimal :class:`ShapeDtypeWorkload` adapter for activation ops."""

    def __init__(self, shape: tuple, dtype: torch.dtype):
        self.shape = shape
        self.dtype = dtype


def _activation_params_from_manifest(op_name: str) -> list:
    """Convert ``input_shape`` / ``dtypes`` workload entries to pytest params.

    The manifest's ``elementwise_unary_activation`` family declares its
    tensor input as ``input`` (PyTorch alignment) so workload entries use
    ``input_shape`` rather than ``x_shape``. The shared
    ``workloads_to_params`` helper assumes ``x_shape``; this local helper
    adapts ``load_workloads`` output for the activation family.
    """
    workloads = load_workloads(op_name)
    params = []
    for w in workloads:
        shape = tuple(w["input_shape"])
        label = w.get("label", "x".join(str(s) for s in shape))
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(shape, dtype, id=f"{label}-{dtype_str}"))
    return params


def _profile_and_record(
    op,
    bm: ManifestBenchmark,
    inputs: tuple,
    baseline_fn: Callable,
    params: dict,
) -> None:
    """Profile op and torch baseline against the same inputs and record both."""
    try:
        result = bm.profile(op, *inputs)
    except ValueError as exc:
        if "No configurations to tune" in str(exc):
            pytest.skip(f"Kernel does not support this shape: {exc}")
        raise
    BenchmarkReport.record(op, params, result, tag="tileops")

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, params, result_bl, tag="torch")


def _randn(shape: tuple, dtype: torch.dtype) -> tuple[torch.Tensor]:
    return (torch.randn(shape, device="cuda", dtype=dtype),)


# ---------------------------------------------------------------------------
# Per-op manifest-driven benches (12 unary activation ops)
# ---------------------------------------------------------------------------

_RELU_OP = "ReluFwdOp"


@pytest.mark.parametrize("shape, dtype", _activation_params_from_manifest(_RELU_OP))
def test_relu_manifest_bench(shape: tuple, dtype: torch.dtype) -> None:
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = ReluFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_RELU_OP, op, _ActivationWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, torch.relu,
        {"shape": shape, "dtype": dtype, "n_total": n_total},
    )


_GELU_OP = "GeluFwdOp"


@pytest.mark.parametrize("shape, dtype", _activation_params_from_manifest(_GELU_OP))
def test_gelu_manifest_bench(shape: tuple, dtype: torch.dtype) -> None:
    import torch.nn.functional as F
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = GeluFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_GELU_OP, op, _ActivationWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, F.gelu,
        {"shape": shape, "dtype": dtype, "n_total": n_total},
    )


_SILU_OP = "SiluFwdOp"


@pytest.mark.parametrize("shape, dtype", _activation_params_from_manifest(_SILU_OP))
def test_silu_manifest_bench(shape: tuple, dtype: torch.dtype) -> None:
    import torch.nn.functional as F
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = SiluFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_SILU_OP, op, _ActivationWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, F.silu,
        {"shape": shape, "dtype": dtype, "n_total": n_total},
    )


_HARDSWISH_OP = "HardswishFwdOp"


@pytest.mark.parametrize(
    "shape, dtype", _activation_params_from_manifest(_HARDSWISH_OP),
)
def test_hardswish_manifest_bench(shape: tuple, dtype: torch.dtype) -> None:
    import torch.nn.functional as F
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = HardswishFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_HARDSWISH_OP, op, _ActivationWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, F.hardswish,
        {"shape": shape, "dtype": dtype, "n_total": n_total},
    )


_HARDSIGMOID_OP = "HardsigmoidFwdOp"


@pytest.mark.parametrize(
    "shape, dtype", _activation_params_from_manifest(_HARDSIGMOID_OP),
)
def test_hardsigmoid_manifest_bench(shape: tuple, dtype: torch.dtype) -> None:
    import torch.nn.functional as F
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = HardsigmoidFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_HARDSIGMOID_OP, op, _ActivationWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, F.hardsigmoid,
        {"shape": shape, "dtype": dtype, "n_total": n_total},
    )


_MISH_OP = "MishFwdOp"


@pytest.mark.parametrize("shape, dtype", _activation_params_from_manifest(_MISH_OP))
def test_mish_manifest_bench(shape: tuple, dtype: torch.dtype) -> None:
    import torch.nn.functional as F
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = MishFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_MISH_OP, op, _ActivationWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, F.mish,
        {"shape": shape, "dtype": dtype, "n_total": n_total},
    )


_SELU_OP = "SeluFwdOp"


@pytest.mark.parametrize("shape, dtype", _activation_params_from_manifest(_SELU_OP))
def test_selu_manifest_bench(shape: tuple, dtype: torch.dtype) -> None:
    import torch.nn.functional as F
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = SeluFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_SELU_OP, op, _ActivationWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, F.selu,
        {"shape": shape, "dtype": dtype, "n_total": n_total},
    )


_LEAKY_RELU_OP = "LeakyReluFwdOp"


@pytest.mark.parametrize(
    "shape, dtype", _activation_params_from_manifest(_LEAKY_RELU_OP),
)
def test_leaky_relu_manifest_bench(shape: tuple, dtype: torch.dtype) -> None:
    import torch.nn.functional as F
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = LeakyReluFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_LEAKY_RELU_OP, op, _ActivationWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, lambda x: F.leaky_relu(x, 0.01),
        {"shape": shape, "dtype": dtype, "n_total": n_total},
    )


_ELU_OP = "EluFwdOp"


@pytest.mark.parametrize("shape, dtype", _activation_params_from_manifest(_ELU_OP))
def test_elu_manifest_bench(shape: tuple, dtype: torch.dtype) -> None:
    import torch.nn.functional as F
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = EluFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_ELU_OP, op, _ActivationWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, F.elu,
        {"shape": shape, "dtype": dtype, "n_total": n_total},
    )


_HARDTANH_OP = "HardtanhFwdOp"


@pytest.mark.parametrize(
    "shape, dtype", _activation_params_from_manifest(_HARDTANH_OP),
)
def test_hardtanh_manifest_bench(shape: tuple, dtype: torch.dtype) -> None:
    import torch.nn.functional as F
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = HardtanhFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_HARDTANH_OP, op, _ActivationWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, F.hardtanh,
        {"shape": shape, "dtype": dtype, "n_total": n_total},
    )


_SOFTPLUS_OP = "SoftplusFwdOp"


@pytest.mark.parametrize(
    "shape, dtype", _activation_params_from_manifest(_SOFTPLUS_OP),
)
def test_softplus_manifest_bench(shape: tuple, dtype: torch.dtype) -> None:
    import torch.nn.functional as F
    inputs = _randn(shape, dtype)
    n_total = inputs[0].numel()
    op = SoftplusFwdOp(N_total=n_total, dtype=dtype)
    bm = ManifestBenchmark(_SOFTPLUS_OP, op, _ActivationWorkload(shape, dtype))
    _profile_and_record(
        op, bm, inputs, F.softplus,
        {"shape": shape, "dtype": dtype, "n_total": n_total},
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
