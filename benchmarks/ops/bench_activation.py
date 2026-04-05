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
from typing import Optional

import pytest
import torch

from benchmarks.benchmark import BenchmarkBase, BenchmarkReport
from tileops.kernels.elementwise import (
    ErfKernel,
    MishKernel,
    ReluKernel,
    _make_unary_explicit,
)
from tileops.ops.elementwise import ErfOp, GeluOp, MishOp, ReluOp
from workloads.base import FixtureBase
from workloads.ops.activation import ReluTest

# ---------------------------------------------------------------------------
# LLM-realistic shapes (LLaMA-family defaults)
# ---------------------------------------------------------------------------

_SHAPES_2D = [
    (1, 4096),       # 4K  -- single-token small
    (1024, 4096),    # 4M  -- small transformer hidden dim
    (1024, 16384),   # 16M -- large (LLaMA 7B intermediate ~11008, rounded)
]
_SIZES = {
    "4K": prod(_SHAPES_2D[0]),
    "4M": prod(_SHAPES_2D[1]),
    "16M": prod(_SHAPES_2D[2]),
}

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_UNARY_STRATEGIES = ("direct", "explicit_parallel", "register_copy")


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


class UnaryBenchCase:
    """Minimal test harness for unary benchmarks."""

    def __init__(self, n_total: int, dtype: torch.dtype):
        self.n_total = n_total
        self.dtype = dtype
        self.output_dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        return (torch.randn(self.n_total, device="cuda", dtype=self.dtype),)


class UnaryBenchmark(BenchmarkBase):
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
        ("n_total, dtype", [
            pytest.param(4096, torch.float16, marks=pytest.mark.smoke),
        ]),
    ]


@R2SmallTensorFixture
def test_r2_small_tensor_unary(n_total: int, dtype: torch.dtype) -> None:
    """R2: Benchmark divmod overhead on small tensors (unary relu, 4K)."""
    test = UnaryBenchCase(n_total, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    op = ReluOp(N_total=n_total, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.relu(x)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ---------------------------------------------------------------------------
# R3: JIT compilation cost
# ---------------------------------------------------------------------------


_R3_SIZES = [
    1_000, 2_000, 4_000, 8_000, 16_000,
    32_000, 64_000, 128_000, 256_000, 512_000,
]


class R3JitFixture(FixtureBase):
    PARAMS = [
        ("n_total", [
            pytest.param(n, marks=pytest.mark.full) for n in _R3_SIZES
        ]),
    ]


@R3JitFixture
def test_r3_jit_compilation_cost(n_total: int) -> None:
    """R3: Benchmark JIT compilation cost — relu with 10 different N values.

    Each test case creates a new kernel (different N -> different codegen),
    measuring both first-call (cold JIT) and subsequent (warm) latency.
    """
    import time

    dtype = torch.float16
    x = torch.randn(n_total, device="cuda", dtype=dtype)

    # Cold: time the first call including JIT compilation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    op = ReluOp(N_total=n_total, dtype=dtype)
    _ = op(x)
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - t0) * 1000.0

    # Warm: profile subsequent calls
    test = UnaryBenchCase(n_total, dtype)
    bm = UnaryBenchmark(test)
    warm_result = bm.profile(op, x)

    BenchmarkReport.record(
        "r3_jit_cost",
        {"n_total": n_total, "cold_ms": round(cold_ms, 2)},
        warm_result,
        tag="relu_jit",
    )


# ---------------------------------------------------------------------------
# R4: DEFAULT_STRATEGY confirmation (full matrix)
# ---------------------------------------------------------------------------


_R4_PARAMS = []
for size_label, n in _SIZES.items():
    for dt in _DTYPES:
        for strategy in _UNARY_STRATEGIES:
            mark = pytest.mark.smoke if (
                size_label == "4M" and dt == torch.float16
                and strategy == "register_copy"
            ) else pytest.mark.full
            _R4_PARAMS.append(
                pytest.param(
                    n, size_label, dt, strategy,
                    id=f"{size_label}-{dt}-{strategy}",
                    marks=mark,
                )
            )


class R4StrategyFixture(FixtureBase):
    PARAMS = [
        ("n_total, size_label, dtype, strategy", _R4_PARAMS),
    ]


@R4StrategyFixture
def test_r4_default_strategy_unary(
    n_total: int,
    size_label: str,
    dtype: torch.dtype,
    strategy: str,
) -> None:
    """R4: Benchmark all 3 unary strategies to confirm DEFAULT_STRATEGY."""
    test = UnaryBenchCase(n_total, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    op = ReluOp(N_total=n_total, dtype=dtype, strategy=strategy)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "r4_strategy_unary",
        locals(),
        result,
        tag=f"relu_{strategy}",
    )


# Also benchmark gelu to verify strategy choice holds for transcendental ops
_R4_GELU_PARAMS = []
for size_label, n in _SIZES.items():
    for dt in _DTYPES:
        for strategy in _UNARY_STRATEGIES:
            _R4_GELU_PARAMS.append(
                pytest.param(
                    n, size_label, dt, strategy,
                    id=f"gelu-{size_label}-{dt}-{strategy}",
                    marks=pytest.mark.full,
                )
            )


class R4GeluStrategyFixture(FixtureBase):
    PARAMS = [
        ("n_total, size_label, dtype, strategy", _R4_GELU_PARAMS),
    ]


@R4GeluStrategyFixture
def test_r4_default_strategy_gelu(
    n_total: int,
    size_label: str,
    dtype: torch.dtype,
    strategy: str,
) -> None:
    """R4: Benchmark gelu strategies (transcendental op) to confirm DEFAULT_STRATEGY."""
    test = UnaryBenchCase(n_total, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    op = GeluOp(N_total=n_total, dtype=dtype, strategy=strategy)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "r4_strategy_gelu",
        locals(),
        result,
        tag=f"gelu_{strategy}",
    )


# ---------------------------------------------------------------------------
# R5: Boundary auto-guard tail vectorization
# ---------------------------------------------------------------------------


# npt=8, threads=256 -> block_size=2048
_BLOCK_SIZE = 256 * 8
_R5_SIZES = [
    (_BLOCK_SIZE * 1000, "aligned"),                   # perfectly aligned
    (_BLOCK_SIZE * 1000 + 1, "unaligned_plus_1"),      # minimal tail
    (_BLOCK_SIZE * 1000 + 127, "unaligned_plus_127"),  # large partial tail
]


class R5BoundaryFixture(FixtureBase):
    PARAMS = [
        ("n_total, align_label", [
            pytest.param(n, label, marks=pytest.mark.full)
            for n, label in _R5_SIZES
        ]),
    ]


@R5BoundaryFixture
def test_r5_boundary_guard(n_total: int, align_label: str) -> None:
    """R5: Benchmark boundary auto-guard tail vectorization.

    Compares aligned vs unaligned sizes under explicit_parallel strategy
    to detect performance cliff from boundary guard overhead.
    """
    dtype = torch.float16
    test = UnaryBenchCase(n_total, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    op = ReluOp(N_total=n_total, dtype=dtype, strategy="explicit_parallel")
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(
        "r5_boundary",
        {"n_total": n_total, "align_label": align_label},
        result,
        tag=f"relu_{align_label}",
    )


# ---------------------------------------------------------------------------
# R6: threads=256 vs 128 for complex ops
# ---------------------------------------------------------------------------


_R6_KERNEL_OPS = [
    ("relu", ReluKernel),
    ("erf", ErfKernel),
    ("mish", MishKernel),
]

_R6_THREADS = [128, 256]

_R6_PARAMS = []
for size_label, n in _SIZES.items():
    for op_name, _ in _R6_KERNEL_OPS:
        for threads in _R6_THREADS:
            mark = pytest.mark.smoke if (
                size_label == "4M" and op_name == "relu" and threads == 256
            ) else pytest.mark.full
            _R6_PARAMS.append(
                pytest.param(
                    n, size_label, op_name, threads,
                    id=f"{op_name}-{size_label}-t{threads}",
                    marks=mark,
                )
            )


class R6ThreadsFixture(FixtureBase):
    PARAMS = [
        ("n_total, size_label, op_name, threads", _R6_PARAMS),
    ]


_R6_KERNEL_MAP = {name: cls for name, cls in _R6_KERNEL_OPS}


@R6ThreadsFixture
def test_r6_threads_comparison(
    n_total: int,
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
    test = UnaryBenchCase(n_total, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    npt = 8  # default for fp16
    kernel_cls = _R6_KERNEL_MAP[op_name]
    # Build kernel directly with the desired threads/npt so block_size is correct
    kernel_fn = _make_unary_explicit(
        n_total, dtype_str, kernel_cls.op_func, threads=threads, num_per_thread=npt,
    )
    # Profile: call the JIT kernel with matching runtime args
    compiled = kernel_fn(threads, npt)
    result = bm.profile(compiled, *inputs)
    BenchmarkReport.record(
        "r6_threads",
        {"n_total": n_total, "size_label": size_label, "op_name": op_name, "threads": threads},
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
    n_total = 1_000_000
    threads = 256
    dtype_str = "float32" if dtype == torch.float32 else "float16"
    test = UnaryBenchCase(n_total, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    # Build kernel directly with the desired threads/npt so block_size is correct
    kernel_fn = _make_unary_explicit(
        n_total, dtype_str, ReluKernel.op_func,
        threads=threads, num_per_thread=num_per_thread,
    )
    compiled = kernel_fn(threads, num_per_thread)
    result = bm.profile(compiled, *inputs)
    BenchmarkReport.record(
        "r7_dtype_npt",
        {"dtype_label": dtype_label, "num_per_thread": num_per_thread},
        result,
        tag=f"relu_{dtype_label}_npt{num_per_thread}",
    )


# ---------------------------------------------------------------------------
# Baseline throughput benchmarks (existing, refined with LLaMA shapes)
# ---------------------------------------------------------------------------


_RELU_BENCH_PARAMS = [
    pytest.param(prod(_SHAPES_2D[1]), torch.float16, id="throughput-fp16"),
    pytest.param(prod(_SHAPES_2D[1]), torch.bfloat16, id="throughput-bf16"),
    pytest.param(prod(_SHAPES_2D[1]), torch.float32, id="baseline-fp32"),
]


@pytest.mark.parametrize("n_total, dtype", _RELU_BENCH_PARAMS)
def test_relu_bench(n_total: int, dtype: torch.dtype) -> None:
    test = ReluTest(n_total, dtype)
    bm = UnaryBenchmark(test)
    inputs = test.gen_inputs()

    op = ReluOp(N_total=n_total, dtype=dtype)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    def baseline_fn(x):
        return torch.relu(x)

    result_bl = bm.profile(baseline_fn, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
