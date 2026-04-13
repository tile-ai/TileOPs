"""Benchmarks for multi-dim reduction paths across all six reduction families.

Covers 3D tensors with multi-dim and non-last-axis dim specifications,
both keepdim=True and keepdim=False variants, to surface performance
regressions and optimization opportunities in multi-dim reduction code.

Groups 1 (reduce), 3 (logical), 4 (vector norm), and 6 (logsumexp) use
true multi-dim reduction (e.g. dim=[0, 2]).

Groups 2 (argreduce) and 5 (cumulative) are architecturally single-dim:
  - Argreduce (argmax/argmin): accepts only scalar dim (int).
    We benchmark dim=0, dim=1, and dim=2 on 3D tensors.
  - Cumulative (cumsum/cumprod): only accepts (M, N, dtype) and always
    operates on dim=-1. We benchmark 3D-shaped inputs reshaped to 2D.
These two groups cannot provide true multi-dim reduction cases.

Shape conventions use LLaMA-family dimensions:
  - (batch=4, seq=128, hidden=4096): 7B inference context
  - (batch=2, seq=512, hidden=4096): 7B longer-context inference
"""

from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, BenchmarkWorkload
from workloads.workload_base import FixtureBase, WorkloadBase

# ===================================================================
# 1. Reduce (sum, mean, amax) — multi-dim
# ===================================================================


class ReduceMultidimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype, op_kind",
            [
                # 3D: (batch=4, seq=128, hidden=4096) — LLaMA-7B inference
                # dim=[0, 2] keepdim=False: per-position stats across batch+hidden
                pytest.param(
                    (4, 128, 4096), [0, 2], False, torch.float16, "sum",
                    id="sum-7B-dim02-nokeepdim",
                ),
                # dim=[0, 2] keepdim=True
                pytest.param(
                    (4, 128, 4096), [0, 2], True, torch.float16, "sum",
                    id="sum-7B-dim02-keepdim",
                ),
                # dim=[0, 1] keepdim=False: per-hidden reduction over batch+seq
                pytest.param(
                    (4, 128, 4096), [0, 1], False, torch.float16, "mean",
                    id="mean-7B-dim01-nokeepdim",
                ),
                # dim=[0, 1] keepdim=True
                pytest.param(
                    (4, 128, 4096), [0, 1], True, torch.bfloat16, "mean",
                    id="mean-7B-dim01-keepdim-bf16",
                ),
                # amax over batch+hidden
                pytest.param(
                    (4, 128, 4096), [0, 2], False, torch.float16, "amax",
                    id="amax-7B-dim02-nokeepdim",
                ),
                # Longer context: (batch=2, seq=512, hidden=4096) — LLaMA-7B
                pytest.param(
                    (2, 512, 4096), [0, 2], False, torch.float16, "sum",
                    id="sum-7B-longctx-dim02",
                ),
            ],
        ),
    ]


class ReduceMultidimTest(WorkloadBase):
    def __init__(
        self,
        shape: tuple,
        dim: list,
        keepdim: bool,
        dtype: torch.dtype,
        op_kind: str,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> object:
        x_f32 = x.float()
        ops = {
            "sum": lambda t: t.sum(dim=self.dim, keepdim=self.keepdim),
            "mean": lambda t: t.mean(dim=self.dim, keepdim=self.keepdim),
            "amax": lambda t: t.amax(dim=self.dim, keepdim=self.keepdim),
        }
        return ops[self.op_kind](x_f32).to(x.dtype)


class ReduceMultidimBenchmark(BenchmarkBase[BenchmarkWorkload]):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        total_elems = 1
        for s in t.shape:
            total_elems *= s
        return total_elems

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        total_elems = 1
        for s in t.shape:
            total_elems *= s
        # Output elements: product of kept dims
        out_elems = 1
        for i, s in enumerate(t.shape):
            if i not in t.dim:
                out_elems *= s
        return (total_elems + out_elems) * elem_bytes


def _make_reduce_op(dtype, op_kind, dim, keepdim):
    from tileops.ops.reduction.reduce import AmaxFwdOp, MeanFwdOp, SumFwdOp

    op_map = {"sum": SumFwdOp, "mean": MeanFwdOp, "amax": AmaxFwdOp}
    cls = op_map[op_kind]
    return cls(dtype=dtype, dim=dim, keepdim=keepdim)


@ReduceMultidimFixture
def test_reduce_multidim_bench(
    shape: tuple,
    dim: list,
    keepdim: bool,
    dtype: torch.dtype,
    op_kind: str,
) -> None:
    test = ReduceMultidimTest(shape, dim, keepdim, dtype, op_kind)
    bm = ReduceMultidimBenchmark(test)
    inputs = test.gen_inputs()

    op = _make_reduce_op(dtype, op_kind, dim, keepdim)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# 2. Argreduce (argmax, argmin) — non-last-axis dims on 3D tensor
#    ArgmaxFwdOp/ArgminFwdOp only accept scalar dim (int), not a list.
#    We cover dim=0, dim=1, and dim=2 on a 3D tensor.
# ===================================================================


class ArgreduceMultidimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype, op_kind",
            [
                # dim=0: reduce across batch — LLaMA-7B (batch=4, seq=128, hidden=4096)
                pytest.param(
                    (4, 128, 4096), 0, False, torch.float16, "argmax",
                    id="argmax-7B-dim0-nokeepdim",
                ),
                pytest.param(
                    (4, 128, 4096), 0, True, torch.bfloat16, "argmin",
                    id="argmin-7B-dim0-keepdim-bf16",
                ),
                # dim=1: reduce across seq — LLaMA-7B (batch=4, seq=128, hidden=4096)
                pytest.param(
                    (4, 128, 4096), 1, False, torch.float16, "argmin",
                    id="argmin-7B-dim1-nokeepdim",
                ),
                pytest.param(
                    (4, 128, 4096), 1, True, torch.bfloat16, "argmin",
                    id="argmin-7B-dim1-keepdim-bf16",
                ),
                # dim=2: reduce across hidden (last axis)
                pytest.param(
                    (4, 128, 4096), 2, False, torch.float16, "argmax",
                    id="argmax-7B-dim2-nokeepdim",
                ),
                pytest.param(
                    (4, 128, 4096), 2, True, torch.float16, "argmin",
                    id="argmin-7B-dim2-keepdim",
                ),
            ],
        ),
    ]


class ArgreduceMultidimTest(WorkloadBase):
    def __init__(
        self,
        shape: tuple,
        dim: int,
        keepdim: bool,
        dtype: torch.dtype,
        op_kind: str,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        if self.op_kind == "argmax":
            return x.argmax(dim=self.dim, keepdim=self.keepdim)
        return x.argmin(dim=self.dim, keepdim=self.keepdim)


class ArgreduceMultidimBenchmark(BenchmarkBase[BenchmarkWorkload]):
    def calculate_flops(self) -> Optional[float]:
        total_elems = 1
        for s in self.workload.shape:
            total_elems *= s
        return total_elems

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        total_elems = 1
        for s in t.shape:
            total_elems *= s
        # Output: int64 (8 bytes) for each position
        out_elems = total_elems // t.shape[t.dim]
        return total_elems * elem_bytes + out_elems * 8


def _make_argreduce_op(dtype, op_kind, dim, keepdim):
    from tileops.ops.reduction.argmax import ArgmaxFwdOp
    from tileops.ops.reduction.argmin import ArgminFwdOp

    cls = ArgmaxFwdOp if op_kind == "argmax" else ArgminFwdOp
    return cls(dtype=dtype, dim=dim, keepdim=keepdim)


@ArgreduceMultidimFixture
def test_argreduce_multidim_bench(
    shape: tuple,
    dim: int,
    keepdim: bool,
    dtype: torch.dtype,
    op_kind: str,
) -> None:
    test = ArgreduceMultidimTest(shape, dim, keepdim, dtype, op_kind)
    bm = ArgreduceMultidimBenchmark(test)
    inputs = test.gen_inputs()

    op = _make_argreduce_op(dtype, op_kind, dim, keepdim)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# 3. Logical reduce (any, all, count_nonzero) — multi-dim
# ===================================================================


class LogicalReduceMultidimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype, op_kind",
            [
                # dim=[0, 2] keepdim=False — LLaMA-7B (batch=4, seq=128, hidden=4096)
                pytest.param(
                    (4, 128, 4096), [0, 2], False, torch.float16, "any",
                    id="any-7B-dim02-nokeepdim",
                ),
                # dim=[0, 2] keepdim=True
                pytest.param(
                    (4, 128, 4096), [0, 2], True, torch.float16, "all",
                    id="all-7B-dim02-keepdim",
                ),
                # dim=[0, 1] — count_nonzero (no keepdim, matches torch semantics)
                pytest.param(
                    (4, 128, 4096), [0, 1], False, torch.int32, "count_nonzero",
                    id="cnt_nz-7B-dim01-i32",
                ),
                # dim=[0, 1] keepdim=True
                pytest.param(
                    (4, 128, 4096), [0, 1], True, torch.float16, "any",
                    id="any-7B-dim01-keepdim",
                ),
            ],
        ),
    ]


class LogicalReduceMultidimTest(WorkloadBase):
    def __init__(
        self,
        shape: tuple,
        dim: list,
        keepdim: bool,
        dtype: torch.dtype,
        op_kind: str,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.dtype in (torch.int32, torch.int64):
            x = torch.randint(-5, 6, self.shape, dtype=self.dtype, device="cuda")
        elif self.dtype == torch.bool:
            x = torch.randint(0, 2, self.shape, dtype=torch.bool, device="cuda")
        else:
            x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        if self.op_kind == "any":
            return x.bool().any(dim=self.dim, keepdim=self.keepdim)
        elif self.op_kind == "all":
            return x.bool().all(dim=self.dim, keepdim=self.keepdim)
        elif self.op_kind == "count_nonzero":
            return torch.count_nonzero(x, dim=self.dim).to(torch.int64)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class LogicalReduceMultidimBenchmark(BenchmarkBase[BenchmarkWorkload]):
    def calculate_flops(self) -> Optional[float]:
        total_elems = 1
        for s in self.workload.shape:
            total_elems *= s
        return total_elems

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        total_elems = 1
        for s in t.shape:
            total_elems *= s
        out_elems = 1
        dims = set(d % len(t.shape) for d in t.dim)
        for i, s in enumerate(t.shape):
            if i not in dims:
                out_elems *= s
        out_elem_bytes = 8 if t.op_kind == "count_nonzero" else 1
        return total_elems * elem_bytes + out_elems * out_elem_bytes


def _make_logical_op(dtype, op_kind, dim, keepdim):
    from tileops.ops.reduction.all_op import AllFwdOp
    from tileops.ops.reduction.any_op import AnyFwdOp
    from tileops.ops.reduction.count_nonzero import CountNonzeroFwdOp

    op_map = {"any": AnyFwdOp, "all": AllFwdOp, "count_nonzero": CountNonzeroFwdOp}
    cls = op_map[op_kind]
    # CountNonzeroFwdOp does not accept keepdim (always removes reduced dim)
    if op_kind == "count_nonzero":
        return cls(dtype=dtype, dim=dim)
    return cls(dtype=dtype, dim=dim, keepdim=keepdim)


@LogicalReduceMultidimFixture
def test_logical_reduce_multidim_bench(
    shape: tuple,
    dim: list,
    keepdim: bool,
    dtype: torch.dtype,
    op_kind: str,
) -> None:
    test = LogicalReduceMultidimTest(shape, dim, keepdim, dtype, op_kind)
    bm = LogicalReduceMultidimBenchmark(test)
    inputs = test.gen_inputs()

    op = _make_logical_op(dtype, op_kind, dim, keepdim)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# 4. Vector norm (l1, l2, inf) — multi-dim
# ===================================================================


class VectorNormMultidimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype, op_kind",
            [
                # dim=[0, 2] keepdim=False — LLaMA-7B (batch=4, seq=128, hidden=4096)
                pytest.param(
                    (4, 128, 4096), [0, 2], False, torch.float16, "l2",
                    id="l2-7B-dim02-nokeepdim",
                ),
                # dim=[0, 2] keepdim=True
                pytest.param(
                    (4, 128, 4096), [0, 2], True, torch.float16, "l2",
                    id="l2-7B-dim02-keepdim",
                ),
                # dim=[0, 1] keepdim=False: per-hidden norm over batch+seq
                pytest.param(
                    (4, 128, 4096), [0, 1], False, torch.float16, "l1",
                    id="l1-7B-dim01-nokeepdim",
                ),
                # inf norm
                pytest.param(
                    (4, 128, 4096), [0, 2], False, torch.bfloat16, "inf",
                    id="inf-7B-dim02-nokeepdim-bf16",
                ),
            ],
        ),
    ]


_ORD_MAP = {"l1": 1, "l2": 2, "inf": float("inf")}


class VectorNormMultidimTest(WorkloadBase):
    def __init__(
        self,
        shape: tuple,
        dim: list,
        keepdim: bool,
        dtype: torch.dtype,
        op_kind: str,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype
        self.op_kind = op_kind

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        ord_val = _ORD_MAP[self.op_kind]
        return torch.linalg.vector_norm(
            x, ord=ord_val, dim=self.dim, keepdim=self.keepdim,
        )


class VectorNormMultidimBenchmark(BenchmarkBase[BenchmarkWorkload]):
    def calculate_flops(self) -> Optional[float]:
        total_elems = 1
        for s in self.workload.shape:
            total_elems *= s
        return total_elems

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        total_elems = 1
        for s in t.shape:
            total_elems *= s
        out_elems = 1
        dims = set(d % len(t.shape) for d in t.dim)
        for i, s in enumerate(t.shape):
            if i not in dims:
                out_elems *= s
        return (total_elems + out_elems) * elem_bytes


def _make_norm_op(dtype, op_kind, dim, keepdim):
    from tileops.ops.reduction.inf_norm import InfNormFwdOp
    from tileops.ops.reduction.l1_norm import L1NormFwdOp
    from tileops.ops.reduction.l2_norm import L2NormFwdOp

    op_map = {"l1": L1NormFwdOp, "l2": L2NormFwdOp, "inf": InfNormFwdOp}
    cls = op_map[op_kind]
    return cls(dtype=dtype, dim=dim, keepdim=keepdim)


@VectorNormMultidimFixture
def test_vector_norm_multidim_bench(
    shape: tuple,
    dim: list,
    keepdim: bool,
    dtype: torch.dtype,
    op_kind: str,
) -> None:
    test = VectorNormMultidimTest(shape, dim, keepdim, dtype, op_kind)
    bm = VectorNormMultidimBenchmark(test)
    inputs = test.gen_inputs()

    op = _make_norm_op(dtype, op_kind, dim, keepdim)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# 5. Cumulative (cumsum, cumprod) — 3D tensor reshaped to (M, N)
#    CumsumFwdOp/CumprodFwdOp accept only (M, N, dtype) and always
#    operate on dim=-1.  Multi-dim reduction is architecturally
#    unsupported.  We benchmark 3D-shaped inputs (reshaped to M=batch*seq,
#    N=hidden) so the benchmark exercises realistic multi-dim-shaped data
#    even though the kernel sees a 2D view.
# ===================================================================


class CumulativeMultidimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dtype, op_kind",
            [
                # 3D: (batch=4, seq=128, hidden=4096) — LLaMA-7B inference
                pytest.param(
                    (4, 128, 4096), torch.float16, "cumsum",
                    id="cumsum-7B-3D",
                ),
                pytest.param(
                    (4, 128, 4096), torch.bfloat16, "cumsum",
                    id="cumsum-7B-3D-bf16",
                ),
                # Longer context: (batch=2, seq=512, hidden=4096)
                pytest.param(
                    (2, 512, 4096), torch.float16, "cumprod",
                    id="cumprod-7B-longctx-3D",
                ),
            ],
        ),
    ]


class CumulativeMultidimTest(WorkloadBase):
    def __init__(self, shape: tuple, dtype: torch.dtype, op_kind: str):
        self.shape = shape
        self.dtype = dtype
        self.op_kind = op_kind
        # M = product of all dims except last
        self.M = 1
        for s in shape[:-1]:
            self.M *= s
        self.N = shape[-1]

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.op_kind == "cumprod":
            x = torch.rand(*self.shape, dtype=self.dtype, device="cuda") * 0.01 + 0.99
        else:
            x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        if self.op_kind == "cumsum":
            return x_f32.cumsum(dim=-1).to(x.dtype)
        elif self.op_kind == "cumprod":
            return x_f32.cumprod(dim=-1).to(x.dtype)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class CumulativeMultidimBenchmark(BenchmarkBase[BenchmarkWorkload]):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        return t.M * t.N

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        # Read + write: 2 * M * N
        return 2 * t.M * t.N * elem_bytes


def _make_cumulative_op(M, N, dtype, op_kind):
    from tileops.ops.reduction.cumprod import CumprodFwdOp
    from tileops.ops.reduction.cumsum import CumsumFwdOp

    op_map = {"cumsum": CumsumFwdOp, "cumprod": CumprodFwdOp}
    cls = op_map[op_kind]
    return cls(M=M, N=N, dtype=dtype)


@CumulativeMultidimFixture
def test_cumulative_multidim_bench(
    shape: tuple,
    dtype: torch.dtype,
    op_kind: str,
) -> None:
    test = CumulativeMultidimTest(shape, dtype, op_kind)
    bm = CumulativeMultidimBenchmark(test)
    inputs = test.gen_inputs()

    op = _make_cumulative_op(test.M, test.N, dtype, op_kind)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


# ===================================================================
# 6. LogSumExp — multi-dim
#    LogSumExpFwdOp supports multi-dim via _supports_multidim = True.
# ===================================================================


class LogSumExpMultidimFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype",
            [
                # 3D: (batch=4, seq=128, hidden=4096) — LLaMA-7B inference
                # dim=[0, 2] keepdim=False: logsumexp across batch+hidden
                pytest.param(
                    (4, 128, 4096), [0, 2], False, torch.float16,
                    id="lse-7B-dim02-nokeepdim",
                ),
                # dim=[0, 2] keepdim=True
                pytest.param(
                    (4, 128, 4096), [0, 2], True, torch.float16,
                    id="lse-7B-dim02-keepdim",
                ),
                # dim=[0, 1] keepdim=False: per-hidden logsumexp over batch+seq
                pytest.param(
                    (4, 128, 4096), [0, 1], False, torch.float16,
                    id="lse-7B-dim01-nokeepdim",
                ),
                # dim=[0, 1] keepdim=True, bfloat16
                pytest.param(
                    (4, 128, 4096), [0, 1], True, torch.bfloat16,
                    id="lse-7B-dim01-keepdim-bf16",
                ),
                # Longer context: (batch=2, seq=512, hidden=4096) — LLaMA-7B
                pytest.param(
                    (2, 512, 4096), [0, 2], False, torch.float16,
                    id="lse-7B-longctx-dim02",
                ),
                # Longer context with keepdim=True
                pytest.param(
                    (2, 512, 4096), [0, 2], True, torch.bfloat16,
                    id="lse-7B-longctx-dim02-keepdim-bf16",
                ),
            ],
        ),
    ]


class LogSumExpMultidimTest(WorkloadBase):
    def __init__(
        self,
        shape: tuple,
        dim: list,
        keepdim: bool,
        dtype: torch.dtype,
    ):
        self.shape = shape
        self.dim = dim
        self.keepdim = keepdim
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(x.float(), dim=self.dim, keepdim=self.keepdim).to(
            x.dtype
        )


class LogSumExpMultidimBenchmark(BenchmarkBase[BenchmarkWorkload]):
    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        total_elems = 1
        for s in t.shape:
            total_elems *= s
        return total_elems

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem_bytes = torch.tensor([], dtype=t.dtype).element_size()
        total_elems = 1
        for s in t.shape:
            total_elems *= s
        out_elems = 1
        dims = set(d % len(t.shape) for d in t.dim)
        for i, s in enumerate(t.shape):
            if i not in dims:
                out_elems *= s
        return (total_elems + out_elems) * elem_bytes


def _make_logsumexp_op(dtype, dim, keepdim):
    from tileops.ops.reduction.logsumexp import LogSumExpFwdOp

    return LogSumExpFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)


@LogSumExpMultidimFixture
def test_logsumexp_multidim_bench(
    shape: tuple,
    dim: list,
    keepdim: bool,
    dtype: torch.dtype,
) -> None:
    test = LogSumExpMultidimTest(shape, dim, keepdim, dtype)
    bm = LogSumExpMultidimBenchmark(test)
    inputs = test.gen_inputs()

    op = _make_logsumexp_op(dtype, dim, keepdim)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
