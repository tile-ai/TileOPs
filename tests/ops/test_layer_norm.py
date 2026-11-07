"""Tests for LayerNorm and fused-add LayerNorm."""

import re

import pytest
import torch
import torch.nn.functional as F
from tilelang import tvm

from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.constants import SM90_REGISTERS
from tileops._kernels.norm import (
    FusedAddLayerNormHeldKernel,
    FusedAddLayerNormKernel,
    LayerNormHeldKernel,
    LayerNormKernel,
    LayerNormWarpRowKernel,
)
from tileops._kernels.norm import layer_norm as layer_norm_kernel_module
from tileops._kernels.norm._config import make_raw_sums_hold
from tileops._ops.norm.fused_add_layer_norm import FusedAddLayerNormFwdOp
from tileops._ops.norm.layer_norm import LayerNormFwdOp
from tileops._utils import WARP_LANES, get_sm_count, get_sm_version
from tileops.backend import BUILTIN
from tileops.backend import _registry as registry
from tileops.backend._registry import register_kernel_type
from workloads.device import run_device
from workloads.norm import (
    FusedAddLayerNormFarRowWorkload,
    FusedAddLayerNormOutlierLeadWorkload,
    FusedAddLayerNormVanishingResidualWorkload,
    FusedAddLayerNormWorkload,
    LayerNormFarFromZeroWorkload,
    LayerNormOutlierLeadWorkload,
    LayerNormWorkload,
    layer_norm_verification,
    normalization_verification,
)
from workloads.numerics import compare_outputs


class LayerNormTest(LayerNormWorkload, TestBase):
    pass


class LayerNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                # Standard aligned shapes -- fp32
                pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
                # A DiT row: narrow, unaligned, one warp per row; a tail CTA holds one row
                pytest.param(1025, 1152, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, False, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.float32, False, marks=pytest.mark.full),
                # Standard aligned shapes -- fp16
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.float16, False, marks=pytest.mark.full),
                # Standard aligned shapes -- bf16
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(8192, 8192, torch.bfloat16, False, marks=pytest.mark.full),
                # Non-power-of-two hidden dims
                pytest.param(1024, 3000, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.float32, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2048, 5120, torch.bfloat16, False, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(1025, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1025, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                # fp32, one warp per row: 250 vectors, so six lanes hold one fewer
                pytest.param(1025, 1000, torch.float32, False, marks=pytest.mark.full),
                # Tuned over 128 and 256 threads: 256 leaves each thread an odd 95 elements
                # of a row walked from shared memory by a 512-row grid.
                pytest.param(512, 24320, torch.float16, True, marks=pytest.mark.full),
            ],
        ),
    ]


@LayerNormFixture
def test_layer_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = LayerNormTest(m, n, dtype)
    op = LayerNormFwdOp(normalized_shape=(n,))
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_layer_norm_tunes_after_a_failed_first_build(monkeypatch) -> None:
    # A failed first compile leaves tuning to the next build. 2048 rows of 16384 queue past
    # the held kernel's registers, so the fragment kernel builds them.
    n = 16384
    op = LayerNormFwdOp(normalized_shape=(n,))
    # Bound first with fp32, so the failing fp16 entry keeps the op's entry table.
    warm = LayerNormTest(1, n, torch.float32)
    op(*warm.gen_inputs())
    build = layer_norm_kernel_module._layer_norm_kernel
    failures = [RuntimeError("injected compile failure")]

    def fails_to_compile(*args):
        jit = build(*args)
        if not failures:
            return jit

        def compile_once(*params):
            raise failures.pop()

        return compile_once

    monkeypatch.setattr(layer_norm_kernel_module, "_layer_norm_kernel", fails_to_compile)
    test = LayerNormTest(2048, n, torch.float16)
    with pytest.raises(RuntimeError, match="injected compile failure"):
        op(*test.gen_inputs())
    op.request_tune()
    op(*test.gen_inputs())
    (kernel,) = [k for k in op.iter_kernels() if k.dtype == torch.float16]
    assert kernel.config["threads"] is not None
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_layer_norm_kernel_handles_unaligned_shape() -> None:
    """The kernel, not the Op layer, owns non-aligned boundary handling."""
    m, n = 16, 3000
    dtype = torch.float16
    test = LayerNormTest(m, n, dtype)
    op = LayerNormFwdOp(normalized_shape=(n,), eps=test.eps, target=BUILTIN)
    test.check(op, *test.gen_inputs())


class LayerNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@LayerNormNonContigFixture
def test_layer_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]  # non-contiguous slice
    weight = torch.randn(n, dtype=dtype, device=run_device())
    bias = torch.randn(n, dtype=dtype, device=run_device())

    op = LayerNormFwdOp(normalized_shape=(n,))

    # Reference using torch.nn.functional.layer_norm
    x_ref = x.contiguous()
    y_ref = F.layer_norm(
        x_ref.float(),
        (n,),
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    compare_outputs(y, y_ref, layer_norm_verification(dtype))


class LayerNorm3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@LayerNorm3DFixture
def test_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    weight = torch.randn(hidden, dtype=dtype, device=run_device())
    bias = torch.randn(hidden, dtype=dtype, device=run_device())

    op = LayerNormFwdOp(normalized_shape=(hidden,))

    # Reference using torch.nn.functional.layer_norm
    y_ref = F.layer_norm(
        x.float(),
        (hidden,),
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)

    y = op(x, weight, bias)
    compare_outputs(y, y_ref, layer_norm_verification(dtype))


class LayerNormFarFromZeroTest(LayerNormFarFromZeroWorkload, TestBase):
    pass


def _far_from_zero_case(rows, n, kernel, dtype, row, mark):
    return pytest.param(
        rows, n, kernel, dtype, row, marks=mark, id=f"{kernel}-{n}-{str(dtype)[6:]}-{row}"
    )


_F32_ROWS = ["constant", "noisy", "huge", "wide_spread"]

# (rows, or rows an SM when a float, n, kernel): "held" holds a row in registers, split
# across a cluster at 4 x 4096 and queued at 2 rows an SM; "warp" holds a row in one warp;
# "fragment" serves rows that queue past a thread's registers, walked from shared memory,
# padded at 8300.
_LAYER_NORM_FAR_CASES = [
    *[_far_from_zero_case(4, 4096, "held", torch.float32, r, pytest.mark.smoke) for r in _F32_ROWS],
    _far_from_zero_case(4, 4096, "held", torch.bfloat16, "huge", pytest.mark.smoke),
    *[
        _far_from_zero_case(4, 3000, "held", torch.float32, r, pytest.mark.smoke)
        for r in ["constant", "huge"]
    ],
    _far_from_zero_case(4, 3000, "warp", torch.bfloat16, "huge", pytest.mark.smoke),
    *[_far_from_zero_case(4, 1020, "warp", torch.float32, r, pytest.mark.smoke) for r in _F32_ROWS],
    _far_from_zero_case(2.0, 4096, "held", torch.float32, "noisy", pytest.mark.full),
    _far_from_zero_case(2.0, 4096, "held", torch.bfloat16, "huge", pytest.mark.full),
    *[
        _far_from_zero_case(2.0, 16384, "fragment", torch.float32, r, pytest.mark.full)
        for r in _F32_ROWS
    ],
    _far_from_zero_case(2.0, 16384, "fragment", torch.bfloat16, "huge", pytest.mark.full),
    *[
        _far_from_zero_case(1024, 8300, "fragment", torch.float32, r, pytest.mark.full)
        for r in ["constant", "huge"]
    ],
]

_LAYER_NORM_KERNELS = {
    "held": LayerNormHeldKernel,
    "warp": LayerNormWarpRowKernel,
    "fragment": LayerNormKernel,
}


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("rows, n, kernel, dtype, row", _LAYER_NORM_FAR_CASES)
def test_layer_norm_normalizes_rows_far_from_zero(rows, n, kernel, dtype, row) -> None:
    m = int(rows * get_sm_count()) if isinstance(rows, float) else rows
    test = LayerNormFarFromZeroTest(m, n, dtype, row=row)
    op = LayerNormFwdOp(normalized_shape=(n,))
    test.check(op, *test.gen_inputs())
    (built,) = op.iter_kernels()
    assert type(built) is _LAYER_NORM_KERNELS[kernel]


class LayerNormOutlierLeadTest(LayerNormOutlierLeadWorkload, TestBase):
    pass


def assert_raw_sums_fail(x: torch.Tensor) -> None:
    """Assert that every row of *x* fails `make_raw_sums_hold` at one add deep, the loosest
    bound any kernel takes, so the kernel takes the shift-robust rule the case tests."""
    rows = x.double().reshape(-1, x.shape[-1])
    var = rows.var(dim=-1, unbiased=False)
    square_mean = rows.square().mean(dim=-1)
    # The row nearest to holding: the least mean square for its variance.
    i = int((square_mean / var).argmin())
    holds = make_raw_sums_hold(str(x.dtype)[6:], 1)(float(square_mean[i]), float(var[i]))
    assert not bool(tvm.arith.Analyzer().simplify(holds))


def warp_leads(held, m: int, n: int, dtype: torch.dtype) -> tuple:
    """Each warp's first element in every cluster CTA's share of a row, as *held*, a
    `LayerNormHeldKernel`-like class, places *m* rows of *n*: the elements its warps take
    their first sums less."""
    sm_count, arch = get_sm_count(), get_sm_version()
    cluster = held._cluster(n, dtype, m, arch, sm_count)
    vec = held._vec(n, dtype, cluster)
    threads = held._threads(n, cluster, vec, m, sm_count)
    return tuple(
        cta * (n // cluster) + warp * WARP_LANES * vec
        for cta in range(cluster)
        for warp in range(threads // WARP_LANES)
    )


def _outlier_case(rows, n, kernel, leads, mark):
    return pytest.param(rows, n, kernel, leads, marks=mark, id=f"{kernel}-{n}-{leads}")


def outlier_rows(rows) -> int:
    """Rows of a case: *rows* itself, rows an SM when a float, or one CTA past a full wave."""
    if rows == "wave+1":
        return get_sm_count() + 1
    return int(rows * get_sm_count()) if isinstance(rows, float) else rows


def op_preferring(op_cls, name: str, kernel_cls, **kwargs):
    """*op_cls* built with *kernel_cls* registered as *name* ahead of its in-tree kernels."""
    state = registry.snapshot()
    try:
        register_kernel_type(op_cls.__name__, name, kernel_cls)
        return op_cls(**kwargs)
    finally:
        registry.restore(state)


class _WideHeld(LayerNormHeldKernel):
    """`LayerNormHeldKernel` at 1024 threads a CTA."""

    general = False
    preferred_over = frozenset(LayerNormFwdOp.kernel_types)

    @property
    def default_config(self) -> dict:
        return {**super().default_config, "threads": 1024}


def check_held_launch(held, kernel: str, m: int, n: int, dtype: torch.dtype) -> None:
    """Assert that *held*'s rule places *m* rows of *n* as *kernel* names: "held" alone on
    its SM, "queued" with CTAs queued, "split" or "queued-split" likewise across a cluster,
    "fragment" past a thread's registers; skip a split where clusters do not exist."""
    sm_count, arch = get_sm_count(), get_sm_version()
    cluster = held._cluster(n, dtype, m, arch, sm_count)
    if "split" in kernel and arch < 90:
        pytest.skip("a row splits across a cluster on SM90 only")
    if kernel == "fragment":
        assert held._queued_past_registers(n, dtype, m, arch, sm_count)
        return
    assert not held._queued_past_registers(n, dtype, m, arch, sm_count)
    assert (m * cluster > sm_count) == kernel.startswith("queued")
    assert (cluster > 1) == ("split" in kernel)


# (rows, n, kernel, leads); rows as `outlier_rows` reads them. "row" puts the lead first in
# each row of `outlier_lead_rows`, "warps" first in each warp's share of the held kernel; kernels as
# `check_held_launch` names them, and "warp" holds a row in one warp, a tail CTA holding one.
# "fragment" walks rows from shared memory, padded at 8300.
_LAYER_NORM_OUTLIER_CASES = [
    _outlier_case(4, 4096, "split", "row", pytest.mark.smoke),
    _outlier_case(5, 1532, "warp", "row", pytest.mark.smoke),
    _outlier_case(1, 262144, "split", "warps", pytest.mark.full),
    _outlier_case(2.0, 6144, "queued", "row", pytest.mark.full),
    _outlier_case(2.0, 3000, "queued", "warps", pytest.mark.full),
    _outlier_case("wave+1", 12288, "queued-split", "row", pytest.mark.full),
    _outlier_case(2.0, 16384, "fragment", "row", pytest.mark.full),
    _outlier_case(1024, 8300, "fragment", "row", pytest.mark.full),
]


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("rows, n, kernel, leads", _LAYER_NORM_OUTLIER_CASES)
def test_layer_norm_normalizes_rows_with_an_outlier_lead(rows, n, kernel, leads) -> None:
    m, dtype = outlier_rows(rows), torch.float32
    if kernel != "warp":
        check_held_launch(LayerNormHeldKernel, kernel, m, n, dtype)
    positions = (0,) if leads == "row" else warp_leads(LayerNormHeldKernel, m, n, dtype)
    test = LayerNormOutlierLeadTest(m, n, dtype, leads=positions)
    op = LayerNormFwdOp(normalized_shape=(n,))
    inputs = test.gen_inputs()
    assert_raw_sums_fail(inputs[0])
    test.check(op, *inputs)
    (built,) = op.iter_kernels()
    assert type(built) is _LAYER_NORM_KERNELS.get(kernel, LayerNormHeldKernel)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "rows, n",
    [
        pytest.param(4, 3000, marks=pytest.mark.smoke),
        pytest.param(2.0, 3000, marks=pytest.mark.full),
    ],
)
def test_layer_norm_held_kernel_leaves_warps_empty_and_normalizes_an_outlier_lead(rows, n) -> None:
    # 1024 threads take at most 750 accesses of a 3000-wide fp32 row: the last warps hold
    # nothing, and the merges must leave them out.
    op = op_preferring(LayerNormFwdOp, "layer_norm_held_wide", _WideHeld, normalized_shape=(n,))
    test = LayerNormOutlierLeadTest(outlier_rows(rows), n, torch.float32, leads=(0,))
    test.check(op, *test.gen_inputs())
    (built,) = op.iter_kernels()
    assert isinstance(built, _WideHeld)
    cluster = built._cluster(n, torch.float32, test.m, get_sm_version(), get_sm_count())
    accesses = n // cluster // built._vec(n, torch.float32, cluster)
    assert accesses <= 1024 - WARP_LANES


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_layer_norm_held_kernel_merges_less_a_reference_near_the_mean() -> None:
    # The first warp holds 128 elements of 13000.04, the others 3000: a merge less the first
    # warp's centre would round 31 terms at 10000 apart.
    n, dtype = 32768, torch.float32
    op = op_preferring(LayerNormFwdOp, "layer_norm_held_wide", _WideHeld, normalized_shape=(n,))
    x = torch.full((1, n), 3000.0, dtype=dtype, device=run_device())
    x[0, :128] = 13000.04
    weight = torch.full((n,), 32.0, dtype=dtype, device=x.device)
    bias = torch.zeros(n, dtype=dtype, device=x.device)
    assert_raw_sums_fail(x)
    expected = F.layer_norm(x.double(), (n,), weight.double(), bias.double(), 1e-5)
    compare_outputs(op(x, weight, bias), expected.to(dtype), layer_norm_verification(dtype))
    (built,) = op.iter_kernels()
    assert isinstance(built, _WideHeld)
    if get_sm_version() >= 90:
        assert built._cluster(n, dtype, 1, get_sm_version(), get_sm_count()) > 1


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "row",
    [
        pytest.param("huge", marks=pytest.mark.smoke),
        pytest.param("constant", marks=pytest.mark.full),
        pytest.param("noisy", marks=pytest.mark.full),
        # `outlier_lead_rows` with the lead first in each row
        pytest.param("lead", marks=pytest.mark.full),
    ],
)
def test_layer_norm_fragment_normalizes_rows_far_from_zero_without_the_walk(row) -> None:
    # 256 threads leave 33 of a padded 8448-wide row to a thread, an odd share no walk takes.
    class Unwalked(LayerNormKernel):
        general = False
        preferred_over = frozenset(LayerNormFwdOp.kernel_types)

        @property
        def default_config(self) -> dict:
            return {"block_m": 1, "threads": 256}

    n = 8300
    op = op_preferring(LayerNormFwdOp, "layer_norm_unwalked", Unwalked, normalized_shape=(n,))
    m = 2 * get_sm_count()
    if row == "lead":
        test = LayerNormOutlierLeadTest(m, n, torch.float32, leads=(0,))
    else:
        test = LayerNormFarFromZeroTest(m, n, torch.float32, row=row)
    test.check(op, *test.gen_inputs())
    (built,) = op.iter_kernels()
    assert isinstance(built, Unwalked)
    assert built.config["threads"] == 256


@pytest.mark.smoke
def test_layer_norm_serves_a_changed_leading_dims_product_from_one_kernel() -> None:
    """The row count is a fact of the call, so it does not specialize the op's kernel."""
    n = 4096
    dtype = torch.float16

    op = LayerNormFwdOp(normalized_shape=(n,))
    weight = torch.randn(n, dtype=dtype, device=run_device())
    bias = torch.randn(n, dtype=dtype, device=run_device())

    x1 = torch.randn(512, n, dtype=dtype, device=run_device())
    y1 = op(x1, weight, bias)
    assert y1.shape == x1.shape

    x2 = torch.randn(1024, n, dtype=dtype, device=run_device())
    y2 = op(x2, weight, bias)
    assert y2.shape == x2.shape

    y_ref = F.layer_norm(
        x2.float(),
        (n,),
        weight=weight.float(),
        bias=bias.float(),
        eps=1e-5,
    ).to(dtype)
    compare_outputs(y2, y_ref, layer_norm_verification(dtype))


@pytest.mark.smoke
@pytest.mark.parametrize("n", [256, 1152])
@pytest.mark.parametrize("give", ["weight", "bias", "neither"])
def test_either_affine_tensor_alone_matches_torch(give: str, n: int) -> None:
    dtype = torch.float16
    x = torch.randn(8, n, dtype=dtype, device=run_device())
    kwargs = {} if give == "neither" else {give: torch.randn(n, dtype=dtype, device=run_device())}
    got = LayerNormFwdOp(normalized_shape=(n,))(x, **kwargs)
    expected = F.layer_norm(x.float(), (n,), **{k: v.float() for k, v in kwargs.items()})
    compare_outputs(got, expected.to(dtype), normalization_verification("LayerNormFwdOp", x.dtype))


def _misaligned(t: torch.Tensor) -> torch.Tensor:
    """*t* copied into a contiguous view that starts one element into its storage."""
    view = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)[1:].view(t.shape)
    view.copy_(t)
    return view


@pytest.mark.smoke
@pytest.mark.parametrize(
    "n", [pytest.param(4096, id="aligned-row"), pytest.param(1000, id="warp-row")]
)
def test_layer_norm_reads_inputs_off_the_vector_boundary(n: int) -> None:
    """A contiguous input, weight and bias may start anywhere in their storage."""
    dtype = torch.bfloat16
    x = _misaligned(torch.randn(2, n, dtype=dtype, device=run_device()))
    weight = _misaligned(torch.randn(n, dtype=dtype, device=x.device))
    bias = _misaligned(torch.randn(n, dtype=dtype, device=x.device))
    compare_outputs(
        LayerNormFwdOp(normalized_shape=(n,))(x, weight, bias),
        F.layer_norm(x.float(), (n,), weight.float(), bias.float()).to(dtype),
        normalization_verification("LayerNormFwdOp", dtype),
    )


class FusedAddLayerNormTest(FusedAddLayerNormWorkload, TestBase):
    pass


class FusedAddLayerNormFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, tune",
            [
                # One CTA a row, 16-byte accesses -- each dtype
                pytest.param(1024, 4096, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(4096, 4096, torch.float32, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.float16, False, marks=pytest.mark.full),
                pytest.param(4096, 4096, torch.bfloat16, False, marks=pytest.mark.full),
                # A share the block's accesses do not divide: the last access skipped
                pytest.param(1024, 3000, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1024, 3000, torch.bfloat16, False, marks=pytest.mark.full),
                # The fragment kernel at a padded width, and accesses narrower than 16 bytes
                pytest.param(264, 6531, torch.float32, False, marks=pytest.mark.full),
                # Tuned mode
                pytest.param(1024, 3072, torch.bfloat16, True, marks=pytest.mark.full),
            ],
        ),
    ]


@FusedAddLayerNormFixture
def test_fused_add_layer_norm_op(m: int, n: int, dtype: torch.dtype, tune: bool) -> None:
    test = FusedAddLayerNormTest(m, n, dtype)
    op = FusedAddLayerNormFwdOp()
    if tune:
        op.request_tune()
        # The first call tunes and returns the untuned result; the check reads the tuned one.
        op(*test.gen_inputs())
    test.check(op, *test.gen_inputs())


def _fused_add_launch(op: FusedAddLayerNormFwdOp, m: int) -> tuple:
    """``(cluster, threads)`` of the program the op's in-tree kernel compiled for *m* rows."""
    (kernel,) = op.iter_kernels()
    (key,) = [key for key in kernel._generation.programs if key[1] == m]
    source = kernel._generation.programs[key].get_kernel_source()
    (threads,) = set(re.findall(r"__launch_bounds__\((\d+)", source))
    return key[2], int(threads)


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "n, launches",
    [
        # A lone 4096-wide row costs less on one CTA than split; 256 threads, two accesses each.
        pytest.param(4096, {1: ((1, 256), (1, 256))}, marks=pytest.mark.smoke),
        # 192 accesses a row: 128 threads would leave half idle in a second round, so 96
        # threads make two each.
        pytest.param(1536, {2048: ((1, 96), (1, 96))}, marks=pytest.mark.full),
    ],
)
def test_fused_add_layer_norm_launch_follows_the_row(n, launches) -> None:
    dtype = torch.bfloat16
    op = FusedAddLayerNormFwdOp()
    for m in launches:
        test = FusedAddLayerNormTest(m, n, dtype)
        test.check(op, *test.gen_inputs())
    sm90 = get_sm_version() >= 90
    for m, (on_sm90, elsewhere) in launches.items():
        assert _fused_add_launch(op, m) == (on_sm90 if sm90 else elsewhere)


class FusedAddLayerNormVanishingResidualTest(FusedAddLayerNormVanishingResidualWorkload, TestBase):
    pass


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "rows_per_sm, n, dtype, held, workload",
    [
        # Queued rows past a thread's registers: the fragment kernel, normalizing the rounded
        # sum (a residual the bf16 rounding drops leaves no offset)
        pytest.param(
            2,
            12288,
            torch.bfloat16,
            False,
            FusedAddLayerNormVanishingResidualTest,
            marks=pytest.mark.smoke,
        ),
        # Past the registers but no CTA queued: the held kernel
        pytest.param(
            0.5, 32768, torch.float32, True, FusedAddLayerNormTest, marks=pytest.mark.smoke
        ),
    ],
)
def test_fused_add_layer_norm_takes_the_fragment_kernel_past_the_registers(
    rows_per_sm, n, dtype, held, workload
) -> None:
    m = int(rows_per_sm * get_sm_count())
    op = FusedAddLayerNormFwdOp()
    test = workload(m, n, dtype)
    test.check(op, *test.gen_inputs())
    (kernel,) = op.iter_kernels()
    assert isinstance(kernel, FusedAddLayerNormHeldKernel) == held


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_fused_add_layer_norm_tunes_from_the_rule(monkeypatch) -> None:
    # Tuning measures the first row count built, seeded with the threads the rule gave it,
    # and the published width serves every row count and cluster width after it.
    n, dtype = 8192, torch.bfloat16
    op = FusedAddLayerNormFwdOp()
    builders = []
    for m in (1, 2048):
        op(*FusedAddLayerNormTest(m, n, dtype).gen_inputs())
        (kernel,) = op.iter_kernels()
        builders.append(kernel.kernel)
    assert builders[0] is not builders[1]
    tuned = []
    tune = kernel.tune_jit_kernel

    def recording_tune(jit_kernel, configs, **kwargs):
        tuned.append((jit_kernel, kwargs["seed_config"], configs))
        return tune(jit_kernel, configs, **kwargs)

    monkeypatch.setattr(kernel, "tune_jit_kernel", recording_tune)
    op.request_tune()
    # On SM90 the lone row's cluster share is one access a thread, so 128 is the only width.
    sm90 = get_sm_version() >= 90
    threads = 128 if sm90 else 512
    widths = [128] if sm90 else [128, 256, 512, 1024]
    # Each width with the compiler's registers (0) and with every CTAs-an-SM count that leaves
    # a thread 64 to 96 registers.
    ctas = {128: [0, 6, 7, 8], 256: [0, 3, 4], 512: [0, 2], 1024: [0, 1]}
    configs = [{"threads": t, "ctas_per_sm": c} for t in widths for c in ctas[t]]
    ((tuned_builder, seed, tuned_configs),) = tuned
    assert (tuned_builder, seed["threads"], tuned_configs) == (builders[0], threads, configs)
    # The untuned CTAs an SM keeps: the rule's, at 64 registers a thread or more.
    assert 1 <= seed["ctas_per_sm"] <= SM90_REGISTERS // (threads * 64)
    published = op.kernel_config()["threads"]
    assert published is not None
    for m in (1, 2048):
        test = FusedAddLayerNormTest(m, n, dtype)
        test.check(op, *test.gen_inputs())
        assert _fused_add_launch(op, m)[1] == published


class FusedAddLayerNormFarRowTest(FusedAddLayerNormFarRowWorkload, TestBase):
    pass


def _far_row_case(rows, n, kernel, dtype, row, mark):
    return pytest.param(
        rows, n, kernel, dtype, row, marks=mark, id=f"{kernel}-{str(dtype)[6:]}-{row}"
    )


# (rows, or rows an SM when a float, n, kernel): "held" holds a row in registers, read in
# accesses that leave the block's last one past it; "split" holds a lone row split across a
# cluster on SM90; "queued" and "queued-split" hold rows whose CTAs queue on every SM, the
# latter split across a cluster on SM90; "fragment" serves rows queued past a thread's
# registers.
_FAR_ROW_CASES = [
    _far_row_case(4, 3000, "held", torch.float32, "constant", pytest.mark.smoke),
    _far_row_case(4, 3000, "held", torch.float32, "noisy", pytest.mark.smoke),
    _far_row_case(4, 3000, "held", torch.float32, "huge", pytest.mark.smoke),
    # A bfloat16 row sums exactly; its mean rounds where 1 / n does, and its sum overflows.
    _far_row_case(4, 3000, "held", torch.bfloat16, "constant", pytest.mark.smoke),
    _far_row_case(4, 3000, "held", torch.bfloat16, "noisy", pytest.mark.smoke),
    _far_row_case(4, 3000, "held", torch.bfloat16, "huge", pytest.mark.smoke),
    _far_row_case(1, 8192, "split", torch.float32, "constant", pytest.mark.full),
    _far_row_case(1, 8192, "split", torch.float32, "noisy", pytest.mark.full),
    _far_row_case(1, 8192, "split", torch.float32, "huge", pytest.mark.full),
    _far_row_case(1, 8192, "split", torch.bfloat16, "huge", pytest.mark.full),
    _far_row_case(2.0, 3000, "queued", torch.float32, "constant", pytest.mark.full),
    _far_row_case(2.0, 3000, "queued", torch.float32, "noisy", pytest.mark.full),
    _far_row_case(2.0, 3000, "queued", torch.float32, "huge", pytest.mark.full),
    _far_row_case(2.0, 3000, "queued", torch.bfloat16, "huge", pytest.mark.full),
    _far_row_case(1.01, 12288, "queued-split", torch.float32, "constant", pytest.mark.full),
    _far_row_case(1.01, 12288, "queued-split", torch.float32, "noisy", pytest.mark.full),
    _far_row_case(1.01, 12288, "queued-split", torch.float32, "huge", pytest.mark.full),
    _far_row_case(2.0, 6531, "fragment", torch.float32, "constant", pytest.mark.full),
    _far_row_case(2.0, 6531, "fragment", torch.float32, "noisy", pytest.mark.full),
    _far_row_case(2.0, 6531, "fragment", torch.float32, "huge", pytest.mark.full),
    _far_row_case(2.0, 12288, "fragment", torch.bfloat16, "huge", pytest.mark.full),
    *[
        _far_row_case(rows, n, kernel, torch.float32, "wide_spread", pytest.mark.full)
        for rows, n, kernel in [
            (4, 3000, "held"),
            (1, 8192, "split"),
            (2.0, 3000, "queued"),
            (1.01, 12288, "queued-split"),
            (2.0, 6531, "fragment"),
        ]
    ],
]


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("rows, n, kernel, dtype, row", _FAR_ROW_CASES)
def test_fused_add_layer_norm_normalizes_rows_far_from_zero(rows, n, kernel, dtype, row) -> None:
    m = int(rows * get_sm_count()) if isinstance(rows, float) else rows
    op = FusedAddLayerNormFwdOp()
    test = FusedAddLayerNormFarRowTest(m, n, dtype, row)
    test.check(op, *test.gen_inputs())
    (built,) = op.iter_kernels()
    assert isinstance(built, FusedAddLayerNormHeldKernel) == (kernel != "fragment")
    if kernel in ("split", "queued-split") and get_sm_version() >= 90:
        assert _fused_add_launch(op, m)[0] > 1


class FusedAddLayerNormOutlierLeadTest(FusedAddLayerNormOutlierLeadWorkload, TestBase):
    pass


# (rows, or rows an SM when a float, n, kernel, leads), as `_LAYER_NORM_OUTLIER_CASES`:
# "split" holds a lone row split across a cluster on SM90, "queued-split" queued rows split.
_FUSED_ADD_OUTLIER_CASES = [
    _outlier_case(4, 4096, "split", "row", pytest.mark.smoke),
    _outlier_case(1, 262144, "split", "warps", pytest.mark.full),
    _outlier_case(2.0, 6144, "queued", "row", pytest.mark.full),
    _outlier_case(2.0, 3000, "queued", "warps", pytest.mark.full),
    _outlier_case("wave+1", 12288, "queued-split", "row", pytest.mark.full),
    _outlier_case(2.0, 6531, "fragment", "row", pytest.mark.full),
]


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("rows, n, kernel, leads", _FUSED_ADD_OUTLIER_CASES)
def test_fused_add_layer_norm_normalizes_rows_with_an_outlier_lead(rows, n, kernel, leads) -> None:
    m, dtype = outlier_rows(rows), torch.float32
    check_held_launch(FusedAddLayerNormHeldKernel, kernel, m, n, dtype)
    positions = (0,) if leads == "row" else warp_leads(FusedAddLayerNormHeldKernel, m, n, dtype)
    op = FusedAddLayerNormFwdOp()
    test = FusedAddLayerNormOutlierLeadTest(m, n, dtype, positions)
    inputs = test.gen_inputs()
    assert_raw_sums_fail(inputs[0] + inputs[1])
    test.check(op, *inputs)
    (built,) = op.iter_kernels()
    held = kernel != "fragment"
    assert type(built) is (FusedAddLayerNormHeldKernel if held else FusedAddLayerNormKernel)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize("rows_per_sm, n", [(1 / 128, 8192), (1.01, 12288)])
def test_fused_add_layer_norm_exchanges_share_no_bytes(rows_per_sm, n) -> None:
    # A peer CTA's store lands in this CTA's shared memory whenever the peer gets there, so
    # every exchange lives in one allocation the planner cannot split into aliased views.
    m = max(1, int(rows_per_sm * get_sm_count()))
    op = FusedAddLayerNormFwdOp()
    op(*FusedAddLayerNormTest(m, n, torch.float32).gen_inputs())
    (kernel,) = op.iter_kernels()
    (key,) = [key for key in kernel._generation.programs if key[1] == m]
    source = kernel._generation.programs[key].get_kernel_source()
    assert "__shared__" in source
    assert re.findall(r"buf_dyn_shmem \+ \d+", source) == []


class FusedAddLayerNormNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FusedAddLayerNormNonContigFixture
def test_fused_add_layer_norm_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    """Test with non-contiguous input (sliced tensor)."""
    x_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    r_full = torch.randn(m, n * 2, dtype=dtype, device=run_device())
    x = x_full[:, :n]  # non-contiguous slice
    residual = r_full[:, :n]
    weight = torch.randn(n, dtype=dtype, device=run_device())
    bias = torch.randn(n, dtype=dtype, device=run_device())

    op = FusedAddLayerNormFwdOp()

    # Reference on contiguous copies
    test = FusedAddLayerNormTest(m, n, dtype)
    y_ref, add_ref = test.ref_program(x.contiguous(), residual.contiguous(), weight, bias)

    y, residual_out = op(x, residual, weight, bias)

    compare_outputs(y, y_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype))
    compare_outputs(
        residual_out, add_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype)
    )


class FusedAddLayerNorm3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@FusedAddLayerNorm3DFixture
def test_fused_add_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    residual = torch.randn(batch, seq, hidden, dtype=dtype, device=run_device())
    weight = torch.randn(hidden, dtype=dtype, device=run_device())
    bias = torch.randn(hidden, dtype=dtype, device=run_device())

    M = batch * seq
    op = FusedAddLayerNormFwdOp()

    test = FusedAddLayerNormTest(M, hidden, dtype)
    y_ref, add_ref = test.ref_program(x, residual, weight, bias)

    y, residual_out = op(x, residual, weight, bias)

    compare_outputs(y, y_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype))
    compare_outputs(
        residual_out, add_ref, normalization_verification("FusedAddLayerNormFwdOp", x.dtype)
    )


@pytest.mark.smoke
def test_fused_add_layer_norm_reads_inputs_off_the_vector_boundary() -> None:
    """A contiguous input, residual, weight and bias may start anywhere in their storage."""
    test = FusedAddLayerNormTest(4, 1024, torch.float16)
    test.check(FusedAddLayerNormFwdOp(), *(_misaligned(t) for t in test.gen_inputs()))
