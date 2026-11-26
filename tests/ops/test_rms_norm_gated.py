"""Public RMSNormGatedFwdOp semantics: both gates, both gate orders, groups, bias, every dtype."""

import pytest
import torch

from tests.cuda_graph_contract import check_replay
from tests.workload_test_base import TestBase, smoke_first
from tileops.norm import RMSNormGatedFwdOp
from workloads.norm import RMSNormGatedCall, RMSNormGatedTinyGateCall
from workloads.workload_base import manifest_call


class RMSNormGatedTest(RMSNormGatedCall, TestBase):
    pass


class RMSNormGatedTinyGateTest(RMSNormGatedTinyGateCall, TestBase):
    pass


_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
_NIGHTLY = pytest.mark.nightly
_BF16 = {"T": "bfloat16", "W": "bfloat16", "Z": "bfloat16"}


def _case(leading, n, dtypes, mark, case_id, *, some=("z",), **params):
    """A call over rows of ``n`` columns; ``some`` lists the optional tensors passed."""
    return pytest.param(leading, n, dtypes, list(some), params, marks=mark, id=case_id)


_DTYPES = (
    _case([3, 7], 128, _BF16, _SMOKE, "bf16"),
    _case([3, 7], 128, {"T": "float16", "W": "float16", "Z": "float16"}, _SMOKE, "fp16"),
    _case([3, 7], 128, {"T": "float32", "W": "float32", "Z": "float32"}, _SMOKE, "fp32"),
    _case([3, 7], 128, {"T": "float16", "W": "bfloat16", "Z": "float32"}, _FULL, "mixed-dtypes"),
    _case(
        [3, 7],
        128,
        {"T": "bfloat16", "W": "bfloat16", "Wb": "bfloat16", "Z": "bfloat16"},
        _FULL,
        "bf16-bias",
        some=("bias", "z"),
    ),
)

_FEATURES = (
    # 128 accesses: 32 lanes of four.
    _case([9], 1024, _BF16, _FULL, "gate-before-norm", norm_before_gate=False),
    _case(
        [5],
        512,
        _BF16,
        _FULL,
        "groups",
        group_size=128,
        norm_before_gate=False,
        activation="sigmoid",
    ),
    _case(
        [6],
        384,
        {"T": "bfloat16", "W": "bfloat16", "Wb": "float16", "Z": "bfloat16"},
        _FULL,
        "bias-and-gate",
        some=("bias", "z"),
        group_size=128,
    ),
    _case(
        [4],
        3000,
        {"T": "float32", "W": "float32", "Wb": "float32"},
        _FULL,
        "bias-no-gate",
        some=("bias",),
        eps=1e-5,
    ),
)

# One case per path through RMSNormGatedKernel's mapping of a group onto lanes.
_PATHS = (
    # 640 accesses over 256 lanes, the last lanes holding two, summed across eight warps.
    _case(
        [3],
        5120,
        {"T": "bfloat16", "W": "float32", "Z": "bfloat16"},
        _FULL,
        "cross-warp",
        norm_before_gate=False,
    ),
    # Rows of one element: 1-element accesses, one lane a group, no shuffle.
    _case([37], 1, _BF16, _FULL, "one-element-rows"),
    # 256 accesses: 64 lanes, two warps a group and two groups a CTA, the last CTA half full.
    _case([3], 2048, _BF16, _FULL, "two-warp-groups"),
    # 5120 accesses: 1024 lanes, the cap, of five.
    _case([2], 40960, _BF16, _FULL, "lane-cap"),
    # Regression: rows past 2^31 need 64-bit row offsets.
    _case(
        [2**31 + 3],
        1,
        {"T": "float16", "W": "float16"},
        _NIGHTLY,
        "rows-past-int32",
        some=(),
    ),
    # Five groups of 12 accesses on 8 lanes of two: a lane past its group's end would read
    # and store the next group's columns of the same row if its slots were not masked.
    _case([3], 480, _BF16, _FULL, "groups-with-empty-slots", group_size=96),
    # 100 is no multiple of 8: 4-element accesses, 25 on 16 lanes of two, seven slots empty.
    _case([5], 100, _BF16, _FULL, "vec-4"),
    # 6 elements: 2-element accesses.
    _case([5], 6, _BF16, _FULL, "vec-2"),
    # 7 elements: 1-element accesses.
    _case([5], 7, {"T": "float16", "W": "float16", "Z": "float16"}, _FULL, "vec-1"),
)


@pytest.mark.parametrize(
    "leading, n, dtypes, some, params", smoke_first([*_DTYPES, *_FEATURES, *_PATHS])
)
def test_rms_norm_gated(leading, n, dtypes, some, params) -> None:
    row = {"bias": None, "z": None, **params}
    for name in some:
        row.pop(name)
    call = manifest_call("RMSNormGatedFwdOp", dtypes, B=leading, N=n, some=some, **row)
    test = RMSNormGatedTest(call)
    op = RMSNormGatedFwdOp(**call.arguments())
    test.check(op, *test.gen_inputs())


def _laid_out(t: torch.Tensor, layout: tuple, heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """*t*, ``(*B, n)``, in *layout*, and the storage the view lies in:
    ``("columns", row, first)``, *heads* of its rows side by side from column *first* of rows
    *row* elements wide; ``("every-other", row)``, every other column of rows *row* elements
    wide; ``("transposed",)``, a ``(n, R)`` tensor's transpose; ``("expanded",)``, its first
    row repeated over every row, zero elements apart."""
    n = t.shape[-1]
    if layout[0] in ("columns", "every-other"):
        width = heads * n
        storage = torch.randn(t.numel() // width, layout[1], device=t.device).to(t.dtype)
        if layout[0] == "columns":
            view = storage[:, layout[2] : layout[2] + width].view(t.shape)
        else:
            view = storage[:, ::2].view(t.shape)
        view.copy_(t)
    elif layout[0] == "transposed":
        storage = t.reshape(-1, n).t().contiguous()
        view = storage.t().view(t.shape)
    else:
        storage = t.reshape(-1, n)[:1].clone()
        view = storage.expand(t.numel() // n, n).view(t.shape)
    return view, storage


@pytest.mark.parametrize(
    "leading, n, group_size, heads, x_layout, z_layout",
    [
        pytest.param(
            [300], 6144, 128, 1, ("columns", 6144, 0), ("columns", 16480, 10240),
            marks=_SMOKE, id="qwen3.8-27b-gate",
        ),
        pytest.param(
            [35], 128, None, 1, ("columns", 130, 0), ("columns", 136, 8), marks=_FULL,
            id="vec-2-strides",
        ),
        pytest.param(
            [1], 6144, 128, 1, ("columns", 6150, 0), ("columns", 16480, 10240), marks=_FULL,
            id="one-row",
        ),
        pytest.param(
            [9, 48], 128, None, 48, ("columns", 6144, 0), ("columns", 16480, 10240),
            marks=_FULL, id="uneven-rows",
        ),
        pytest.param(
            [35], 128, None, 1, ("every-other", 256), ("columns", 136, 8), marks=_FULL,
            id="every-other-x",
        ),
        pytest.param(
            [35], 128, None, 1, ("transposed",), ("columns", 136, 8), marks=_FULL,
            id="transposed-x",
        ),
        pytest.param(
            [35], 128, None, 1, ("columns", 128, 0), ("expanded",), marks=_FULL,
            id="expanded-z",
        ),
    ],
)  # fmt: skip
def test_rms_norm_gated_reads_strided_rows(leading, n, group_size, heads, x_layout, z_layout):
    """x and z whose rows are contiguous and evenly spaced are read where they lie, and in
    any other layout from a packed copy: the output matches the reference, and neither
    storage changes.

    The cells: Qwen3.8-27B's Gated DeltaNet output norm, whose gate is columns 10240 to 16384
    of the 16480-wide input projection, normed in 48 groups of 128; x's rows 130 elements
    apart, off 16-byte alignment past the first, and z's 136 from column 8, so that the
    call's tensors start aligned and none is staged; one row, whose stride is no layout and
    which the packed program reads; the gate's 48 heads split off its columns, rows not
    evenly spaced; every other column of 256-wide rows, whose rows lie a row apart but are
    not contiguous; a transposed x, whose rows are not contiguous either; and an expanded z,
    whose rows lie zero elements apart and are read in place.
    """
    call = manifest_call(
        "RMSNormGatedFwdOp", _BF16, B=leading, N=n, some=["z"], bias=None, group_size=group_size
    )
    test = RMSNormGatedTest(call)
    x, weight, bias, z = test.gen_inputs()
    x, x_storage = _laid_out(x, x_layout, heads)
    z, z_storage = _laid_out(z, z_layout, heads)
    before = x_storage.clone(), z_storage.clone()
    op = RMSNormGatedFwdOp(**call.arguments())

    def run(*args):
        out = op(*args)
        assert torch.equal(x_storage, before[0]) and torch.equal(z_storage, before[1])
        return out

    test.check(op, x, weight, bias, z, runs=run)


@pytest.mark.parametrize(
    "activation", [pytest.param("silu", marks=_SMOKE), pytest.param("sigmoid", marks=_FULL)]
)
def test_rms_norm_gated_keeps_a_tiny_gate(activation) -> None:
    """Regression: a gate before the norm that is subnormal or below the float32 range of
    exp(-z), on z below -87.3 and x near the float32 maximum, must not zero a row the norm
    rescales."""
    call = manifest_call(
        "RMSNormGatedFwdOp",
        {"T": "float32", "W": "float32", "Z": "float32"},
        B=[8],
        N=16,
        some=["z"],
        bias=None,
        norm_before_gate=False,
        activation=activation,
    )
    test = RMSNormGatedTinyGateTest(call)
    test.check(RMSNormGatedFwdOp(**call.arguments()), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="RMSNormGatedFwdOp")
@pytest.mark.smoke
def test_rms_norm_gated_graph_replay() -> None:
    """Qwen3.8-27B's decode call, the Gated DeltaNet output norm of each of a token's 48 value
    heads gated by silu after the norm."""
    call = manifest_call(
        "RMSNormGatedFwdOp", _BF16, B=[1], N=6144, some=["z"], bias=None, group_size=128
    )
    check_replay(RMSNormGatedTest(call), RMSNormGatedFwdOp(**call.arguments()))
