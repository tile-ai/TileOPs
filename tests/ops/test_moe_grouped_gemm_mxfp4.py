"""Tests for the MXFP4 M-grouped GEMM op."""

import re

import pytest
import torch

from tests.ops.moe_grouped_gemm_helpers import (
    SENTINEL,
    SIZES,
    contiguous_layout,
    run_into_guarded_out,
    with_counts,
)
from tileops._kernels.constants import BLOCK_SHARED_BYTES_OPT_IN
from tileops._kernels.moe import MGroupedGEMMMXFP4Call, MoEGroupedGEMMMXFP4SharedDecodeKernel
from tileops._kernels.moe.moe_grouped_gemm_mxfp4 import (
    _shared_decode_smem_bytes,
    _smem_bytes,
    _uses_tables,
)
from tileops._ops.moe import MaskedLayoutSpec, MoEGroupedGEMMMXFP4FwdOp
from tileops.backend import BUILTIN
from tileops.manifest import load_workloads
from workloads.device import run_device
from workloads.moe import MoEGroupedGEMMMXFP4Workload
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call

_OP = "MoEGroupedGEMMMXFP4FwdOp"
_E8M0 = {"S": "float8_e8m0fnu"}
_F32 = {"S": "float32"}


def _workload(max_m: int, dtype_case: dict = _E8M0, **dims) -> MoEGroupedGEMMMXFP4Workload:
    # T only feeds the generated counts; half the slabs' capacity keeps every count in range.
    dims.setdefault("T", dims["E"] * max_m // 2)
    return MoEGroupedGEMMMXFP4Workload(
        manifest_call(_OP, dtype_case, layout={"masked": {"max_m": max_m}}, **dims)
    )


_ROWS = {row["label"]: row for row in load_workloads(_OP)}


@pytest.mark.sm90
@pytest.mark.parametrize(
    ("label", "dtype_case"),
    [
        pytest.param("dsv4-dec-b64-up", _E8M0, marks=pytest.mark.smoke),
        pytest.param("dsv4-dec-b256-up", _F32, marks=pytest.mark.smoke),
        pytest.param("dsv4-dec-b256-down", _E8M0, marks=pytest.mark.full),
        pytest.param("dsv4-ep8-4k-down", _E8M0, marks=pytest.mark.full),
    ],
)
def test_mxfp4_grouped_gemm_runs_the_manifest_rows(label: str, dtype_case: dict) -> None:
    """Representative manifest rows, with their generated valid counts, match the reference."""
    row = {key: value for key, value in _ROWS[label].items() if key not in ("label", "dtype_cases")}
    workload = MoEGroupedGEMMMXFP4Workload(manifest_call(_OP, dtype_case, **row))
    inputs = workload.gen_inputs()
    out = MoEGroupedGEMMMXFP4FwdOp(**workload.call.arguments())(*inputs)
    assert out.dtype is torch.bfloat16
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))


@pytest.mark.sm90
@pytest.mark.parametrize(
    ("max_m", "dims", "counts", "dtype_case"),
    [
        # One K step, counts past max_m and an empty expert.
        pytest.param(
            64,
            {"N": 128, "K": 128},
            [64, 0, 1, 99],
            _E8M0,
            marks=pytest.mark.smoke,
            id="max_m-64-k128",
        ),
        # A slab whose rows end inside a row tile, and a negative count.
        pytest.param(50, {}, [50, 17, 0, -5], _F32, marks=pytest.mark.full, id="max_m-50-float32"),
        pytest.param(0, {"N": 128, "K": 128}, [0, 0], _E8M0, marks=pytest.mark.full, id="max_m-0"),
        # K a multiple of 2048, so the e8m0 code and token tables come by TMA: slabs ending
        # inside a row tile (token rows past max_m zero-filled) and N past the last weight tile.
        pytest.param(
            20,
            {"N": 328, "K": 2048},
            [20, 3, 0, 13],
            _E8M0,
            marks=pytest.mark.full,
            id="max_m-20-k2048-tables",
        ),
        # K = 6144: rows of 192 code bytes, which the table copies in three 64-byte boxes.
        pytest.param(
            20,
            {"N": 256, "K": 6144},
            [20, 3, 0, 13],
            _E8M0,
            marks=pytest.mark.full,
            id="max_m-20-k6144-tables",
        ),
        # Seventeen row tiles of a slab, past the per-count unrolled task walk.
        pytest.param(
            272,
            {"N": 128, "K": 128},
            [272, 17, 0, 255] + [0] * 30,
            _E8M0,
            marks=pytest.mark.full,
            id="max_m-272-tiles",
        ),
    ],
)
def test_mxfp4_grouped_gemm_valid_rows_at_tile_and_slab_boundaries(
    max_m: int, dims: dict, counts: list, dtype_case: dict
) -> None:
    dims = {"N": 256, "K": 512, **dims}
    workload = _workload(max_m, dtype_case, E=len(counts), **dims)
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=max_m))
    run_into_guarded_out(op, workload, *with_counts(workload, counts))


@pytest.mark.sm90
@pytest.mark.parametrize(
    "dtype_case",
    [
        pytest.param(_E8M0, marks=pytest.mark.smoke, id="e8m0"),
        pytest.param(_F32, marks=pytest.mark.full, id="float32"),
    ],
)
def test_mxfp4_grouped_gemm_tiles_past_n_over_persistent_waves(dtype_case: dict) -> None:
    """More row and weight tiles than one wave of SMs, the last weight tile past N, and eleven
    K steps, which no ring depth divides: the ring's slots carry over from step to step and
    from task to task."""
    counts = [64, 0, 50, 64, 17, -1, 99, 33]
    workload = _workload(64, dtype_case, E=len(counts), N=1184, K=1408)
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=64))
    out = run_into_guarded_out(op, workload, *with_counts(workload, counts))
    assert bool((out[1] == SENTINEL).all()) and bool((out[5] == SENTINEL).all())


@pytest.mark.sm90
@pytest.mark.parametrize(
    "k",
    [
        pytest.param(128, marks=pytest.mark.smoke, id="k128"),
        # The e8m0 code tables do not fit beside the ring: the scales come a stage at a time
        # and every task, its check read from global memory, contracts sub-block by sub-block.
        pytest.param(16384, marks=pytest.mark.full, id="k16384-per-stage"),
    ],
)
def test_mxfp4_grouped_gemm_far_sub_block_alone(k: int) -> None:
    """The rows' only nonzero products lie in a sub-block thirteen binades under its weight
    row's largest scale: they keep their full value (32 * 2 ** 10 * 2 ** -13 = 4)."""
    workload = _workload(16, E=1, N=128, K=k)
    a, a_scale, b, b_scale, metadata = with_counts(workload, [16])
    a = torch.zeros_like(a.float())
    a[..., 32:64] = 1.0
    a = a.to(torch.float8_e4m3fn)
    a_scale = torch.full_like(a_scale.view(torch.uint8), 137).view(a_scale.dtype)
    b = torch.zeros_like(b)
    b[..., 16:32] = 0x22
    codes = torch.tensor([127, 114, 127, 127], dtype=torch.uint8, device=b.device).repeat(k // 128)
    b_scale = codes.expand_as(b_scale.view(torch.uint8)).contiguous().view(b_scale.dtype)
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=16))
    out = run_into_guarded_out(op, workload, a, a_scale, b, b_scale, metadata)
    assert bool((out == 4).all())


@pytest.mark.sm90
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_scale_code_zero() -> None:
    """An e8m0 weight scale of code 0 is 2 ** -127, a float32 subnormal: under row scales of
    code 254 each product is 1, not flushed to zero."""
    workload = _workload(16, E=1, N=128, K=256)
    a, a_scale, b, b_scale, metadata = with_counts(workload, [16])
    a = torch.ones_like(a.float()).to(a.dtype)
    a_scale = torch.full_like(a_scale.view(torch.uint8), 254).view(a_scale.dtype)
    b = torch.full_like(b, 0x22)
    b_scale = torch.zeros_like(b_scale.view(torch.uint8)).view(b_scale.dtype)
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=16))
    out = run_into_guarded_out(op, workload, a, a_scale, b, b_scale, metadata)
    assert bool((out == 256).all())


@pytest.mark.sm90
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_row_scale_code_zero() -> None:
    """An a_scale of code 0 (2 ** -127) on fast tiles may flush to zero: each output, here
    K * 2 ** 63 * 2 ** -127, is within the documented ``K * 2 ** -52`` of its value."""
    workload = _workload(16, E=1, N=128, K=128)
    a, a_scale, b, b_scale, metadata = with_counts(workload, [16])
    a = torch.ones_like(a.float()).to(a.dtype)
    a_scale = torch.zeros_like(a_scale.view(torch.uint8)).view(a_scale.dtype)
    b = torch.full_like(b, 0x22)
    b_scale = torch.full_like(b_scale.view(torch.uint8), 190).view(b_scale.dtype)
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=16))
    out = op(a, a_scale, b, b_scale, metadata)
    assert bool(((out.float() - 128 * 2.0**-64).abs() <= 128 * 2.0**-52).all())


@pytest.mark.sm90
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_weight_dequantized_past_float32() -> None:
    """A weight of 2 under a scale of 2 ** 127 dequantizes past float32's range, where the
    reference is infinite; under a_scale 2 ** -27 the output is the products' finite sum,
    128 * 2 * 2 ** 127 * 2 ** -27 = 2 ** 108."""
    workload = _workload(16, E=1, N=128, K=128)
    a, a_scale, b, b_scale, metadata = with_counts(workload, [16])
    a = torch.ones_like(a.float()).to(a.dtype)
    a_scale = torch.full_like(a_scale.view(torch.uint8), 100).view(a_scale.dtype)
    b = torch.full_like(b, 0x44)
    b_scale = torch.full_like(b_scale.view(torch.uint8), 254).view(b_scale.dtype)
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=16))
    out = op(a, a_scale, b, b_scale, metadata)
    assert bool((out == 2.0**108).all())


@pytest.mark.sm90
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_scale_code_nan() -> None:
    """An e8m0 scale of code 255 is NaN: a weight row's makes its column NaN, a token's makes
    its row NaN, and no other output."""
    workload = _workload(16, E=1, N=128, K=128)
    a, a_scale, b, b_scale, metadata = with_counts(workload, [16])
    a_codes = torch.full_like(a_scale.view(torch.uint8), 127)
    b_codes = torch.full_like(b_scale.view(torch.uint8), 127)
    a_codes[0, 3, 0] = 255
    b_codes[0, 5, 2] = 255
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=16))
    out = op(a, a_codes.view(a_scale.dtype), b, b_codes.view(b_scale.dtype), metadata)[0]
    expected = torch.zeros_like(out, dtype=torch.bool)
    expected[3, :] = True
    expected[:, 5] = True
    assert torch.equal(out.isnan(), expected)


@pytest.mark.sm90
@pytest.mark.parametrize(
    "dtype_case",
    [
        pytest.param(_E8M0, marks=pytest.mark.smoke, id="e8m0"),
        pytest.param(_F32, marks=pytest.mark.smoke, id="float32"),
    ],
)
def test_mxfp4_grouped_gemm_scales_whose_product_overflows(dtype_case: dict) -> None:
    """Both scales 2 ** 64, whose product overflows float32, on operands of 0.5: each output,
    2 ** 126, is finite and comes out exact."""
    workload = _workload(16, dtype_case, E=1, N=128, K=128)
    a, a_scale, b, b_scale, metadata = with_counts(workload, [16])
    a = torch.zeros_like(a.float())
    a[..., 0] = 0.5
    a = a.to(torch.float8_e4m3fn)
    b = torch.zeros_like(b)
    b[..., 0] = 0x01
    if dtype_case is _E8M0:
        a_scale = torch.full_like(a_scale.view(torch.uint8), 191).view(a_scale.dtype)
        b_scale = torch.full_like(b_scale.view(torch.uint8), 191).view(b_scale.dtype)
    else:
        a_scale = torch.full_like(a_scale, 2.0**64)
        b_scale = torch.full_like(b_scale, 2.0**64)
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=16))
    out = run_into_guarded_out(op, workload, a, a_scale, b, b_scale, metadata)
    assert bool((out == 2.0**126).all())


@pytest.mark.sm90
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_folded_and_split_tasks_on_one_cta() -> None:
    """Persistent CTAs running several tasks each, one weight tile of one expert holding a row
    whose step scales span seven binades: that task contracts its sub-blocks alone, the
    others fold, through the same ring and token-scale slices."""
    workload = _workload(64, E=8, N=3072, K=512)
    a, a_scale, b, b_scale, metadata = workload.gen_inputs()
    codes = b_scale.view(torch.uint8).clone()
    codes[3, 1000, 1] = codes[3, 1000, 0] - 7
    inputs = (a, a_scale, b, codes.view(torch.float8_e8m0fnu), metadata)
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=64))
    run_into_guarded_out(op, workload, *inputs)


@pytest.mark.sm90
@pytest.mark.parametrize(
    ("layout", "dtype_case", "n", "k"),
    [
        pytest.param(
            contiguous_layout("tight", "physical_psum"),
            _E8M0,
            256,
            512,
            marks=pytest.mark.smoke,
            id="tight-psum",
        ),
        pytest.param(
            contiguous_layout("tight", "per_row"),
            _E8M0,
            256,
            512,
            marks=pytest.mark.full,
            id="tight-per-row",
        ),
        pytest.param(
            contiguous_layout("aligned", "per_row", 128),
            _E8M0,
            256,
            512,
            marks=pytest.mark.full,
            id="aligned-per-row-128",
        ),
        pytest.param(
            contiguous_layout("aligned", "physical_psum", 128),
            _E8M0,
            256,
            512,
            marks=pytest.mark.full,
            id="aligned-psum-128",
        ),
        pytest.param(
            contiguous_layout("aligned", "per_row", 32),
            _F32,
            256,
            512,
            marks=pytest.mark.full,
            id="aligned-per-row-32-float32",
        ),
        # 128-row tiles, where each float32 sub-block takes the one accumulator in turn.
        pytest.param(
            contiguous_layout("aligned", "per_row", 128),
            _F32,
            256,
            512,
            marks=pytest.mark.full,
            id="aligned-per-row-128-float32",
        ),
        # K a multiple of 2048 (the e8m0 tables by TMA): token rows past m zero-filled.
        pytest.param(
            contiguous_layout("tight", "per_row"),
            _E8M0,
            256,
            2048,
            marks=pytest.mark.full,
            id="tight-per-row-k2048",
        ),
        # K = 16384: the e8m0 code tables do not fit beside a ring of 128 x 128 tiles, so the
        # scales come a stage at a time and the check reads them from global memory.
        pytest.param(
            contiguous_layout("aligned", "per_row", 128),
            _E8M0,
            256,
            16384,
            marks=pytest.mark.full,
            id="aligned-per-row-128-k16384-per-stage",
        ),
        # Tight tails stored by rows, clipped at an N tail.
        pytest.param(
            contiguous_layout("tight", "per_row"),
            _F32,
            136,
            512,
            marks=pytest.mark.full,
            id="tight-per-row-n136",
        ),
        # Experts found from their row ends, a tight tail stored by rows clipped at an N tail.
        pytest.param(
            contiguous_layout("tight", "physical_psum"),
            _E8M0,
            136,
            512,
            marks=pytest.mark.full,
            id="tight-psum-n136",
        ),
    ],
)
def test_mxfp4_grouped_gemm_contiguous_rows_of_every_expert(
    layout: dict, dtype_case: dict, n: int, k: int
) -> None:
    spec = layout["contiguous"]
    align = spec["alignment"]
    rows = sum(SIZES) if spec["packing"] == "tight" else sum(-(-n // align) * align for n in SIZES)
    stated = {"P": rows} if spec["metadata_kind"] == "physical_psum" else {}
    workload = MoEGroupedGEMMMXFP4Workload(
        manifest_call(_OP, dtype_case, layout=layout, E=len(SIZES), sizes=SIZES, N=n, K=k, **stated)
    )
    op = MoEGroupedGEMMMXFP4FwdOp(workload.call.arguments()["layout"])
    run_into_guarded_out(op, workload, *workload.gen_inputs())


def _aligned_workload(
    n: int, k: int, metadata_kind: str = "per_row"
) -> MoEGroupedGEMMMXFP4Workload:
    """Every expert's rows of ``SIZES``, runs aligned to 128 rows, under e8m0 scales."""
    rows = sum(-(-size // 128) * 128 for size in SIZES)
    stated = {"P": rows} if metadata_kind == "physical_psum" else {}
    return MoEGroupedGEMMMXFP4Workload(
        manifest_call(
            _OP,
            _E8M0,
            layout=contiguous_layout("aligned", metadata_kind, 128),
            E=len(SIZES),
            sizes=SIZES,
            N=n,
            K=k,
            **stated,
        )
    )


def _shared_decode_op(workload: MoEGroupedGEMMMXFP4Workload, inputs: tuple):
    """The op on *workload*'s layout, built on *inputs*, which decode to shared memory."""
    op = MoEGroupedGEMMMXFP4FwdOp(workload.call.arguments()["layout"], target=BUILTIN)
    op(*inputs)
    assert isinstance(next(op.iter_kernels()), MoEGroupedGEMMMXFP4SharedDecodeKernel)
    return op


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("metadata_kind", "n", "k"),
    [
        # 64-byte code-table rows; the last weight tile holds 72 of N's columns.
        pytest.param("per_row", 328, 2048, marks=pytest.mark.smoke, id="per-row-n328-k2048"),
        # 128-byte code-table rows; experts found from their physical row ends.
        pytest.param("physical_psum", 256, 4096, marks=pytest.mark.full, id="psum-k4096"),
        # Code tables that leave room for three token and decoded stages (48 steps).
        pytest.param("per_row", 256, 6144, marks=pytest.mark.full, id="per-row-k6144"),
        # The largest K whose shallowest ring, two stages, fits beside its code tables.
        pytest.param("per_row", 256, 10240, marks=pytest.mark.full, id="per-row-k10240"),
    ],
)
def test_mxfp4_grouped_gemm_decodes_aligned_rows_into_shared_memory(
    metadata_kind: str, n: int, k: int
) -> None:
    """Aligned 128-row runs under e8m0 scales, K a multiple of 2048, decode each weight tile once
    into shared memory and match the reference."""
    workload = _aligned_workload(n, k, metadata_kind)
    inputs = workload.gen_inputs()
    run_into_guarded_out(_shared_decode_op(workload, inputs), workload, *inputs)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_shared_decode_unfolded_tasks() -> None:
    """288 tasks on persistent CTAs, so each runs two or three and reuses its code tables, flags
    and column scales: one weight row whose codes span twelve binades over K puts its tasks on
    the sub-block by sub-block contraction, the others fold. The rows' only nonzero products
    lie in one sub-block, under codes whose largest, so each column's scale, steps from one
    weight tile to the next (by sevens, so a CTA's tasks, four tiles apart, differ); the
    spanning row's sub-block keeps its full value (32 * 0.5 * its scale), which folding would
    lose six binades under its row's reference."""
    workload = _aligned_workload(4096, 2048)
    a, a_scale, b, b_scale, ids = workload.gen_inputs()
    a = torch.zeros_like(a.float())
    a[:, 288:320] = 1.0
    a = a.to(torch.float8_e4m3fn)
    b = torch.full_like(b, 0x11)
    a_codes = torch.full_like(a_scale.view(torch.uint8), 127)
    steps = torch.arange(b.shape[1], device=b.device).div(128, rounding_mode="floor").remainder(7)
    steps = steps.to(torch.uint8)
    b_codes = torch.randint_like(b_scale.view(torch.uint8), 0, 4) + 115 + steps[:, None]
    b_codes[4, 700, 9] = b_codes[4, 700].max() - 12
    inputs = (a, a_codes.view(a_scale.dtype), b, b_codes.view(b_scale.dtype), ids)
    out = run_into_guarded_out(_shared_decode_op(workload, inputs), workload, *inputs)
    assert bool((out[ids == 4, 700] == 2.0 ** (int(b_codes[4, 700, 9]) - 123)).all())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_shared_decode_folds_a_row_spanning_eleven_binades() -> None:
    """Weight row 77's codes span eleven binades over K, the most a fold takes: its lowest
    sub-block sits five binades under the row's reference, its highest six over. Even tokens'
    only nonzero products lie in the lowest sub-block, odd tokens' in the highest, and each
    keeps its exact value: 32 * 0.5 * 2 ** -12 and 32 * 0.5 * 2 ** -1."""
    workload = _aligned_workload(256, 2048)
    a, a_scale, b, b_scale, ids = workload.gen_inputs()
    a = torch.zeros_like(a.float())
    a[0::2, 288:320] = 1.0
    a[1::2, 1280:1312] = 1.0
    a = a.to(torch.float8_e4m3fn)
    b = torch.full_like(b, 0x11)
    a_codes = torch.full_like(a_scale.view(torch.uint8), 127)
    b_codes = torch.full_like(b_scale.view(torch.uint8), 120)
    b_codes[:, 77, 9] = 115
    b_codes[:, 77, 40] = 126
    inputs = (a, a_codes.view(a_scale.dtype), b, b_codes.view(b_scale.dtype), ids)
    out = _shared_decode_op(workload, inputs)(*inputs)
    owned = ids < len(SIZES)
    even = owned & (torch.arange(ids.numel(), device=ids.device) % 2 == 0)
    assert bool((out[even, 77] == 2.0**-8).all())
    assert bool((out[owned & ~even, 77] == 8.0).all())
    assert bool((out[owned][:, :77] == 2.0**-3).all())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_shared_decode_rows_of_distinct_references() -> None:
    """Weight row r's codes are all 120 + r % 5, so neighbouring rows fold to different
    references and a sub-block decoded under another row's reference lands binades off. Every
    product is 0.5 * 2 ** (code - 127) under unit token scales, and each column's exact sum over
    K = 2048 is 2 ** (code - 117)."""
    workload = _aligned_workload(256, 2048)
    a, a_scale, b, b_scale, ids = workload.gen_inputs()
    a = torch.ones_like(a.float()).to(a.dtype)
    b = torch.full_like(b, 0x11)
    a_codes = torch.full_like(a_scale.view(torch.uint8), 127)
    rows = torch.arange(b.shape[1], device=b.device)
    codes = (120 + rows % 5).to(torch.uint8)
    b_codes = codes[None, :, None].expand_as(b_scale.view(torch.uint8)).contiguous()
    inputs = (a, a_codes.view(a_scale.dtype), b, b_codes.view(b_scale.dtype), ids)
    out = _shared_decode_op(workload, inputs)(*inputs)[ids < len(SIZES)]
    expected = torch.exp2(codes.float() - 117).to(out.dtype)
    assert bool((out == expected[None, :]).all())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_shared_decode_weight_scales_past_the_fold() -> None:
    """Whole weight rows under e8m0 codes the fold does not take, each on the sub-block
    contraction with the other rows of its tile: code 255 (NaN) makes its column NaN, and
    codes 0 (2 ** -127, subnormal), 30 and 200 give the exact sums 2048 * 2 ** -127,
    2048 * 2 ** -97 and 2048 * 2 ** 73; the second weight tile folds."""
    workload = _aligned_workload(256, 2048)
    a, a_scale, b, b_scale, ids = workload.gen_inputs()
    a = torch.ones_like(a.float()).to(a.dtype)
    b = torch.full_like(b, 0x22)
    a_codes = torch.full_like(a_scale.view(torch.uint8), 127)
    b_codes = torch.full_like(b_scale.view(torch.uint8), 127)
    for row, code in ((3, 255), (7, 0), (11, 30), (13, 200)):
        b_codes[:, row, :] = code
    inputs = (a, a_codes.view(a_scale.dtype), b, b_codes.view(b_scale.dtype), ids)
    out = _shared_decode_op(workload, inputs)(*inputs)[ids < len(SIZES)]
    assert bool(out[:, 3].isnan().all())
    for row, value in ((7, 2.0**-116), (11, 2.0**-86), (13, 2.0**84)):
        assert bool((out[:, row] == value).all())
    assert bool((out[:, 128:] == 2048).all())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_shared_decode_token_scale_past_the_fold() -> None:
    """A token scale of 2 ** 123 beside weight scales of 2 ** -63: folded to the row's
    reference, a step's products under the token scale alone overflow float32. The task
    contracts sub-block by sub-block instead, and each output is the products' finite sum,
    2048 * 448 * 6 * 2 ** 60."""
    workload = _aligned_workload(256, 2048)
    a, a_scale, b, b_scale, ids = workload.gen_inputs()
    a = torch.full_like(a.float(), 448.0).to(a.dtype)
    b = torch.full_like(b, 0x77)
    a_scale = torch.full_like(a_scale.view(torch.uint8), 250).view(a_scale.dtype)
    b_scale = torch.full_like(b_scale.view(torch.uint8), 64).view(b_scale.dtype)
    inputs = (a, a_scale, b, b_scale, ids)
    out = _shared_decode_op(workload, inputs)(*inputs)
    expected = torch.tensor(2048 * 448 * 6 * 2.0**60, dtype=torch.bfloat16)
    assert bool((out[ids < len(SIZES)] == expected.to(out.device)).all())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_shared_decode_token_code_zero() -> None:
    """A token code of 0 (2 ** -127), an all-zero block's, keeps its task folded, where the
    scale flushes to zero as the register path's does: each output, K * 2 ** 63 * 2 ** -127,
    comes out 0, within the documented ``K * 2 ** -52``; the unfolded contraction would keep
    it."""
    workload = _aligned_workload(256, 2048)
    a, a_scale, b, b_scale, ids = workload.gen_inputs()
    a = torch.ones_like(a.float()).to(a.dtype)
    a_scale = torch.zeros_like(a_scale.view(torch.uint8)).view(a_scale.dtype)
    b = torch.full_like(b, 0x22)
    b_scale = torch.full_like(b_scale.view(torch.uint8), 190).view(b_scale.dtype)
    inputs = (a, a_scale, b, b_scale, ids)
    out = _shared_decode_op(workload, inputs)(*inputs)[ids < len(SIZES)]
    assert bool((out == 0).all())


@pytest.mark.sm90
@pytest.mark.parametrize(
    "align",
    [
        # Runs of whole 128-row tiles: the shared-decode kernel.
        pytest.param(256, marks=pytest.mark.smoke, id="shared-decode"),
        # 64-row tiles: the register-fragment kernel.
        pytest.param(64, marks=pytest.mark.full, id="register-fragments"),
    ],
)
def test_mxfp4_grouped_gemm_skips_trailing_padding_tiles(align: int) -> None:
    """Aligned per-row ids ending in padding row tiles marked E: those tiles are skipped, and
    with more tasks than SMs (64 weight tiles a row tile) each CTA meets skipped tasks after
    the ones it runs, which load no code table."""
    sizes, pad = [5, 300, 0, 77], 512
    workload = MoEGroupedGEMMMXFP4Workload(
        manifest_call(
            _OP,
            _E8M0,
            layout=contiguous_layout("aligned", "per_row", align),
            E=len(sizes),
            sizes=sizes,
            N=8192,
            K=2048,
        )
    )
    a, a_scale, b, b_scale, ids = workload.gen_inputs()
    a = torch.cat([a, a[:pad]])
    a_scale = torch.cat([a_scale, a_scale[:pad]])
    ids = torch.cat([ids, torch.full((pad,), len(sizes), dtype=ids.dtype, device=ids.device)])
    op = MoEGroupedGEMMMXFP4FwdOp(workload.call.arguments()["layout"])
    run_into_guarded_out(op, workload, a, a_scale, b, b_scale, ids)


@pytest.mark.sm90
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_tune_picks_a_correct_config() -> None:
    """The config a tuning picks computes the GEMM correctly."""
    workload = _workload(64, E=4, N=256, K=256)
    inputs = with_counts(workload, [64, 0, 33, 10])
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=64))
    op.request_tune()
    out = op(*inputs)
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_shared_decode_tune_picks_a_correct_config() -> None:
    """Tuning aligned rows under e8m0 scales, which decode into shared memory, picks a config
    that computes the GEMM correctly."""
    workload = _aligned_workload(256, 2048)
    inputs = workload.gen_inputs()
    op = MoEGroupedGEMMMXFP4FwdOp(workload.call.arguments()["layout"], target=BUILTIN)
    op.request_tune()
    run_into_guarded_out(op, workload, *inputs)
    assert isinstance(next(op.iter_kernels()), MoEGroupedGEMMMXFP4SharedDecodeKernel)


@pytest.mark.smoke
def test_mxfp4_grouped_gemm_rejects_scales_of_two_dtypes() -> None:
    """Both scales take one dtype: an e8m0 a_scale beside a float32 b_scale is refused."""
    workload = _workload(16, E=2, N=128, K=128)
    a, a_scale, b, _, metadata = workload.gen_inputs()
    b_scale = torch.ones(2, 128, 4, dtype=torch.float32, device=a.device)
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=16))
    with pytest.raises(ValueError, match="b_scale"):
        op(a, a_scale, b, b_scale, metadata)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_refuses_n_off_the_store_unit() -> None:
    """The output's TMA store moves whole 16-byte units: N off a multiple of 8 is refused
    before anything is built."""
    workload = _workload(16, E=2, N=100, K=128)
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=16))
    with pytest.raises(Exception, match="multiple of 8"):
        op(*workload.gen_inputs())
    assert not list(op.iter_kernels())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_refuses_an_empty_contraction() -> None:
    """K = 0 satisfies the signature (K a multiple of 128) but leaves no K step to contract: it
    is refused before anything is built."""
    device = run_device()
    a = torch.empty(2, 16, 0, dtype=torch.float8_e4m3fn, device=device)
    a_scale = torch.empty(2, 16, 0, dtype=torch.float32, device=device)
    b = torch.empty(2, 128, 0, dtype=torch.uint8, device=device)
    b_scale = torch.empty(2, 128, 0, dtype=torch.float32, device=device)
    metadata = torch.tensor([16, 3], dtype=torch.int32, device=device)
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=16))
    with pytest.raises(ValueError, match="positive multiple of 128"):
        op(a, a_scale, b, b_scale, metadata)
    assert not list(op.iter_kernels())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_refuses_experts_whose_row_ends_overflow_shared_memory() -> None:
    """Tight per-row ids keep every expert's row end in shared memory: 65536 experts' ends
    leave no room beside the default tile, and the call is refused before anything is built."""
    experts = 65536
    workload = MoEGroupedGEMMMXFP4Workload(
        manifest_call(
            _OP,
            _E8M0,
            layout=contiguous_layout("tight", "per_row"),
            E=experts,
            sizes=[0] * (experts - 1) + [16],
            N=8,
            K=128,
        )
    )
    op = MoEGroupedGEMMMXFP4FwdOp(workload.call.arguments()["layout"])
    with pytest.raises(ValueError, match="row end in shared memory.*beside the default tile"):
        op(*workload.gen_inputs())
    assert not list(op.iter_kernels())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_refuses_runs_aligned_off_the_row_tile() -> None:
    """Aligned runs are tiled in whole 8-row tiles: alignment 4 is refused before anything is
    built."""
    workload = MoEGroupedGEMMMXFP4Workload(
        manifest_call(
            _OP,
            _E8M0,
            layout=contiguous_layout("aligned", "per_row", 4),
            E=2,
            sizes=[5, 3],
            N=128,
            K=128,
        )
    )
    op = MoEGroupedGEMMMXFP4FwdOp(workload.call.arguments()["layout"])
    with pytest.raises(ValueError, match="aligned runs"):
        op(*workload.gen_inputs())
    assert not list(op.iter_kernels())


def _compiled_smem(program) -> int:
    """The dynamic shared memory a built program launches with plus its static arrays."""
    host = program.get_host_source()
    # The launch's last integer argument is its dynamic shared memory.
    dynamic = int(
        re.findall(r"\(\(int64_t\)(\d+)\)", host[: host.rfind("_kernel_packed == NULL")])[-1]
    )
    element_bytes = {"uint64_t": 8, "float": 4, "int": 4, "uchar": 1}
    arrays = re.findall(
        r"__shared__ (?:__align__\(\d+\) )?(\w+) \w+\[(\d+)\]", program.get_kernel_source()
    )
    return dynamic + sum(element_bytes[dtype] * int(count) for dtype, count in arrays)


@pytest.mark.sm90
@pytest.mark.parametrize(
    ("k", "scale_dtype", "block_m", "block_n"),
    [
        pytest.param(2048, torch.float8_e8m0fnu, 64, 256, marks=pytest.mark.smoke, id="tma-tables"),
        # Code-table rows of 176 bytes come by one bulk copy and take no token table.
        pytest.param(1408, torch.float8_e8m0fnu, 64, 192, marks=pytest.mark.full, id="bulk-tables"),
        pytest.param(512, torch.float32, 64, 192, marks=pytest.mark.full, id="scales-a-stage"),
    ],
)
def test_mxfp4_grouped_gemm_smem_model_matches_the_build(
    k: int, scale_dtype: torch.dtype, block_m: int, block_n: int
) -> None:
    """``_smem_bytes``, which the config refusal and the stage choice read, counts the shared
    memory the built program takes, plus 1024 bytes of code-table alignment under the tables."""
    call = MGroupedGEMMMXFP4Call(
        device=torch.device(run_device()),
        kind="masked",
        max_m=64,
        scale_dtype=scale_dtype,
        num_groups=4,
        m=256,
        n=256,
        k=k,
    )
    op = MoEGroupedGEMMMXFP4FwdOp(MaskedLayoutSpec(max_m=64))
    kernel = op.kernel_for("grouped_gemm_mxfp4", lambda: call)
    config = {"block_m": block_m, "block_n": block_n, "num_stages": 4, "num_sms": 4}
    scale_bytes, scale_k = kernel.scale_bytes, kernel.scale_k
    tables = _uses_tables(block_m, block_n, 4, scale_bytes, scale_k, kernel.smem_limit)
    assert tables == (scale_bytes == 1)
    modelled = _smem_bytes(block_m, block_n, 4, scale_bytes, scale_k, tables)
    assert modelled - _compiled_smem(kernel.kernel(**config)) == (1024 if tables else 0)


# SM90's opt-in shared memory a block, less the kernels' headroom.
_SM90_SMEM_LIMIT = (
    BLOCK_SHARED_BYTES_OPT_IN[90] - MoEGroupedGEMMMXFP4SharedDecodeKernel._SMEM_HEADROOM
)


def _shared_decode_refusal(num_stages: int, scale_k: int, smem_limit: int) -> str | None:
    """Why a 128 x 128 shared-decode config of ``num_stages`` is refused on 132 SMs."""
    config = {"block_m": 128, "block_n": 128, "num_stages": num_stages, "num_sms": 132}
    return MoEGroupedGEMMMXFP4SharedDecodeKernel.config_refusal(
        config, sm_count=132, smem_limit=smem_limit, scale_k=scale_k, alignment=128
    )


def _aligned_kernel(k: int):
    """The implementation the op selects for 512 rows of four experts in aligned 128-row runs
    under e8m0 scales, N = 256."""
    call = MGroupedGEMMMXFP4Call(
        device=torch.device(run_device()),
        kind="contiguous",
        packing="aligned",
        metadata_kind="per_row",
        alignment=128,
        scale_dtype=torch.float8_e8m0fnu,
        num_groups=4,
        m=512,
        n=256,
        k=k,
    )
    layout = _aligned_workload(256, k).call.arguments()["layout"]
    op = MoEGroupedGEMMMXFP4FwdOp(layout, target=BUILTIN)
    return op.kernel_for("grouped_gemm_mxfp4", lambda: call)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "k",
    [
        pytest.param(2048, marks=pytest.mark.smoke, id="k2048"),
        pytest.param(4096, marks=pytest.mark.full, id="k4096"),
    ],
)
def test_mxfp4_grouped_gemm_shared_decode_smem_model_matches_the_build(k: int) -> None:
    """``_shared_decode_smem_bytes``, which the dispatch, the config refusal and the stage choice
    read, counts the shared memory the built program takes, plus 1024 bytes of code-table
    alignment."""
    kernel = _aligned_kernel(k)
    assert isinstance(kernel, MoEGroupedGEMMMXFP4SharedDecodeKernel)
    config = kernel.default_config
    modelled = _shared_decode_smem_bytes(config["block_n"], config["num_stages"], kernel.scale_k)
    assert modelled - _compiled_smem(kernel.kernel(**config)) == 1024


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_mxfp4_grouped_gemm_shared_decode_takes_the_modelled_shared_memory() -> None:
    """A shared-decode config is refused one byte under the shared memory
    ``_shared_decode_smem_bytes`` models and taken at it; a call whose shallowest ring does not
    fit (K = 16384, whose code tables alone pass 160 KiB) keeps the register-fragment kernel."""
    scale_k = 4096 // 128
    need = _shared_decode_smem_bytes(128, 4, scale_k)
    assert "shared memory" in _shared_decode_refusal(4, scale_k, need - 1)
    assert _shared_decode_refusal(4, scale_k, need) is None
    # The tuning candidates at K = 2048 are the ring depths that fit beside the tiles and whose
    # unrolled loop tiles its 16 steps; depth 8 tiles them but does not fit.
    assert [c["num_stages"] for c in _aligned_kernel(2048).autotune_configs] == [2, 4]
    assert not isinstance(_aligned_kernel(16384), MoEGroupedGEMMMXFP4SharedDecodeKernel)


@pytest.mark.smoke
def test_mxfp4_grouped_gemm_shared_decode_refuses_a_ring_its_unrolled_loop_cannot_tile() -> None:
    """Each task of the shared decode starts every ring at its first stage, which holds while its
    K loop, unrolled by twice the ring depths' least common multiple, tiles the K steps: a token
    ring 3 deep at K = 2048 (16 steps, an unroll of 12) is refused, the refusal naming the
    unroll, and rings 2 and 4 deep are taken."""
    scale_k = 2048 // 128
    assert "unrolls the K steps by 12" in _shared_decode_refusal(3, scale_k, _SM90_SMEM_LIMIT)
    assert _shared_decode_refusal(2, scale_k, _SM90_SMEM_LIMIT) is None
    assert _shared_decode_refusal(4, scale_k, _SM90_SMEM_LIMIT) is None


@pytest.mark.smoke
def test_mxfp4_grouped_gemm_takes_bulk_tables_under_the_tma_bound() -> None:
    """A code table whose rows no TMA box divides comes by one bulk copy without token code
    tables; it is taken only where those token tables would fit beside it too. On a 128 x 128
    tile and a 7-deep ring, K = 1792's tables fit by their own size but not with token tables,
    so the scales come a stage at a time; a smaller tile takes them."""
    smem_limit = _SM90_SMEM_LIMIT
    scale_k = 1792 // 128
    assert _smem_bytes(128, 128, 7, 1, scale_k, True) <= smem_limit
    assert not _uses_tables(128, 128, 7, 1, scale_k, smem_limit)
    assert _uses_tables(64, 128, 4, 1, scale_k, smem_limit)
