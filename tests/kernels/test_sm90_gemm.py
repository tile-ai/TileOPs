"""Correctness tests for SM90GemmFwdKernel, the DeepGEMM sm90 bf16 template port."""

import math

import pytest
import torch

from tileops.kernels.moe.sm90_gemm import GemmType, Major, SM90GemmFwdKernel
from tileops.kernels.moe.sm90_gemm_heuristics import (
    GemmDesc,
    SM90GemmSpec,
    get_best_config,
    layout_candidates,
)

pytestmark = pytest.mark.hopper


def _operands(m, n, k, major_a="k", major_b="k"):
    """Logical ``a[M, K]`` and ``b[N, K]``; an MN-major one is a transposed view."""
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    if major_a == "mn":
        a = a.T.contiguous().T
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    if major_b == "mn":
        b = b.T.contiguous().T
    return a, b


def _assert_gemm(out, ref):
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=1e-1)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "m,n,k,config",
    [
        pytest.param(512, 512, 512, dict(block_m=64, block_n=128), id="one-math-warpgroup"),
        pytest.param(1024, 1024, 1024, dict(block_m=128, block_n=128), id="two-math-warpgroups"),
        pytest.param(2048, 2048, 1024, dict(block_m=256, block_n=128), id="block-m-256"),
    ],
)
def test_dense_tile_shapes(m, n, k, config):
    """Each math warp-group arrangement the template offers."""
    a, b = _operands(m, n, k)
    kernel = SM90GemmFwdKernel(GemmType.NORMAL, config=config)
    _assert_gemm(kernel(a, b), a.float() @ b.float().T)


@pytest.mark.parametrize(
    "m,n,cluster",
    [
        pytest.param(4096, 4096, dict(cluster_m=2), id="multicast-b"),
        pytest.param(4096, 4096, dict(cluster_n=2), id="multicast-a"),
        pytest.param(4096 + 3 * 128, 4096 + 5 * 128, dict(cluster_m=2), id="odd-groups-split"),
        pytest.param(1000, 1000, dict(cluster_n=2), id="ragged-with-dead-peer"),
    ],
)
@pytest.mark.full
def test_dense_tma_multicast(m, n, cluster):
    """A 2-CTA cluster shares one operand; odd block counts exercise the peer checks."""
    a, b = _operands(m, n, 1024)
    kernel = SM90GemmFwdKernel(GemmType.NORMAL, config=dict(block_m=128, block_n=128, **cluster))
    _assert_gemm(kernel(a, b), a.float() @ b.float().T)


@pytest.mark.parametrize(
    "major_a,major_b",
    [
        pytest.param("mn", "k", id="tn"),
        pytest.param("k", "mn", id="nn"),
        pytest.param("mn", "mn", id="tt"),
    ],
)
@pytest.mark.full
def test_dense_layouts_from_strides(major_a, major_b):
    """Majorness is read off the operands and picks the transposed TMA/WGMMA path."""
    a, b = _operands(1024, 1024, 1024, major_a, major_b)
    kernel = SM90GemmFwdKernel(GemmType.NORMAL)
    spec = kernel.spec_for(a, b)
    assert spec.major_a is Major(major_a) and spec.major_b is Major(major_b)
    _assert_gemm(kernel(a, b), a.float() @ b.float().T)


@pytest.mark.full
def test_dense_fp32_output():
    a, b = _operands(1024, 1024, 1024)
    kernel = SM90GemmFwdKernel(GemmType.NORMAL, cd_dtype=torch.float32)
    out = kernel(a, b)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, a.float() @ b.float().T, rtol=1e-3, atol=1e-2)


@pytest.mark.full
def test_dynamic_m_shares_one_spec():
    """M is dynamic by default: two row counts resolve to one spec, hence one compiled kernel."""
    kernel = SM90GemmFwdKernel(GemmType.NORMAL, config=dict(block_m=128, block_n=128))
    a1, b = _operands(1024, 1024, 1024)
    a2 = torch.randn(1000, 1024, device="cuda", dtype=torch.bfloat16)
    spec1, spec2 = kernel.spec_for(a1, b), kernel.spec_for(a2, b)
    assert spec1 == spec2 and spec1.shape_m == 0
    _assert_gemm(kernel(a1, b), a1.float() @ b.float().T)
    _assert_gemm(kernel(a2, b), a2.float() @ b.float().T)


def _grouped_operands(sizes, n, k, layout, *, alignment=128, major_b="k", dtype=torch.bfloat16):
    """Per-group rows of ``a`` and their metadata for one grouped layout.

    ``per_row``: each group's segment padded to ``alignment``, a sentinel tail of one
    tile, metadata is the expert id per row. ``tight``: rows packed, metadata is the
    physical end per group. ``aligned``: each group starts at the previous end rounded
    up to ``alignment``, metadata is the physical end per group. Returns
    ``(a, b, metadata, ref, valid)``; ``ref`` is fp32 and defined on ``valid`` rows.
    """
    torch.manual_seed(0)
    groups = len(sizes)
    starts, ends, row = [], [], 0
    for size in sizes:
        start = row if layout == "tight" else math.ceil(row / alignment) * alignment
        starts.append(start)
        ends.append(start + size)
        row = start + (math.ceil(size / alignment) * alignment if layout == "per_row" else size)
    total = row + alignment if layout == "per_row" else row
    if layout == "aligned":
        total = math.ceil(total / alignment) * alignment
    a = torch.randn(total, k, device="cuda", dtype=dtype)
    b = torch.randn(groups, n, k, device="cuda", dtype=dtype)
    if major_b == "mn":
        b = b.transpose(1, 2).contiguous().transpose(1, 2)
    ref = torch.zeros(total, n, dtype=torch.float32, device="cuda")
    valid = torch.zeros(total, dtype=torch.bool, device="cuda")
    for g in range(groups):
        ref[starts[g] : ends[g]] = a[starts[g] : ends[g]].float() @ b[g].float().T
        valid[starts[g] : ends[g]] = True
    if layout == "per_row":
        metadata = torch.full((total,), groups, dtype=torch.int32)
        for g in range(groups):
            metadata[starts[g] : starts[g] + math.ceil(sizes[g] / alignment) * alignment] = g
    else:
        metadata = torch.tensor(ends, dtype=torch.int32)
    return a, b, metadata.cuda(), ref, valid


@pytest.mark.smoke
@pytest.mark.parametrize(
    "sizes,n,k,major_b,config",
    [
        pytest.param(
            [100, 0, 300, 128, 7],
            512,
            512,
            "k",
            dict(block_m=128, block_n=128),
            id="empty-and-partial-groups",
        ),
        pytest.param([300] * 32, 2048, 1024, "mn", None, id="mn-major-b"),
        pytest.param(
            [100, 200, 300, 400, 500, 600, 700, 800],
            2048,
            1024,
            "k",
            dict(block_m=128, block_n=128, cluster_m=2),
            id="multicast-b-group-boundary",
        ),
    ],
)
def test_m_grouped_aligned_per_row(sizes, n, k, major_b, config):
    """Rows follow their group's B; padding rows and the sentinel tail are never read back."""
    a, b, ids, ref, valid = _grouped_operands(sizes, n, k, "per_row", major_b=major_b)
    kernel = SM90GemmFwdKernel(
        GemmType.M_GROUPED_ALIGNED_PER_ROW, num_groups=len(sizes), m_alignment=128, config=config
    )
    out = kernel(a, b, grouped_layout=ids)
    _assert_gemm(out[valid], ref[valid])


@pytest.mark.full
def test_grouped_requires_layout_and_alignment():
    a, b, ids, _, _ = _grouped_operands([64, 64], 256, 256, "per_row")
    kernel = SM90GemmFwdKernel(GemmType.M_GROUPED_ALIGNED_PER_ROW, num_groups=2)
    with pytest.raises(ValueError, match="grouped_layout"):
        kernel(a, b)
    with pytest.raises(ValueError, match="m_alignment"):
        kernel(a[:-64], b, grouped_layout=ids[:-64])


@pytest.mark.full
def test_refuses_operands_tma_cannot_address():
    a = torch.randn(64, 60, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 60, device="cuda", dtype=torch.bfloat16)
    assert "multiple of 8" in SM90GemmFwdKernel.refusal_for(a, b)
    with pytest.raises(ValueError, match="multiple of 8"):
        SM90GemmFwdKernel(GemmType.NORMAL)(a, b)


def _desc(m, n, k, **kw):
    fields = dict(
        gemm_type=GemmType.NORMAL,
        m=m,
        n=n,
        k=k,
        num_groups=1,
        major_a=Major.K,
        major_b=Major.K,
        cd_dtype="bfloat16",
        num_sms=132,
    )
    fields.update(kw)
    return GemmDesc(**fields)


@pytest.mark.full
def test_selector_prunes_like_deepgemm():
    """Every candidate satisfies the register, block_n step and stage-count rules."""
    for layout in layout_candidates(_desc(4096, 4096, 4096)):
        assert not (layout.block_m > 128 and layout.block_n > 128)
        assert layout.block_n % 64 == 0
    fp32 = layout_candidates(_desc(4096, 4096, 4096, cd_dtype="float32"))
    assert all(layout.block_m <= 128 for layout in fp32)


@pytest.mark.full
def test_selector_merges_surplus_stages_for_one_warpgroup():
    """A dense NT GEMM with a single math warp-group trades stages for a wider block_k."""
    spec = get_best_config(_desc(8, 4096, 7168))
    assert spec.num_math_threads == 128
    assert spec.block_k > 64 and spec.num_stages >= 5


@pytest.mark.full
def test_spec_rejects_inconsistent_template_parameters():
    fields = dict(
        gemm_type=GemmType.NORMAL,
        major_a=Major.K,
        major_b=Major.K,
        ab_dtype="bfloat16",
        cd_dtype="bfloat16",
        num_groups=1,
        shape_m=0,
        shape_n=4096,
        shape_k=4096,
        block_m=128,
        block_n=128,
        block_k=64,
        num_stages=4,
        num_math_threads=256,
        num_tma_multicast=1,
        is_tma_multicast_on_a=False,
        num_sms=132,
    )
    SM90GemmSpec(**fields)
    with pytest.raises(ValueError, match="num_math_threads"):
        SM90GemmSpec(**{**fields, "block_m": 64})
    with pytest.raises(ValueError, match="both exceed 128"):
        SM90GemmSpec(**{**fields, "block_m": 256, "block_n": 256})
    with pytest.raises(ValueError, match="K-major A"):
        SM90GemmSpec(
            **{
                **fields,
                "gemm_type": GemmType.M_GROUPED_ALIGNED_PER_ROW,
                "num_groups": 4,
                "major_a": Major.MN,
            }
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "sizes,config",
    [
        pytest.param([100, 0, 300, 128, 7, 64], dict(block_m=128, block_n=128), id="two-wgs"),
        pytest.param([100, 0, 300, 128, 7, 64], dict(block_m=64, block_n=128), id="one-wg"),
    ],
)
def test_m_grouped_tight_psum_masks_each_groups_last_tile(sizes, config):
    """Tight rows: a group's ragged last tile is stored under a row mask, not over its neighbour."""
    a, b, ends, ref, _ = _grouped_operands(sizes, 2048, 1024, "tight")
    kernel = SM90GemmFwdKernel(GemmType.M_GROUPED_TIGHT_PSUM, num_groups=len(sizes), config=config)
    out = torch.full_like(ref, 1e4, dtype=torch.bfloat16)  # poison: every row must be written
    kernel(a, b, grouped_layout=ends, out=out)
    _assert_gemm(out, ref)


@pytest.mark.full
@pytest.mark.parametrize(
    "sizes,config",
    [
        pytest.param([100, 0, 300, 128, 7, 64], dict(block_m=128, block_n=128), id="padded-groups"),
        pytest.param([300] * 32, None, id="selected"),
    ],
)
def test_m_grouped_aligned_psum(sizes, config):
    """Aligned psum rows: each group starts at the previous end rounded up to block_m."""
    a, b, ends, ref, valid = _grouped_operands(sizes, 2048, 1024, "aligned")
    kernel = SM90GemmFwdKernel(
        GemmType.M_GROUPED_ALIGNED_PSUM, num_groups=len(sizes), m_alignment=128, config=config
    )
    out = kernel(a, b, grouped_layout=ends)
    _assert_gemm(out[valid], ref[valid])


@pytest.mark.full
@pytest.mark.parametrize(
    "masked,max_m,config",
    [
        pytest.param([100, 0, 256, 33], 256, dict(block_m=64, block_n=128), id="one-wg"),
        pytest.param([100, 200, 300, 400], 512, dict(block_m=128, block_n=128), id="two-wgs"),
    ],
)
def test_m_grouped_masked(masked, max_m, config):
    """Masked slabs: only the first ``masked_m[g]`` rows of each group are meaningful."""
    torch.manual_seed(0)
    groups = len(masked)
    a = torch.randn(groups, max_m, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(groups, 1024, 512, device="cuda", dtype=torch.bfloat16)
    kernel = SM90GemmFwdKernel(GemmType.M_GROUPED_MASKED, num_groups=groups, config=config)
    out = kernel(a, b, grouped_layout=torch.tensor(masked, dtype=torch.int32).cuda())
    assert out.shape == (groups, max_m, 1024)
    for g, mm in enumerate(masked):
        if mm:
            _assert_gemm(out[g, :mm], a[g, :mm].float() @ b[g].float().T)


@pytest.mark.full
@pytest.mark.parametrize(
    "major_a,major_b,config",
    [
        pytest.param("k", "k", dict(block_m=128, block_n=128), id="nt"),
        pytest.param("mn", "mn", None, id="tt-selected"),
    ],
)
def test_batched(major_a, major_b, config):
    """Batched: one tile grid per batch, no scheduler swizzle, no multicast."""
    torch.manual_seed(0)
    a = torch.randn(4, 1000, 512, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4, 1000, 512, device="cuda", dtype=torch.bfloat16)
    if major_a == "mn":
        a = a.transpose(1, 2).contiguous().transpose(1, 2)
    if major_b == "mn":
        b = b.transpose(1, 2).contiguous().transpose(1, 2)
    kernel = SM90GemmFwdKernel(GemmType.BATCHED, num_groups=4, config=config)
    _assert_gemm(kernel(a, b), torch.bmm(a.float(), b.float().transpose(1, 2)))


@pytest.mark.smoke
def test_fp16_operands_dense_and_tight():
    """fp16 operands take the same kernel; the output follows the operand dtype."""
    a, b = _operands(1024, 1024, 1024)
    a, b = a.half(), b.half()
    out = SM90GemmFwdKernel(GemmType.NORMAL)(a, b)
    assert out.dtype is torch.float16
    _assert_gemm(out, a.float() @ b.float().T)
    ta, tb, ends, ref, _ = _grouped_operands(
        [100, 0, 300, 128, 7, 64], 512, 512, "tight", dtype=torch.float16
    )
    kernel = SM90GemmFwdKernel(GemmType.M_GROUPED_TIGHT_PSUM, num_groups=6)
    _assert_gemm(kernel(ta, tb, grouped_layout=ends), ref)
    with pytest.raises(ValueError, match="one dtype"):
        SM90GemmFwdKernel(GemmType.NORMAL)(a, b.bfloat16())
