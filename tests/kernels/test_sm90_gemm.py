"""Correctness tests for SM90GemmFwdKernel, the DeepGEMM ``sm90_bf16_gemm_impl`` port."""

import math

import pytest
import torch

from tileops.kernels.moe.sm90_gemm import GemmType, Major, SM90GemmFwdKernel
from tileops.kernels.moe.sm90_gemm_heuristics import (
    GemmDesc,
    SM90GemmSpec,
    layout_candidates,
    spec_from_config,
)

pytestmark = pytest.mark.hopper


def _batched_operands(g, m, n, k, major_a="k", major_b="k", dtype=torch.bfloat16):
    """Logical ``a[G, M, K]`` and ``b[G, N, K]``; an MN-major one is a transposed view."""
    torch.manual_seed(0)
    a = torch.randn(g, m, k, device="cuda", dtype=dtype)
    if major_a == "mn":
        a = a.transpose(1, 2).contiguous().transpose(1, 2)
    b = torch.randn(g, n, k, device="cuda", dtype=dtype)
    if major_b == "mn":
        b = b.transpose(1, 2).contiguous().transpose(1, 2)
    return a, b


def _bmm_ref(a, b):
    return torch.bmm(a.float(), b.float().transpose(1, 2))


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
def test_batched_tile_shapes(m, n, k, config):
    """Each math warp-group arrangement the template offers, on the batched scheduler."""
    a, b = _batched_operands(2, m, n, k)
    kernel = SM90GemmFwdKernel(GemmType.BATCHED, num_groups=2, config=config)
    _assert_gemm(kernel(a, b), _bmm_ref(a, b))


@pytest.mark.parametrize(
    "major_a,major_b",
    [
        pytest.param("mn", "k", id="tn"),
        pytest.param("k", "mn", id="nn"),
        pytest.param("mn", "mn", id="tt"),
    ],
)
@pytest.mark.full
def test_batched_layouts_from_strides(major_a, major_b):
    """Majorness is read off the operands and picks the transposed TMA/WGMMA path."""
    a, b = _batched_operands(3, 1000, 1000, 1024, major_a, major_b)
    kernel = SM90GemmFwdKernel(GemmType.BATCHED, num_groups=3)
    spec = kernel.spec_for(a, b)
    assert spec.major_a is Major(major_a) and spec.major_b is Major(major_b)
    _assert_gemm(kernel(a, b), _bmm_ref(a, b))


@pytest.mark.full
def test_fp32_output():
    a, b = _batched_operands(2, 1000, 1000, 1024)
    kernel = SM90GemmFwdKernel(GemmType.BATCHED, num_groups=2, cd_dtype=torch.float32)
    out = kernel(a, b)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, _bmm_ref(a, b), rtol=1e-3, atol=1e-2)


@pytest.mark.full
def test_dynamic_m_shares_one_spec():
    """M is dynamic by default: two row counts resolve to one spec, hence one compiled kernel."""
    kernel = SM90GemmFwdKernel(
        GemmType.BATCHED, num_groups=2, config=dict(block_m=128, block_n=128)
    )
    a1, b = _batched_operands(2, 1024, 1024, 1024)
    a2 = torch.randn(2, 1000, 1024, device="cuda", dtype=torch.bfloat16)
    spec1, spec2 = kernel.spec_for(a1, b), kernel.spec_for(a2, b)
    assert spec1 == spec2 and spec1.shape_m == 0
    _assert_gemm(kernel(a1, b), _bmm_ref(a1, b))
    _assert_gemm(kernel(a2, b), _bmm_ref(a2, b))


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
    a = torch.randn(2, 64, 60, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 60, device="cuda", dtype=torch.bfloat16)
    assert "multiple of 8" in SM90GemmFwdKernel.refusal_for(a, b)
    with pytest.raises(ValueError, match="multiple of 8"):
        SM90GemmFwdKernel(GemmType.BATCHED, num_groups=2)(a, b)


def _desc(m, n, k, **kw):
    fields = dict(
        gemm_type=GemmType.BATCHED,
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
    # Every candidate takes the 64-wide K block; 128 is pinned-config only (see
    # layout_candidates for the power-cap measurement behind that).
    grouped = dict(gemm_type=GemmType.M_GROUPED_TIGHT_PSUM, num_groups=128)
    decode = layout_candidates(_desc(4096, 4096, 7168, **grouped))
    assert {layout.block_k for layout in decode} == {64}


@pytest.mark.full
def test_spec_rejects_inconsistent_template_parameters():
    fields = dict(
        gemm_type=GemmType.BATCHED,
        major_a=Major.K,
        major_b=Major.K,
        ab_dtype="bfloat16",
        cd_dtype="bfloat16",
        num_groups=2,
        shape_m=0,
        shape_n=4096,
        shape_k=4096,
        block_m=128,
        block_n=128,
        block_k=64,
        num_stages=4,
        num_math_threads=256,
        num_sms=132,
    )
    SM90GemmSpec(**fields)
    with pytest.raises(ValueError, match="num_math_threads"):
        SM90GemmSpec(**{**fields, "block_m": 64})
    with pytest.raises(ValueError, match="both exceed 128"):
        SM90GemmSpec(**{**fields, "block_m": 256, "block_n": 256})
    with pytest.raises(ValueError, match="block_k must be 64 or 128"):
        SM90GemmSpec(**{**fields, "block_k": 96})
    # A pinned pipeline past the shared-memory budget, or the single stage the
    # 128x256x128 tile has room for, is refused, not launched (it deadlocked).
    with pytest.raises(ValueError, match="at most"):
        spec_from_config(_desc(4096, 4096, 4096), dict(block_m=128, block_n=256, num_stages=8))
    with pytest.raises(ValueError, match="at least 2"):
        spec_from_config(_desc(4096, 4096, 4096), dict(block_m=128, block_n=256, block_k=128))
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


@pytest.mark.smoke
def test_m_grouped_tight_per_row_recovers_the_psum_schedule():
    """Per-row ids on tight rows: the recovered ends give the psum type's exact output."""
    sizes = [100, 0, 300, 128, 7, 64]
    a, b, ends, ref, _ = _grouped_operands(sizes, 2048, 1024, "tight")
    ids = torch.repeat_interleave(
        torch.arange(len(sizes), dtype=torch.int32, device="cuda"),
        torch.tensor(sizes, device="cuda"),
    )
    per_row = SM90GemmFwdKernel(GemmType.M_GROUPED_TIGHT_PER_ROW, num_groups=len(sizes))
    out = per_row(a, b, grouped_layout=ids)
    _assert_gemm(out, ref)
    psum = SM90GemmFwdKernel(GemmType.M_GROUPED_TIGHT_PSUM, num_groups=len(sizes))
    assert torch.equal(out, psum(a, b, grouped_layout=ends))


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


@pytest.mark.smoke
def test_fp16_operands_batched_and_tight():
    """fp16 operands take the same kernel; the output follows the operand dtype."""
    a, b = _batched_operands(2, 1024, 1024, 1024, dtype=torch.float16)
    out = SM90GemmFwdKernel(GemmType.BATCHED, num_groups=2)(a, b)
    assert out.dtype is torch.float16
    _assert_gemm(out, _bmm_ref(a, b))
    ta, tb, ends, ref, _ = _grouped_operands(
        [100, 0, 300, 128, 7, 64], 512, 512, "tight", dtype=torch.float16
    )
    kernel = SM90GemmFwdKernel(GemmType.M_GROUPED_TIGHT_PSUM, num_groups=6)
    _assert_gemm(kernel(ta, tb, grouped_layout=ends), ref)
    with pytest.raises(ValueError, match="one dtype"):
        SM90GemmFwdKernel(GemmType.BATCHED, num_groups=2)(a, b.bfloat16())


def _gated_ref(ref, activation):
    """``act(gate) * up`` over a GEMM reference whose columns stack gate then up."""
    gate, up = ref.chunk(2, dim=-1)
    act = torch.nn.functional.silu if activation == "silu_and_mul" else torch.nn.functional.gelu
    return act(gate) * up


@pytest.mark.smoke
@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(dict(block_m=128, block_n=128), id="two-wgs"),
        pytest.param(dict(block_m=64, block_n=256), id="one-wg"),
    ],
)
def test_fused_gated_activation_tight(activation, config):
    """The fused epilogue half-loads gate and up into one B tile and stores N / 2 columns.

    Tight ragged groups take the row-masked store path with the halved tile width.
    """
    sizes = [100, 0, 300, 128, 7, 64]
    a, b, ends, ref, _ = _grouped_operands(sizes, 1024, 512, "tight")
    kernel = SM90GemmFwdKernel(
        GemmType.M_GROUPED_TIGHT_PSUM, num_groups=len(sizes), activation=activation, config=config
    )
    out = kernel(a, b, grouped_layout=ends)
    assert out.shape == (a.shape[0], 512)
    _assert_gemm(out, _gated_ref(ref, activation))


@pytest.mark.full
def test_fused_gated_activation_masked_and_fp32_output():
    """Masked groups and an fp32 ``out`` take the fused epilogue without a cast."""
    masked = [64, 0, 17, 33]
    a, b = _batched_operands(len(masked), 64, 1024, 512)
    counts = torch.tensor(masked, dtype=torch.int32, device="cuda")
    kernel = SM90GemmFwdKernel(
        GemmType.M_GROUPED_MASKED, num_groups=len(masked), activation="silu_and_mul"
    )
    out = kernel(a, b, grouped_layout=counts)
    ref = _gated_ref(_bmm_ref(a, b), "silu_and_mul")
    for g, rows in enumerate(masked):
        _assert_gemm(out[g, :rows], ref[g, :rows])
    fp32 = SM90GemmFwdKernel(
        GemmType.BATCHED, num_groups=len(masked), activation="silu_and_mul", cd_dtype=torch.float32
    )(a, b)
    assert fp32.dtype is torch.float32
    _assert_gemm(fp32, ref)


@pytest.mark.full
def test_fused_gated_activation_refusals():
    """A fused call needs an even split of N into gate and up, a K-major B, a known name."""
    with pytest.raises(ValueError, match="activation"):
        SM90GemmFwdKernel(GemmType.BATCHED, num_groups=2, activation="relu")
    a, b = _batched_operands(2, 64, 1032, 512)
    with pytest.raises(ValueError, match="multiple of 16"):
        SM90GemmFwdKernel(GemmType.BATCHED, num_groups=2, activation="silu_and_mul")(a, b)
    a, b = _batched_operands(2, 64, 1024, 512, major_b="mn")
    with pytest.raises(ValueError, match="K-major B"):
        SM90GemmFwdKernel(GemmType.BATCHED, num_groups=2, activation="silu_and_mul")(a, b)


def _k_grouped_operands(sizes, m, n, major_a="mn", major_b="mn", dtype=torch.bfloat16):
    """Groups packed along K: logical ``a[M, sum_k]``, ``b[N, sum_k]``, fp32 ``ref[G, M, N]``.

    The MN-major storage is ``[sum_k, M]`` / ``[sum_k, N]`` (``GroupedGemmFwdOp``'s
    TN); a K-major ``b`` is ``[N, sum_k]`` (its TT).
    """
    torch.manual_seed(0)
    sum_k = sum(sizes)
    a = torch.randn(m, sum_k, device="cuda", dtype=dtype)
    if major_a == "mn":
        a = a.T.contiguous().T
    b = torch.randn(n, sum_k, device="cuda", dtype=dtype)
    if major_b == "mn":
        b = b.T.contiguous().T
    ref = torch.zeros(len(sizes), m, n, device="cuda")
    start = 0
    for g, size in enumerate(sizes):
        ref[g] = a[:, start : start + size].float() @ b[:, start : start + size].float().T
        start += size
    return a, b, torch.tensor(sizes, dtype=torch.int32, device="cuda"), ref


@pytest.mark.smoke
@pytest.mark.parametrize(
    "major_a,major_b,config",
    [
        pytest.param("mn", "mn", dict(block_m=64, block_n=128), id="tn-one-wg"),
        pytest.param("mn", "mn", dict(block_m=128, block_n=128), id="tn-two-wgs"),
        pytest.param("mn", "k", None, id="tt-k-major-b"),
        pytest.param("k", "mn", None, id="k-major-a"),
    ],
)
def test_k_grouped_contiguous(major_a, major_b, config):
    """Groups along K: each group's K tail is masked in shared memory, a K-major
    operand's group start is rounded down to 8 and its head masked, a group with
    no tokens stores zeros, and ragged M / N edges are clipped by the TMA store.
    """
    sizes = [100, 0, 300, 64, 7, 1]
    a, b, ks, ref = _k_grouped_operands(sizes, 200, 136, major_a, major_b)
    kernel = SM90GemmFwdKernel(GemmType.K_GROUPED_CONTIGUOUS, num_groups=len(sizes), config=config)
    out = torch.full_like(ref, 1e4, dtype=torch.bfloat16)  # poison: every tile must be written
    kernel(a, b, grouped_layout=ks, out=out)
    _assert_gemm(out, ref)


@pytest.mark.full
def test_k_grouped_contiguous_without_tokens_and_refusals():
    """Every group empty gives zeros without a launch; the fused epilogue is not offered."""
    a, b, ks, ref = _k_grouped_operands([0, 0, 0], 64, 64)
    out = SM90GemmFwdKernel(GemmType.K_GROUPED_CONTIGUOUS, num_groups=3)(a, b, grouped_layout=ks)
    assert out.shape == (3, 64, 64) and not out.any()
    with pytest.raises(ValueError, match="per-group B"):
        SM90GemmFwdKernel(GemmType.K_GROUPED_CONTIGUOUS, num_groups=3, activation="silu_and_mul")
