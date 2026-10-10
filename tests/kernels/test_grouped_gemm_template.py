"""The persistent GEMM template's own config mechanism: its selector, and its specs.

Every condition an op reaches runs through dispatch in ``tests/ops``. What stays here is
what no op can request yet: the selector's choices, and results at a spec pinned by an
explicit config.
"""

# FIXME(staged-rollout): spec-pinned results bypass dispatch
#
# Broken invariant: a numerical test reaches its kernel through an op's dispatch.
# Why: the template picks its config per call outside ``Kernel.default_config``, so an op
#   cannot pin a spec; the tests below construct ``GemmTemplate`` with one.
# Cleanup: when config selection moves into the standard Kernel mechanism, register each
#   spec through ``register_implementation`` and move these tests to ``tests/ops``.

import dataclasses
import math

import pytest
import torch

from tileops.kernels.gemm.persistent.heuristics import (
    GemmDesc,
    GroupedGemmSpec,
    get_best_config,
    layout_candidates,
    spec_from_config,
)
from tileops.kernels.gemm.persistent.template import GemmTemplate, GemmType, Major
from workloads.device import run_device

pytestmark = pytest.mark.sm90


def _assert_gemm(out, ref):
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=1e-1)


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
    a = torch.randn(total, k, device=run_device(), dtype=dtype)
    b = torch.randn(groups, n, k, device=run_device(), dtype=dtype)
    if major_b == "mn":
        b = b.transpose(1, 2).contiguous().transpose(1, 2)
    ref = torch.zeros(total, n, dtype=torch.float32, device=run_device())
    valid = torch.zeros(total, dtype=torch.bool, device=run_device())
    for g in range(groups):
        ref[starts[g] : ends[g]] = a[starts[g] : ends[g]].float() @ b[g].float().T
        valid[starts[g] : ends[g]] = True
    if layout == "per_row":
        metadata = torch.full((total,), groups, dtype=torch.int32)
        for g in range(groups):
            metadata[starts[g] : starts[g] + math.ceil(sizes[g] / alignment) * alignment] = g
    else:
        metadata = torch.tensor(ends, dtype=torch.int32)
    return a, b, metadata.to(run_device()), ref, valid


@pytest.mark.cuda_only
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
    ],
)
def test_m_grouped_aligned_per_row(sizes, n, k, major_b, config):
    """Rows follow their group's B; padding rows and the sentinel tail are never read back."""
    a, b, ids, ref, valid = _grouped_operands(sizes, n, k, "per_row", major_b=major_b)
    kernel = GemmTemplate(
        GemmType.M_GROUPED_ALIGNED_PER_ROW, num_groups=len(sizes), m_alignment=128, config=config
    )
    out = kernel(a, b, grouped_layout=ids)
    _assert_gemm(out[valid], ref[valid])


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
def test_selector_prunes_invalid_layouts():
    """Every candidate satisfies the register, block_n step and stage-count rules."""
    for layout in layout_candidates(_desc(4096, 4096, 4096)):
        assert not (layout.block_m > 128 and layout.block_n > 128)
        assert layout.block_n % 64 == 0
    fp32 = layout_candidates(_desc(4096, 4096, 4096, cd_dtype="float32"))
    assert all(layout.block_m <= 128 for layout in fp32)
    grouped = dict(gemm_type=GemmType.M_GROUPED_TIGHT_PSUM, num_groups=128)
    decode = layout_candidates(_desc(4096, 4096, 7168, **grouped))
    assert {layout.block_k for layout in decode} == {64}


@pytest.mark.smoke
def test_selector_short_group_h200_band():
    grouped = dict(
        gemm_type=GemmType.M_GROUPED_TIGHT_PSUM,
        num_groups=160,
        activation="silu_and_mul",
    )
    short = get_best_config(_desc(4096, 3072, 5120, device_name="NVIDIA H200", **grouped))
    assert (short.block_m, short.block_n, short.block_k) == (64, 128, 128)
    assert (
        get_best_config(_desc(16384, 3072, 5120, device_name="NVIDIA H200", **grouped)).block_k
        == 64
    )
    assert (
        get_best_config(_desc(4096, 3072, 5120, device_name="NVIDIA H100", **grouped)).block_k == 64
    )
    plain = {**grouped, "activation": "none"}
    assert (
        get_best_config(_desc(4096, 5120, 1536, device_name="NVIDIA H200", **plain)).block_k == 128
    )
    assert (
        get_best_config(_desc(4096, 7168, 2048, device_name="NVIDIA H200", **plain)).block_k == 64
    )


@pytest.mark.smoke
def test_selector_stages_h200_batched_epilogue():
    """H200 batched 128x256 tiles trade half-width stores for a fourth K stage."""
    h200 = get_best_config(_desc(1024, 1024, 1024, num_groups=8, device_name="NVIDIA H200"))
    assert (h200.block_m, h200.block_n, h200.num_stages, h200.epilogue_stage_n) == (
        128,
        256,
        4,
        128,
    )
    h100 = get_best_config(_desc(1024, 1024, 1024, num_groups=8, device_name="NVIDIA H100"))
    assert h100.epilogue_stage_n == 0
    # The band is fitted on the board, not on the exact name CUDA reports for it.
    nvl = get_best_config(_desc(1024, 1024, 1024, num_groups=8, device_name="NVIDIA H200 NVL"))
    assert nvl.epilogue_stage_n == 128


@pytest.mark.smoke
def test_grouped_selector_calibration_preserves_dense_and_k_grouped_choices():
    """Keep the measured dense and K-grouped winners while fitting M-grouped."""
    dense = get_best_config(
        _desc(
            512,
            36864,
            7168,
            gemm_type=GemmType.DENSE,
            activation="silu_and_mul",
            device_name="NVIDIA H200",
        )
    )
    assert (dense.block_m, dense.block_n, dense.num_stages) == (128, 192, 4)

    k_grouped = get_best_config(
        _desc(
            4096,
            4096,
            4096,
            gemm_type=GemmType.K_GROUPED_CONTIGUOUS,
            num_groups=16,
            major_a=Major.MN,
            major_b=Major.MN,
            device_name="NVIDIA H200",
        )
    )
    assert (k_grouped.block_m, k_grouped.block_n, k_grouped.num_stages) == (128, 256, 4)
    assert k_grouped.epilogue_stage_n == 64


@pytest.mark.smoke
def test_grouped_selector_calibration_stays_on_physical_psum_layouts():
    tight = _desc(
        16 * 128,
        1536,
        2048,
        gemm_type=GemmType.M_GROUPED_TIGHT_PSUM,
        num_groups=16,
        device_name="NVIDIA H200",
    )
    assert all(layout.block_n != 192 for layout in layout_candidates(tight))

    padded = _desc(
        16 * 3328,
        1536,
        2048,
        gemm_type=GemmType.M_GROUPED_ALIGNED_PSUM,
        num_groups=16,
        m_alignment=128,
        device_name="NVIDIA H200",
    )
    assert all(layout.block_n != 192 for layout in layout_candidates(padded))
    assert get_best_config(padded).epilogue_stage_n == 64
    beyond = dataclasses.replace(padded, m=16 * (3328 + 128))
    assert get_best_config(beyond).epilogue_stage_n == 0


def _mg(gemm_type, rows_per_group, n, k, groups=16, **kw):
    """One m-grouped or K-grouped descriptor, given the rows each group carries."""
    return dict(gemm_type=gemm_type, m=rows_per_group * groups, n=n, k=k, num_groups=groups, **kw)


_MN_MAJOR = dict(major_a=Major.MN, major_b=Major.MN)

# The spec the selector returns today for one descriptor per gemm type, per
# device branch and per layout family, as (block_m, block_n, block_k,
# num_stages, epilogue_stage_n). Purpose: regression. The selection is a fitted
# model, so a change that moves any row here changes what ships for that whole
# family and has to be re-measured rather than re-recorded.
_GOLDEN_SPECS = {
    "dense": (dict(gemm_type=GemmType.DENSE, m=4096, n=4096, k=4096), (128, 256, 64, 4, 128)),
    "dense-gated": (
        dict(gemm_type=GemmType.DENSE, m=512, n=36864, k=7168, activation="silu_and_mul"),
        (128, 192, 64, 4, 0),
    ),
    "dense-h100": (
        dict(gemm_type=GemmType.DENSE, m=4096, n=4096, k=4096, device_name="NVIDIA H100"),
        (128, 256, 64, 3, 0),
    ),
    "batched": (dict(m=1024, n=1024, k=1024, num_groups=8), (128, 256, 64, 4, 128)),
    "batched-fp32": (
        dict(m=1024, n=1024, k=1024, num_groups=8, cd_dtype="float32"),
        (128, 192, 64, 4, 96),
    ),
    "batched-mn-major": (
        dict(m=2048, n=2048, k=512, num_groups=4, **_MN_MAJOR),
        (128, 256, 64, 4, 128),
    ),
    "tight-psum-decode": (
        _mg(GemmType.M_GROUPED_TIGHT_PSUM, 8, 4096, 7168),
        (64, 128, 128, 4, 0),
    ),
    "tight-psum-prefill": (
        _mg(GemmType.M_GROUPED_TIGHT_PSUM, 2048, 4096, 7168),
        (128, 256, 64, 3, 0),
    ),
    "tight-per-row": (
        _mg(GemmType.M_GROUPED_TIGHT_PER_ROW, 512, 7168, 2048),
        (128, 256, 64, 3, 0),
    ),
    "aligned-psum-staged": (
        _mg(GemmType.M_GROUPED_ALIGNED_PSUM, 512, 1536, 2048),
        (128, 256, 64, 4, 64),
    ),
    "aligned-psum-many-waves": (
        _mg(GemmType.M_GROUPED_ALIGNED_PSUM, 4096, 4096, 7168),
        (128, 256, 64, 3, 0),
    ),
    "aligned-per-row-64": (
        _mg(GemmType.M_GROUPED_ALIGNED_PER_ROW, 512, 3072, 5120, m_alignment=64),
        (64, 256, 64, 5, 128),
    ),
    "aligned-per-row-256": (
        _mg(GemmType.M_GROUPED_ALIGNED_PER_ROW, 512, 3072, 5120, m_alignment=256),
        (256, 128, 64, 4, 32),
    ),
    "masked": (
        dict(
            gemm_type=GemmType.M_GROUPED_MASKED,
            m=1024,
            n=4096,
            k=7168,
            num_groups=8,
            expected_m=256,
        ),
        (128, 256, 64, 4, 64),
    ),
    "k-grouped": (
        _mg(GemmType.K_GROUPED_CONTIGUOUS, 256, 4096, 4096, **_MN_MAJOR),
        (128, 256, 64, 4, 64),
    ),
    "k-grouped-short": (
        _mg(GemmType.K_GROUPED_CONTIGUOUS, 256, 2048, 512, groups=8, **_MN_MAJOR),
        (128, 256, 64, 4, 64),
    ),
}


@pytest.mark.full
@pytest.mark.parametrize("fields,expected", _GOLDEN_SPECS.values(), ids=_GOLDEN_SPECS)
def test_selector_golden_specs(fields, expected):
    spec = get_best_config(_desc(**{"device_name": "NVIDIA H200", **fields}))
    got = (spec.block_m, spec.block_n, spec.block_k, spec.num_stages, spec.epilogue_stage_n)
    assert got == expected


@pytest.mark.full
@pytest.mark.parametrize(
    "rows_per_group,expected",
    [
        # Rows per group either side of each block_m the m-grouped types offer:
        # one row past a tile is a whole extra tile per group, and which tile the
        # model then prefers is what these pin.
        (32, (64, 128, 128, 4, 0)),
        (33, (64, 256, 64, 4, 0)),
        (64, (64, 256, 64, 4, 0)),
        (65, (128, 256, 64, 3, 0)),
        (127, (128, 256, 64, 3, 0)),
        (128, (128, 256, 64, 3, 0)),
        (129, (128, 256, 64, 3, 0)),
    ],
)
def test_selector_golden_rows_per_group_boundaries(rows_per_group, expected):
    spec = get_best_config(
        _desc(
            16 * rows_per_group,
            4096,
            7168,
            gemm_type=GemmType.M_GROUPED_TIGHT_PSUM,
            num_groups=16,
            device_name="NVIDIA H200",
        )
    )
    got = (spec.block_m, spec.block_n, spec.block_k, spec.num_stages, spec.epilogue_stage_n)
    assert got == expected


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
    GroupedGemmSpec(**fields)
    with pytest.raises(ValueError, match="num_math_threads"):
        GroupedGemmSpec(**{**fields, "block_m": 64})
    with pytest.raises(ValueError, match="both exceed 128"):
        GroupedGemmSpec(**{**fields, "block_m": 256, "block_n": 256})
    with pytest.raises(ValueError, match="block_k must be 64 or 128"):
        GroupedGemmSpec(**{**fields, "block_k": 96})
    # A pinned pipeline past the shared-memory budget, or the single stage the
    # 128x256x128 tile has room for, is refused, not launched (it deadlocked).
    with pytest.raises(ValueError, match="at most"):
        spec_from_config(_desc(4096, 4096, 4096), dict(block_m=128, block_n=256, num_stages=8))
    with pytest.raises(ValueError, match="at least 2"):
        spec_from_config(_desc(4096, 4096, 4096), dict(block_m=128, block_n=256, block_k=128))
    with pytest.raises(ValueError, match="K-major A"):
        GroupedGemmSpec(
            **{
                **fields,
                "gemm_type": GemmType.M_GROUPED_ALIGNED_PER_ROW,
                "num_groups": 4,
                "major_a": Major.MN,
            }
        )


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "sizes,config",
    [
        pytest.param([100, 0, 300, 128, 7, 64], dict(block_m=128, block_n=128), id="two-wgs"),
        pytest.param([100, 0, 300, 128, 7, 64], dict(block_m=64, block_n=128), id="one-wg"),
        pytest.param(
            [100, 0, 300, 128, 7, 64],
            dict(block_m=128, block_n=256, epilogue_stage_n=64, num_stages=4),
            id="chunked-epilogue",
        ),
    ],
)
def test_m_grouped_tight_psum_masks_each_groups_last_tile(sizes, config):
    """Tight rows: a group's ragged last tile is stored under a row mask, not over its neighbour."""
    a, b, ends, ref, _ = _grouped_operands(sizes, 2048, 1024, "tight")
    kernel = GemmTemplate(GemmType.M_GROUPED_TIGHT_PSUM, num_groups=len(sizes), config=config)
    out = torch.full_like(ref, 1e4, dtype=torch.bfloat16)  # poison: every row must be written
    kernel(a, b, grouped_layout=ends, out=out)
    _assert_gemm(out, ref)


@pytest.mark.cuda_only
@pytest.mark.full
@pytest.mark.parametrize(
    "sizes,config",
    [
        pytest.param([100, 0, 300, 128, 7, 64], dict(block_m=128, block_n=128), id="padded-groups"),
    ],
)
def test_m_grouped_aligned_psum(sizes, config):
    """Aligned psum rows: each group starts at the previous end rounded up to block_m."""
    a, b, ends, ref, valid = _grouped_operands(sizes, 2048, 1024, "aligned")
    kernel = GemmTemplate(
        GemmType.M_GROUPED_ALIGNED_PSUM, num_groups=len(sizes), m_alignment=128, config=config
    )
    out = kernel(a, b, grouped_layout=ends)
    _assert_gemm(out[valid], ref[valid])


@pytest.mark.cuda_only
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
    kernel = GemmTemplate(GemmType.M_GROUPED_MASKED, num_groups=groups, config=config)
    out = kernel(a, b, grouped_layout=torch.tensor(masked, dtype=torch.int32).cuda())
    assert out.shape == (groups, max_m, 1024)
    for g, mm in enumerate(masked):
        if mm:
            _assert_gemm(out[g, :mm], a[g, :mm].float() @ b[g].float().T)


def _gated_ref(ref, activation):
    """``act(gate) * up`` over a GEMM reference whose columns stack gate then up."""
    gate, up = ref.chunk(2, dim=-1)
    act = torch.nn.functional.silu if activation == "silu_and_mul" else torch.nn.functional.gelu
    return act(gate) * up


@pytest.mark.cuda_only
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
    kernel = GemmTemplate(
        GemmType.M_GROUPED_TIGHT_PSUM, num_groups=len(sizes), activation=activation, config=config
    )
    out = kernel(a, b, grouped_layout=ends)
    assert out.shape == (a.shape[0], 512)
    _assert_gemm(out, _gated_ref(ref, activation))


def _k_grouped_operands(sizes, m, n, major_a="mn", major_b="mn", dtype=torch.bfloat16):
    """Groups packed along K: logical ``a[M, sum_k]``, ``b[N, sum_k]``, fp32 ``ref[G, M, N]``.

    The MN-major storage is ``[sum_k, M]`` / ``[sum_k, N]`` (``GroupedGemmFwdOp``'s
    TN); a K-major ``b`` is ``[N, sum_k]`` (its TT).
    """
    torch.manual_seed(0)
    sum_k = sum(sizes)
    a = torch.randn(m, sum_k, device=run_device(), dtype=dtype)
    if major_a == "mn":
        a = a.T.contiguous().T
    b = torch.randn(n, sum_k, device=run_device(), dtype=dtype)
    if major_b == "mn":
        b = b.T.contiguous().T
    ref = torch.zeros(len(sizes), m, n, device=run_device())
    start = 0
    for g, size in enumerate(sizes):
        ref[g] = a[:, start : start + size].float() @ b[:, start : start + size].float().T
        start += size
    return a, b, torch.tensor(sizes, dtype=torch.int32, device=run_device()), ref


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "major_a,major_b,config",
    [
        pytest.param("mn", "mn", dict(block_m=64, block_n=128), id="tn-one-wg"),
        pytest.param("mn", "mn", dict(block_m=128, block_n=128), id="tn-two-wgs"),
    ],
)
def test_k_grouped_contiguous(major_a, major_b, config):
    """Groups along K: each group's K tail is masked in shared memory, a K-major
    operand's group start is rounded down to 8 and its head masked, a group with
    no tokens stores zeros, and ragged M / N edges are clipped by the TMA store.
    """
    sizes = [100, 0, 300, 64, 7, 1]
    a, b, ks, ref = _k_grouped_operands(sizes, 200, 136, major_a, major_b)
    kernel = GemmTemplate(GemmType.K_GROUPED_CONTIGUOUS, num_groups=len(sizes), config=config)
    out = torch.full_like(ref, 1e4, dtype=torch.bfloat16)  # poison: every tile must be written
    kernel(a, b, grouped_layout=ks, out=out)
    _assert_gemm(out, ref)
