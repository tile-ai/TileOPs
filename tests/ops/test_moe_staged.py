"""Tests for the staged MoE public boundary and its grouped GEMM and expert MLP ops."""

import dataclasses

import pytest
import torch

from tests.test_base import served_in_tree
from tileops.backend import BUILTIN
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import (
    MGroupedGemmCall,
    MGroupedGemmFwdInterface,
    MoEGroupedGemmKernel,
    MoEGroupedGemmMMAKernel,
    PostPermuteCall,
    PrePermuteCall,
)
from tileops.ops import moe as public_moe
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    MaskedLayoutSpec,
    MoEExpertMLPFwdOp,
    MoEGroupedGemmFwdOp,
    MoEPostPermuteFwdOp,
    MoEPrePermuteFwdOp,
    RoutingEpilogueSpec,
)
from tileops.utils import get_sm_version
from workloads.device import run_device, run_device_available
from workloads.moe import (
    MoEExpertMLPWorkload,
    MoEGroupedGemmWorkload,
    moe_call,
    post_permute_verification,
)
from workloads.numerics import compare_outputs

_TIGHT = ContiguousLayoutSpec.tight_physical_psum()


@pytest.mark.smoke
def test_layout_presets_expose_only_supported_semantics() -> None:
    physical = ContiguousLayoutSpec.tight_physical_psum()
    per_row = ContiguousLayoutSpec.tight_per_row()
    aligned = ContiguousLayoutSpec.aligned_per_row(128)
    masked = MaskedLayoutSpec(max_m=4)

    assert physical.selection_key == "tight_physical_psum"
    assert per_row.selection_key == "tight_per_row"
    assert aligned.selection_key == "aligned_per_row"
    assert aligned.alignment == 128
    assert masked.selection_key == "masked_predicated"
    assert repr(physical) == "ContiguousLayoutSpec.tight_physical_psum()"
    assert repr(per_row) == "ContiguousLayoutSpec.tight_per_row()"
    assert repr(aligned) == "ContiguousLayoutSpec.aligned_per_row(128)"
    aligned_psum = ContiguousLayoutSpec.aligned_physical_psum(64)
    assert aligned_psum.selection_key == "aligned_physical_psum"
    assert repr(aligned_psum) == "ContiguousLayoutSpec.aligned_physical_psum(64)"
    assert (physical.kind, masked.kind) == ("contiguous", "masked")
    with pytest.raises(ValueError, match="alignment > 1"):
        ContiguousLayoutSpec.aligned_per_row(1)
    with pytest.raises(ValueError, match="non-negative"):
        MaskedLayoutSpec(max_m=-1)


@pytest.mark.smoke
def test_aligned_pre_permute_capacity_is_a_gemm_legal_row_extent() -> None:
    """Worst-case padding capacity still satisfies the aligned GEMM contract."""
    alignment = 128
    op = MoEPrePermuteFwdOp(ContiguousLayoutSpec.aligned_per_row(alignment), num_local_experts=16)
    shapes = op._infer_output_shapes(
        (256, 2048), (256, 2), dtypes={"hidden_states": torch.bfloat16}
    )
    rows = shapes["expert_input"][0]
    assert rows == 2560
    assert rows % alignment == 0
    assert shapes["layout_metadata"] == (rows,)


@pytest.mark.smoke
def test_default_public_surface_hides_kernel_author_and_metadata_types() -> None:
    for name in (
        "MGroupedGemmCall",
        "PrePermuteCall",
        "PostPermuteCall",
        "MaterializedExpertLayout",
        "NoScaleComputeSpec",
    ):
        assert not hasattr(public_moe, name)


@pytest.mark.smoke
def test_epilogue_spec_is_minimal_and_frozen() -> None:
    epilogue = RoutingEpilogueSpec()
    assert epilogue.accumulation_dtype is torch.float32
    # The output dtype is the op's ``out_dtype``, not the epilogue's.
    assert not hasattr(epilogue, "output_dtype")
    with pytest.raises(ValueError, match="finite and positive"):
        RoutingEpilogueSpec(routed_scaling_factor=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        epilogue.routed_scaling_factor = 2.0


@pytest.mark.smoke
def test_family_call_specs_are_frozen_and_keep_selection_axes_separate() -> None:
    pre = PrePermuteCall(arch=90, sm_count=1, layout=ContiguousLayoutSpec.tight_physical_psum())
    aligned_pre = dataclasses.replace(pre, layout=ContiguousLayoutSpec.aligned_per_row(128))
    gemm = MGroupedGemmCall(
        arch=90, sm_count=1, kind="contiguous", packing="tight", metadata_kind="physical_psum"
    )
    post = PostPermuteCall(
        arch=90,
        sm_count=1,
        layout_key="tight_physical_psum",
        epilogue=RoutingEpilogueSpec(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        pre.arch = 100
    assert aligned_pre != pre
    assert len({pre, aligned_pre}) == 2
    assert post.layout_key == "tight_physical_psum"
    # ``m`` selects but does not build, and the device facts follow from ``device``, so
    # they are outside this record's identity; every other field is inside it.
    taller = dataclasses.replace(gemm, m=4096)
    assert taller == gemm
    assert len({gemm, taller}) == 1
    assert dataclasses.replace(gemm, n=8) != gemm
    assert dataclasses.replace(gemm, device=torch.device("meta")) != gemm
    assert dataclasses.replace(gemm, arch=100) == gemm


class _ExecutableGroupedCandidate(Kernel, MGroupedGemmFwdInterface):
    """A replacement that writes zeros of the GEMM's output shape instead of compiling."""

    builds = 0

    def __init__(self, call: MGroupedGemmCall) -> None:
        super().__init__()
        type(self).builds += 1
        self.call = call

    def forward(self, a, b, layout_metadata, out=None):
        zeros = a.new_zeros((*a.shape[:-1], b.shape[1]), dtype=self.call.cd_dtype)
        return zeros if out is None else out.copy_(zeros)


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="candidate test uses CUDA calls")
def test_injected_candidate_uses_common_selection_and_call_spec_cache() -> None:
    device = torch.device("cuda")
    ends = torch.tensor([1], dtype=torch.int32, device=device)
    # A shape the in-tree kernel serves: the replacement runs under the key it selects.
    a = torch.ones(1, 8, dtype=torch.bfloat16, device=device)
    b = torch.ones(1, 8, 8, dtype=torch.bfloat16, device=device)
    _ExecutableGroupedCandidate.builds = 0
    op = MoEGroupedGemmFwdOp(
        _TIGHT, kernel_map={"grouped_gemm": _ExecutableGroupedCandidate}, target=BUILTIN, tune=True
    )

    first = op(a, b, ends)
    second = op(a, b, ends)
    # More rows for the same experts is the same specialization.
    taller = op(torch.ones(3, 8, dtype=torch.bfloat16, device=device), b, ends * 3)

    assert first.shape == second.shape == (1, 8)
    assert taller.shape == (3, 8)
    assert _ExecutableGroupedCandidate.builds == 1
    # The op's flag reaches the resolved entry, not the constructor.
    assert next(iter(op.built_kernels("grouped_gemm").values()))._tune_requested
    assert len(op.built_kernels("grouped_gemm")) == 1
    assert op.eval_roofline() == (2 * 3 * 8 * 8, (3 * 8 + 1 * 8 * 8 + 3 * 8) * 2 + 4)

    out = torch.empty(1, 8, dtype=torch.bfloat16, device=device)
    assert op(a, b, ends, out=out) is out
    with pytest.raises(ValueError, match="out does not have the dtype of output"):
        op(a, b, ends, out=torch.empty(1, 8, dtype=torch.float32, device=device))


@pytest.mark.smoke
def test_expert_mlp_forwards_caller_replacements_to_both_gemms() -> None:
    mlp = MoEExpertMLPFwdOp(_TIGHT, kernel_map={"grouped_gemm": _ExecutableGroupedCandidate})
    assert mlp.gate_up.kernel_map["grouped_gemm"] is _ExecutableGroupedCandidate
    assert mlp.down.kernel_map["grouped_gemm"] is _ExecutableGroupedCandidate
    assert MoEExpertMLPFwdOp(_TIGHT, "gelu_and_mul").gate_up.activation == "gelu_and_mul"


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "layout",
    [
        ContiguousLayoutSpec.tight_physical_psum(),
        ContiguousLayoutSpec.aligned_per_row(8),
    ],
)
def test_pre_permute_ships_one_contiguous_candidate(
    layout: ContiguousLayoutSpec,
) -> None:
    op = MoEPrePermuteFwdOp(layout, num_local_experts=1)
    call = PrePermuteCall(arch=90, layout=layout, input_dtype=torch.bfloat16)
    assert op.select_implementation("pre_permute", call) == "pre_permute_contiguous"


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="staged kernels require CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_staged_tight_pre_post_round_trip(dtype: torch.dtype) -> None:
    """The tensor-only staged boundary preserves every routed contribution."""
    tokens, top_k, experts, hidden = 4, 2, 4, 64
    x = torch.randn(tokens, hidden, dtype=dtype, device=run_device())
    local_ids = torch.tensor(
        [[0, 1], [2, 3], [0, 2], [1, 3]], dtype=torch.int32, device=run_device()
    )
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=run_device())
    layout = ContiguousLayoutSpec.tight_physical_psum()

    pre = MoEPrePermuteFwdOp(layout, num_local_experts=experts)
    expert_input, metadata, inverse = pre(x, local_ids)
    assert expert_input.shape == (tokens * top_k, hidden)
    assert metadata.shape == (experts,)
    assert inverse.shape == (tokens * top_k,)

    token_rows = torch.arange(tokens * top_k, device=run_device()) // top_k
    assert torch.equal(expert_input[inverse.long()], x[token_rows])

    post = MoEPostPermuteFwdOp(layout)
    output = post(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    compare_outputs(output.float(), expected, post_permute_verification())
    assert pre.eval_roofline() == (0, 1616)
    assert post.eval_roofline() == (1024, 1600)

    # A routing scale other than one multiplies every output element once more.
    scaled = MoEPostPermuteFwdOp(layout, RoutingEpilogueSpec(routed_scaling_factor=2.0))
    output = scaled(expert_input, weights, inverse)
    compare_outputs(
        output.float(),
        2 * expected,
        post_permute_verification(scaled.epilogue.routed_scaling_factor),
    )
    assert scaled.eval_roofline() == (1024 + tokens * hidden, 1600)


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="staged kernels require CUDA")
@pytest.mark.parametrize(
    "tokens,top_k,experts,hidden",
    [(512, 8, 128, 128), (32, 8, 128, 7168)],
)
def test_staged_tight_optimized_shapes_round_trip(
    tokens: int, top_k: int, experts: int, hidden: int
) -> None:
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=run_device())
    local_ids = (
        torch.arange(tokens * top_k, dtype=torch.int32, device=run_device())
        .remainder(experts)
        .reshape(tokens, top_k)
    )
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=run_device())
    layout = ContiguousLayoutSpec.tight_physical_psum()
    pre = MoEPrePermuteFwdOp(layout, num_local_experts=experts)

    expert_input, physical_ends, inverse = pre(x, local_ids)

    token_rows = torch.arange(tokens * top_k, device=run_device()) // top_k
    assert torch.equal(expert_input[inverse.long()], x[token_rows])
    counts = torch.bincount(local_ids.flatten().long(), minlength=experts)
    assert torch.equal(physical_ends, counts.cumsum(0).int())
    output = MoEPostPermuteFwdOp(layout)(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    compare_outputs(output.float(), expected, post_permute_verification())


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="staged kernels require CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_staged_aligned_per_row_pre_post_round_trip(dtype: torch.dtype) -> None:
    tokens, top_k, experts, hidden, alignment = 4, 2, 4, 64, 4
    x = torch.randn(tokens, hidden, dtype=dtype, device=run_device())
    local_ids = torch.tensor(
        [[0, 0], [0, 2], [2, 2], [2, 0]], dtype=torch.int32, device=run_device()
    )
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device=run_device())
    layout = ContiguousLayoutSpec.aligned_per_row(alignment)
    capacity = tokens * top_k + experts * (alignment - 1)

    pre = MoEPrePermuteFwdOp(layout, num_local_experts=experts)
    expert_input, row_expert_ids, inverse = pre(x, local_ids)
    assert expert_input.shape == (capacity, hidden)
    assert row_expert_ids.shape == (capacity,)
    assert inverse.shape == (tokens * top_k,)

    token_rows = torch.arange(tokens * top_k, device=run_device()) // top_k
    assert torch.equal(expert_input[inverse.long()], x[token_rows])
    assert row_expert_ids.tolist() == [0] * 4 + [2] * 4 + [experts] * (capacity - 8)
    assert torch.count_nonzero(expert_input[8:]) == 0

    post = MoEPostPermuteFwdOp(layout)
    output = post(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    compare_outputs(output.float(), expected, post_permute_verification())


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="selection records CUDA architecture")
def test_grouped_gemm_call_no_candidate_serves_reports_no_implementation() -> None:
    """A call outside every shipped candidate's region says so, rather than crashing."""
    device = torch.device("cuda")
    op = MoEGroupedGemmFwdOp(ContiguousLayoutSpec.tight_per_row())  # not claimed yet
    assert set(op.kernel_map) == {"grouped_gemm", "grouped_gemm_mma"}
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op(
            torch.empty(2, 8, dtype=torch.bfloat16, device=device),
            torch.empty(2, 4, 8, dtype=torch.bfloat16, device=device),
            torch.tensor([0, 1], dtype=torch.int32, device=device),
        )


# MoEGroupedGemmFwdOp selects MoEGroupedGemmKernel, the adapter over the shared template, through the
# staged candidate protocol; these tests route every layout it claims through the op and check the
# rows the layout defines against the workload's per-expert reference.


_TIGHT_MOE_GROUPED_GEMM = {
    "contiguous": {"packing": "tight", "metadata_kind": "physical_psum", "alignment": 1}
}


def _aligned(metadata_kind: str) -> dict:
    return {"contiguous": {"packing": "aligned", "metadata_kind": metadata_kind, "alignment": 128}}


def _gemm_call(dtype: torch.dtype, layout: dict, **row):
    return moe_call(
        "MoEGroupedGemmFwdOp", {"D": str(dtype).removeprefix("torch.")}, layout=layout, **row
    )


def _run(workload) -> tuple:
    """The op built from the call, its output on the call's inputs, and the reference."""
    inputs = workload.gen_inputs()
    cls = MoEGroupedGemmFwdOp if len(inputs) == 3 else MoEExpertMLPFwdOp
    op = cls(**workload.call.arguments({}))
    out = op(*inputs)
    compare_outputs(out, workload.ref_program(*inputs), workload.verification(*inputs))
    return op, out


@pytest.mark.parametrize(
    "layout,rows,dtype",
    [
        pytest.param(
            _TIGHT_MOE_GROUPED_GEMM,
            {"P": 600},
            torch.bfloat16,
            marks=pytest.mark.smoke,
            id="tight-psum",
        ),
        pytest.param(
            _aligned("per_row"),
            {"P": 128 * 9},
            torch.float16,
            marks=pytest.mark.smoke,
            id="aligned-per-row",
        ),
        pytest.param(
            _aligned("physical_psum"),
            {"P": 128 * 9},
            torch.bfloat16,
            marks=pytest.mark.full,
            id="aligned-psum",
        ),
        pytest.param(
            {"masked": {"max_m": 128}}, {}, torch.bfloat16, marks=pytest.mark.full, id="masked"
        ),
    ],
)
def test_grouped_gemm_runs_each_layout_through_the_op(layout, rows, dtype):
    """Each claimed layout: the op builds the device's kernel once and matches the reference."""
    num_experts = 6
    op, out = _run(
        MoEGroupedGemmWorkload(_gemm_call(dtype, layout, K=512, E=num_experts, N=256, **rows))
    )
    assert out.dtype is dtype and out.shape[-1] == 256
    if served_in_tree(op):
        (kernel,) = op.built_kernels("grouped_gemm").values()
        arch = get_sm_version(out.device.index)
        expected = MoEGroupedGemmKernel if arch == 90 else MoEGroupedGemmMMAKernel
        assert type(kernel) is expected


@pytest.mark.smoke
def test_grouped_gemm_reuses_its_kernel_across_row_counts():
    """The materialized row count is not in the build identity: a second M reuses the kernel."""
    num_experts = 6
    op = MoEGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    for rows in (600, 300):
        workload = MoEGroupedGemmWorkload(
            _gemm_call(torch.bfloat16, _TIGHT_MOE_GROUPED_GEMM, P=rows, K=512, E=num_experts, N=256)
        )
        op(*workload.gen_inputs())
    if served_in_tree(op):
        assert len(op.built_kernels("grouped_gemm")) == 1


@pytest.mark.smoke
@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
def test_grouped_gemm_fuses_the_gated_activation(activation):
    """With ``activation`` the op hands the kernel a gate||up ``b`` and gets ffn columns back."""
    num_experts = 6
    call = _gemm_call(
        torch.bfloat16,
        _TIGHT_MOE_GROUPED_GEMM,
        P=600,
        K=512,
        E=num_experts,
        N=192,
        activation=activation,
    )
    op, out = _run(MoEGroupedGemmWorkload(call))
    assert out.shape == (600, 192)
    if served_in_tree(op):
        (kernel,) = op.built_kernels("grouped_gemm").values()
        assert kernel.call.activation == activation


@pytest.mark.smoke
def test_grouped_gemm_dims_off_the_tile_grid():
    """Cover non-tile-aligned K and N."""
    num_experts = 6
    _run(
        MoEGroupedGemmWorkload(
            _gemm_call(torch.bfloat16, _TIGHT_MOE_GROUPED_GEMM, P=200, K=96, E=num_experts, N=192)
        )
    )


@pytest.mark.smoke
def test_grouped_gemm_fp32_output_and_preallocated_out():
    num_experts = 6
    call = _gemm_call(
        torch.bfloat16,
        _TIGHT_MOE_GROUPED_GEMM,
        P=600,
        K=512,
        E=num_experts,
        N=256,
        out_dtype="float32",
    )
    workload = MoEGroupedGemmWorkload(call)
    a, b, metadata = workload.gen_inputs()
    op = MoEGroupedGemmFwdOp(**call.arguments({}))
    out = torch.empty(600, 256, dtype=torch.float32, device=run_device())
    assert op(a, b, metadata, out=out) is out
    compare_outputs(
        out, workload.ref_program(a, b, metadata), workload.verification(a, b, metadata)
    )


@pytest.mark.smoke
def test_grouped_gemm_refuses_a_strided_out():
    """The output buffer is declared contiguous; a strided one is refused before the kernel."""
    workload = MoEGroupedGemmWorkload(
        _gemm_call(torch.bfloat16, _TIGHT_MOE_GROUPED_GEMM, P=64, K=64, E=2, N=64)
    )
    op = MoEGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    out = torch.empty(64, 128, dtype=torch.bfloat16, device=run_device())[:, ::2]
    with pytest.raises(ValueError, match="out must be contiguous"):
        op(*workload.gen_inputs(), out=out)


@pytest.mark.smoke
def test_grouped_gemm_refuses_what_the_template_cannot_run_at_selection():
    """Calls outside the adapter's region are refused by selection, naming the reason."""
    op = MoEGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    a = torch.randn(8, 60, dtype=torch.bfloat16, device=run_device())
    b = torch.randn(2, 16, 60, dtype=torch.bfloat16, device=run_device())
    ends = torch.tensor([4, 8], dtype=torch.int32, device=run_device())
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op(a, b, ends)  # K not a multiple of 8
    fused = MoEGroupedGemmFwdOp(
        ContiguousLayoutSpec.tight_physical_psum(), activation="silu_and_mul"
    )
    a = torch.randn(8, 64, dtype=torch.bfloat16, device=run_device())
    b = torch.randn(2, 24, 64, dtype=torch.bfloat16, device=run_device())
    with pytest.raises(ValueError, match="no implementation serves this call"):
        fused(a, b, ends)  # fused N must be a multiple of 16
    # An aligned layout whose alignment is not a tile height has no instantiation either.
    op = MoEGroupedGemmFwdOp(ContiguousLayoutSpec.aligned_per_row(8))
    a = torch.randn(16, 64, dtype=torch.bfloat16, device=run_device())
    b = torch.randn(2, 16, 64, dtype=torch.bfloat16, device=run_device())
    ids = torch.tensor([0] * 8 + [1] * 8, dtype=torch.int32, device=run_device())
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op(a, b, ids)


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "kind, packing, metadata_kind, alignment, max_m, rows, num_groups",
    [
        pytest.param("contiguous", "tight", "physical_psum", 1, None, 600, 6, id="tight-psum"),
        pytest.param(
            "contiguous", "aligned", "physical_psum", 128, None, 1152, 6, id="aligned-psum"
        ),
        pytest.param("contiguous", "aligned", "per_row", 128, None, 1152, 6, id="aligned-per-row"),
        pytest.param("masked", None, None, 1, 128, 6 * 128, 6, id="masked"),
    ],
)
def test_mma_grouped_gemm_tunes_on_a_layout_its_call_could_carry(
    kind: str,
    packing: "str | None",
    metadata_kind: "str | None",
    alignment: int,
    max_m: "int | None",
    rows: int,
    num_groups: int,
):
    """The metadata is an int32 input that sets which tiles run, so tuning cannot take it
    random: the supply builds the call's own rows in its layout."""
    call = MGroupedGemmCall(
        kind=kind,
        packing=packing,
        metadata_kind=metadata_kind,
        alignment=alignment,
        max_m=max_m,
        ab_dtype=torch.bfloat16,
        cd_dtype=torch.bfloat16,
        num_groups=num_groups,
        m=rows,
        n=256,
        k=512,
    )
    supply = MoEGroupedGemmMMAKernel._supply_prog_for(call)
    a, b, layout, c = supply([None] * 4)

    lead = [num_groups, max_m] if kind == "masked" else [rows]
    assert [list(t.shape) for t in (a, b, c)] == [
        [*lead, 512],
        [num_groups, 256, 512],
        [*lead, 256],
    ]
    if kind == "masked":
        assert layout.shape == (num_groups,) and 0 < int(layout.max()) <= max_m
    elif metadata_kind == "per_row":
        assert layout.shape == (rows,) and int(layout.min()) >= 0 and int(layout.max()) < num_groups
    else:
        assert layout.shape == (num_groups,) and bool((layout.diff() > 0).all())
        assert int(layout[-1]) <= rows
    with pytest.raises(RuntimeError, match="expects 4 parameters"):
        supply([None] * 5)


@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype,activation",
    [
        pytest.param(torch.bfloat16, "silu_and_mul", id="bf16-silu"),
        pytest.param(torch.float16, "gelu_and_mul", id="fp16-gelu"),
    ],
)
def test_expert_mlp_composes_two_template_gemms(dtype, activation):
    """The MLP is two template GEMMs; the first carries the activation and halves its width."""
    num_experts = 6
    call = moe_call(
        "MoEExpertMLPFwdOp",
        {"D": str(dtype).removeprefix("torch.")},
        layout=_TIGHT_MOE_GROUPED_GEMM,
        activation=activation,
        P=600,
        H=256,
        E=num_experts,
        F=192,
    )
    op, out = _run(MoEExpertMLPWorkload(call))
    assert out.dtype is dtype and out.shape == (600, 256)
    if served_in_tree(op):
        (gate_up,) = op.gate_up.built_kernels("grouped_gemm").values()
        (down,) = op.down.built_kernels("grouped_gemm").values()
        assert (gate_up.inner.activation, down.inner.activation) == (activation, "none")
