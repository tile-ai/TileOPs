"""Tests for the staged MoE public boundary and its grouped GEMM and expert MLP ops."""

import dataclasses

import pytest
import torch

from tests.test_base import served_in_tree
from tileops.backend import BUILTIN
from tileops.kernels.grouped_gemm import GemmTemplate
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe import (
    MGroupedGemmCall,
    MGroupedGemmFwdInterface,
    MoeGroupedGemmKernel,
    PostPermuteCall,
    PrePermuteCall,
)
from tileops.ops import moe as public_moe
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    MaskedLayoutSpec,
    MoeExpertMLPFwdOp,
    MoeGroupedGemmFwdOp,
    MoePostPermuteFwdOp,
    MoePrePermuteFwdOp,
    RoutingEpilogueSpec,
)
from workloads.device import run_device, run_device_available
from workloads.moe import MoeExpertMLPWorkload, MoeGroupedGemmWorkload, moe_call, valid_rows

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
    op = MoePrePermuteFwdOp(ContiguousLayoutSpec.aligned_per_row(alignment), num_local_experts=16)
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
    op = MoeGroupedGemmFwdOp(
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
    mlp = MoeExpertMLPFwdOp(_TIGHT, kernel_map={"grouped_gemm": _ExecutableGroupedCandidate})
    assert mlp.gate_up.kernel_map["grouped_gemm"] is _ExecutableGroupedCandidate
    assert mlp.down.kernel_map["grouped_gemm"] is _ExecutableGroupedCandidate
    assert MoeExpertMLPFwdOp(_TIGHT, "gelu_and_mul").gate_up.activation == "gelu_and_mul"


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
    op = MoePrePermuteFwdOp(layout, num_local_experts=1)
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

    pre = MoePrePermuteFwdOp(layout, num_local_experts=experts)
    expert_input, metadata, inverse = pre(x, local_ids)
    assert expert_input.shape == (tokens * top_k, hidden)
    assert metadata.shape == (experts,)
    assert inverse.shape == (tokens * top_k,)

    token_rows = torch.arange(tokens * top_k, device=run_device()) // top_k
    torch.testing.assert_close(expert_input[inverse.long()], x[token_rows])

    post = MoePostPermuteFwdOp(layout)
    output = post(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)
    assert pre.eval_roofline() == (0, 1616)
    assert post.eval_roofline() == (1024, 1600)

    # A routing scale other than one multiplies every output element once more.
    scaled = MoePostPermuteFwdOp(layout, RoutingEpilogueSpec(routed_scaling_factor=2.0))
    output = scaled(expert_input, weights, inverse)
    torch.testing.assert_close(output.float(), 2 * expected, rtol=2e-2, atol=4e-2)
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
    pre = MoePrePermuteFwdOp(layout, num_local_experts=experts)

    expert_input, physical_ends, inverse = pre(x, local_ids)

    token_rows = torch.arange(tokens * top_k, device=run_device()) // top_k
    torch.testing.assert_close(expert_input[inverse.long()], x[token_rows], rtol=0, atol=0)
    counts = torch.bincount(local_ids.flatten().long(), minlength=experts)
    torch.testing.assert_close(physical_ends, counts.cumsum(0).int(), rtol=0, atol=0)
    output = MoePostPermuteFwdOp(layout)(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)


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

    pre = MoePrePermuteFwdOp(layout, num_local_experts=experts)
    expert_input, row_expert_ids, inverse = pre(x, local_ids)
    assert expert_input.shape == (capacity, hidden)
    assert row_expert_ids.shape == (capacity,)
    assert inverse.shape == (tokens * top_k,)

    token_rows = torch.arange(tokens * top_k, device=run_device()) // top_k
    torch.testing.assert_close(expert_input[inverse.long()], x[token_rows])
    assert row_expert_ids.tolist() == [0] * 4 + [2] * 4 + [experts] * (capacity - 8)
    torch.testing.assert_close(expert_input[8:], torch.zeros_like(expert_input[8:]))

    post = MoePostPermuteFwdOp(layout)
    output = post(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="selection records CUDA architecture")
def test_grouped_gemm_call_no_candidate_serves_reports_no_implementation() -> None:
    """A call outside every shipped candidate's region says so, rather than crashing."""
    device = torch.device("cuda")
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_per_row())  # not claimed yet
    assert set(op.kernel_map) == {"grouped_gemm"}
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op(
            torch.empty(2, 8, dtype=torch.bfloat16, device=device),
            torch.empty(2, 4, 8, dtype=torch.bfloat16, device=device),
            torch.tensor([0, 1], dtype=torch.int32, device=device),
        )


# MoeGroupedGemmFwdOp selects MoeGroupedGemmKernel, the adapter over the shared template, through the
# staged candidate protocol; these tests route every layout it claims through the op and check the
# rows the layout defines against the workload's per-expert reference.
_E = 6


_TIGHT_MOE_GROUPED_GEMM = {
    "contiguous": {"packing": "tight", "metadata_kind": "physical_psum", "alignment": 1}
}


def _aligned(metadata_kind: str) -> dict:
    return {"contiguous": {"packing": "aligned", "metadata_kind": metadata_kind, "alignment": 128}}


def _gemm_call(dtype: torch.dtype, layout: dict, **row):
    return moe_call(
        "MoeGroupedGemmFwdOp", {"D": str(dtype).removeprefix("torch.")}, layout=layout, **row
    )


def _run(workload) -> tuple:
    """The op built from the call, its output on the call's inputs, and the reference."""
    inputs = workload.gen_inputs()
    cls = MoeGroupedGemmFwdOp if len(inputs) == 3 else MoeExpertMLPFwdOp
    op = cls(**workload.call.arguments({}))
    out = op(*inputs)
    rows = inputs[0].numel() // inputs[0].shape[-1]
    valid = valid_rows(op.layout, inputs[-1], rows, inputs[1].shape[0])
    ref = workload.ref_program(*inputs)
    torch.testing.assert_close(
        out.reshape(-1, out.shape[-1])[valid].float(),
        ref.reshape(-1, ref.shape[-1])[valid].float(),
        rtol=2e-2,
        atol=1e-1,
    )
    return op, out


@pytest.mark.sm90
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
    """Each claimed layout: the op selects the template, builds it once, matches the reference."""
    op, out = _run(MoeGroupedGemmWorkload(_gemm_call(dtype, layout, K=512, E=_E, N=256, **rows)))
    assert out.dtype is dtype and out.shape[-1] == 256
    if served_in_tree(op):
        (kernel,) = op.built_kernels("grouped_gemm").values()
        assert isinstance(kernel, MoeGroupedGemmKernel)
        assert isinstance(kernel.inner, GemmTemplate)


@pytest.mark.sm90
@pytest.mark.smoke
def test_grouped_gemm_reuses_its_kernel_across_row_counts():
    """The materialized row count is not in the build identity: a second M reuses the kernel."""
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    for rows in (600, 300):
        workload = MoeGroupedGemmWorkload(
            _gemm_call(torch.bfloat16, _TIGHT_MOE_GROUPED_GEMM, P=rows, K=512, E=_E, N=256)
        )
        op(*workload.gen_inputs())
    if served_in_tree(op):
        assert len(op.built_kernels("grouped_gemm")) == 1


@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
def test_grouped_gemm_fuses_the_gated_activation(activation):
    """With ``activation`` the op hands the template a gate||up ``b`` and gets ffn columns back."""
    call = _gemm_call(
        torch.bfloat16, _TIGHT_MOE_GROUPED_GEMM, P=600, K=512, E=_E, N=192, activation=activation
    )
    op, out = _run(MoeGroupedGemmWorkload(call))
    assert out.shape == (600, 192)
    if served_in_tree(op):
        (kernel,) = op.built_kernels("grouped_gemm").values()
        assert kernel.inner.activation == activation


@pytest.mark.sm90
@pytest.mark.smoke
def test_grouped_gemm_dims_off_the_tile_grid():
    """Cover non-tile-aligned K and N."""
    _run(
        MoeGroupedGemmWorkload(
            _gemm_call(torch.bfloat16, _TIGHT_MOE_GROUPED_GEMM, P=200, K=96, E=_E, N=192)
        )
    )


@pytest.mark.sm90
@pytest.mark.smoke
def test_grouped_gemm_fp32_output_and_preallocated_out():
    call = _gemm_call(
        torch.bfloat16, _TIGHT_MOE_GROUPED_GEMM, P=600, K=512, E=_E, N=256, out_dtype="float32"
    )
    workload = MoeGroupedGemmWorkload(call)
    a, b, metadata = workload.gen_inputs()
    op = MoeGroupedGemmFwdOp(**call.arguments({}))
    out = torch.empty(600, 256, dtype=torch.float32, device=run_device())
    assert op(a, b, metadata, out=out) is out
    torch.testing.assert_close(out, workload.ref_program(a, b, metadata), rtol=1e-3, atol=1e-2)


@pytest.mark.sm90
@pytest.mark.smoke
def test_grouped_gemm_refuses_a_strided_out():
    """The output buffer is declared contiguous; a strided one is refused before the kernel."""
    workload = MoeGroupedGemmWorkload(
        _gemm_call(torch.bfloat16, _TIGHT_MOE_GROUPED_GEMM, P=64, K=64, E=2, N=64)
    )
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    out = torch.empty(64, 128, dtype=torch.bfloat16, device=run_device())[:, ::2]
    with pytest.raises(ValueError, match="out must be contiguous"):
        op(*workload.gen_inputs(), out=out)


@pytest.mark.sm90
@pytest.mark.smoke
def test_grouped_gemm_refuses_what_the_template_cannot_run_at_selection():
    """Calls outside the adapter's region are refused by selection, naming the reason."""
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    a = torch.randn(8, 60, dtype=torch.bfloat16, device=run_device())
    b = torch.randn(2, 16, 60, dtype=torch.bfloat16, device=run_device())
    ends = torch.tensor([4, 8], dtype=torch.int32, device=run_device())
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op(a, b, ends)  # K not a multiple of 8
    fused = MoeGroupedGemmFwdOp(
        ContiguousLayoutSpec.tight_physical_psum(), activation="silu_and_mul"
    )
    a = torch.randn(8, 64, dtype=torch.bfloat16, device=run_device())
    b = torch.randn(2, 24, 64, dtype=torch.bfloat16, device=run_device())
    with pytest.raises(ValueError, match="no implementation serves this call"):
        fused(a, b, ends)  # fused N must be a multiple of 16
    # An aligned layout whose alignment is not a tile height has no instantiation either.
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.aligned_per_row(8))
    a = torch.randn(16, 64, dtype=torch.bfloat16, device=run_device())
    b = torch.randn(2, 16, 64, dtype=torch.bfloat16, device=run_device())
    ids = torch.tensor([0] * 8 + [1] * 8, dtype=torch.int32, device=run_device())
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op(a, b, ids)


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
    call = moe_call(
        "MoeExpertMLPFwdOp",
        {"D": str(dtype).removeprefix("torch.")},
        layout=_TIGHT_MOE_GROUPED_GEMM,
        activation=activation,
        P=600,
        H=256,
        E=_E,
        F=192,
    )
    op, out = _run(MoeExpertMLPWorkload(call))
    assert out.dtype is dtype and out.shape == (600, 256)
    if served_in_tree(op):
        (gate_up,) = op.gate_up.built_kernels("grouped_gemm").values()
        (down,) = op.down.built_kernels("grouped_gemm").values()
        assert (gate_up.inner.activation, down.inner.activation) == (activation, "none")
