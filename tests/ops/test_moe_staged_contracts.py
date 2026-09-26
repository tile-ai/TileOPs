"""Behavioral contract tests for the staged MoE public boundary."""

import dataclasses

import pytest
import torch

import tileops.ops.moe.staged as staged_module
from tileops.backend import BUILTIN
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import MGroupedGemmCall, PostPermuteCall, PrePermuteCall
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

pytestmark = pytest.mark.smoke

_TIGHT = ContiguousLayoutSpec.tight_physical_psum()


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


def test_default_public_surface_hides_kernel_author_and_metadata_types() -> None:
    for name in (
        "MGroupedGemmCall",
        "PrePermuteCall",
        "PostPermuteCall",
        "MaterializedExpertLayout",
        "NoScaleComputeSpec",
    ):
        assert not hasattr(public_moe, name)


def test_epilogue_spec_is_minimal_and_frozen() -> None:
    epilogue = RoutingEpilogueSpec()
    assert epilogue.accumulation_dtype is torch.float32
    # The output dtype is the op's ``out_dtype``, not the epilogue's.
    assert not hasattr(epilogue, "output_dtype")
    with pytest.raises(ValueError, match="finite and positive"):
        RoutingEpilogueSpec(routed_scaling_factor=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        epilogue.routed_scaling_factor = 2.0


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
    # ``m`` selects but does not build, so it is outside this record's identity;
    # every other field is inside it.
    taller = dataclasses.replace(gemm, m=4096)
    assert taller == gemm
    assert len({gemm, taller}) == 1
    assert dataclasses.replace(gemm, n=8) != gemm
    assert dataclasses.replace(gemm, arch=100) != gemm
    # The flattened layout fields admit only what a layout spec can express.
    with pytest.raises(ValueError, match="no max_m"):
        dataclasses.replace(gemm, max_m=4)
    with pytest.raises(ValueError, match="alignment"):
        dataclasses.replace(gemm, packing="aligned")
    with pytest.raises(ValueError, match="carry no packing"):
        MGroupedGemmCall(arch=90, sm_count=1, kind="masked", packing="tight", max_m=4)
    with pytest.raises(ValueError, match="kind is"):
        MGroupedGemmCall(arch=90, sm_count=1, kind="padded")
    # Only the record that names no layout skips the checks.
    assert MGroupedGemmCall(arch=90, sm_count=1).kind == ""
    with pytest.raises(ValueError, match="kind is"):
        MGroupedGemmCall(arch=90, sm_count=1, max_m=4)


def _layout_key_of(call: MGroupedGemmCall) -> str | None:
    """The contiguous-layout key a test candidate claims, off the GEMM call."""
    if call.kind != "contiguous":
        return None
    return f"{call.packing}_{call.metadata_kind}"


class _PhysicalPsumCandidate(Kernel):
    supported_archs = [90]

    @classmethod
    def applies(cls, call: object) -> bool:
        return _layout_key_of(call) == "tight_physical_psum"

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _PerRowCandidate(Kernel):
    supported_archs = [90]

    @classmethod
    def applies(cls, call: object) -> bool:
        return _layout_key_of(call) == "tight_per_row"

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _GeneralCandidate(Kernel):
    general = True
    supported_archs = [90]

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _NeverCandidate(Kernel):
    @classmethod
    def applies(cls, call: object) -> bool:
        return False

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


def _zero_output(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None, dtype=None):
    """What an executable fake writes: zeros of the GEMM's output shape."""
    result = a.new_zeros((*a.shape[:-1], b.shape[1]), dtype=dtype)
    return result if out is None else out.copy_(result)


class _ExecutableGroupedCandidate(Kernel):
    builds = 0

    def __init__(self, call: MGroupedGemmCall) -> None:
        super().__init__()
        type(self).builds += 1
        self.call = call

    def forward(self, a, b, layout_metadata, *, out=None):
        return _zero_output(a, b, out, dtype=self.call.cd_dtype)


def _grouped_op_with_declared_candidates(**candidates: type[Kernel]) -> MoeGroupedGemmFwdOp:
    class DeclaringGroupedGemmOp(MoeGroupedGemmFwdOp):
        @property
        def default_kernel_map(self) -> dict[str, Kernel]:
            return dict(candidates)

    return DeclaringGroupedGemmOp(_TIGHT)


def _tight_call(metadata_kind: str = "physical_psum", **fields: object) -> MGroupedGemmCall:
    return MGroupedGemmCall(
        arch=90,
        sm_count=1,
        kind="contiguous",
        packing="tight",
        metadata_kind=metadata_kind,
        **fields,
    )


def test_grouped_gemm_selection_behavior_table() -> None:
    physical_call = _tight_call()
    per_row_call = _tight_call("per_row")
    op = _grouped_op_with_declared_candidates(
        physical=_PhysicalPsumCandidate,
        per_row=_PerRowCandidate,
        general=_GeneralCandidate,
    )

    assert op.select_kernel_key(("physical", "per_row", "general"), physical_call) == "physical"
    assert op.select_kernel_key(("physical", "per_row", "general"), per_row_call) == "per_row"
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op.select_kernel_key(
            ("physical", "per_row", "general"), dataclasses.replace(physical_call, arch=80)
        )


def test_grouped_gemm_ambiguous_and_incompatible_override_fail_explicitly() -> None:
    call = _tight_call()
    ambiguous = _grouped_op_with_declared_candidates(
        first=_PhysicalPsumCandidate,
        second=_PhysicalPsumCandidate,
    )
    with pytest.raises(ValueError, match="dispatch is ambiguous"):
        ambiguous.select_kernel_key(("first", "second"), call)

    class OverrideableOp(MoeGroupedGemmFwdOp):
        @property
        def default_kernel_map(self) -> dict[str, Kernel]:
            return {"special": _PhysicalPsumCandidate, "general": _GeneralCandidate}

    overridden = OverrideableOp(_TIGHT, kernel_map={"special": _NeverCandidate})
    with pytest.raises(ValueError, match="does not fall back"):
        overridden.select_kernel_key(("special", "general"), call)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CallSpec records CUDA architecture")
def test_call_architecture_comes_from_the_input_device(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    observed_indices: list[int | None] = []

    def fake_sm_version(index: int | None = None) -> int:
        observed_indices.append(index)
        return 90

    monkeypatch.setattr(staged_module, "get_sm_version", fake_sm_version)
    op = MoeGroupedGemmFwdOp(_TIGHT, kernel_map={"grouped_gemm": _ExecutableGroupedCandidate})
    op(
        torch.empty(1, 4, dtype=torch.bfloat16, device=device),
        torch.empty(1, 2, 4, dtype=torch.bfloat16, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
    )

    assert observed_indices == [device.index]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="candidate test uses CUDA calls")
def test_injected_candidate_uses_common_selection_and_call_spec_cache() -> None:
    device = torch.device("cuda")
    ends = torch.tensor([1], dtype=torch.int32, device=device)
    a = torch.ones(1, 4, dtype=torch.bfloat16, device=device)
    b = torch.ones(1, 2, 4, dtype=torch.bfloat16, device=device)
    _ExecutableGroupedCandidate.builds = 0
    op = MoeGroupedGemmFwdOp(
        _TIGHT, kernel_map={"grouped_gemm": _ExecutableGroupedCandidate}, target=BUILTIN, tune=True
    )

    first = op(a, b, ends)
    second = op(a, b, ends)
    # More rows for the same experts is the same specialization.
    taller = op(torch.ones(3, 4, dtype=torch.bfloat16, device=device), b, ends * 3)

    assert first.shape == second.shape == (1, 2)
    assert taller.shape == (3, 2)
    assert _ExecutableGroupedCandidate.builds == 1
    # The build is told to tune: the op's flag travels in the call record.
    assert next(iter(op.built_kernels("grouped_gemm").values())).call.tune
    assert len(op.built_kernels("grouped_gemm")) == 1
    assert op.eval_roofline() == (2 * 3 * 2 * 4, (3 * 4 + 1 * 2 * 4 + 3 * 2) * 2 + 4)

    out = torch.empty(1, 2, dtype=torch.bfloat16, device=device)
    assert op(a, b, ends, out=out) is out
    with pytest.raises(ValueError, match="out does not have the dtype of output"):
        op(a, b, ends, out=torch.empty(1, 2, dtype=torch.float32, device=device))


def test_expert_mlp_forwards_caller_replacements_to_both_gemms() -> None:
    mlp = MoeExpertMLPFwdOp(_TIGHT, kernel_map={"grouped_gemm": _ExecutableGroupedCandidate})
    assert mlp.gate_up.forwarded_overrides() == {"grouped_gemm": _ExecutableGroupedCandidate}
    assert mlp.down.forwarded_overrides() == {"grouped_gemm": _ExecutableGroupedCandidate}
    assert MoeExpertMLPFwdOp(_TIGHT, "gelu_and_mul").gate_up.activation == "gelu_and_mul"


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
    assert op.select_kernel_key(tuple(op.kernel_map), call) == "contiguous"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="staged kernels require CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_staged_tight_pre_post_round_trip(dtype: torch.dtype) -> None:
    """The tensor-only staged boundary preserves every routed contribution."""
    tokens, top_k, experts, hidden = 4, 2, 4, 64
    x = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
    local_ids = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], dtype=torch.int32, device="cuda")
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device="cuda")
    layout = ContiguousLayoutSpec.tight_physical_psum()

    pre = MoePrePermuteFwdOp(layout, num_local_experts=experts)
    expert_input, metadata, inverse = pre(x, local_ids)
    assert expert_input.shape == (tokens * top_k, hidden)
    assert metadata.shape == (experts,)
    assert inverse.shape == (tokens * top_k,)

    token_rows = torch.arange(tokens * top_k, device="cuda") // top_k
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="staged kernels require CUDA")
@pytest.mark.parametrize(
    "tokens,top_k,experts,hidden",
    [(512, 8, 128, 128), (32, 8, 128, 7168)],
)
def test_staged_tight_optimized_shapes_round_trip(
    tokens: int, top_k: int, experts: int, hidden: int
) -> None:
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    local_ids = (
        torch.arange(tokens * top_k, dtype=torch.int32, device="cuda")
        .remainder(experts)
        .reshape(tokens, top_k)
    )
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device="cuda")
    layout = ContiguousLayoutSpec.tight_physical_psum()
    pre = MoePrePermuteFwdOp(layout, num_local_experts=experts)

    expert_input, physical_ends, inverse = pre(x, local_ids)

    token_rows = torch.arange(tokens * top_k, device="cuda") // top_k
    torch.testing.assert_close(expert_input[inverse.long()], x[token_rows], rtol=0, atol=0)
    counts = torch.bincount(local_ids.flatten().long(), minlength=experts)
    torch.testing.assert_close(physical_ends, counts.cumsum(0).int(), rtol=0, atol=0)
    output = MoePostPermuteFwdOp(layout)(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="staged kernels require CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_staged_aligned_per_row_pre_post_round_trip(dtype: torch.dtype) -> None:
    tokens, top_k, experts, hidden, alignment = 4, 2, 4, 64, 4
    x = torch.randn(tokens, hidden, dtype=dtype, device="cuda")
    local_ids = torch.tensor([[0, 0], [0, 2], [2, 2], [2, 0]], dtype=torch.int32, device="cuda")
    weights = torch.rand(tokens, top_k, dtype=torch.float32, device="cuda")
    layout = ContiguousLayoutSpec.aligned_per_row(alignment)
    capacity = tokens * top_k + experts * (alignment - 1)

    pre = MoePrePermuteFwdOp(layout, num_local_experts=experts)
    expert_input, row_expert_ids, inverse = pre(x, local_ids)
    assert expert_input.shape == (capacity, hidden)
    assert row_expert_ids.shape == (capacity,)
    assert inverse.shape == (tokens * top_k,)

    token_rows = torch.arange(tokens * top_k, device="cuda") // top_k
    torch.testing.assert_close(expert_input[inverse.long()], x[token_rows])
    assert row_expert_ids.tolist() == [0] * 4 + [2] * 4 + [experts] * (capacity - 8)
    torch.testing.assert_close(expert_input[8:], torch.zeros_like(expert_input[8:]))

    post = MoePostPermuteFwdOp(layout)
    output = post(expert_input, weights, inverse)
    expected = x.float() * weights.sum(dim=1, keepdim=True)
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-2)


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
