"""Op-level tests for the staged grouped GEMM and expert MLP template.

``MoeGroupedGemmFwdOp`` selects ``MoeGroupedGemmKernel``, the adapter over the
shared template, through the staged candidate protocol; these tests route every layout
it claims through the op and check the rows the layout defines against the
workload's per-expert reference.
"""

import pytest
import torch

from tests.test_base import served_in_tree
from tileops.kernels.grouped_gemm import GemmTemplate
from tileops.kernels.moe import MoeGroupedGemmKernel
from tileops.ops.moe import ContiguousLayoutSpec, MoeExpertMLPFwdOp, MoeGroupedGemmFwdOp
from workloads.moe import MoeExpertMLPWorkload, MoeGroupedGemmWorkload, moe_call, valid_rows

pytestmark = pytest.mark.sm90

_E = 6
_TIGHT = {"contiguous": {"packing": "tight", "metadata_kind": "physical_psum", "alignment": 1}}


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


@pytest.mark.parametrize(
    "layout,rows,dtype",
    [
        pytest.param(_TIGHT, {"P": 600}, torch.bfloat16, marks=pytest.mark.smoke, id="tight-psum"),
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


@pytest.mark.smoke
def test_grouped_gemm_reuses_its_kernel_across_row_counts():
    """The materialized row count is not in the build identity: a second M reuses the kernel."""
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    for rows in (600, 300):
        workload = MoeGroupedGemmWorkload(
            _gemm_call(torch.bfloat16, _TIGHT, P=rows, K=512, E=_E, N=256)
        )
        op(*workload.gen_inputs())
    if served_in_tree(op):
        assert len(op.built_kernels("grouped_gemm")) == 1


@pytest.mark.smoke
@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
def test_grouped_gemm_fuses_the_gated_activation(activation):
    """With ``activation`` the op hands the template a gate||up ``b`` and gets ffn columns back."""
    call = _gemm_call(torch.bfloat16, _TIGHT, P=600, K=512, E=_E, N=192, activation=activation)
    op, out = _run(MoeGroupedGemmWorkload(call))
    assert out.shape == (600, 192)
    if served_in_tree(op):
        (kernel,) = op.built_kernels("grouped_gemm").values()
        assert kernel.inner.activation == activation


@pytest.mark.smoke
def test_grouped_gemm_dims_off_the_tile_grid():
    """Cover non-tile-aligned K and N."""
    _run(MoeGroupedGemmWorkload(_gemm_call(torch.bfloat16, _TIGHT, P=200, K=96, E=_E, N=192)))


@pytest.mark.smoke
def test_grouped_gemm_fp32_output_and_preallocated_out():
    call = _gemm_call(torch.bfloat16, _TIGHT, P=600, K=512, E=_E, N=256, out_dtype="float32")
    workload = MoeGroupedGemmWorkload(call)
    a, b, metadata = workload.gen_inputs()
    op = MoeGroupedGemmFwdOp(**call.arguments({}))
    out = torch.empty(600, 256, dtype=torch.float32, device="cuda")
    assert op(a, b, metadata, out=out) is out
    torch.testing.assert_close(out, workload.ref_program(a, b, metadata), rtol=1e-3, atol=1e-2)


@pytest.mark.smoke
def test_grouped_gemm_refuses_a_strided_out():
    """The output buffer is declared contiguous; a strided one is refused before the kernel."""
    workload = MoeGroupedGemmWorkload(_gemm_call(torch.bfloat16, _TIGHT, P=64, K=64, E=2, N=64))
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    out = torch.empty(64, 128, dtype=torch.bfloat16, device="cuda")[:, ::2]
    with pytest.raises(ValueError, match="out must be contiguous"):
        op(*workload.gen_inputs(), out=out)


@pytest.mark.smoke
def test_grouped_gemm_refuses_what_the_template_cannot_run_at_selection():
    """Calls outside the adapter's region are refused by selection, naming the reason."""
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())
    a = torch.randn(8, 60, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2, 16, 60, dtype=torch.bfloat16, device="cuda")
    ends = torch.tensor([4, 8], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op(a, b, ends)  # K not a multiple of 8
    fused = MoeGroupedGemmFwdOp(
        ContiguousLayoutSpec.tight_physical_psum(), activation="silu_and_mul"
    )
    a = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2, 24, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="no implementation serves this call"):
        fused(a, b, ends)  # fused N must be a multiple of 16
    # An aligned layout whose alignment is not a tile height has no instantiation either.
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.aligned_per_row(8))
    a = torch.randn(16, 64, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2, 16, 64, dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([0] * 8 + [1] * 8, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="no implementation serves this call"):
        op(a, b, ids)


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
        layout=_TIGHT,
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
