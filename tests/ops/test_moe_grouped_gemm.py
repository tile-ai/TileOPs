"""Op-level tests for the staged grouped GEMM and expert MLP template.

``MoeGroupedGemmFwdOp`` selects ``MoeGroupedGemmKernel``, the adapter over the
shared template, through the staged candidate protocol; these tests route every layout
it claims through the op and check the rows the layout defines against the
workload's per-expert reference.
"""

import pytest
import torch

from tileops.kernels.grouped_gemm import GroupedGemmTemplate
from tileops.kernels.moe import MoeGroupedGemmKernel
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    MaskedLayoutSpec,
    MoeExpertMLPFwdOp,
    MoeGroupedGemmFwdOp,
)
from workloads.moe import MoeExpertMLPStagedWorkload, MoeGroupedGemmStagedWorkload

pytestmark = pytest.mark.hopper

_E = 6  # experts; the skewed distribution leaves five of them nearly empty


def _assert_valid_rows(out: torch.Tensor, ref: torch.Tensor, valid: torch.Tensor) -> None:
    flat_out = out.reshape(-1, out.shape[-1])[valid].float()
    flat_ref = ref.reshape(-1, ref.shape[-1])[valid].float()
    torch.testing.assert_close(flat_out, flat_ref, rtol=2e-2, atol=1e-1)


def _layout(name: str, args: dict):
    if name == "masked":
        return MaskedLayoutSpec(max_m=args["max_m"])
    return getattr(ContiguousLayoutSpec, name)(*args.values())


@pytest.mark.parametrize(
    "layout_name,layout_args,a_shape,dtype",
    [
        pytest.param(
            "tight_physical_psum",
            {},
            (600, 512),
            torch.bfloat16,
            marks=pytest.mark.smoke,
            id="tight-psum",
        ),
        pytest.param(
            "aligned_per_row",
            {"alignment": 128},
            (128 * 9, 512),
            torch.float16,
            marks=pytest.mark.smoke,
            id="aligned-per-row",
        ),
        pytest.param(
            "aligned_physical_psum",
            {"alignment": 128},
            (128 * 9, 512),
            torch.bfloat16,
            marks=pytest.mark.full,
            id="aligned-psum",
        ),
        pytest.param(
            "masked",
            {"max_m": 128},
            (_E, 128, 512),
            torch.bfloat16,
            marks=pytest.mark.full,
            id="masked",
        ),
    ],
)
def test_grouped_gemm_runs_each_layout_through_the_op(layout_name, layout_args, a_shape, dtype):
    """Each claimed layout: the op selects the template, builds it once, matches the reference."""
    workload = MoeGroupedGemmStagedWorkload(
        a_shape, (_E, 256, a_shape[-1]), layout_name, dtype, **layout_args
    )
    a, b, metadata = workload.gen_inputs()
    op = MoeGroupedGemmFwdOp(_layout(layout_name, layout_args))
    torch._assert_async(op.layout_guard(a, b, metadata))

    out = op(a, b, metadata)
    assert out.dtype is dtype and out.shape == (*a_shape[:-1], 256)
    _assert_valid_rows(out, workload.ref_program(a, b, metadata), workload.valid_rows)
    (kernel,) = op.built_kernels("grouped_gemm").values()
    assert isinstance(kernel, MoeGroupedGemmKernel)
    assert isinstance(kernel.inner, GroupedGemmTemplate)
    # A second call with a new row count reuses the instance: M is not in the key.
    if layout_name == "masked":
        a2_shape = a_shape
    elif layout_name == "tight_physical_psum":
        a2_shape = (a_shape[0] // 2, a_shape[1])
    else:
        a2_shape = (a_shape[0] - 2 * layout_args["alignment"], a_shape[1])
    a2, b2, metadata2 = MoeGroupedGemmStagedWorkload(
        a2_shape,
        (_E, 256, a_shape[-1]),
        layout_name,
        dtype,
        distribution="uniform",
        **layout_args,
    ).gen_inputs()
    op(a2, b2, metadata2)
    assert len(op.built_kernels("grouped_gemm")) == 1


@pytest.mark.smoke
@pytest.mark.parametrize("activation", ["silu_and_mul", "gelu_and_mul"])
def test_grouped_gemm_fuses_the_gated_activation(activation):
    """With ``activation`` the op hands the template a gate||up ``b`` and gets ffn columns back."""
    workload = MoeGroupedGemmStagedWorkload(
        (600, 512), (_E, 2 * 192, 512), "tight_physical_psum", torch.bfloat16, activation=activation
    )
    a, b, metadata = workload.gen_inputs()
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum(), activation=activation)
    out = op(a, b, metadata)
    assert out.shape == (600, 192)
    _assert_valid_rows(out, workload.ref_program(a, b, metadata), workload.valid_rows)
    (kernel,) = op.built_kernels("grouped_gemm").values()
    assert kernel.inner.activation == activation


@pytest.mark.smoke
def test_grouped_gemm_dims_off_the_tile_grid():
    """Cover non-tile-aligned K and N."""
    workload = MoeGroupedGemmStagedWorkload(
        (200, 96), (_E, 192, 96), "tight_physical_psum", torch.bfloat16
    )
    a, b, metadata = workload.gen_inputs()
    out = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum())(a, b, metadata)
    _assert_valid_rows(out, workload.ref_program(a, b, metadata), workload.valid_rows)


@pytest.mark.smoke
def test_grouped_gemm_fp32_output_and_preallocated_out():
    workload = MoeGroupedGemmStagedWorkload(
        (600, 512), (_E, 256, 512), "tight_physical_psum", torch.bfloat16
    )
    a, b, metadata = workload.gen_inputs()
    op = MoeGroupedGemmFwdOp(ContiguousLayoutSpec.tight_physical_psum(), out_dtype=torch.float32)
    out = torch.empty(600, 256, dtype=torch.float32, device="cuda")
    assert op(a, b, metadata, out=out) is out
    ref = workload.ref_program(a, b, metadata).float()
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-2)


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
    ids = torch.tensor([0] * 8 + [2] * 8, dtype=torch.int32, device="cuda")
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
    workload = MoeExpertMLPStagedWorkload(
        (600, 256),
        (_E, 2 * 192, 256),
        (_E, 256, 192),
        "tight_physical_psum",
        dtype,
        activation=activation,
    )
    x, w_gate_up, w_down, metadata = workload.gen_inputs()
    op = MoeExpertMLPFwdOp(ContiguousLayoutSpec.tight_physical_psum(), activation)
    out = op(x, w_gate_up, w_down, metadata)
    assert out.dtype is dtype and out.shape == (600, 256)
    _assert_valid_rows(
        out, workload.ref_program(x, w_gate_up, w_down, metadata), workload.valid_rows
    )
    (gate_up,) = op.gate_up.built_kernels("grouped_gemm").values()
    (down,) = op.down.built_kernels("grouped_gemm").values()
    assert (gate_up.inner.activation, down.inner.activation) == (activation, "none")
