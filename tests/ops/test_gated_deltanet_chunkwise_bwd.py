import pytest
import torch

from tests.test_base import FixtureBase
from tileops.kernels.gated_deltanet.gated_deltanet_bwd import GatedDeltaNetBwdKernel
from tileops.ops import GatedDeltaNetBwdOp


def _differentiable_fwd(q, k, v, g_raw, beta, chunk_size):
    """Fully differentiable chunked forward matching paper (Eq. 10 via WY)."""
    B, H, S, DK = q.shape
    DV = v.shape[-1]
    BC = chunk_size
    NC = S // BC
    g_cum = g_raw.float().reshape(B, H, NC, BC).cumsum(-1).reshape(B, H, S)
    h = q.new_zeros(B, H, DK, DV)
    o_chunks = []
    eye = torch.eye(BC, device=q.device, dtype=torch.float32)
    mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.float32))
    for c in range(NC):
        sl = slice(c * BC, (c + 1) * BC)
        qc = q[:, :, sl, :].float()
        kc = k[:, :, sl, :].float()
        vc = v[:, :, sl, :].float()
        gc = g_cum[:, :, sl]
        bc = beta[:, :, sl].float()
        Gram = torch.einsum("bhik,bhjk->bhij", kc, kc)
        Gamma = torch.exp(gc.unsqueeze(-1) - gc.unsqueeze(-2))
        M = bc.unsqueeze(-1) * (Gamma * Gram)
        A = eye + torch.tril(M, diagonal=-1)
        A_inv = torch.linalg.inv(A)
        wc = A_inv @ (kc * bc.unsqueeze(-1))
        uc = A_inv @ (vc * bc.unsqueeze(-1))
        g_last = gc[:, :, -1:]
        v_new = uc - (wc * torch.exp(gc + g_last).unsqueeze(-1)) @ h
        o_part = (qc @ h) * torch.exp(gc).unsqueeze(-1)
        attn = (qc @ kc.transpose(-2, -1)) * Gamma * mask
        o_c = o_part + attn @ v_new
        o_chunks.append(o_c)
        k_sc = kc * torch.exp(g_last - gc).unsqueeze(-1)
        h = h * torch.exp(g_last).unsqueeze(-1) + k_sc.transpose(-2, -1) @ v_new
    return torch.cat(o_chunks, dim=2)


def gated_deltanet_autograd_bwd_torch(do, q, k, v, g, beta, chunk_size):
    """Compute backward gradients via autograd on the differentiable forward."""
    q_ = q.float().detach().requires_grad_(True)
    k_ = k.float().detach().requires_grad_(True)
    v_ = v.float().detach().requires_grad_(True)
    g_ = g.float().detach().requires_grad_(True)
    beta_ = beta.float().detach().requires_grad_(True)

    o = _differentiable_fwd(q_, k_, v_, g_, beta_, chunk_size)
    loss = (o * do.float()).sum()
    dq, dk, dv, dg, dbeta = torch.autograd.grad(loss, [q_, k_, v_, g_, beta_])
    return dq, dk, dv, dg, dbeta


# Autograd-based reference: differentiable forward → torch.autograd.grad


# Backward correctness tests


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-2, "rtol": 1e-2}
    elif dtype == torch.float16:
        return {"atol": 5e-2, "rtol": 5e-2}
    else:  # bfloat16 — wider tolerance due to compounding chunk-boundary
        # rounding in bf16 (7-bit mantissa); validated against FLA at 0.998+ cosine.
        return {"atol": 1e-1, "rtol": 1e-1}


class GatedDeltaNetBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune",
            [
                pytest.param(2, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(2, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(1, 128, 4, 64, 64, 32, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1, 128, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1, 128, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(1, 128, 2, 128, 128, 64, torch.float16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@GatedDeltaNetBwdFixture
def test_gated_deltanet_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    torch.manual_seed(42)
    B, H, S, DK, DV, BC = batch, heads, seq_len, dim_k, dim_v, chunk_size
    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, H, S, device="cuda", dtype=dtype)
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5

    # Forward to get S for backward kernel
    from tileops.ops import GatedDeltaNetFwdOp

    fwd_op = GatedDeltaNetFwdOp(chunk_size=BC)
    _o, S_fwd, _Aw, _Au = fwd_op.forward(q, k, v, g, beta)
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    # Reference via autograd
    ref_dq, ref_dk, ref_dv, ref_dg, ref_dbeta = gated_deltanet_autograd_bwd_torch(
        do, q, k, v, g, beta, BC
    )
    ref_outputs = (ref_dq, ref_dk, ref_dv, ref_dg, ref_dbeta)

    # Kernel
    op = GatedDeltaNetBwdOp(chunk_size=BC, tune=tune)
    op_outputs = op.forward(do, q, k, v, g, beta, S_fwd)

    tols = _get_tolerances(dtype)
    names = ["dq", "dk", "dv", "dg", "dbeta"]
    for name, ref_out, op_out in zip(names, ref_outputs, op_outputs, strict=True):
        torch.testing.assert_close(
            op_out,
            ref_out.to(dtype),
            **tols,
            msg=lambda m, n=name: f"{n}: {m}",
        )
        if dtype == torch.float16 and dim_k == 128 and dim_v == 128:
            diff_norm = torch.linalg.vector_norm(op_out.float() - ref_out.float())
            ref_norm = torch.linalg.vector_norm(ref_out.float()).clamp_min(1e-12)
            relative_l2 = float((diff_norm / ref_norm).item())
            l2_limit = 1.5e-2 if op.kernel.default_config["recurrence_segmented_carry"] else 3e-2
            assert relative_l2 < l2_limit, (
                f"{name}: relative L2 error {relative_l2:.6f} exceeds {l2_limit}; "
                "this usually indicates a missing prepare-A gradient path"
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("dtype", "block_v"),
    [
        pytest.param(torch.float16, 64, marks=pytest.mark.smoke),
        pytest.param(torch.bfloat16, 128, marks=pytest.mark.smoke),
    ],
)
def test_gated_deltanet_bwd_segmented_carry_matches_sequential_d128(
    dtype: torch.dtype,
    block_v: int,
) -> None:
    torch.manual_seed(456)
    B, H, S, DK, DV, BC = 1, 1, 1024, 128, 128, 64
    q = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, H, S, DK, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, H, S, device="cuda", dtype=dtype)
    beta = torch.rand(B, H, S, device="cuda", dtype=dtype) * 0.5
    do = torch.randn(B, H, S, DV, device="cuda", dtype=dtype) * 0.1

    from tileops.ops import GatedDeltaNetFwdOp

    fwd_op = GatedDeltaNetFwdOp(chunk_size=BC)
    _o, S_fwd, _Aw, _Au = fwd_op.forward(q, k, v, g, beta)

    common_config = {
        "num_stages": 2,
        "threads": 128,
        "parallel_threads": 256,
        "recurrence_threads": 128,
        "recurrence_segment_chunks": 8,
    }
    baseline = GatedDeltaNetBwdKernel(
        B,
        H,
        S,
        BC,
        DK,
        DV,
        str(dtype).removeprefix("torch."),
        config={
            **common_config,
            "recurrence_block_v": 64,
            "recurrence_segmented_carry": 0,
        },
    )
    split = GatedDeltaNetBwdKernel(
        B,
        H,
        S,
        BC,
        DK,
        DV,
        str(dtype).removeprefix("torch."),
        config={
            **common_config,
            "recurrence_block_v": block_v,
            "recurrence_segmented_carry": 1,
        },
    )
    baseline_outputs = baseline.forward(do, q, k, v, g, beta, S_fwd)
    split_outputs = split.forward(do, q, k, v, g, beta, S_fwd)

    for name, expected, actual in zip(
        ["dq", "dk", "dv", "dg", "dbeta"],
        baseline_outputs,
        split_outputs,
        strict=True,
    ):
        torch.testing.assert_close(
            actual,
            expected,
            atol=1e-3,
            rtol=1e-3,
            msg=lambda m, n=name: f"{n}: {m}",
        )


@pytest.mark.parametrize(
    (
        "batch",
        "heads",
        "seq_len",
        "chunk_size",
        "dim_v",
        "expected_mode",
        "expected_block_v",
    ),
    [
        (1, 16, 1024, 64, 128, 0, 32),
        (1, 16, 2048, 64, 128, 1, 128),
        (2, 16, 4096, 64, 128, 1, 128),
        (4, 16, 4096, 64, 128, 1, 128),
        (1, 16, 4160, 64, 128, 0, 32),
        (1, 16, 4096, 64, 64, 0, 0),
    ],
)
@pytest.mark.smoke
def test_gated_deltanet_bwd_default_carry_dispatch(
    batch: int,
    heads: int,
    seq_len: int,
    chunk_size: int,
    dim_v: int,
    expected_mode: int,
    expected_block_v: int,
) -> None:
    kernel = GatedDeltaNetBwdKernel(
        batch=batch,
        head=heads,
        seq_len=seq_len,
        chunk_size=chunk_size,
        dim_k=128,
        dim_v=dim_v,
        dtype="float16",
    )
    assert kernel.default_config["recurrence_segmented_carry"] == expected_mode
    assert kernel.default_config["recurrence_block_v"] == expected_block_v


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
