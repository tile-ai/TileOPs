import pytest
import torch

from tests.ops.test_gated_deltanet_fwd import (
    compute_w_u_torch,
    kernel2_gated_deltanet_torch,
    prepare_wy_repr_gated_torch,
)
from tests.test_base import FixtureBase, TestBase
from tileops.ops import GatedDeltaNetPrefillFwdOp
from workloads.gated_deltanet import (
    GatedDeltaNetPrefillFwdTest as _GatedDeltaNetPrefillFwdTestWorkload,
)


def dense_chunk_transitions_torch(
    k: torch.Tensor,
    g: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense per-chunk affine transitions for scan-path validation."""
    B, H, S, DK = k.shape
    DV = u.shape[-1]
    BC = chunk_size
    num_chunks = S // BC

    k = k.float().reshape(B, H, num_chunks, BC, DK)
    g = g.float().reshape(B, H, num_chunks, BC)
    w = w.float().reshape(B, H, num_chunks, BC, DK)
    u = u.float().reshape(B, H, num_chunks, BC, DV)

    A = torch.empty(B, H, num_chunks, DK, DK, dtype=torch.float32, device=k.device)
    bias = torch.empty(B, H, num_chunks, DK, DV, dtype=torch.float32, device=k.device)
    eye = torch.eye(DK, dtype=torch.float32, device=k.device).view(1, 1, DK, DK)

    for c in range(num_chunks):
        k_c = k[:, :, c]
        g_c = g[:, :, c]
        w_c = w[:, :, c]
        u_c = u[:, :, c]
        g_last = g_c[:, :, -1]

        k_scaled = k_c * torch.exp(g_last.unsqueeze(-1) - g_c).unsqueeze(-1)
        w_scaled = w_c * torch.exp(g_c + g_last.unsqueeze(-1)).unsqueeze(-1)
        A[:, :, c] = torch.exp(g_last).view(B, H, 1, 1) * eye - torch.einsum(
            "bhnk,bhnj->bhkj", k_scaled, w_scaled
        )
        bias[:, :, c] = torch.einsum("bhnk,bhnv->bhkv", k_scaled, u_c)

    return A, bias


def compose_dense_transitions_torch(
    prev_A: torch.Tensor,
    prev_bias: torch.Tensor,
    next_A: torch.Tensor,
    next_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose ``next`` after ``prev`` for H' = A @ H + B."""
    composed_A = torch.einsum("bhij,bhjk->bhik", next_A, prev_A)
    composed_bias = torch.einsum("bhij,bhjv->bhiv", next_A, prev_bias) + next_bias
    return composed_A, composed_bias


def serial_prefix_dense_transitions_torch(
    A: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference prefix scan over dense chunk transitions."""
    B, H, num_chunks, DK, _ = A.shape
    DV = bias.shape[-1]
    running_A = torch.eye(DK, dtype=torch.float32, device=A.device).expand(B, H, DK, DK)
    running_bias = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=A.device)
    prefix_A = torch.empty_like(A)
    prefix_bias = torch.empty_like(bias)

    for c in range(num_chunks):
        running_A, running_bias = compose_dense_transitions_torch(
            running_A, running_bias, A[:, :, c], bias[:, :, c]
        )
        prefix_A[:, :, c] = running_A
        prefix_bias[:, :, c] = running_bias

    return prefix_A, prefix_bias


def butterfly_prefix_dense_transitions_torch(
    A: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hillis-Steele style inclusive scan over dense chunk transitions."""
    num_chunks = A.shape[2]
    prefix_A = A.clone()
    prefix_bias = bias.clone()

    offset = 1
    while offset < num_chunks:
        prev_A = prefix_A.clone()
        prev_bias = prefix_bias.clone()
        for c in range(offset, num_chunks):
            prefix_A[:, :, c], prefix_bias[:, :, c] = compose_dense_transitions_torch(
                prev_A[:, :, c - offset],
                prev_bias[:, :, c - offset],
                prev_A[:, :, c],
                prev_bias[:, :, c],
            )
        offset *= 2

    return prefix_A, prefix_bias


def structured_chunk_transitions_torch(
    k: torch.Tensor,
    g: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build structured per-chunk transitions A = alpha I + U V^T."""
    B, H, S, DK = k.shape
    DV = u.shape[-1]
    BC = chunk_size
    num_chunks = S // BC

    k = k.float().reshape(B, H, num_chunks, BC, DK)
    g = g.float().reshape(B, H, num_chunks, BC)
    w = w.float().reshape(B, H, num_chunks, BC, DK)
    u = u.float().reshape(B, H, num_chunks, BC, DV)

    alpha = torch.empty(B, H, num_chunks, dtype=torch.float32, device=k.device)
    U = torch.empty(B, H, num_chunks, DK, BC, dtype=torch.float32, device=k.device)
    V = torch.empty(B, H, num_chunks, DK, BC, dtype=torch.float32, device=k.device)
    bias = torch.empty(B, H, num_chunks, DK, DV, dtype=torch.float32, device=k.device)

    for c in range(num_chunks):
        k_c = k[:, :, c]
        g_c = g[:, :, c]
        w_c = w[:, :, c]
        u_c = u[:, :, c]
        g_last = g_c[:, :, -1]

        k_scaled = k_c * torch.exp(g_last.unsqueeze(-1) - g_c).unsqueeze(-1)
        w_scaled = w_c * torch.exp(g_c + g_last.unsqueeze(-1)).unsqueeze(-1)
        alpha[:, :, c] = torch.exp(g_last)
        U[:, :, c] = -k_scaled.transpose(-1, -2)
        V[:, :, c] = w_scaled.transpose(-1, -2)
        bias[:, :, c] = torch.einsum("bhnk,bhnv->bhkv", k_scaled, u_c)

    return alpha, U, V, bias


def materialize_structured_A_torch(
    alpha: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """Materialize a structured transition for reference checks."""
    DK = U.shape[-2]
    eye = torch.eye(DK, dtype=torch.float32, device=U.device).view(1, 1, DK, DK)
    return alpha[..., None, None] * eye + torch.einsum("...kr,...jr->...kj", U, V)


def compose_structured_transition_torch(
    prev: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    next_: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose ``next`` after ``prev`` while keeping A = alpha I + U V^T."""
    prev_alpha, prev_U, prev_V, prev_bias = prev
    next_alpha, next_U, next_V, next_bias = next_

    alpha = next_alpha * prev_alpha
    cross = torch.einsum("bhkr,bhks->bhrs", prev_U, next_V)
    prev_V_part = next_alpha.view(*next_alpha.shape, 1, 1) * prev_V
    next_V_part = (
        prev_alpha.view(*prev_alpha.shape, 1, 1) * next_V
        + torch.einsum("bhkr,bhrs->bhks", prev_V, cross)
    )
    U = torch.cat([prev_U, next_U], dim=-1)
    V = torch.cat([prev_V_part, next_V_part], dim=-1)

    next_inner = torch.einsum("bhkr,bhkv->bhrv", next_V, prev_bias)
    bias = (
        next_alpha.view(*next_alpha.shape, 1, 1) * prev_bias
        + torch.einsum("bhkr,bhrv->bhkv", next_U, next_inner)
        + next_bias
    )
    return alpha, U, V, bias


def butterfly_prefix_structured_transitions_torch(
    alpha: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    bias: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Hillis-Steele inclusive scan over structured chunk transitions."""
    num_chunks = alpha.shape[2]
    prefix = [
        (alpha[:, :, c], U[:, :, c], V[:, :, c], bias[:, :, c])
        for c in range(num_chunks)
    ]

    offset = 1
    while offset < num_chunks:
        prev_prefix = prefix.copy()
        for c in range(offset, num_chunks):
            prefix[c] = compose_structured_transition_torch(
                prev_prefix[c - offset], prev_prefix[c]
            )
        offset *= 2

    return prefix


class GatedDeltaNetPrefillFwdTest(_GatedDeltaNetPrefillFwdTestWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, S, DK = k.shape
        _, _, _, DV = v.shape
        BC = self.chunk_size
        g_cum = g.float().reshape(B, H, S // BC, BC).cumsum(-1).reshape(B, H, S).to(g.dtype)
        Aw, Au = prepare_wy_repr_gated_torch(k, g_cum, beta, BC)
        w, u = compute_w_u_torch(Aw, Au, k, v, beta, BC)
        S_0 = torch.zeros(B, H, DK, DV, dtype=torch.float32, device=q.device)
        final_state, o = kernel2_gated_deltanet_torch(q, k, g_cum, w, u, S_0, BC)
        return o.to(self.dtype), final_state.to(self.dtype)


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-2, "rtol": 1e-2}
    if dtype == torch.float16:
        return {"atol": 5e-2, "rtol": 5e-2}
    return {"atol": 1e-1, "rtol": 1e-1}


class GatedDeltaNetPrefillFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(1, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(1, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


@GatedDeltaNetPrefillFwdFixture
def test_gated_deltanet_prefill_fwd(
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
    test = GatedDeltaNetPrefillFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    op = GatedDeltaNetPrefillFwdOp(
        batch,
        heads,
        seq_len,
        dim_k,
        dim_v,
        chunk_size,
        dtype,
        layout="bhtd",
        tune=tune,
    )
    test.check(op, *test.gen_inputs(), **_get_tolerances(dtype))


@pytest.mark.smoke
def test_gated_deltanet_prefill_bthd_layout_matches_bhtd() -> None:
    torch.manual_seed(42)
    B, H, S, DK, DV, BC = 1, 4, 128, 64, 64, 64
    dtype = torch.float16

    test = GatedDeltaNetPrefillFwdTest(B, H, S, DK, DV, BC, dtype)
    q, k, v, g, beta = test.gen_inputs()

    bhtd_op = GatedDeltaNetPrefillFwdOp(
        B,
        H,
        S,
        DK,
        DV,
        BC,
        dtype,
        layout="bhtd",
        tune=False,
    )
    bthd_op = GatedDeltaNetPrefillFwdOp(
        B,
        H,
        S,
        DK,
        DV,
        BC,
        dtype,
        tune=False,
    )

    o_bhtd, state_bhtd = bhtd_op(q, k, v, g, beta)
    q_bthd = q.permute(0, 2, 1, 3)
    k_bthd = k.permute(0, 2, 1, 3)
    v_bthd = v.permute(0, 2, 1, 3)
    g_bthd = g.permute(0, 2, 1)
    beta_bthd = beta.permute(0, 2, 1)
    assert not q_bthd.is_contiguous()
    assert not g_bthd.is_contiguous()
    o_bthd, state_bthd = bthd_op(
        q_bthd,
        k_bthd,
        v_bthd,
        g_bthd,
        beta_bthd,
    )

    torch.testing.assert_close(
        o_bthd.permute(0, 2, 1, 3).contiguous(),
        o_bhtd,
        **_get_tolerances(dtype),
    )
    torch.testing.assert_close(state_bthd, state_bhtd, **_get_tolerances(dtype))


@pytest.mark.smoke
def test_gated_deltanet_prefill_bthd_blocksolve_path_matches_bhtd() -> None:
    torch.manual_seed(42)
    B, H, S, DK, DV, BC = 1, 16, 128, 128, 128, 64
    dtype = torch.float16

    test = GatedDeltaNetPrefillFwdTest(B, H, S, DK, DV, BC, dtype)
    q, k, v, g, beta = test.gen_inputs()

    bhtd_op = GatedDeltaNetPrefillFwdOp(
        B,
        H,
        S,
        DK,
        DV,
        BC,
        dtype,
        layout="bhtd",
        tune=False,
    )
    bthd_op = GatedDeltaNetPrefillFwdOp(
        B,
        H,
        S,
        DK,
        DV,
        BC,
        dtype,
        tune=False,
        layout="bthd",
    )

    o_bhtd, state_bhtd = bhtd_op(q, k, v, g, beta)
    o_bthd, state_bthd = bthd_op(
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        g.permute(0, 2, 1).contiguous(),
        beta.permute(0, 2, 1).contiguous(),
    )

    torch.testing.assert_close(
        o_bthd.permute(0, 2, 1, 3).contiguous(),
        o_bhtd,
        **_get_tolerances(dtype),
    )
    torch.testing.assert_close(state_bthd, state_bhtd, **_get_tolerances(dtype))


@pytest.mark.smoke
def test_gated_deltanet_prefill_dense_transition_reference() -> None:
    torch.manual_seed(42)
    B, H, S, DK, DV, BC = 1, 2, 64, 16, 16, 16
    dtype = torch.float32
    device = torch.device("cuda")

    test = GatedDeltaNetPrefillFwdTest(B, H, S, DK, DV, BC, dtype)
    q, k, v, g, beta = test.gen_inputs()
    q, k, v, g, beta = (
        q.to(device),
        k.to(device),
        v.to(device),
        g.to(device),
        beta.to(device),
    )

    g_cum = g.float().reshape(B, H, S // BC, BC).cumsum(-1).reshape(B, H, S)
    Aw, Au = prepare_wy_repr_gated_torch(k, g_cum, beta, BC)
    w, u = compute_w_u_torch(Aw, Au, k, v, beta, BC)

    S_0 = torch.randn(B, H, DK, DV, dtype=torch.float32, device=device) * 0.01
    serial_final_state, _ = kernel2_gated_deltanet_torch(q, k, g_cum, w, u, S_0, BC)

    A, bias = dense_chunk_transitions_torch(k, g_cum, w, u, BC)
    prefix_A, prefix_bias = serial_prefix_dense_transitions_torch(A, bias)
    butterfly_A, butterfly_bias = butterfly_prefix_dense_transitions_torch(A, bias)
    structured_alpha, structured_U, structured_V, structured_bias = (
        structured_chunk_transitions_torch(k, g_cum, w, u, BC)
    )
    structured_A = materialize_structured_A_torch(
        structured_alpha, structured_U, structured_V
    )
    structured_prefix = butterfly_prefix_structured_transitions_torch(
        structured_alpha, structured_U, structured_V, structured_bias
    )
    final_alpha, final_U, final_V, final_bias = structured_prefix[-1]
    structured_final_A = materialize_structured_A_torch(final_alpha, final_U, final_V)
    scan_final_state = torch.einsum(
        "bhij,bhjv->bhiv", prefix_A[:, :, -1], S_0
    ) + prefix_bias[:, :, -1]

    torch.testing.assert_close(structured_A, A, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(structured_bias, bias, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(butterfly_A, prefix_A, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(butterfly_bias, prefix_bias, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(structured_final_A, prefix_A[:, :, -1], atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(final_bias, prefix_bias[:, :, -1], atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(scan_final_state, serial_final_state, atol=5e-4, rtol=5e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
