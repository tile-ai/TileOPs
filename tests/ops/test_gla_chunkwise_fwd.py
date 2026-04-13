import pytest
import torch

from tests.ops.gla_test_utils import cosine_sim, get_tolerances
from tests.test_base import FixtureBase
from tileops.ops import GLAFwdOp


def gla_fwd_chunked_torch(q, k, v, g, chunk_size, scale=None):
    """Fully differentiable chunked GLA forward in float32."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BC = chunk_size
    NC = T // BC

    if scale is None:
        scale = K ** -0.5

    q = q.float() * scale
    k = k.float()
    v = v.float()
    g = g.float()

    g_cum = g.reshape(B, NC, BC, H, K).cumsum(dim=2).reshape(B, T, H, K)

    h = q.new_zeros(B, H, K, V)
    mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.float32))

    o_chunks = []
    for c in range(NC):
        sl = slice(c * BC, (c + 1) * BC)
        qc = q[:, sl, :, :]
        kc = k[:, sl, :, :]
        vc = v[:, sl, :, :]
        gc = g_cum[:, sl, :, :]
        g_last = gc[:, -1:, :, :]

        q_gated = qc * torch.exp(gc)
        o_inter = torch.einsum("bthk,bhkv->bthv", q_gated, h)

        k_ungated = kc * torch.exp(-gc)
        A = torch.einsum("bihk,bjhk->bhij", q_gated, k_ungated)
        A = A * mask.unsqueeze(0).unsqueeze(0)
        o_intra = torch.einsum("bhij,bjhv->bihv", A, vc)

        o_chunks.append(o_inter + o_intra)

        k_adj = kc * torch.exp(g_last - gc)
        h = h * torch.exp(g_last).permute(0, 2, 3, 1).squeeze(-1).unsqueeze(-1)
        h = h + torch.einsum("bthk,bthv->bhkv", k_adj, vc)

    return torch.cat(o_chunks, dim=1)

try:
    from fla.ops.gla import chunk_gla
except ImportError:
    chunk_gla = None


# =============================================================================
# Forward correctness tests
# =============================================================================

class GLAFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 64, 2, 64, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(2, 64, 2, 64, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(2, 64, 2, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(1, 128, 4, 64, 64, 64, torch.float32, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 256, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


@GLAFwdFixture
def test_gla_fwd(
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
    B, T, H, K, V, BC = batch, seq_len, heads, dim_k, dim_v, chunk_size
    scale = K ** -0.5

    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype)

    # --- Torch reference ---
    ref_o = gla_fwd_chunked_torch(q, k, v, g, BC, scale=scale)

    # --- FLA reference (if available) ---
    if chunk_gla is not None:
        fla_o, _ = chunk_gla(q.float(), k.float(), v.float(), g.float(), scale=scale)
        cos = cosine_sim(ref_o, fla_o)
        print(f"  FLA vs ref o: cosine={cos:.6f}")
        assert cos > 0.99, f"FLA vs ref o cosine too low: {cos:.6f}"

    # --- TileOPs ---
    fwd_op = GLAFwdOp(B, T, H, K, V, BC, scale=scale,
                       output_final_state=False, dtype=dtype, tune=tune)
    op_o, _ = fwd_op.forward(q, k, v, g)

    tols = get_tolerances(dtype)
    cos = cosine_sim(ref_o, op_o)
    print(f"  TileOPs vs ref o: cosine={cos:.6f}")
    torch.testing.assert_close(
        op_o.float(), ref_o.float(), **tols,
        msg=lambda m: f"o: {m}",
    )

    # --- TileOPs vs FLA ---
    if chunk_gla is not None:
        cos = cosine_sim(fla_o, op_o)
        print(f"  TileOPs vs FLA o: cosine={cos:.6f}")
        assert cos > 0.99, f"TileOPs vs FLA o cosine too low: {cos:.6f}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
