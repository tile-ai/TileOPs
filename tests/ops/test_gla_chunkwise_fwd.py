import pytest
import torch

from tests.ops.gla_test_utils import (
    cosine_sim,
    get_tolerances,
    gla_fwd_chunked_torch,
)
from tests.test_base import FixtureBase
from tileops.kernels.linear_attention.gla import GLAPartitionedFwdKernel
from tileops.ops import GLAFwdOp

try:
    from fla.ops.gla import chunk_gla
except ImportError:
    chunk_gla = None


class GLAFwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune",
            [
                pytest.param(2, 64, 2, 64, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(2, 64, 2, 64, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(2, 64, 2, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(1, 128, 4, 64, 64, 64, torch.float32, False, marks=pytest.mark.full),
                pytest.param(1, 128, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
                pytest.param(1, 128, 4, 64, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
                pytest.param(2, 256, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            ],
        ),
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
    scale = K**-0.5

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

    fwd_op = GLAFwdOp(
        chunk_size=BC,
        scale=scale,
        tune=tune,
    )
    op_o, _ = fwd_op.forward(q, k, v, g)

    tols = get_tolerances(dtype)
    cos = cosine_sim(ref_o, op_o)
    print(f"  TileOPs vs ref o: cosine={cos:.6f}")
    torch.testing.assert_close(
        op_o.float(),
        ref_o.float(),
        **tols,
        msg=lambda m: f"o: {m}",
    )

    # --- TileOPs vs FLA ---
    if chunk_gla is not None:
        cos = cosine_sim(fla_o, op_o)
        print(f"  TileOPs vs FLA o: cosine={cos:.6f}")
        assert cos > 0.99, f"TileOPs vs FLA o cosine too low: {cos:.6f}"


@pytest.mark.smoke
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="partitioned GLA forward uses Hopper instructions",
)
def test_gla_partitioned_fwd() -> None:
    """The long-context specialization preserves GLAFwdOp's FP32 state ABI."""
    torch.manual_seed(42)
    batch, seq_len, heads, dim_k, dim_v, chunk_size = 1, 2048, 2, 128, 128, 64
    dtype = torch.bfloat16
    q = torch.randn(batch, seq_len, heads, dim_k, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(batch, seq_len, heads, dim_k, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(batch, seq_len, heads, dim_v, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(batch, seq_len, heads, dim_k, device="cuda", dtype=dtype)
    ref_o, ref_state = gla_fwd_chunked_torch(
        q,
        k,
        v,
        g,
        chunk_size,
        return_final_state=True,
    )
    kernel = GLAPartitionedFwdKernel(
        batch,
        seq_len,
        heads,
        dim_k,
        dim_v,
        chunk_size=chunk_size,
        dtype=dtype,
        config={
            "g_num_stages": 2,
            "g_threads": 128,
            "h_num_stages": 2,
            "h_threads": 128,
            "num_v_partitions": 2,
            "num_k_partitions": 2,
            "partition_chunks": 32,
            "partition_min_chunks": 0,
            "scan_threads": 128,
        },
    )
    o, final_state = kernel(q, k, v, g)

    assert final_state.dtype == torch.float32
    torch.testing.assert_close(o.float(), ref_o, **get_tolerances(dtype))
    torch.testing.assert_close(final_state, ref_state, **get_tolerances(dtype))
