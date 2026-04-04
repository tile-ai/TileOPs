
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GatedDeltaNetFwdOp
from workloads.ops.gated_deltanet_chunkwise_fwd import (
    GatedDeltaNetFwdTest as _GatedDeltaNetFwdTestWorkload,
)


class GatedDeltaNetFwdTest(_GatedDeltaNetFwdTestWorkload, TestBase):
    pass


# =============================================================================
# Torch reference implementations (test-only)
# =============================================================================


# =============================================================================
# Forward correctness tests
# =============================================================================

def _get_tolerances(dtype: torch.dtype) -> dict:
    # Tolerances are looser than docs/testing.md defaults (fp16: 1e-3, bf16: 1.6e-2)
    # because Gated DeltaNet uses sequential chunk recurrence: each chunk's hidden
    # state h depends on all prior chunks, so fp32 rounding errors accumulate across
    # the chunk chain. With seq_len=128 and chunk_size=32 that is 4 serial steps of
    # matmul + exp + state update, which amplifies per-element error well beyond
    # single-kernel tolerances.
    if dtype == torch.float32:
        return {"atol": 1e-2, "rtol": 1e-2}
    elif dtype == torch.float16:
        return {"atol": 5e-2, "rtol": 5e-2}
    else:  # bfloat16
        return {"atol": 1e-1, "rtol": 1e-1}


class GatedDeltaNetFwdFixture(FixtureBase):
    PARAMS = [
        ("batch, seq_len, heads, dim_k, dim_v, chunk_size, dtype, tune", [
            pytest.param(2, 64, 2, 64, 64, 32, torch.float32, False, marks=pytest.mark.smoke),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float32, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 64, 2, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(1, 128, 4, 64, 64, 32, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(2, 8192, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
            pytest.param(2, 16384, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
        ]),
    ]


@GatedDeltaNetFwdFixture
def test_gated_deltanet_fwd(
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
    test = GatedDeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    op = GatedDeltaNetFwdOp(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype, tune=tune)
    tols = _get_tolerances(dtype)
    inputs = test.gen_inputs()
    ref_o = test.ref_program(*inputs)
    op_o, _S, _Aw, _Au = op(*inputs)
    torch.testing.assert_close(op_o, ref_o, **tols)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
