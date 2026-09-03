import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import DeltaNetFwdOp
from workloads.linear_attention import DeltaNetFwdWorkload


class DeltaNetFwdTest(DeltaNetFwdWorkload, TestBase):
    pass


def _get_tolerances(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.float16:
        return {"atol": 2e-2, "rtol": 2e-2}
    else:  # bfloat16
        return {"atol": 5e-2, "rtol": 5e-2}


class DeltaNetFwdFixture(FixtureBase):
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
                pytest.param(2, 8192, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2, 16384, 4, 64, 64, 64, torch.float16, False, marks=pytest.mark.full),
                # chunk_size=64 is where the untuned default takes a tiled width, so
                # the tuned run has to beat a tiled baseline rather than no tiling.
                pytest.param(
                    2,
                    128,
                    2,
                    64,
                    64,
                    64,
                    torch.bfloat16,
                    True,
                    marks=pytest.mark.full,
                    id="full-bf16-tuned",
                ),
            ],
        ),
    ]


@DeltaNetFwdFixture
def test_deltanet_fwd(
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
    test = DeltaNetFwdTest(batch, heads, seq_len, dim_k, dim_v, chunk_size, dtype)
    op = DeltaNetFwdOp(chunk_size=chunk_size, tune=tune)
    tols = _get_tolerances(dtype)
    inputs = test.gen_inputs()
    ref_o = test.ref_program(*inputs)
    op_o, _S, _Aw, _Au, _w, _u = op(*inputs)
    torch.testing.assert_close(op_o, ref_o, **tols)
    if tune:
        # The forward above already proves the selected config builds and runs;
        # this pins it to the declared candidate set the sweep draws from.
        assert op.kernel.config in op.kernel.autotune_configs
