import pytest
import torch

from tileops.kernels.gla import GLAPrefillFwdKernel
from tileops.ops import GLAPrefillFwdOp
from workloads.linear_attention import GLAPrefillFwdWorkload


def _tolerances(dtype: torch.dtype) -> dict[str, float]:
    if dtype == torch.bfloat16:
        return {"atol": 1e-1, "rtol": 1e-1}
    if dtype == torch.float16:
        return {"atol": 5e-2, "rtol": 5e-2}
    return {"atol": 1e-2, "rtol": 1e-2}


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gla_prefill_fwd(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    test = GLAPrefillFwdWorkload(1, 128, 2, 64, 64, 64, dtype)
    inputs = test.gen_inputs()
    ref_o, ref_state = test.ref_program(*inputs)
    op = GLAPrefillFwdOp(chunk_size=64)
    o, state = op(*inputs)

    torch.testing.assert_close(o, ref_o, **_tolerances(dtype))
    torch.testing.assert_close(state, ref_state, **_tolerances(dtype))


@pytest.mark.smoke
def test_gla_prefill_partitioned_recurrence() -> None:
    torch.manual_seed(42)
    dtype = torch.bfloat16
    test = GLAPrefillFwdWorkload(1, 2048, 2, 128, 128, 64, dtype)
    inputs = test.gen_inputs()
    ref_o, ref_state = test.ref_program(*inputs)
    kernel = GLAPrefillFwdKernel(
        1,
        2048,
        2,
        128,
        128,
        chunk_size=64,
        dtype=dtype,
        config={
            "g_num_stages": 2,
            "g_threads": 128,
            "h_num_stages": 2,
            "h_threads": 128,
            "a_inter_threads": 64,
            "a_intra_threads": 128,
            "o_num_stages": 2,
            "o_threads": 256,
            "num_v_partitions": 2,
            "num_k_partitions": 2,
            "partition_chunks": 32,
            "partition_min_chunks": 0,
            "scan_threads": 128,
        },
    )
    assert kernel._partition_chunks == 32
    o, state = kernel(*inputs)

    torch.testing.assert_close(o, ref_o, **_tolerances(dtype))
    torch.testing.assert_close(state, ref_state, **_tolerances(dtype))
