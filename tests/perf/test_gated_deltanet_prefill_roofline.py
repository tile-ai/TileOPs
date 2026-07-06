import pytest

from tileops.perf.formulas import gated_deltanet_prefill_fwd_roofline

pytestmark = pytest.mark.smoke


def _expected_roofline(
    batch: int,
    heads: int,
    seq_len: int,
    dim_k: int,
    dim_v: int,
    chunk_size: int,
    elem_bytes: int,
) -> tuple[int, int]:
    num_chunks = seq_len // chunk_size
    state_flops = 4 * batch * heads * num_chunks * chunk_size * dim_k * dim_v
    intra_flops = 4 * batch * heads * num_chunks * chunk_size * chunk_size * (
        dim_k + dim_v
    )
    input_elems = (
        3 * batch * heads * seq_len * dim_k
        + batch * heads * seq_len * dim_v
        + 2 * batch * heads * seq_len
    )
    output_elems = batch * heads * seq_len * dim_v + batch * heads * dim_k * dim_v
    return state_flops + intra_flops, (input_elems + output_elems) * elem_bytes


def test_gated_deltanet_prefill_roofline_manifest_bthd_layout() -> None:
    flops, nbytes = gated_deltanet_prefill_fwd_roofline(
        q_shape=[1, 512, 16, 128],
        v_shape=[1, 512, 16, 128],
        chunk_size=64,
        layout="bthd",
        dtype="float16",
    )

    assert (flops, nbytes) == _expected_roofline(
        batch=1,
        heads=16,
        seq_len=512,
        dim_k=128,
        dim_v=128,
        chunk_size=64,
        elem_bytes=2,
    )
    assert flops > 0
    assert flops > 0


def test_gated_deltanet_prefill_roofline_head_major_layout() -> None:
    flops, nbytes = gated_deltanet_prefill_fwd_roofline(
        q_shape=[1, 16, 512, 128],
        v_shape=[1, 16, 512, 128],
        chunk_size=64,
        layout="bhtd",
        dtype="float16",
    )

    assert (flops, nbytes) == _expected_roofline(
        batch=1,
        heads=16,
        seq_len=512,
        dim_k=128,
        dim_v=128,
        chunk_size=64,
        elem_bytes=2,
    )
