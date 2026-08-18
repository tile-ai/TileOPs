from types import SimpleNamespace

import pytest
import torch

from tileops.ops import GroupedQueryAttentionPrefillDenseFwdOp
from tileops.perf.formulas import gqa_fwd_roofline

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("pos_encoding_mode", ["none", "rope"])
def test_gqa_dense_roofline_requires_runtime_inputs(pos_encoding_mode: str) -> None:
    op = GroupedQueryAttentionPrefillDenseFwdOp(
        batch=1,
        heads=8,
        heads_kv=2,
        seq_len=64,
        dim=128,
        dtype=torch.float16,
        pos_encoding_mode=pos_encoding_mode,
    )

    with pytest.raises(RuntimeError, match="requires a prior forward"):
        op.eval_roofline()


def test_gqa_dense_roofline_uses_distinct_input_and_output_dtypes() -> None:
    op = SimpleNamespace(
        batch=1,
        seq_len=3,
        seq_len_kv=5,
        heads=4,
        heads_kv=2,
        dim=8,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        input_dtype=torch.float8_e4m3fn,
        output_dtype=torch.float16,
        dtype=torch.float16,
    )

    flops, nbytes = gqa_fwd_roofline(op)

    assert flops == 4 * 1 * 4 * (3 * 5) * 8
    q_elems = 1 * 3 * 4 * 8
    kv_elems = 1 * 5 * 2 * 8
    assert nbytes == q_elems + 2 * kv_elems + 2 * q_elems + 3 * 1 * 2 * 4


def test_gqa_dense_roofline_counts_rectangular_window_visibility() -> None:
    flops, _ = gqa_fwd_roofline(
        q_shape=(1, 2, 4, 8),
        kv_shape=(1, 4, 2, 8),
        is_causal=True,
        window_size_left=1,
        window_size_right=-1,
        dtypes=["float16"],
    )

    # Bottom-right alignment gives query centers 2 and 3; each sees two keys.
    assert flops == 4 * 1 * 4 * 4 * 8


def test_gqa_dense_roofline_counts_fused_rope() -> None:
    op = SimpleNamespace(
        batch=1,
        seq_len=3,
        seq_len_kv=5,
        heads=4,
        heads_kv=2,
        dim=8,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        dtype=torch.float16,
        fuse_rope=True,
        rotary_dim=4,
        max_position=7,
    )

    flops, nbytes = gqa_fwd_roofline(op)

    attention_flops = 4 * 1 * 4 * (3 * 5) * 8
    rope_flops = 3 * 1 * (3 * 4 + 5 * 2) * 4
    assert flops == attention_flops + rope_flops
    q_elems = 1 * 3 * 4 * 8
    kv_elems = 1 * 5 * 2 * 8
    attention_bytes = 2 * (q_elems + 2 * kv_elems + q_elems)
    rope_table_bytes = 7 * 4 * 2
    assert nbytes == attention_bytes + rope_table_bytes
