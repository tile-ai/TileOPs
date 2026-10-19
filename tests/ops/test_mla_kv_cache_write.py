"""Public MLA latent cache write semantics: cache dtypes, both RoPE layouts and the skip path."""

import pytest
import torch

from tests.compile_contract import assert_op_owns_graph_nodes, register_compile_contract
from tests.workload_test_base import TestBase
from tileops.ops import MLAKVCacheWriteFwdOp
from workloads.attention.mla_cache_write import (
    MLAKVCacheWriteWorkload,
    mla_cache_write_result,
)


class MLAKVCacheWriteTest(MLAKVCacheWriteWorkload, TestBase):
    """The workload as a test case: the reference and its policy come from the workload."""


BF16, FP16, FP32, FP8 = torch.bfloat16, torch.float16, torch.float32, torch.float8_e4m3fn


def _check(workload, op):
    inputs = workload.gen_inputs()
    TestBase.check(workload, op, *inputs, runs=lambda *args: mla_cache_write_result(op, *args))


@pytest.mark.parametrize(
    "tokens,latent_dim,rope_dim,page_size,num_pages,dtype,cache_dtype,slots",
    [
        pytest.param(
            5,
            64,
            16,
            7,
            3,
            BF16,
            BF16,
            [20, -1, 0, 6, 13],
            marks=pytest.mark.smoke,
            id="bf16-odd-page-skip-and-page-edges",
        ),
        pytest.param(
            4,
            64,
            16,
            4,
            2,
            FP16,
            FP16,
            [7, 0, 3, 4],
            marks=pytest.mark.smoke,
            id="fp16-last-row",
        ),
        pytest.param(
            3,
            64,
            16,
            16,
            2,
            BF16,
            FP8,
            [1, -1, 31],
            marks=pytest.mark.smoke,
            id="fp8-bf16-scaled",
        ),
        pytest.param(
            2,
            64,
            16,
            16,
            4,
            FP16,
            FP8,
            [63, 0],
            marks=pytest.mark.smoke,
            id="fp8-fp16-page-boundary",
        ),
        # 20 + 6 columns are off the 8-element vector, so the columns move narrower.
        pytest.param(
            5,
            20,
            6,
            7,
            3,
            BF16,
            BF16,
            [3, 19, -1, 0, 8],
            marks=pytest.mark.full,
            id="bf16-columns-off-the-vector",
        ),
        # No latent columns: the latent copy writes nothing and is never built.
        pytest.param(
            3,
            0,
            16,
            7,
            3,
            FP16,
            FP8,
            [1, -1, 20],
            marks=pytest.mark.full,
            id="no-latent-columns",
        ),
        # An odd token count leaves the last block with one token.
        pytest.param(
            4095,
            512,
            64,
            64,
            128,
            BF16,
            BF16,
            None,
            marks=pytest.mark.full,
            id="bf16-prefill-odd-tail",
        ),
    ],
)
def test_mla_kv_cache_write_copies_the_named_rows(
    tokens, latent_dim, rope_dim, page_size, num_pages, dtype, cache_dtype, slots
):
    """Every row a slot names holds kv_c then k_pe; every other row is untouched."""
    workload = MLAKVCacheWriteTest(
        tokens, latent_dim, rope_dim, page_size, num_pages, dtype, cache_dtype, slots=slots
    )
    _check(workload, MLAKVCacheWriteFwdOp())


@pytest.mark.parametrize(
    "tokens,latent_dim,rope_dim,layout,dtype,cache_dtype,cos_sin_dtype,slots,tune",
    [
        pytest.param(
            5,
            64,
            16,
            "interleaved",
            BF16,
            BF16,
            BF16,
            [20, -1, 0, 6, 13],
            False,
            marks=pytest.mark.smoke,
            id="interleaved-bf16",
        ),
        pytest.param(
            4,
            64,
            16,
            "neox",
            FP16,
            FP16,
            FP32,
            [7, 0, 3, 4],
            False,
            marks=pytest.mark.smoke,
            id="neox-fp16-float32-table",
        ),
        pytest.param(
            3,
            64,
            16,
            "interleaved",
            BF16,
            FP8,
            FP32,
            [1, -1, 20],
            False,
            marks=pytest.mark.smoke,
            id="interleaved-fp8",
        ),
        pytest.param(
            3,
            64,
            16,
            "neox",
            FP16,
            FP8,
            FP16,
            [9, 2, -1],
            False,
            marks=pytest.mark.smoke,
            id="neox-fp8",
        ),
        # Half of a 6-wide k_pe is odd: a neox thread holds one element of each half.
        pytest.param(
            5,
            20,
            6,
            "neox",
            BF16,
            BF16,
            BF16,
            [3, 19, -1, 0, 8],
            False,
            marks=pytest.mark.full,
            id="neox-one-element-spans",
        ),
        # Half of a 12-wide k_pe is 6: a neox thread holds two elements of each half.
        pytest.param(
            4,
            64,
            12,
            "neox",
            FP16,
            FP16,
            FP16,
            [0, 11, 5, 20],
            False,
            marks=pytest.mark.full,
            id="neox-two-element-spans",
        ),
        # No rope columns: the rotation writes nothing and is never built.
        pytest.param(
            3,
            64,
            0,
            "neox",
            BF16,
            BF16,
            BF16,
            [1, -1, 20],
            False,
            marks=pytest.mark.full,
            id="no-rope-columns",
        ),
        # A 6-wide interleaved k_pe holds two-element spans; 17 latent columns put them off
        # the vector boundary, so they move one element at a time.
        pytest.param(
            4,
            17,
            6,
            "interleaved",
            FP16,
            FP16,
            FP16,
            [0, 11, 5, 20],
            False,
            marks=pytest.mark.full,
            id="interleaved-unaligned-spans",
        ),
        # 512 rotated columns need more rope threads than a block holds, so each takes
        # several rounds.
        pytest.param(
            3,
            64,
            512,
            "interleaved",
            BF16,
            BF16,
            FP32,
            [1, 0, 13],
            False,
            marks=pytest.mark.full,
            id="interleaved-several-rounds",
        ),
        pytest.param(
            3,
            64,
            1024,
            "neox",
            FP16,
            FP16,
            FP32,
            [2, 0, 17],
            False,
            marks=pytest.mark.full,
            id="neox-several-rounds",
        ),
        pytest.param(
            6,
            64,
            16,
            "interleaved",
            BF16,
            FP8,
            FP32,
            [4, 20, 1, -1, 13, 6],
            True,
            marks=pytest.mark.full,
            id="interleaved-fp8-tuned",
        ),
    ],
)
def test_mla_kv_cache_write_rotates_k_pe(
    tokens, latent_dim, rope_dim, layout, dtype, cache_dtype, cos_sin_dtype, slots, tune
):
    """The k_pe half of every named row holds k_pe rotated at the token's position."""
    workload = MLAKVCacheWriteTest(
        tokens,
        latent_dim,
        rope_dim,
        7,
        3,
        dtype,
        cache_dtype,
        True,
        layout,
        cos_sin_dtype,
        max_position=37,
        slots=slots,
    )
    op = MLAKVCacheWriteFwdOp(fuse_rope=True, rope_layout=layout)
    if tune:
        op.request_tune()
    _check(workload, op)


@pytest.mark.parametrize(
    "fuse_rope",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.smoke)],
)
def test_mla_kv_cache_write_saturates_fp8_overflow(fuse_rope):
    """FP8 preserves NaN, saturates overflow/infinities, and rounds ties to even.

    With RoPE, position 0 rotates by angle 0, so the rotated columns carry the same values
    through the rotation's own store.
    """
    workload = MLAKVCacheWriteTest(
        1,
        16,
        16,
        2,
        1,
        FP16,
        FP8,
        fuse_rope,
        "interleaved",
        FP32,
        max_position=1,
        slots=[1],
        positions=[0],
    )
    inputs = workload.gen_inputs()
    kv_c, k_pe, _, _, scale, _, _ = inputs
    values = torch.tensor(
        [
            1000,
            -1000,
            448,
            -448,
            1.2,
            -1.2,
            1.0625,
            1.1875,
            float("nan"),
            float("inf"),
            -float("inf"),
            0,
            -0.0,
            1,
            -1,
            0.5,
        ],
        dtype=kv_c.dtype,
        device=kv_c.device,
    )
    kv_c.copy_(values.reshape_as(kv_c))
    k_pe.copy_(-values.reshape_as(k_pe))
    scale.fill_(1)
    op = MLAKVCacheWriteFwdOp(fuse_rope=fuse_rope, rope_layout="interleaved")
    TestBase.check(workload, op, *inputs, runs=lambda *args: mla_cache_write_result(op, *args))


@pytest.mark.smoke
def test_mla_kv_cache_write_divides_by_the_scale():
    """FP8 stores value / scale, on both the copied and the rotated columns.

    With a scale of 0.9 these bfloat16 values land on a different E4M3 code when multiplied
    by the float32 reciprocal of the scale. Position 0 rotates by angle 0, so the
    rotated columns carry the same values.
    """
    workload = MLAKVCacheWriteTest(
        1,
        16,
        16,
        2,
        1,
        BF16,
        FP8,
        True,
        "interleaved",
        FP32,
        max_position=1,
        slots=[1],
        positions=[0],
    )
    inputs = workload.gen_inputs()
    kv_c, k_pe, _, _, scale, _, _ = inputs
    values = torch.tensor(
        [0.02197265625, 0.0439453125, 0.087890625, 0.17578125, 0.3515625, 0.703125, 1.0, 2.0],
        dtype=kv_c.dtype,
        device=kv_c.device,
    )
    kv_c.copy_(torch.cat([values, -values]).reshape_as(kv_c))
    k_pe.copy_(torch.cat([values, -values]).reshape_as(k_pe))
    scale.fill_(0.9)
    op = MLAKVCacheWriteFwdOp(fuse_rope=True, rope_layout="interleaved")
    TestBase.check(workload, op, *inputs, runs=lambda *args: mla_cache_write_result(op, *args))


@pytest.mark.smoke
def test_mla_kv_cache_write_skips_every_token():
    """All-negative slots leave a sentinel-filled cache unchanged."""
    workload = MLAKVCacheWriteTest(3, 64, 16, 7, 4, BF16, slots=[-1, -1, -1])
    kv_c, k_pe, kv_cache, slots, scale, positions, cos_sin = workload.gen_inputs()
    kv_cache.fill_(3.0)
    before = kv_cache.clone()
    op = MLAKVCacheWriteFwdOp()
    assert op(kv_c, k_pe, kv_cache, slots, scale, positions, cos_sin) is None
    assert torch.equal(kv_cache, before)


@pytest.mark.smoke
def test_mla_kv_cache_write_of_no_tokens_writes_nothing():
    """A call with no token writes nothing; the cache it is handed is non-empty."""
    workload = MLAKVCacheWriteTest(0, 64, 16, 16, 2, BF16)
    kv_c, k_pe, kv_cache, slots, scale, positions, cos_sin = workload.gen_inputs()
    before = kv_cache.clone()
    op = MLAKVCacheWriteFwdOp()
    assert op(kv_c, k_pe, kv_cache, slots, scale, positions, cos_sin) is None
    assert torch.equal(kv_cache, before)


@pytest.mark.smoke
def test_mla_kv_cache_write_leaves_k_pe_unrotated():
    """Rotating into the cache leaves the caller's k_pe as it was."""
    workload = MLAKVCacheWriteTest(4, 64, 16, 7, 3, BF16, BF16, True, "neox", BF16, max_position=37)
    inputs = workload.gen_inputs()
    k_pe = inputs[1].clone()
    MLAKVCacheWriteFwdOp(fuse_rope=True)(*inputs)
    assert torch.equal(inputs[1], k_pe)


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("scale-missing", marks=pytest.mark.smoke),
        pytest.param("scale-present", marks=pytest.mark.smoke),
        pytest.param("rope-tensors-missing", marks=pytest.mark.smoke),
        pytest.param("input-dtype", marks=pytest.mark.full),
        pytest.param("cache-dtype", marks=pytest.mark.full),
    ],
)
def test_mla_kv_cache_write_refuses_an_invalid_tensor_contract(invalid):
    """Unsupported dtypes, a scale that does not match the cache, and missing rope inputs."""
    cache_dtype = FP8 if invalid == "scale-missing" else BF16
    workload = MLAKVCacheWriteTest(2, 64, 16, 16, 2, BF16, cache_dtype)
    inputs = list(workload.gen_inputs())
    op = MLAKVCacheWriteFwdOp()
    if invalid == "scale-missing":
        inputs[4] = None
    elif invalid == "scale-present":
        inputs[4] = torch.ones(1, dtype=torch.float32, device=inputs[0].device)
    elif invalid == "rope-tensors-missing":
        op = MLAKVCacheWriteFwdOp(fuse_rope=True)
    elif invalid == "input-dtype":
        inputs[0], inputs[1] = inputs[0].float(), inputs[1].float()
    else:
        inputs[2] = inputs[2].half()
    with pytest.raises(ValueError, match="dtype" if invalid.endswith("dtype") else None):
        op(*inputs)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_mla_kv_cache_write_compiles_cold_to_its_operator():
    """A cold fullgraph compile owns the graph and writes the same rows as eager."""
    workload = MLAKVCacheWriteTest(
        5,
        64,
        16,
        7,
        3,
        BF16,
        FP8,
        True,
        "interleaved",
        FP32,
        max_position=37,
        slots=[20, -1, 0, 6, 13],
    )
    op = MLAKVCacheWriteFwdOp(fuse_rope=True, rope_layout="interleaved")
    assert_op_owns_graph_nodes(op, *workload.gen_inputs())
    compiled_inputs = workload.gen_inputs()
    torch.compile(op, fullgraph=True)(*compiled_inputs)
    eager_inputs = workload.gen_inputs()
    MLAKVCacheWriteFwdOp(fuse_rope=True, rope_layout="interleaved")(*eager_inputs)
    assert torch.equal(compiled_inputs[2].view(torch.uint8), eager_inputs[2].view(torch.uint8))


register_compile_contract(MLAKVCacheWriteFwdOp)
