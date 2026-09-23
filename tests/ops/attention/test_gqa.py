from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.test_base import FixtureBase, TestBase
from tileops.backend import BUILTIN
from tileops.kernels.attention import (
    GQADecodeBs1Kernel,
    GQADecodeKernel,
    GQADecodeLongContextKernel,
    GQADenseFP8DecodeKernel,
    GQADenseFP8Kernel,
    GQADenseSlidingWindowKernel,
    GQADenseWsKernel,
)
from tileops.kernels.attention.gqa_decode import (
    _effective_dense_num_split,
    _gqa_decode_no_split_run,
    _gqa_decode_split_run,
)
from tileops.kernels.kernel_base import Kernel
from tileops.ops import (
    GroupedQueryAttentionBwdOp,
    GroupedQueryAttentionDenseFwdOp,
)
from tileops.utils import get_sm_version
from workloads.attention.gqa import (
    GroupedQueryAttentionBwdWorkload,
    apply_dense_rope,
    dense_gqa_ref,
)


class GroupedQueryAttentionBwdTest(GroupedQueryAttentionBwdWorkload, TestBase):
    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        grad_output: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal, enable_gqa=True
            )
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(grad_output)
        return q.grad, k.grad, v.grad


@pytest.mark.parametrize(
    "is_causal, rope_layout, rotary_dim, dtype",
    [
        (True, None, None, torch.float16),
        (True, "neox", 64, torch.float16),
        (True, "interleaved", 64, torch.float16),
        (True, "neox", None, torch.bfloat16),
        (False, None, None, torch.float16),
        (False, "neox", 64, torch.float16),
    ],
)
@pytest.mark.smoke
def test_gqa_dense_sm90_main_kernel_matches_reference(
    is_causal: bool,
    rope_layout: Optional[str],
    rotary_dim: Optional[int],
    dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available() or get_sm_version() != 90:
        pytest.skip("Dense warp-specialized prefill requires SM90")
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 160, 270, 8, 2, 128
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    if rope_layout is None:
        op = GroupedQueryAttentionDenseFwdOp(is_causal=is_causal, target=BUILTIN)
        output = op(q, k, v)
        q_ref, k_ref = q, k
    else:
        resolved_rotary_dim = dim if rotary_dim is None else rotary_dim
        angles = torch.randn(seq_len_kv, resolved_rotary_dim // 2, device="cuda") * 0.1
        rope_cos, rope_sin = angles.cos().to(dtype), angles.sin().to(dtype)
        op = GroupedQueryAttentionDenseFwdOp(
            is_causal=is_causal,
            pos_encoding_mode="rope",
            rotary_dim=rotary_dim,
            rope_layout=rope_layout,
            target=BUILTIN,
        )
        output = op(q, k, v, rope_cos=rope_cos, rope_sin=rope_sin)
        q_positions = torch.arange(seq_len_kv - seq_len_q, seq_len_kv, device="cuda")
        k_positions = torch.arange(seq_len_kv, device="cuda")
        q_ref = apply_dense_rope(
            q,
            q_positions,
            rope_cos,
            rope_sin,
            rotary_dim=resolved_rotary_dim,
            layout=rope_layout,
        )
        k_ref = apply_dense_rope(
            k,
            k_positions,
            rope_cos,
            rope_sin,
            rotary_dim=resolved_rotary_dim,
            layout=rope_layout,
        )

    torch.testing.assert_close(
        output,
        dense_gqa_ref(
            q_ref,
            k_ref,
            v,
            heads=heads,
            heads_kv=heads_kv,
            is_causal=is_causal,
        ),
        atol=2e-2 if dtype == torch.bfloat16 else 5e-3,
        rtol=1e-5,
    )
    assert isinstance(next(iter(op.iter_kernels())), GQADenseWsKernel)


@pytest.mark.smoke
@pytest.mark.parametrize(
    (
        "seq_len_q",
        "seq_len_kv",
        "sm_scale",
        "softcap",
        "rope_layout",
        "rotary_dim",
        "out_dtype",
    ),
    [
        (1, 2049, 0.125, 50.0, None, None, torch.bfloat16),
        (256, 1792, 0.125, 0.0, None, None, torch.float16),
        (255, 1793, 0.0625, 50.0, None, None, torch.float16),
        (256, 1792, 0.125, 0.0, "neox", 64, torch.float16),
        (255, 1793, 0.0625, 50.0, "interleaved", 128, torch.bfloat16),
    ],
)
def test_gqa_dense_fp8_causal_rectangular_matches_reference(
    seq_len_q: int,
    seq_len_kv: int,
    sm_scale: float,
    softcap: float,
    rope_layout: Optional[str],
    rotary_dim: Optional[int],
    out_dtype: torch.dtype,
) -> None:
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp8 is None or not torch.cuda.is_available() or get_sm_version() != 90:
        pytest.skip("native FP8 Dense GQA requires SM90 and float8_e4m3fn")
    batch = 1
    heads, heads_kv, dim = 8, 2, 128
    q = (torch.randn(batch, seq_len_q, heads, dim, device="cuda") * 0.2).to(fp8)
    k = (torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda") * 0.2).to(fp8)
    v = (torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda") * 0.2).to(fp8)
    scale = torch.ones((batch, heads_kv), device="cuda", dtype=torch.float32)
    rope_cos = rope_sin = None
    if rope_layout is not None:
        assert rotary_dim is not None
        angles = torch.randn(seq_len_kv, rotary_dim // 2, device="cuda") * 0.1
        rope_cos, rope_sin = angles.cos().to(out_dtype), angles.sin().to(out_dtype)

    op = GroupedQueryAttentionDenseFwdOp(
        is_causal=True,
        out_dtype=out_dtype,
        sm_scale=sm_scale,
        softcap=softcap,
        pos_encoding_mode="rope" if rope_layout is not None else "none",
        rotary_dim=rotary_dim,
        rope_layout="neox" if rope_layout is None else rope_layout,
        target=BUILTIN,
    )
    output = op(
        q,
        k,
        v,
        scale,
        scale,
        scale,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )
    q_ref = q.to(out_dtype)
    k_ref = k.to(out_dtype)
    if rope_layout is not None:
        assert rope_cos is not None and rope_sin is not None and rotary_dim is not None
        q_positions = torch.arange(seq_len_kv - seq_len_q, seq_len_kv, device="cuda")
        k_positions = torch.arange(seq_len_kv, device="cuda")
        q_ref = apply_dense_rope(
            q_ref,
            q_positions,
            rope_cos,
            rope_sin,
            rotary_dim=rotary_dim,
            layout=rope_layout,
        )
        k_ref = apply_dense_rope(
            k_ref,
            k_positions,
            rope_cos,
            rope_sin,
            rotary_dim=rotary_dim,
            layout=rope_layout,
        )
    reference = dense_gqa_ref(
        q_ref,
        k_ref,
        v.to(out_dtype),
        heads=heads,
        heads_kv=heads_kv,
        is_causal=True,
        sm_scale=sm_scale,
        softcap=softcap,
    )
    torch.testing.assert_close(output, reference, atol=8e-2, rtol=2e-2)
    expected_kernel = GQADenseFP8DecodeKernel if seq_len_q == 1 else GQADenseFP8Kernel
    assert isinstance(next(iter(op.iter_kernels())), expected_kernel)


@pytest.mark.smoke
@pytest.mark.parametrize("batch", [1, 2])
def test_gqa_dense_reuses_one_kernel_across_sequence_lengths(batch: int) -> None:
    if not torch.cuda.is_available() or get_sm_version() != 90:
        pytest.skip("Dense warp-specialized prefill requires SM90")
    heads, heads_kv, dim = 8, 2, 128
    op = GroupedQueryAttentionDenseFwdOp(target=BUILTIN)

    for seq_len_q, seq_len_kv in (
        (2, 270),
        (2, 271),
        (160, 270),
        (896, 896),
        (2, 270),
    ):
        q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        output = op(q, k, v)
        torch.testing.assert_close(
            output,
            dense_gqa_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True),
            atol=5e-3,
            rtol=1e-5,
        )

    assert len(list(op.iter_kernels())) == 1


@pytest.mark.parametrize(
    "batch, head_shape, dtype, seq_lens_kv, rope_layout, rotary_dim, kernel_type",
    [
        pytest.param(
            1,
            (8, 2),
            torch.float16,
            (257, 1057),
            None,
            None,
            GQADecodeBs1Kernel,
            id="bs1-fp16-other-head-ratio",
        ),
        pytest.param(
            1,
            (32, 4),
            torch.float16,
            (1024, 1057),
            None,
            None,
            GQADecodeLongContextKernel,
            id="measured-bs1-long-generic",
        ),
        pytest.param(
            2,
            (8, 2),
            torch.bfloat16,
            (257, 511),
            None,
            None,
            GQADecodeKernel,
            id="batched-bf16",
        ),
        pytest.param(
            32,
            (32, 8),
            torch.float16,
            (4096, 4097),
            None,
            None,
            GQADecodeKernel,
            id="batched-fp16-unsplit-tail",
        ),
        pytest.param(
            16,
            (64, 8),
            torch.bfloat16,
            (4096, 4161),
            None,
            None,
            GQADecodeKernel,
            id="batched-bf16-split-tail",
        ),
        pytest.param(
            1,
            (8, 2),
            torch.float16,
            (271,),
            "neox",
            64,
            GQADecodeBs1Kernel,
            id="bs1-fp16-neox-partial",
        ),
        pytest.param(
            1,
            (8, 2),
            torch.float16,
            (1057,),
            "neox",
            128,
            GQADecodeBs1Kernel,
            id="bs1-fp16-neox-full-ctx-tail",
        ),
        pytest.param(
            2,
            (8, 2),
            torch.bfloat16,
            (257,),
            "interleaved",
            128,
            GQADecodeKernel,
            id="batched-bf16-interleaved-full",
        ),
    ],
)
@pytest.mark.smoke
def test_gqa_dense_decode_dispatch_and_dynamic_sequence_lengths(
    batch: int,
    head_shape: tuple[int, int],
    dtype: torch.dtype,
    seq_lens_kv: tuple[int, ...],
    rope_layout: Optional[str],
    rotary_dim: Optional[int],
    kernel_type: type[Kernel],
) -> None:
    if not torch.cuda.is_available() or get_sm_version() != 90:
        pytest.skip("Dense decode requires SM90")
    heads, heads_kv = head_shape
    dim = 128
    op = GroupedQueryAttentionDenseFwdOp(
        pos_encoding_mode="rope" if rope_layout is not None else "none",
        rotary_dim=rotary_dim,
        rope_layout="neox" if rope_layout is None else rope_layout,
        target=BUILTIN,
    )

    for seq_len_kv in seq_lens_kv:
        q = torch.randn(batch, 1, heads, dim, device="cuda", dtype=dtype)
        k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        if rope_layout is None:
            rope_cos = rope_sin = None
            q_ref, k_ref = q, k
        else:
            assert rotary_dim is not None
            angles = torch.randn(seq_len_kv, rotary_dim // 2, device="cuda") * 0.1
            rope_cos, rope_sin = angles.cos().to(dtype), angles.sin().to(dtype)
            q_ref = apply_dense_rope(
                q,
                torch.tensor([seq_len_kv - 1], device="cuda"),
                rope_cos,
                rope_sin,
                rotary_dim=rotary_dim,
                layout=rope_layout,
            )
            k_ref = apply_dense_rope(
                k,
                torch.arange(seq_len_kv, device="cuda"),
                rope_cos,
                rope_sin,
                rotary_dim=rotary_dim,
                layout=rope_layout,
            )
        output = op(q, k, v, rope_cos=rope_cos, rope_sin=rope_sin)
        torch.testing.assert_close(
            output,
            dense_gqa_ref(q_ref, k_ref, v, heads=heads, heads_kv=heads_kv, is_causal=True),
            atol=1.6e-2 if dtype == torch.bfloat16 else 5e-3,
            rtol=1.6e-2 if dtype == torch.bfloat16 else 1e-5,
        )

    kernels = list(op.iter_kernels())
    assert len(kernels) == 1
    assert isinstance(kernels[0], kernel_type)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "num_split, block_N, real_seqlen_kv, expected",
    [
        pytest.param(32, 64, 1024, 16, id="issue-shape-clamps-to-tiles"),
        pytest.param(32, 64, 1000, 8, id="non-candidate-count-rounded-down"),
        pytest.param(7, 64, 100000, 4, id="non-candidate-ceiling-rounded-down"),
        pytest.param(16, 128, 1024, 8, id="block-128-clamps"),
        pytest.param(32, 64, 100, 1, id="short-sequence-no-split"),
        pytest.param(1, 64, 100000, 1, id="tuned-no-split-stays-no-split"),
    ],
)
def test_gqa_dense_decode_effective_num_split(
    num_split: int, block_N: int, real_seqlen_kv: int, expected: int
) -> None:
    """The tuned num_split is a ceiling shrunk to the runtime KV extent."""
    assert _effective_dense_num_split(num_split, block_N, real_seqlen_kv) == expected


@pytest.mark.smoke
def test_gqa_dense_long_context_reuses_configuration_tiers() -> None:
    """A reused op crosses the tile-size boundary in both directions, including KV tails."""
    if not torch.cuda.is_available() or get_sm_version() != 90:
        pytest.skip("Long-context decode defaults are measured on SM90")
    op = GroupedQueryAttentionDenseFwdOp(target=BUILTIN)
    q = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.float16)
    for seq_len in (131072, 131073, 262145, 131071):
        k = torch.randn(1, seq_len, 4, 128, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        out = op(q, k, v)
        # Grouped FP32 matmuls avoid materializing eight copies of the long KV.
        q_grouped = q[0, 0].reshape(4, 8, 128).float()
        scores = q_grouped @ k[0].permute(1, 2, 0).float() * (128**-0.5)
        ref = (scores.softmax(-1) @ v[0].transpose(0, 1).float()).reshape_as(out)
        torch.testing.assert_close(out, ref.to(out.dtype), atol=1e-3, rtol=1e-3)
    kernels = list(op.iter_kernels())
    assert len(kernels) == 2
    assert {kernel.config["block_N"] for kernel in kernels} == {64, 128}


@pytest.mark.smoke
@pytest.mark.parametrize("seqlen_kv", [1, 63, 128, 1024])
def test_gqa_decode_autotune_configs_keep_full_tiles_per_split(seqlen_kv: int) -> None:
    """Every swept config leaves each split one full KV tile; num_split=1 stays comparable."""
    if not torch.cuda.is_available() or get_sm_version() not in (80, 89, 90):
        pytest.skip("GQA decode requires SM80/89/90")
    kernel = GQADecodeKernel(2, 8, 2, seqlen_kv, 128, dtype=torch.float16)
    configs = kernel.autotune_configs
    assert configs, "the sweep must stay non-empty for any positive sequence length"
    for config in configs:
        assert config["num_split"] <= max(1, seqlen_kv // config["block_N"])
    assert any(config["num_split"] == 1 for config in configs)


@pytest.mark.smoke
def test_gqa_decode_tuned_split_count_tracks_runtime_sequence(monkeypatch) -> None:
    """A tuned num_split the sequence cannot fill shrinks instead of pushing
    dispatch into the never-tuned no-split kernel (the reported issue)."""
    if not torch.cuda.is_available() or get_sm_version() not in (80, 89, 90):
        pytest.skip("GQA decode requires SM80/89/90")
    batch, heads, heads_kv, dim = 2, 32, 4, 128
    kernel = GQADecodeKernel(
        batch,
        heads,
        heads_kv,
        1024,
        dim,
        dtype=torch.float16,
        config={"block_H": 64, "block_N": 64, "num_split": 32, "num_stages": 2, "threads": 128},
    )

    calls: list[tuple[str, int]] = []

    def split_spy(*args, **kwargs):
        # num_split is the 12th positional argument of _gqa_decode_split_run
        calls.append(("split", args[11]))
        return _gqa_decode_split_run(*args, **kwargs)

    def no_split_spy(*args, **kwargs):
        calls.append(("no_split", 0))
        return _gqa_decode_no_split_run(*args, **kwargs)

    monkeypatch.setattr("tileops.kernels.attention.gqa_decode._gqa_decode_split_run", split_spy)
    monkeypatch.setattr(
        "tileops.kernels.attention.gqa_decode._gqa_decode_no_split_run", no_split_spy
    )

    # 1024 tokens fill 16 of the tuned 32 splits; 100 cannot fill two
    for seq_len_kv, expected in ((1024, ("split", 16)), (100, ("no_split", 0))):
        q = torch.randn(batch, 1, heads, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        output = kernel(q, k, v)
        torch.testing.assert_close(
            output,
            dense_gqa_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True),
            atol=5e-3,
            rtol=1e-5,
        )
        assert calls[-1] == expected


@pytest.mark.parametrize(
    "is_causal,use_rope",
    [
        pytest.param(True, False, id="causal"),
        pytest.param(True, True, id="causal-rope"),
        pytest.param(False, False, id="noncausal"),
        pytest.param(False, True, id="noncausal-rope"),
    ],
)
@pytest.mark.smoke
def test_gqa_dense_sm90_sliding_window_kernel_matches_reference(
    is_causal: bool, use_rope: bool
) -> None:
    if not torch.cuda.is_available() or get_sm_version() != 90:
        pytest.skip("Dense sliding-window prefill requires SM90")
    batch, seq_len, heads, heads_kv, dim = 1, 270, 8, 2, 128
    window_size_left = 64
    window_size_right = 0 if is_causal else 32
    sm_scale = 0.125
    softcap = 2.0
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)

    rotary_dim = 64
    if use_rope:
        angles = torch.randn(seq_len, rotary_dim // 2, device="cuda") * 0.1
        rope_cos, rope_sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
    else:
        rope_cos = rope_sin = None
    op = GroupedQueryAttentionDenseFwdOp(
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        sm_scale=sm_scale,
        softcap=softcap,
        pos_encoding_mode="rope" if use_rope else "none",
        rotary_dim=rotary_dim if use_rope else None,
        target=BUILTIN,
    )
    output = op(
        q,
        k,
        v,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
    )
    if use_rope:
        assert rope_cos is not None and rope_sin is not None
        positions = torch.arange(seq_len, device="cuda")
        q = apply_dense_rope(q, positions, rope_cos, rope_sin, rotary_dim=rotary_dim, layout="neox")
        k = apply_dense_rope(k, positions, rope_cos, rope_sin, rotary_dim=rotary_dim, layout="neox")

    torch.testing.assert_close(
        output,
        dense_gqa_ref(
            q,
            k,
            v,
            heads=heads,
            heads_kv=heads_kv,
            is_causal=is_causal,
            sm_scale=sm_scale,
            softcap=softcap,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        ),
        atol=5e-3,
        rtol=1e-5,
    )
    assert isinstance(next(iter(op.iter_kernels())), GQADenseSlidingWindowKernel)


class GroupedQueryAttentionBwdFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq_len, heads, heads_kv, dim, causal, dtype, tune",
            [
                pytest.param(
                    1, 1024, 8, 4, 64, False, torch.float16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    1, 1024, 8, 4, 64, False, torch.bfloat16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    4, 2048, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.full
                ),
                pytest.param(
                    4, 2048, 64, 4, 128, False, torch.bfloat16, False, marks=pytest.mark.full
                ),
            ],
        ),
    ]


@GroupedQueryAttentionBwdFixture
def test_gqa_bwd(
    batch: int,
    seq_len: int,
    heads: int,
    heads_kv: int,
    dim: int,
    causal: bool,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = GroupedQueryAttentionBwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GroupedQueryAttentionBwdOp(batch, heads, heads_kv, seq_len, dim, causal, tune=tune)
    test.check(op, *test.gen_inputs(), atol=5e-3, rtol=1e-5)
