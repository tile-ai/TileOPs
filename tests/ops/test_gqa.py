from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.backend import BUILTIN
from tileops.kernels.attention import (
    GQADecodeBs1Kernel,
    GQADecodeKernel,
    GQADecodeLongContextKernel,
)
from tileops.kernels.kernel_base import Kernel
from tileops.ops import (
    GQABwdOp,
    GQADenseFwdOp,
    GQAVarlenFwdOp,
)
from workloads.attention.gqa.bwd import GQABwdWorkload
from workloads.attention.gqa.dense import (
    GQADensePrefillWorkload,
    dense_gqa_ref,
    dense_gqa_verification,
)
from workloads.attention.gqa.rope import apply_dense_rope
from workloads.attention.gqa.varlen import GQAVarlenScaledWorkload
from workloads.device import run_device
from workloads.numerics import compare_outputs


class GQABwdTest(GQABwdWorkload, TestBase):
    pass


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "_architecture",
    [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)],
)
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
def test_gqa_dense_prefill_matches_reference(
    _architecture: int,
    is_causal: bool,
    rope_layout: Optional[str],
    rotary_dim: Optional[int],
    dtype: torch.dtype,
) -> None:
    batch, seq_len_q, seq_len_kv, heads, heads_kv, dim = 1, 160, 270, 8, 2, 128
    q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    if rope_layout is None:
        op = GQADenseFwdOp(is_causal=is_causal, target=BUILTIN)
        output = op(q, k, v)
        q_ref, k_ref = q, k
    else:
        resolved_rotary_dim = dim if rotary_dim is None else rotary_dim
        angles = torch.randn(seq_len_kv, resolved_rotary_dim // 2, device="cuda") * 0.1
        rope_cos, rope_sin = angles.cos().to(dtype), angles.sin().to(dtype)
        op = GQADenseFwdOp(
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

    compare_outputs(
        output,
        dense_gqa_ref(q_ref, k_ref, v, heads=heads, heads_kv=heads_kv, is_causal=is_causal),
        dense_gqa_verification(q.dtype),
    )


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    (
        "_architecture",
        "seq_len_q",
        "seq_len_kv",
        "sm_scale",
        "softcap",
        "rope_layout",
        "rotary_dim",
        "out_dtype",
    ),
    [
        pytest.param(arch, *case, marks=pytest.mark.sm89 if arch == 89 else pytest.mark.sm90)
        for case in [
            (1, 2049, 0.125, 50.0, None, None, torch.bfloat16),
            (256, 1792, 0.125, 0.0, None, None, torch.float16),
            (255, 1793, 0.0625, 50.0, None, None, torch.float16),
            (256, 1792, 0.125, 0.0, "neox", 64, torch.float16),
            (255, 1793, 0.0625, 50.0, "interleaved", 128, torch.bfloat16),
        ]
        for arch in (89, 90)
        if arch == 90 or case[0] != 1
    ],
)
def test_gqa_dense_fp8_causal_rectangular_matches_reference(
    _architecture: int,
    seq_len_q: int,
    seq_len_kv: int,
    sm_scale: float,
    softcap: float,
    rope_layout: Optional[str],
    rotary_dim: Optional[int],
    out_dtype: torch.dtype,
) -> None:
    """Two batches with their own per-KV-head scales: each row reads its batch's scales."""
    workload = GQADensePrefillWorkload(
        2,
        seq_len_q,
        seq_len_kv,
        8,
        2,
        128,
        torch.float8_e4m3fn,
        out_dtype=out_dtype,
        sm_scale=sm_scale,
        softcap=softcap,
        rotary_dim=rotary_dim,
        rope_layout="neox" if rope_layout is None else rope_layout,
    )
    op = GQADenseFwdOp(
        is_causal=True,
        out_dtype=out_dtype,
        sm_scale=sm_scale,
        softcap=softcap,
        pos_encoding_mode="rope" if rope_layout is not None else "none",
        rotary_dim=rotary_dim,
        rope_layout="neox" if rope_layout is None else rope_layout,
        target=BUILTIN,
    )
    TestBase.check(workload, op, *workload.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize("batch", [1, 2])
def test_gqa_dense_reuses_one_kernel_across_sequence_lengths(batch: int) -> None:
    heads, heads_kv, dim = 8, 2, 128
    op = GQADenseFwdOp(target=BUILTIN)

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
        compare_outputs(
            output,
            dense_gqa_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True),
            dense_gqa_verification(q.dtype),
        )


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
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
        # The last key tile holds 15 rows; a per-row guard around the consumer's key
        # rotation leaves the barrier inside it short of threads, and the kernel hangs.
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
    heads, heads_kv = head_shape
    dim = 128
    op = GQADenseFwdOp(
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
        compare_outputs(
            output,
            dense_gqa_ref(q_ref, k_ref, v, heads=heads, heads_kv=heads_kv, is_causal=True),
            dense_gqa_verification(q.dtype),
        )


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
@pytest.mark.parametrize(
    "heads, heads_kv",
    [
        pytest.param(72, 1, id="group-over-a-block"),
        pytest.param(71, 1, id="group-over-a-block-odd"),
        pytest.param(142, 2, id="group-over-a-block-two-kv-heads"),
    ],
)
@pytest.mark.parametrize(
    "dim,dtype",
    [
        pytest.param(64, torch.bfloat16, marks=pytest.mark.sm90),
        pytest.param(256, torch.float16, marks=pytest.mark.sm89),
        pytest.param(256, torch.bfloat16, marks=pytest.mark.sm89),
    ],
)
def test_gqa_dense_decode_covers_a_group_wider_than_one_head_block(
    heads: int, heads_kv: int, *, dim: int, dtype: torch.dtype
) -> None:
    """Every output head is written when the group needs more than one head block of 64."""
    lengths = (1, 63, 2048) if dim == 256 else (2048,)
    for seq_len_kv in lengths:
        q = torch.randn(2, 1, heads, dim, device="cuda", dtype=dtype)
        k = torch.randn(2, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype)
        v = torch.randn_like(k)
        reference = dense_gqa_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True)
        compare_outputs(
            GQADenseFwdOp(target=BUILTIN)(q, k, v), reference, dense_gqa_verification(dtype)
        )
        if dim == 256:
            kernel = GQADecodeKernel(  # lint: test-placement allow: pins block_H to 64
                batch=2,
                heads=heads,
                heads_kv=heads_kv,
                seq_len_kv=seq_len_kv,
                dim=dim,
                dtype=dtype,
                config={
                    "block_H": 64,
                    "block_N": 64,
                    "num_split": 1,
                    "num_stages": 1,
                    "threads": 128,
                },
            )
            compare_outputs(kernel(q, k, v), reference, dense_gqa_verification(dtype))


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gqa_dense_decode_combines_splits_at_large_scores() -> None:
    """Queries and keys scaled by 4 put each split's base-2 log-sum-exp between about 40
    and 90, where bfloat16 steps by 0.25 or 0.5: the combine must read it in float32."""
    torch.manual_seed(0)
    batch, heads, heads_kv, dim, seq_len_kv = 2, 8, 2, 128, 2048
    q = torch.randn(batch, 1, heads, dim, device=run_device(), dtype=torch.bfloat16) * 4
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device=run_device(), dtype=torch.bfloat16) * 4
    v = torch.randn_like(k)
    compare_outputs(
        GQADenseFwdOp(target=BUILTIN)(q, k, v),
        dense_gqa_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True),
        dense_gqa_verification(q.dtype),
    )


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gqa_dense_decode_tunes_the_fused_rope_programs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tuned fused-RoPE decode times the programs forward launches, which read the RoPE
    tables, rather than the plain split program. Each program's last candidate stands in
    for its tuning result, and the program timed last wins."""
    timed_parameters = []

    def record(self, jit_kernel, configs, **kwargs):
        prim_func = jit_kernel.get_tir(**configs[-1])
        timed_parameters.append({buffer.name for buffer in prim_func.buffer_map.values()})
        return SimpleNamespace(config=configs[-1], latency=-float(len(timed_parameters)))

    monkeypatch.setattr(Kernel, "tune_jit_kernel", record)
    torch.manual_seed(0)
    batch, heads, heads_kv, dim, seq_len_kv, dtype = 2, 8, 2, 128, 257, torch.bfloat16
    op = GQADenseFwdOp(pos_encoding_mode="rope", rotary_dim=dim, target=BUILTIN)
    op.request_tune()
    q = torch.randn(batch, 1, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    angles = torch.randn(seq_len_kv, dim // 2, device="cuda") * 0.1
    rope_cos, rope_sin = angles.cos().to(dtype), angles.sin().to(dtype)
    output = op(q, k, v, rope_cos=rope_cos, rope_sin=rope_sin)

    assert timed_parameters
    assert all("rope_cos" in parameters for parameters in timed_parameters)
    q_ref, k_ref = (
        apply_dense_rope(x, positions, rope_cos, rope_sin, rotary_dim=dim, layout="neox")
        for x, positions in (
            (q, torch.tensor([seq_len_kv - 1], device="cuda")),
            (k, torch.arange(seq_len_kv, device="cuda")),
        )
    )
    compare_outputs(
        output,
        dense_gqa_ref(q_ref, k_ref, v, heads=heads, heads_kv=heads_kv, is_causal=True),
        dense_gqa_verification(dtype),
    )


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gqa_dense_long_context_reuses_configuration_tiers() -> None:
    """A reused op crosses the tile-size boundary in both directions, including KV tails."""
    op = GQADenseFwdOp(target=BUILTIN)
    q = torch.randn(1, 1, 32, 128, device="cuda", dtype=torch.float16)
    for seq_len in (131072, 131073, 262145, 131071):
        k = torch.randn(1, seq_len, 4, 128, device="cuda", dtype=q.dtype)
        v = torch.randn_like(k)
        out = op(q, k, v)
        # Grouped FP32 matmuls avoid materializing eight copies of the long KV.
        q_grouped = q[0, 0].reshape(4, 8, 128).float()
        scores = q_grouped @ k[0].permute(1, 2, 0).float() * (128**-0.5)
        ref = (scores.softmax(-1) @ v[0].transpose(0, 1).float()).reshape_as(out)
        compare_outputs(out, ref.to(out.dtype), dense_gqa_verification(q.dtype))


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "_architecture",
    [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)],
)
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
def test_gqa_dense_sliding_window_matches_reference(
    _architecture: int, is_causal: bool, use_rope: bool
) -> None:
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
    op = GQADenseFwdOp(
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

    compare_outputs(
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
        dense_gqa_verification(q.dtype),
    )


class GQABwdFixture(FixtureBase):
    # heads_kv == heads cases reach the pipelined kernel's one-group path and, at head
    # dim 128, the warp-specialized kernel.
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
                    1, 1024, 8, 8, 64, False, torch.float16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    1, 1024, 8, 8, 64, False, torch.bfloat16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    1, 256, 4, 4, 128, True, torch.float16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    4, 2048, 64, 4, 128, False, torch.float16, False, marks=pytest.mark.full
                ),
                pytest.param(
                    4, 2048, 64, 4, 128, False, torch.bfloat16, False, marks=pytest.mark.full
                ),
                pytest.param(
                    16, 2048, 16, 16, 128, False, torch.float16, False, marks=pytest.mark.full
                ),
                pytest.param(
                    4, 4096, 16, 16, 128, False, torch.bfloat16, True, marks=pytest.mark.full
                ),
            ],
        ),
    ]


@GQABwdFixture
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
    test = GQABwdTest(batch, heads, heads_kv, seq_len, dim, causal, dtype)
    op = GQABwdOp(causal)
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize("dim", [16, 80, 192, 256])
def test_gqa_bwd_serves_head_dims_off_the_widest_default(dim: int) -> None:
    """Dims the 256-thread, two-stage default cannot lay out or fit step down to a narrower
    default; 100 rows end part way through a block."""
    torch.manual_seed(123)
    test = GQABwdTest(1, 4, 2, 100, dim, True, torch.float16)
    op = GQABwdOp(target=BUILTIN)
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gqa_bwd_refuses_a_head_dim_off_the_contraction_before_building() -> None:
    """Every backward kernel contracts the head dim in steps of 16, so head dim 24 is refused."""
    test = GQABwdTest(1, 8, 2, 128, 24, True, torch.float16)
    op = GQABwdOp(target=BUILTIN)
    with pytest.raises(ValueError, match="head dim must be a multiple of 16"):
        op(*test.gen_inputs())
    for interface in GQABwdOp.interfaces:
        assert not op.built_entries(interface)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("q_lens", "kv_lens", "params", "out_dtype"),
    [
        pytest.param([512] * 2, [512] * 2, {}, torch.bfloat16, id="uniform-causal"),
        pytest.param([256], [1024], {"is_causal": False}, torch.float16, id="non-causal"),
        pytest.param(
            [192, 64],
            [192, 320],
            # The cap has to sit inside the score range to be observable: these rows
            # draw scores of magnitude about 2, and a cap of 30 is the identity on them.
            {"sm_scale": 0.05, "softcap": 1.0},
            torch.float16,
            id="sm-scale-softcap",
        ),
        pytest.param(
            [200] * 2,
            [200] * 2,
            {"is_causal": False, "window_size_left": 48, "window_size_right": 16},
            torch.bfloat16,
            id="window-both",
        ),
        # The window leaves the first key tile the block scans entirely invisible to
        # its last rows, so that tile's scores are all -infinity while the running
        # maximum still is: row 319 sees no key below 255, and the tile ends at 223.
        pytest.param(
            [320],
            [320],
            {"window_size_left": 64},
            torch.bfloat16,
            id="window-skips-first-tile",
        ),
        pytest.param(
            [1, 17, 0, 129], [1, 17, 3000, 129], {}, torch.float16, id="ragged-with-empty"
        ),
        pytest.param([4, 4], [0, 0], {"is_causal": False}, torch.bfloat16, id="empty-kv"),
        pytest.param(
            [128, 1],
            [384, 65],
            {"pos_encoding_mode": "rope", "rotary_dim": 64, "rope_layout": "interleaved"},
            torch.float16,
            id="rope-partial-interleaved",
        ),
        pytest.param(
            [128, 1], [384, 65], {"pos_encoding_mode": "rope"}, torch.bfloat16, id="rope-full-neox"
        ),
    ],
)
def test_gqa_varlen_fp8_matches_reference(
    q_lens: list[int], kv_lens: list[int], params: dict, out_dtype: torch.dtype
) -> None:
    """Every branch the FP8 packed-varlen kernel serves agrees with the float32 reference."""
    heads, heads_kv, dim = 16, 4, 128
    workload = GQAVarlenScaledWorkload(
        len(q_lens),
        q_lens,
        kv_lens,
        heads,
        heads_kv,
        dim,
        params.get("is_causal", True),
        params.get("window_size_left", -1),
        params.get("window_size_right", -1),
        torch.float8_e4m3fn,
        sm_scale=params.get("sm_scale"),
        softcap=params.get("softcap"),
        out_dtype=out_dtype,
        rotary_dim=(
            params.get("rotary_dim", dim) if params.get("pos_encoding_mode") == "rope" else None
        ),
        rope_layout=params.get("rope_layout", "neox"),
    )
    inputs = workload.gen_inputs()
    op = GQAVarlenFwdOp(out_dtype=out_dtype, target=BUILTIN, **params)

    # One e4m3 rounding of the softmax weights, and one more of Q and K under fused
    # rotation; the dense FP8 cases are held to the same bound.

    TestBase.check(workload, op, *inputs)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gqa_varlen_refuses_fp8_off_head_dim_128() -> None:
    from tileops.kernels.attention.call_spec import AttentionCall

    call = AttentionCall(
        arch=90,
        sm_count=132,
        dtype=torch.bfloat16,
        batch=2,
        heads=16,
        heads_kv=4,
        dim=64,
        is_causal=True,
        is_fp8=True,
        is_uniform=False,
    )
    with pytest.raises(ValueError, match="requires head dimension 128"):
        GQAVarlenFwdOp().key_for("gqa_varlen", call)


@pytest.mark.parametrize(
    "tune",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
)
def test_gqa_dense_decode_under_tuning(tune: bool) -> None:
    batch, heads, heads_kv, dim = 2, 32, 4, 128
    q = torch.randn(batch, 1, heads, dim, device=run_device(), dtype=torch.float16)
    k = torch.randn(batch, 4096, heads_kv, dim, device=run_device(), dtype=torch.float16)
    v = torch.randn_like(k)
    op = GQADenseFwdOp()
    if tune:
        op.request_tune()
    compare_outputs(
        op(q, k, v),
        dense_gqa_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True),
        dense_gqa_verification(q.dtype),
    )
