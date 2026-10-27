from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._kernels.attention import (
    GQADecodeBs1Kernel,
    GQADecodeKernel,
    GQADecodeLongContextKernel,
    GQADecodeWSKernel,
    GQADensePersistentWSKernel,
    GQADenseSlidingWindowBandKernel,
)
from tileops._kernels.attention.call_spec import AttentionCall
from tileops._kernels.kernel_base import Kernel
from tileops.attention import GQADenseBwdOp, GQADenseFwdOp, GQADenseWithLSEFwdOp, GQAVarlenFwdOp
from tileops.backend import BUILTIN, OpNotAvailableError
from workloads.attention.gqa.bwd import GQABwdWorkload
from workloads.attention.gqa.dense import (
    GQADensePrefillWorkload,
    GQADenseWithLSECall,
    dense_gqa_ref,
    dense_gqa_verification,
)
from workloads.attention.gqa.rope import apply_dense_rope
from workloads.attention.gqa.varlen import GQAVarlenScaledWorkload
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call


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
@pytest.mark.parametrize(
    "batch, seq_len_q, seq_len_kv, heads, dim, is_causal, options",
    [
        pytest.param(1, 1000, 1500, 32, 64, True, {"sinks": True}, id="d64-causal-sinks-lead"),
        pytest.param(2, 600, 600, 32, 64, False, {}, id="d64-noncausal-tail"),
        pytest.param(1, 777, 3001, 32, 128, True, {"softcap": 2.0}, id="d128-causal-softcap"),
        pytest.param(1, 800, 800, 32, 64, True, {"rotary_dim": 64}, id="d64-causal-rope"),
        # Sinks on the two-consumer D128 program, softcap on the three-consumer D64 program.
        pytest.param(1, 300, 900, 32, 128, True, {"sinks": True}, id="d128-causal-sinks"),
        pytest.param(1, 500, 700, 32, 64, True, {"softcap": 2.0}, id="d64-causal-softcap"),
        # Its query slots and key/value stages exceed shared memory: the WS kernel serves it.
        pytest.param(1, 300, 600, 16, 144, True, {"persistent": False}, id="d144-causal-ws"),
        # Keys and values of 32 MiB, over half of an H200's L2: a head's query blocks run on
        # neighbouring tiles, 18 blocks being one full group of 16 and two past it.
        pytest.param(1, 2304, 8192, 32, 128, True, {"grouped": True}, id="d128-causal-grouped"),
        # 32 tiles of eight key steps, the last partly past the keys: each splits in parts.
        pytest.param(1, 128, 1000, 32, 128, False, {"split": True}, id="d128-noncausal-split"),
    ],
)
def test_gqa_dense_persistent_prefill_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
    seq_len_q: int,
    seq_len_kv: int,
    heads: int,
    dim: int,
    is_causal: bool,
    options: dict,
) -> None:
    heads_kv = heads // 4
    served = []
    forward = GQADensePersistentWSKernel.forward

    def counted_forward(self, *args, **kwargs):
        served.append(type(self))
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(GQADensePersistentWSKernel, "forward", counted_forward)
    softcap = options.get("softcap")
    rotary_dim = options.get("rotary_dim")
    op = GQADenseFwdOp(
        is_causal=is_causal,
        softcap=softcap,
        pos_encoding_mode="rope" if rotary_dim else "none",
        rotary_dim=rotary_dim,
        target=BUILTIN,
    )

    def check(seq_len_q: int, persistent: bool) -> None:
        served.clear()
        q = torch.randn(batch, seq_len_q, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        kwargs = {}
        sinks = torch.randn(heads, device="cuda") if options.get("sinks") else None
        if sinks is not None:
            kwargs["sinks"] = sinks
        q_ref, k_ref = q, k
        if rotary_dim:
            angles = torch.randn(seq_len_kv, rotary_dim // 2, device="cuda") * 0.1
            kwargs["rope_cos"], kwargs["rope_sin"] = (
                angles.cos().to(q.dtype),
                angles.sin().to(q.dtype),
            )
            rope = dict(rotary_dim=rotary_dim, layout="neox")
            q_positions = torch.arange(seq_len_kv - seq_len_q, seq_len_kv, device="cuda")
            q_ref = apply_dense_rope(q, q_positions, kwargs["rope_cos"], kwargs["rope_sin"], **rope)
            k_positions = torch.arange(seq_len_kv, device="cuda")
            k_ref = apply_dense_rope(k, k_positions, kwargs["rope_cos"], kwargs["rope_sin"], **rope)
        output = op(q, k, v, **kwargs)
        assert bool(served) == persistent
        compare_outputs(
            output,
            dense_gqa_ref(
                q_ref,
                k_ref,
                v,
                heads=heads,
                heads_kv=heads_kv,
                is_causal=is_causal,
                softcap=softcap,
                sinks=sinks,
            ),
            dense_gqa_verification(q.dtype),
        )

    # One wave of query tiles and several: the persistent kernel serves both where it fits.
    persistent = options.get("persistent", True)
    check(128, persistent=persistent)
    check(seq_len_q, persistent=persistent)
    if options.get("grouped"):
        assert {kernel.rank_group for kernel in op.iter_kernels()} == {16}
    if options.get("split"):
        assert all(kernel.num_splits > 1 for kernel in op.iter_kernels())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize("batch", [1, 2])
def test_gqa_dense_reuses_one_kernel_across_sequence_lengths(batch: int) -> None:
    heads, heads_kv, dim = 8, 2, 128
    op = GQADenseFwdOp(is_causal=True, target=BUILTIN)

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
            GQADecodeWSKernel,
            id="batched-bf16",
        ),
        pytest.param(
            32,
            (32, 8),
            torch.float16,
            (4096, 4097),
            None,
            None,
            GQADecodeWSKernel,
            id="batched-fp16-unsplit-tail",
        ),
        pytest.param(
            16,
            (64, 8),
            torch.bfloat16,
            (4096, 4161),
            None,
            None,
            GQADecodeWSKernel,
            id="batched-bf16-split-tail",
        ),
        # The last key tile holds 15 rows; a per-row guard around the consumer's key
        # rotation leaves the barrier inside it short of threads, and the kernel hangs.
        pytest.param(
            1,
            (8, 2),
            torch.float16,
            (271, 1025),
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
        is_causal=True,
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
        call = op._dense_call((q, k, v, None, None, None, rope_cos, rope_sin, None))
        assert type(op.kernel_for("gqa_dense", lambda call=call: call)) is kernel_type
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
            GQADenseFwdOp(is_causal=True, target=BUILTIN)(q, k, v),
            reference,
            dense_gqa_verification(dtype),
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
@pytest.mark.sm90
@pytest.mark.parametrize(
    "shape, dtype, options, split",
    [
        pytest.param(
            (2, 1, 257, 8, 2, 128),
            torch.float16,
            {"softcap": 2.0},
            False,
            id="fused-softcap",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 1, 1025, 64, 8, 64),
            torch.bfloat16,
            {"has_sinks": True},
            True,
            id="split-sinks",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 1, 129, 64, 8, 64),
            torch.float16,
            {"has_sinks": True},
            False,
            id="fused-sinks",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (2, 1, 1025, 8, 2, 128),
            torch.bfloat16,
            {"softcap": 2.0},
            True,
            id="split-softcap",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (1, 1, 513, 8, 1, 256),
            torch.float16,
            {"sm_scale": 0.05, "has_sinks": True},
            True,
            id="split-d256-sinks",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (2, 1, 1025, 8, 2, 128),
            torch.float16,
            {"sm_scale": 1e-50},
            True,
            id="split-vanishing-scale",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (2, 1, 2049, 71, 1, 64),
            torch.bfloat16,
            {},
            True,
            id="split-wide-group",
            marks=pytest.mark.full,
        ),
    ],
)
def test_gqa_dense_decode_ws_matches_reference(
    shape: tuple, dtype: torch.dtype, options: dict, split: bool
) -> None:
    """The warp-specialized decode serves batched SM90 calls with one split or several."""
    torch.manual_seed(0)
    workload = GQADensePrefillWorkload(*shape, dtype, **options)
    op = GQADenseFwdOp(is_causal=True, sm_scale=workload.sm_scale, softcap=options.get("softcap"))
    inputs = workload.gen_inputs()
    if options.get("has_sinks"):
        inputs[-1][0] = -float("inf")
        inputs[-1][1] = float("inf")
        inputs[-1][-1] = 20.0
    padded = (*inputs[:3], *([None] * 6))
    if options.get("has_sinks"):
        padded = (*padded[:8], inputs[-1])
    kernel = op.kernel_for("gqa_dense", lambda: op._dense_call(padded))
    assert isinstance(kernel, GQADecodeWSKernel)
    assert (kernel.num_split(shape[2]) > 1) == split
    TestBase.check(workload, op, *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gqa_dense_decode_ws_leaves_a_scale_fp32_cannot_hold() -> None:
    """A scale whose base-2 form overflows FP32 stays on the general decode kernel."""
    q = torch.randn(2, 1, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 1025, 2, 128, device="cuda", dtype=torch.bfloat16)
    op = GQADenseFwdOp(is_causal=True, sm_scale=3e38)
    call = op._dense_call((q, k, k, *([None] * 6)))
    assert op.key_for("gqa_dense", call) == "gqa_dense_decode"


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
        GQADenseFwdOp(is_causal=True, target=BUILTIN)(q, k, v),
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
    op = GQADenseFwdOp(is_causal=True, pos_encoding_mode="rope", rotary_dim=dim, target=BUILTIN)
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
    op = GQADenseFwdOp(is_causal=True, target=BUILTIN)
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


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gqa_dense_sliding_window_tuned_tile_skips_two_masked_key_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tuned 128x64 tile under a 129-key causal window starts each query tile's walk at
    its first row's window, so its last row meets two fully masked key tiles before any
    score: the online softmax must carry both without NaN. The tuner also only times
    tiles the kernel builds, two 64-row warpgroup halves for 256 threads."""
    tile = {"block_m": 128, "block_n": 64, "num_stages": 2, "threads": 128}
    timed = []

    def record(self, jit_kernel, configs, **kwargs):
        timed.extend(configs)
        return SimpleNamespace(config=tile, latency=1.0)

    monkeypatch.setattr(Kernel, "tune_jit_kernel", record)
    torch.manual_seed(0)
    batch, seq_len, heads, heads_kv, dim, window = 1, 512, 8, 2, 64, 129
    op = GQADenseFwdOp(is_causal=True, window_size_left=window, target=BUILTIN)
    op.request_tune()
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    output = op(q, k, v)

    assert tile in timed
    assert all(config["block_m"] * 2 >= config["threads"] for config in timed)
    compare_outputs(
        output,
        dense_gqa_ref(
            q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True, window_size_left=window
        ),
        dense_gqa_verification(q.dtype),
    )


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.parametrize(
    ("shape", "dtype", "options", "band"),
    [
        pytest.param(
            (2, 300, 300, 16, 2, 64),
            torch.bfloat16,
            {"window_size_left": 127, "has_sinks": True},
            True,
            id="band-sinks-tail-group8",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (1, 300, 300, 8, 2, 64),
            torch.float16,
            {"window_size_left": 0},
            True,
            id="band-diagonal-only",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            (2, 300, 300, 16, 2, 64),
            torch.bfloat16,
            {"window_size_left": 127, "has_sinks": True, "tune": True},
            True,
            id="band-tuned",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (1, 300, 300, 8, 2, 64),
            torch.float16,
            {"window_size_left": 128},
            True,
            id="band-widest",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (1, 300, 300, 8, 2, 64),
            torch.float16,
            {"window_size_left": 129},
            False,
            id="past-band",
            marks=pytest.mark.full,
        ),
        pytest.param(
            (1, 300, 300, 4, 4, 128),
            torch.bfloat16,
            {"window_size_left": 127, "softcap": 2.0},
            True,
            id="band-mha-d128-softcap",
            marks=pytest.mark.full,
        ),
    ],
)
def test_gqa_dense_window_band(shape, dtype, options, band):
    """A causal window of at most one key tile packs the KV head's query heads into one CTA."""
    options = dict(options)
    tune = options.pop("tune", False)
    workload = GQADensePrefillWorkload(*shape, dtype, is_causal=True, sm_scale=0.125, **options)
    op_options = {k: v for k, v in options.items() if k != "has_sinks"}
    op = GQADenseFwdOp(is_causal=True, sm_scale=workload.sm_scale, target=BUILTIN, **op_options)
    if tune:
        op.request_tune()
    inputs = workload.gen_inputs()
    TestBase.check(workload, op, *inputs)
    call = op._dense_call(tuple(inputs) + (None,) * (9 - len(inputs)))
    assert (
        isinstance(
            op.kernel_for("gqa_dense", lambda call=call: call), GQADenseSlidingWindowBandKernel
        )
        is band
    )


class GQABwdFixture(FixtureBase):
    # Head dim 64 reaches the pipelined kernel, its heads_kv == heads cases the one-group
    # path; head dim 128 reaches the warp-specialized kernel. On 132 SMs its grouped cases
    # take, in order: one query head per tile with dK/dV partials, two heads per tile with
    # partials, and the whole group per tile.
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
                    1, 256, 8, 2, 128, True, torch.bfloat16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    2, 1024, 32, 8, 128, True, torch.float16, False, marks=pytest.mark.smoke
                ),
                pytest.param(
                    4, 512, 32, 16, 128, True, torch.bfloat16, False, marks=pytest.mark.full
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
    op = GQADenseBwdOp(causal)
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
    op = GQADenseBwdOp(is_causal=True, target=BUILTIN)
    test.check(op, *test.gen_inputs())


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gqa_bwd_partials_return_to_zero_between_calls() -> None:
    """Key blocks whose query heads span several tiles add dK/dV partials into accumulators
    the last tile leaves zeroed, so a second call on the same op starts from zero; each call
    adds dQ into one of two accumulators and zeroes the other, so the third call adds into
    the first call's again. Non-causal full-length key blocks take the same partial path as
    causal ones."""
    test = GQABwdTest(1, 8, 2, 256, 128, False, torch.bfloat16)
    op = GQADenseBwdOp(False, target=BUILTIN)
    inputs = test.gen_inputs()
    for _ in range(3):
        test.check(op, *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize(
    "heads, heads_kv, causal",
    [pytest.param(8, 2, False, id="grouped-partials"), pytest.param(4, 4, True, id="mha")],
)
def test_gqa_bwd_graph_replays_alternate_dq_accumulators(
    heads: int, heads_kv: int, causal: bool
) -> None:
    """The dQ accumulator a call adds into is chosen on the device, so each replay of a
    captured call adds into the one the previous replay zeroed."""
    test = GQABwdTest(1, heads, heads_kv, 256, 128, causal, torch.bfloat16)
    op = GQADenseBwdOp(causal, target=BUILTIN)
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = op(*inputs)
    for _ in range(3):
        graph.replay()
        test.check(op, *inputs, runs=lambda *args: outputs)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gqa_bwd_refuses_a_head_dim_off_the_contraction_before_building() -> None:
    """Every backward kernel contracts the head dim in steps of 16, so head dim 24 is refused."""
    test = GQABwdTest(1, 8, 2, 128, 24, True, torch.float16)
    op = GQADenseBwdOp(is_causal=True, target=BUILTIN)
    with pytest.raises(OpNotAvailableError, match="head dim must be a multiple of 16"):
        op(*test.gen_inputs())
    for interface in GQADenseBwdOp.interfaces:
        assert not op._entries_built(interface)


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
    op = GQAVarlenFwdOp(out_dtype=out_dtype, target=BUILTIN, **{"is_causal": True, **params})

    # One e4m3 rounding of the softmax weights, and one more of Q and K under fused
    # rotation; the dense FP8 cases are held to the same bound.

    TestBase.check(workload, op, *inputs)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gqa_varlen_refuses_fp8_off_head_dim_128() -> None:
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
        GQAVarlenFwdOp(is_causal=True).key_for("gqa_varlen", call)


@pytest.mark.parametrize(
    "tune",
    [pytest.param(False, marks=pytest.mark.smoke), pytest.param(True, marks=pytest.mark.full)],
)
def test_gqa_dense_decode_under_tuning(tune: bool) -> None:
    batch, heads, heads_kv, dim = 2, 32, 4, 128
    q = torch.randn(batch, 1, heads, dim, device=run_device(), dtype=torch.float16)
    k = torch.randn(batch, 4096, heads_kv, dim, device=run_device(), dtype=torch.float16)
    v = torch.randn_like(k)
    op = GQADenseFwdOp(is_causal=True)
    if tune:
        op.request_tune()
    compare_outputs(
        op(q, k, v),
        dense_gqa_ref(q, k, v, heads=heads, heads_kv=heads_kv, is_causal=True),
        dense_gqa_verification(q.dtype),
    )


@pytest.mark.sm90
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            ((2, 129, 257, 8, 2, 64), torch.float16, {}), id="prefill-fp16", marks=pytest.mark.smoke
        ),
        pytest.param(
            (
                (1, 257, 257, 8, 2, 128),
                torch.bfloat16,
                {
                    "is_causal": False,
                    "window_size_left": 32,
                    "window_size_right": 16,
                    "rotary_dim": 64,
                    "softcap": 2.0,
                },
            ),
            id="window-rope",
            marks=pytest.mark.full,
        ),
        pytest.param(
            ((2, 1, 17, 8, 2, 64), torch.float16, {}), id="decode-unsplit", marks=pytest.mark.full
        ),
        pytest.param(
            ((1, 1, 641, 8, 2, 128), torch.float16, {}), id="decode-bs1", marks=pytest.mark.full
        ),
        pytest.param(
            ((1, 129, 225, 16, 2, 128), torch.float8_e4m3fn, {}),
            id="fp8-prefill",
            marks=pytest.mark.full,
        ),
        pytest.param(
            ((1, 1, 2049, 8, 2, 128), torch.float8_e4m3fn, {}),
            id="fp8-decode",
            marks=pytest.mark.full,
        ),
        pytest.param(
            ((1, 129, 257, 8, 2, 128), torch.bfloat16, {"is_causal": False, "softcap": 2.0}),
            id="prefill-bf16-softcap",
            marks=pytest.mark.full,
        ),
        pytest.param(
            ((2, 1, 257, 6, 2, 64), torch.bfloat16, {}),
            id="decode-split-tail",
            marks=pytest.mark.full,
        ),
        pytest.param(
            ((1, 1, 1025, 8, 2, 128), torch.float16, {"rotary_dim": 64}),
            id="decode-rope-split",
            marks=pytest.mark.full,
        ),
        pytest.param(
            ((1, 1, 1025, 32, 4, 128), torch.float16, {}), id="decode-long", marks=pytest.mark.full
        ),
        pytest.param(
            ((1, 129, 225, 8, 2, 128), torch.float8_e4m3fn, {"softcap": 2.0, "rotary_dim": 64}),
            id="fp8-softcap-rope",
            marks=pytest.mark.full,
        ),
        pytest.param(
            ((1, 129, 3136, 8, 1, 128), torch.float8_e4m3fn, {}),
            id="fp8-causal-long",
            marks=pytest.mark.full,
        ),
        pytest.param(
            ((1, 6945, 6945, 4, 1, 128), torch.float8_e4m3fn, {"is_causal": False}),
            id="fp8-noncausal-long",
            marks=pytest.mark.full,
        ),
    ],
)
def test_gqa_dense_sinks(case):
    """Keep one small feature smoke; cover dtype and kernel variants in full."""
    shape, dtype, options = case
    workload = GQADensePrefillWorkload(
        *shape,
        dtype,
        out_dtype=torch.bfloat16 if dtype == torch.float8_e4m3fn else None,
        sm_scale=0.125,
        has_sinks=True,
        **options,
    )
    op_options = dict(options)
    if "rotary_dim" in options:
        op_options["pos_encoding_mode"] = "rope"
    op = GQADenseFwdOp(
        **{"is_causal": True, **op_options},
        sm_scale=workload.sm_scale,
        out_dtype=workload.out_dtype if dtype == torch.float8_e4m3fn else None,
    )
    inputs = workload.gen_inputs()
    # Large finite sinks must suppress output without overflow; -inf disables
    # only that head's sink. Other heads keep distinct, finite logits.
    inputs[-1][0] = -float("inf")
    inputs[-1][1] = float("inf")
    inputs[-1][-1] = 3e38
    TestBase.check(workload, op, *inputs)


@pytest.mark.sm90
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            ((2, 17, 33, 8, 2, 64), torch.float16, True),
            id="prefill-fp16",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            ((1, 6945, 6945, 4, 1, 128), torch.float8_e4m3fn, False),
            id="fp8-noncausal-long",
            marks=pytest.mark.full,
        ),
    ],
)
def test_gqa_dense_sink_reuse(case):
    """One op accepts absent, strided and changed sinks in successive calls."""
    shape, dtype, is_causal = case
    out_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else None
    workload = GQADensePrefillWorkload(
        *shape, dtype, out_dtype=out_dtype, is_causal=is_causal, has_sinks=True
    )
    op = GQADenseFwdOp(is_causal=is_causal, out_dtype=out_dtype)
    inputs = workload.gen_inputs()
    sinks = inputs[-1].repeat_interleave(2)[::2]
    for current_sinks in (None, sinks, sinks + 1, None):
        TestBase.check(workload, op, *inputs[:-1], current_sinks)


@pytest.mark.sm90
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            ((2, 1, 17, 8, 2, 64), torch.float16, True),
            id="decode-unsplit-equal",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            ((2, 1, 257, 6, 2, 64), torch.bfloat16, False),
            id="decode-split-tail",
            marks=pytest.mark.full,
        ),
        pytest.param(
            ((1, 1, 641, 8, 2, 128), torch.float16, False),
            id="decode-bs1",
            marks=pytest.mark.full,
        ),
        pytest.param(
            ((1, 1, 1025, 32, 4, 128), torch.float16, False),
            id="decode-long",
            marks=pytest.mark.full,
        ),
        pytest.param(
            ((1, 1, 2049, 8, 2, 128), torch.float8_e4m3fn, False),
            id="fp8-decode",
            marks=pytest.mark.full,
        ),
    ],
)
def test_gqa_dense_sink_large_logits(case):
    """Sink mass survives when absolute logits dwarf the log denominator."""
    shape, dtype, all_equal = case
    out_dtype = torch.bfloat16 if dtype == torch.float8_e4m3fn else None
    workload = GQADensePrefillWorkload(
        *shape, dtype, out_dtype=out_dtype, sm_scale=0.125, has_sinks=True
    )
    op = GQADenseFwdOp(is_causal=True, sm_scale=workload.sm_scale, out_dtype=out_dtype)
    inputs = workload.gen_inputs()
    if all_equal:
        # Equal ordinary and sink logits give 17 / 18 even when log(17)
        # is below one ULP of the FP32 maximum.
        inputs[0].fill_(4096)
        inputs[1].fill_(4096)
        inputs[2].fill_(1)
        inputs[-1].fill_(2**27)
    else:
        # One dominant token and an equal sink must each receive half the
        # weight, including when the absolute LSE cannot represent log2(2).
        is_fp8 = dtype == torch.float8_e4m3fn
        inputs[0].fill_(1 if is_fp8 else 4096)
        inputs[1].zero_()
        inputs[1][:, -1].fill_(1 if is_fp8 else 4096)
        inputs[2].fill_(1)
        if is_fp8:
            for scale, value in zip(inputs[3:6], (4096, 2048, 1), strict=True):
                scale.fill_(value)
        inputs[-1].fill_(2**27 if is_fp8 else shape[-1] * workload.sm_scale * 4096**2)
    TestBase.check(workload, op, *inputs[:-1], None)
    TestBase.check(workload, op, *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize(
    "dim, is_causal, sm_scale",
    [
        # Head dim 128 reaches the warp-specialized backward, 64 the pipelined one.
        pytest.param(128, True, None, id="d128-causal"),
        pytest.param(128, False, 0.1, id="d128-scaled"),
        pytest.param(64, True, 0.1, id="d64-causal-scaled"),
    ],
)
def test_gqa_dense_with_lse_fwd_feeds_the_bwd(
    dim: int, is_causal: bool, sm_scale: Optional[float]
) -> None:
    """GQADenseWithLSEFwdOp's ``o`` and natural-log ``lse`` go into GQADenseBwdOp as they
    are, and the gradients are torch autograd's of the same attention."""
    torch.manual_seed(0)
    batch, seq_len, heads, heads_kv, dtype = 2, 256, 8, 2, torch.bfloat16
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len, heads_kv, dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    forward = GQADenseWithLSEFwdOp(is_causal=is_causal, sm_scale=sm_scale, target=BUILTIN)
    o, lse = forward(q, k, v)

    leaves = [t.float().requires_grad_() for t in (q, k, v)]
    reference = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in leaves), is_causal=is_causal, scale=sm_scale, enable_gqa=True
    ).transpose(1, 2)
    scores = (
        leaves[0].detach().transpose(1, 2)
        @ leaves[1].detach().repeat_interleave(heads // heads_kv, dim=2).permute(0, 2, 3, 1)
        * (dim**-0.5 if sm_scale is None else sm_scale)
    )
    if is_causal:
        scores = scores.masked_fill(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device="cuda").triu(1), float("-inf")
        )
    torch.testing.assert_close(lse, scores.logsumexp(-1), atol=1e-3, rtol=1e-4)
    compare_outputs(o, reference.detach().to(dtype), dense_gqa_verification(dtype))

    do = torch.randn_like(o)
    dq, dk, dv = GQADenseBwdOp(is_causal=is_causal, sm_scale=sm_scale, target=BUILTIN)(
        do, q, k, v, o, lse
    )
    reference.backward(do.float())
    tolerance = GQABwdTest(batch, heads, heads_kv, seq_len, dim, is_causal, dtype).verification(do)
    for got, leaf in zip((dq, dk, dv), leaves, strict=True):
        compare_outputs(got, leaf.grad.to(dtype), tolerance)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gqa_bwd_gives_a_row_with_lse_minus_inf_zero_gradients() -> None:
    """``lse`` ``-inf`` marks a row that saw no key, as the forward writes it: the row gets a
    zero ``dq`` and adds nothing to ``dk`` and ``dv``."""
    torch.manual_seed(0)
    test = GQABwdTest(1, 4, 2, 256, 64, True, torch.bfloat16)
    do, q, k, v, o, lse = (t.detach() for t in test.gen_inputs())
    lse = lse.clone()
    lse[:, :, 7] = float("-inf")
    dq, dk, dv = GQADenseBwdOp(is_causal=True, target=BUILTIN)(do, q, k, v, o, lse)
    assert (dq[:, 7] == 0).all()
    silent = do.clone()
    silent[:, 7] = 0
    expected = test.ref_program(silent, q, k, v, o, lse)
    compare_outputs((dq, dk, dv), expected, test.verification(silent))


@pytest.mark.smoke
def test_gqa_dense_bwd_refuses_what_its_kernels_do_not_serve() -> None:
    """A window or softcap of the forward is refused by name, not silently dropped."""
    test = GQABwdTest(1, 4, 2, 128, 64, True, torch.bfloat16)
    inputs = [t.detach() for t in test.gen_inputs()]
    with pytest.raises(OpNotAvailableError, match="softcap"):
        GQADenseBwdOp(is_causal=True, softcap=30.0)(*inputs)
    with pytest.raises(OpNotAvailableError, match="sliding window"):
        GQADenseBwdOp(is_causal=True, window_size_left=32)(*inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize(
    "row, plan",
    [
        # Both lengths on 128-row boundaries: the first group scores its diagonal tile's lower
        # 64 keys alone.
        pytest.param(
            dict(B=1, S_q=256, S_kv=256, H=8, H_kv=2, D=128), dict(diag_aligned=True), id="square"
        ),
        # More tiles than SMs, rows past the last full tile, and a softcap that bends the scores.
        pytest.param(
            dict(B=2, S_q=700, S_kv=1100, H=32, H_kv=8, D=128, softcap=2.0),
            {},
            id="d128-rect-softcap",
        ),
        # Keys and values over half of an H200's L2: the grouped tile order.
        pytest.param(
            dict(B=1, S_q=2304, S_kv=8192, H=32, H_kv=8, D=128),
            dict(rank_group=16, diag_aligned=True),
            id="d128-grouped",
        ),
        # Eight tiles, the first starting at row -64: each tile's keys split in parts, merged
        # with their log-sum-exp.
        pytest.param(
            dict(B=1, S_q=150, S_kv=1100, H=4, H_kv=1, D=128), dict(split=True), id="d128-split"
        ),
        pytest.param(
            dict(B=2, S_q=128, S_kv=384, H=8, H_kv=2, D=64, softcap=30.0), {}, id="rect-softcap"
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_gqa_dense_with_lse_fwd_matches_reference(row: dict, plan: dict, dtype: str) -> None:
    """A manifest call's ``(o, lse)`` against the FP32 reference; the persistent kernel
    serves it at head dimension 128 only, with the diagonal width, and where ``plan`` names
    them the tile order and key parts, that ``plan`` states."""
    torch.manual_seed(0)
    call = manifest_call("GQADenseWithLSEFwdOp", {"T": dtype}, is_causal=True, **row)
    workload = GQADenseWithLSECall(call)
    op = GQADenseWithLSEFwdOp(**workload.arguments(), target=BUILTIN)
    TestBase.check(workload, op, *workload.gen_inputs())
    persistent = [k for k in op.iter_kernels() if isinstance(k, GQADensePersistentWSKernel)]
    assert bool(persistent) == (row["D"] == 128)
    for kernel in persistent:
        assert kernel.diag_aligned == plan.get("diag_aligned", False)
        if "rank_group" in plan:
            assert kernel.rank_group == plan["rank_group"]
        if plan.get("split"):
            assert kernel.num_splits > 1


@pytest.mark.smoke
@pytest.mark.parametrize(
    "batch, seq_len_q, seq_len_kv, heads, is_causal, splits",
    [
        # 32 tiles of 16 key steps on 132 SMs: four parts each.
        pytest.param(1, 128, 2048, 32, True, 4, id="few-tiles"),
        # 256 tiles fill the SMs.
        pytest.param(8, 128, 2048, 32, True, 1, id="tiles-fill-sms"),
        # Two parts would take one key step off two: less than the merge costs.
        pytest.param(1, 128, 256, 32, True, 1, id="saves-one-step"),
        # Three key steps cap the parts at three, which take two steps off.
        pytest.param(1, 128, 384, 32, False, 3, id="capped-by-steps"),
        # The first causal tile of a square call has one key step.
        pytest.param(1, 512, 512, 8, True, 1, id="shortest-tile-one-step"),
        # 150 rows align the first tile to row -64: it has eight key steps, the others nine.
        pytest.param(1, 150, 1100, 4, True, 8, id="partial-first-tile"),
    ],
)
def test_gqa_dense_persistent_splits_the_keys_of_few_tiles(
    batch: int, seq_len_q: int, seq_len_kv: int, heads: int, is_causal: bool, splits: int
) -> None:
    """Parts a tile's key steps divide into: the SMs over the tiles, at most the shortest
    tile's key steps, and only where they take two key steps off the longest tile."""
    call = AttentionCall(
        arch=90,
        sm_count=132,
        dtype=torch.bfloat16,
        batch=batch,
        heads=heads,
        heads_kv=heads,
        dim=128,
        max_seqlen_q=seq_len_q,
        seqlen_kv=seq_len_kv,
        is_causal=is_causal,
    )
    assert GQADensePersistentWSKernel._num_splits(call) == splits


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
@pytest.mark.parametrize(
    "case, dtype",
    [
        # Three parts of 128 keys average 100.015625, -50 and -50: with the first rounded to
        # 16 bits, they merge to 0 instead of 0.0052.
        pytest.param("cancel", torch.float16, id="cancel-f16"),
        pytest.param("cancel", torch.bfloat16, id="cancel-bf16"),
        # Logits of 2^28 over parts of 256, 256, 256 and 232 keys: an FP32 log-sum-exp per
        # part rounds the key counts away and weighs the parts alike.
        pytest.param("large", torch.float16, id="large-logits"),
        # The same logits beside a sink of 2^28 over three parts of 128 keys.
        pytest.param("sink", torch.bfloat16, id="large-logits-sink"),
    ],
)
def test_gqa_dense_split_merges_parts_without_rounding(case: str, dtype: torch.dtype) -> None:
    """Merged parts match the reference where 16-bit part outputs or an FP32 log-sum-exp
    per part would lose the result."""
    heads, heads_kv, dim = 32, 8, 128
    seq_len_kv = 1000 if case == "large" else 384
    q = torch.zeros(1, 128, heads, dim, device="cuda", dtype=dtype)
    k = torch.zeros(1, seq_len_kv, heads_kv, dim, device="cuda", dtype=dtype)
    v = torch.zeros_like(k)
    sm_scale, sinks = None, None
    if case == "cancel":
        v[:, :32] = 100.0625 if dtype == torch.float16 else 100.5
        v[:, 32:128] = 100
        v[:, 128:] = -50
    else:
        # 4096 * 4096 * 128 * 0.125: every logit is 2^28.
        q.fill_(4096)
        k.fill_(4096)
        sm_scale = 0.125
        if case == "large":
            v[:, 768:] = 8
        else:
            v.fill_(1)
            sinks = torch.full((heads,), 2.0**28, device="cuda")
    op = GQADenseFwdOp(is_causal=False, sm_scale=sm_scale, target=BUILTIN)
    output = op(q, k, v, sinks=sinks) if sinks is not None else op(q, k, v)
    assert all(kernel.num_splits > 1 for kernel in op.iter_kernels())
    expected = dense_gqa_ref(
        q, k, v, heads=heads, heads_kv=heads_kv, is_causal=False, sm_scale=sm_scale, sinks=sinks
    )
    compare_outputs(output, expected, dense_gqa_verification(dtype))


@pytest.mark.in_tree_kernels
@pytest.mark.sm90
@pytest.mark.smoke
def test_gqa_dense_persistent_masks_the_keys_past_a_narrow_last_tile() -> None:
    """897 keys in four parts: the last part's last key tile holds one key, so its softmax
    covers only the lower 64 keys of that tile. Equal logits and unit values give an output
    of exactly 1; a key past the end that escapes the mask lowers it to 897 / 960."""
    heads, heads_kv, dim, seq_len_kv = 32, 8, 128, 897
    q = torch.zeros(1, 128, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, seq_len_kv, heads_kv, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.ones_like(k)
    op = GQADenseFwdOp(is_causal=False, target=BUILTIN)
    output = op(q, k, v)
    assert all(kernel.num_splits == 4 for kernel in op.iter_kernels())
    torch.testing.assert_close(output, torch.ones_like(output), atol=0, rtol=0)


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_gqa_dense_with_lse_fwd_refuses_a_window_by_name() -> None:
    """No windowed kernel writes the log-sum-exp, so a windowed call is refused, naming it."""
    q = torch.randn(1, 128, 8, 64, device=run_device(), dtype=torch.bfloat16)
    k = torch.randn(1, 128, 2, 64, device=run_device(), dtype=torch.bfloat16)
    with pytest.raises(OpNotAvailableError, match="log-sum-exp"):
        GQADenseWithLSEFwdOp(is_causal=True, window_size_left=32, target=BUILTIN)(q, k, k)
