"""Tests for the GLA ops: chunkwise forward and backward, inference, decode."""

import itertools

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops.backend import TensorSpec, registry
from tileops.kernels.linear_attention.gla.call_spec import GLACall
from tileops.kernels.linear_attention.gla.dense_decode import GLADenseDecodeFwdKernel
from tileops.kernels.linear_attention.gla.dense_prefill_partitioned import (
    GLADensePrefillPartitionedKernel,
)
from tileops.kernels.linear_attention.gla.dense_prefill_subchunk import (
    GLADensePrefillSubchunkKernel,
)
from tileops.ops import GLAChunkBwdOp, GLAChunkFwdOp, GLAFwdOp, GLARecurrentFwdOp
from workloads.device import run_device, run_device_is_cuda
from workloads.linear_attention.gla import (
    GLADecodeWorkload,
    GLAFwdWorkload,
    chunkwise_verification,
    decode_verification,
    gla_autograd_bwd_torch,
    gla_decode_torch,
    gla_fwd_chunked_torch,
    inference_verification,
)
from workloads.numerics import compare_outputs

try:
    from fla.ops.gla import chunk_gla
except ImportError:
    chunk_gla = None

try:
    from fla.ops.gla import fused_recurrent_gla
except ImportError:
    fused_recurrent_gla = None


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors (flattened)."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm() + 1e-12)).item()


class GLAChunkFwdFixture(FixtureBase):
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


@GLAChunkFwdFixture
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

    q = torch.randn(B, T, H, K, device=run_device(), dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device=run_device(), dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device=run_device(), dtype=dtype) * 0.1
    g = -torch.rand(B, T, H, K, device=run_device(), dtype=dtype)

    # --- Torch reference ---
    ref_o, _ref_state = gla_fwd_chunked_torch(q, k, v, g, BC, scale=scale)

    # --- FLA reference (if available; its Triton kernels need CUDA) ---
    fla = chunk_gla is not None and run_device_is_cuda()
    if fla:
        fla_o, _ = chunk_gla(q.float(), k.float(), v.float(), g.float(), scale=scale)
        cos = cosine_sim(ref_o, fla_o)
        print(f"  FLA vs ref o: cosine={cos:.6f}")
        compare_outputs(fla_o.to(dtype), ref_o.to(dtype), chunkwise_verification(dtype))

    fwd_op = GLAChunkFwdOp(
        chunk_size=BC,
        scale=scale,
    )
    if tune:
        fwd_op.request_tune()
    op_o, _ = fwd_op(q, k, v, g)

    cos = cosine_sim(ref_o, op_o)
    print(f"  TileOPs vs ref o: cosine={cos:.6f}")
    compare_outputs(op_o, ref_o.to(op_o.dtype), chunkwise_verification(dtype))

    # --- TileOPs vs FLA ---
    if fla:
        cos = cosine_sim(fla_o, op_o)
        print(f"  TileOPs vs FLA o: cosine={cos:.6f}")
        compare_outputs(op_o, fla_o.to(op_o.dtype), chunkwise_verification(dtype))


def _fla_autograd_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GLA backward gradients via FLA's chunk_gla + autograd.

    FLA uses the same BTHD layout and g shape [B, T, H, K] as TileOPs.

    Returns:
        (dq, dk, dv, dg) all in float32.
    """
    q_ = q.float().detach().requires_grad_(True)
    k_ = k.float().detach().requires_grad_(True)
    v_ = v.float().detach().requires_grad_(True)
    g_ = g.float().detach().requires_grad_(True)

    o, _ = chunk_gla(q_, k_, v_, g_, scale=scale)
    loss = (o * do.float()).sum()
    dq, dk, dv, dg = torch.autograd.grad(loss, [q_, k_, v_, g_])
    return dq, dk, dv, dg


class GLAChunkBwdFixture(FixtureBase):
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
            ],
        ),
    ]


@pytest.mark.cuda_only
@GLAChunkBwdFixture
def test_gla_bwd(
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

    # GLA layout: BTHD — [B, T, H, K/V]
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype) * 0.1
    v = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(B, T, H, K, device="cuda", dtype=dtype)
    do = torch.randn(B, T, H, V, device="cuda", dtype=dtype) * 0.1

    scale = K**-0.5

    # --- Torch reference via autograd ---
    ref_dq, ref_dk, ref_dv, ref_dg = gla_autograd_bwd_torch(do, q, k, v, g, BC, scale=scale)
    ref_grads = {"dq": ref_dq, "dk": ref_dk, "dv": ref_dv, "dg": ref_dg}

    # --- FLA reference via autograd (if available) ---
    if chunk_gla is not None:
        fla_dq, fla_dk, fla_dv, fla_dg = _fla_autograd_bwd(do, q, k, v, g, scale=scale)
        fla_grads = {"dq": fla_dq, "dk": fla_dk, "dv": fla_dv, "dg": fla_dg}

        compare_outputs(
            tuple(fla_grads.values()),
            tuple(ref_grads.values()),
            chunkwise_verification(dtype, backward=True),
        )

    # --- TileOPs backward, fed the reference's chunk states ---
    _, h = gla_fwd_chunked_torch(q, k, v, g, BC, scale=scale, with_chunk_states=True)

    dht = torch.zeros(B, H, K, V, device="cuda", dtype=torch.float32)
    bwd_op = GLAChunkBwdOp(chunk_size=BC, scale=scale)
    if tune:
        bwd_op.request_tune()
    op_dq, op_dk, op_dv, op_dg = bwd_op(q, k, v, g, h, do, dht)
    op_grads = {"dq": op_dq, "dk": op_dk, "dv": op_dv, "dg": op_dg}

    compare_outputs(
        tuple(op_grads.values()),
        tuple(ref_grads.values()),
        chunkwise_verification(dtype, backward=True),
    )

    if chunk_gla is not None:
        compare_outputs(
            tuple(op_grads.values()),
            tuple(fla_grads.values()),
            chunkwise_verification(dtype, backward=True),
        )


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_gla_refuses_extents_its_gemms_do_not_tile() -> None:
    """dim_v=72 splits into 18-column state tiles, which no 8-column MMA tile covers."""
    B, T, H, K, V = 1, 64, 2, 64, 72
    q, k, g = (torch.randn(B, T, H, K, device="cuda", dtype=torch.float16) for _ in range(3))
    v, do = (torch.randn(B, T, H, V, device="cuda", dtype=torch.float16) for _ in range(2))
    with pytest.raises(ValueError, match="dim_v=72"):
        GLAChunkFwdOp(chunk_size=64)(q, k, v, g)
    h = torch.zeros(B, 2, H, K, V, device="cuda")
    dht = torch.zeros(B, H, K, V, device="cuda")
    with pytest.raises(ValueError, match="dim_v=72"):
        GLAChunkBwdOp(chunk_size=64)(q, k, v, g, h, do, dht)


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_gla_bwd_refuses_what_99_kb_cannot_hold() -> None:
    """SM89 gives a block 99 KB of opt-in shared memory, too little for the float32 backward
    at head dims 128 / 128, so the op refuses it before anything is built."""
    B, T, H, K, V = 1, 128, 2, 128, 128
    device = run_device()
    q, k = (torch.randn(B, T, H, K, device=device) for _ in range(2))
    g = -torch.rand(B, T, H, K, device=device)
    v, do = (torch.randn(B, T, H, V, device=device) for _ in range(2))
    h = torch.zeros(B, T // 64 + 1, H, K, V, device=device)
    dht = torch.zeros(B, H, K, V, device=device)
    op = GLAChunkBwdOp(chunk_size=64)
    with pytest.raises(ValueError, match="bytes of shared memory per block"):
        op(q, k, v, g, h, do, dht)
    for interface in GLAChunkBwdOp.interfaces:
        assert not op.built_entries(interface)


def _skip_unless_kernel_serves(kernel_cls: type, test: GLAFwdWorkload) -> None:
    """Skip when *kernel_cls* declares that it does not serve *test*'s call on the run device."""
    call = GLACall(
        batch=test.batch,
        seq_len=test.seq_len,
        heads=test.heads,
        dim_k=test.dim_k,
        dim_v=test.dim_v,
        dtype=test.dtype,
        device=torch.device(run_device()),
    )
    reason = kernel_cls.unavailable(call) or kernel_cls.refusal(call)
    if reason is not None:
        pytest.skip(f"{kernel_cls.__name__}: {reason}")


# The public GLA inference contract and its in-tree dense-prefill path.
class GLAFwdTest(GLAFwdWorkload, TestBase):
    pass


@pytest.fixture
def isolated_registry():
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True
    yield
    registry.restore(state)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_registry")
def test_gla_fwd_reaches_external_target_with_optional_inputs() -> None:
    calls = []

    def build_kernel(*specs, **params):
        calls.append((specs, params))

        def kernel(q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu):
            del k, g, initial_state, cu_seqlens_cpu
            state_batch = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
            return (
                torch.empty_like(v),
                torch.empty(state_batch, q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32),
            )

        return kernel

    registry.register_kernel_builder("GLAFwdOp", "gla_test", build_kernel)
    q = torch.randn(1, 7, 2, 8, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(1, 7, 2, 6, dtype=torch.float16)
    g = -torch.rand_like(q)
    state = torch.zeros(2, 2, 8, 6, dtype=torch.float32)
    cu = torch.tensor([0, 3, 7], dtype=torch.int64)
    op = GLAFwdOp(scale=0.125, target="gla_test")

    o, final_state = op(q, k, v, g, state, cu, cu.clone())
    assert o.shape == v.shape
    assert final_state.shape == state.shape

    decode_q = torch.randn(1, 1, 2, 8, dtype=torch.float16)
    decode_v = torch.randn(1, 1, 2, 6, dtype=torch.float16)
    decode_state = torch.zeros(1, 2, 8, 6, dtype=torch.float32)
    decode_o, decode_final = op(decode_q, decode_q, decode_v, decode_q, decode_state)
    assert decode_o.shape == decode_v.shape
    assert decode_final.shape == decode_state.shape
    assert calls == [
        (
            tuple(TensorSpec.of(tensor) for tensor in (q, k, v, g, state, cu, cu)),
            {"scale": 0.125},
        ),
        (
            tuple(
                None if tensor is None else TensorSpec.of(tensor)
                for tensor in (decode_q, decode_q, decode_v, decode_q, decode_state, None, None)
            ),
            {"scale": 0.125},
        ),
    ]


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_registry")
@pytest.mark.parametrize("seeded", [False, True])
def test_gla_fwd_roofline_counts_packed_states(seeded: bool) -> None:
    def build_kernel(*specs, **params):
        def kernel(q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu):
            state_batch = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
            return torch.empty_like(v), torch.empty(
                state_batch, q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32
            )

        return kernel

    registry.register_kernel_builder("GLAFwdOp", "gla_test", build_kernel)
    op = GLAFwdOp(target="gla_test")
    q, k, g = (torch.empty(1, 7, 2, 8, dtype=torch.float16) for _ in range(3))
    v = torch.empty(1, 7, 2, 6, dtype=torch.float16)
    packed_state = torch.empty(2, 2, 8, 6, dtype=torch.float32) if seeded else None
    cu = torch.tensor([0, 3, 7], dtype=torch.int64)

    # Reuse the Op for dense input too: packed sequence metadata must not leak.
    for state, lengths, expected_bytes in (
        (packed_state, cu, 2568 if seeded else 1800),
        (packed_state[:1] if seeded else None, None, 1776 if seeded else 1392),
    ):
        op(q, k, v, g, state, lengths)
        assert op.eval_roofline() == (7 * 2 * (5 * 8 * 6 + 8 + 6), expected_bytes)


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("seq_len, dim", [(128, 64), (128, 128), (1024, 64)])
def test_gla_dense_prefill_matches_fla(dtype: torch.dtype, seq_len: int, dim: int) -> None:
    torch.manual_seed(2160)
    test = GLAFwdTest(2, seq_len, 4, dim, dim, dtype, has_initial_state=True)
    _skip_unless_kernel_serves(GLADensePrefillSubchunkKernel, test)
    inputs = test.gen_inputs()
    op = GLAFwdOp()
    test.check(op, *inputs)
    test.check(op, *inputs[:4])
    inputs[3].mul_(3.0)
    test.check(op, *inputs)


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("dtype,dim", [(torch.bfloat16, 64), (torch.float16, 128)])
@pytest.mark.parametrize("scale", [None, 0.3])
def test_gla_packed_varlen_matches_fla(dtype: torch.dtype, dim: int, scale: float | None) -> None:
    """Sequence lengths from one token up, with the state and the host offsets each absent.

    Two heads is the fewest at which a per-sequence state walk oversubscribes the device at
    width 128 and not at width 64, so the float16 case runs the partitioned walk and the
    bfloat16 case the per-sequence one. The longest row spans several partitions, so the
    state a chunk is read with is one the scan composed.
    """
    if chunk_gla is None:
        pytest.skip("FLA not installed")
    torch.manual_seed(2237)
    lengths = [1, 7, 63, 64, 100, 600]
    total, heads = sum(lengths), 2
    q, k = (torch.randn(1, total, heads, dim, device="cuda", dtype=dtype) * 0.1 for _ in range(2))
    v = torch.randn(1, total, heads, dim, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(1, total, heads, dim, device="cuda", dtype=dtype)
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int64, device="cuda")
    seeded = torch.randn(len(lengths), heads, dim, dim, device="cuda", dtype=torch.float32) * 0.1
    op = GLAFwdOp(scale)
    for state, host in ((seeded, cu_seqlens.cpu()), (None, None)):
        o, final_state = op(q, k, v, g, state, cu_seqlens, host)
        ref_o, ref_state = chunk_gla(
            q,
            k,
            v,
            g,
            scale=dim**-0.5 if scale is None else scale,
            initial_state=state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        compare_outputs((o, final_state), (ref_o, ref_state), inference_verification(dtype))


@pytest.mark.sm89
@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_gla_packed_varlen_refuses_offsets_99_kb_cannot_hold() -> None:
    """SM89 gives a block 99 KB of opt-in shared memory, and the prefill programs hold every
    sequence's offset, so 16384 sequences at head dim 128 are refused before anything is built."""
    sequences, heads, dim = 16384, 2, 128
    device = run_device()
    q, k, v = (
        torch.randn(1, 2 * sequences, heads, dim, device=device, dtype=torch.float16)
        for _ in range(3)
    )
    g = -torch.rand(1, 2 * sequences, heads, dim, device=device, dtype=torch.float16)
    cu_seqlens = torch.arange(0, 2 * sequences + 1, 2, dtype=torch.int64, device=device)
    op = GLAFwdOp()
    with pytest.raises(ValueError, match="shared memory per block for 16384 sequences"):
        op(q, k, v, g, None, cu_seqlens)
    for interface in GLAFwdOp.interfaces:
        assert not op.built_entries(interface)


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_gla_prefill_stays_finite_when_the_gate_outruns_a_split_exponent(
    dtype: torch.dtype,
) -> None:
    """A chunk's causal product is two factors whose exponents cancel.

    The signature bounds the gate nowhere, so a gate of ten per token drives one factor past
    the largest bfloat16 and the other below the smallest, and their product would be a NaN
    the cancelled exponent never has.
    """
    if chunk_gla is None:
        pytest.skip("FLA not installed")
    torch.manual_seed(2237)
    lengths = [100, 156]
    total, heads, dim = sum(lengths), 2, 64
    q, k = (torch.randn(1, total, heads, dim, device="cuda", dtype=dtype) * 0.1 for _ in range(2))
    v = torch.randn(1, total, heads, dim, device="cuda", dtype=dtype) * 0.1
    g = -torch.rand(1, total, heads, dim, device="cuda", dtype=dtype) * 10.0
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int64, device="cuda")
    o, final_state = GLAFwdOp()(q, k, v, g, None, cu_seqlens, None)
    ref_o, ref_state = chunk_gla(
        q, k, v, g, scale=dim**-0.5, output_final_state=True, cu_seqlens=cu_seqlens
    )
    assert torch.isfinite(o).all()
    compare_outputs((o, final_state), (ref_o, ref_state), inference_verification(dtype))


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
def test_gla_prefill_rows_shorter_than_a_whole_chunk_match_fla() -> None:
    """An equal-length call whose rows are not a multiple of 64 runs the packed kernel."""
    torch.manual_seed(2237)
    test = GLAFwdTest(2, 100, 4, 64, 64, torch.bfloat16, has_initial_state=True)
    op = GLAFwdOp()
    test.check(op, *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "dtype,has_initial_state,gate_scale",
    [(torch.bfloat16, True, 1.0), (torch.float16, False, 3.0)],
)
def test_gla_long_prefill_uses_partitioned_kernel(
    dtype: torch.dtype, has_initial_state: bool, gate_scale: float
) -> None:
    torch.manual_seed(2160)
    test = GLAFwdTest(2, 16384, 4, 64, 64, dtype, has_initial_state)
    _skip_unless_kernel_serves(GLADensePrefillPartitionedKernel, test)
    inputs = test.gen_inputs()
    inputs[3].mul_(gate_scale)
    op = GLAFwdOp()
    test.check(
        op,
        *inputs,
    )


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "dtype,dim,has_initial_state,scale",
    [
        (torch.float16, 128, True, None),
        (torch.bfloat16, 64, False, None),
        (torch.bfloat16, 64, True, 0.3),
    ],
)
def test_gla_dense_decode_matches_fla(
    dtype: torch.dtype, dim: int, has_initial_state: bool, scale: float | None
) -> None:
    torch.manual_seed(2174)
    test = GLAFwdTest(2, 1, 4, dim, dim, dtype, has_initial_state, scale)
    _skip_unless_kernel_serves(GLADenseDecodeFwdKernel, test)
    inputs = test.gen_inputs()
    op = GLAFwdOp(scale)
    # The workload checks FP32 state at 3e-7 and permits one output rounding unit.
    test.check(op, *inputs)


@pytest.mark.smoke
@pytest.mark.cuda_only
@pytest.mark.in_tree_kernels
def test_gla_dense_decode_steps_match_one_recurrence() -> None:
    """Feeding each step's final_state back matches one recurrence over all the steps."""
    torch.manual_seed(2174)
    steps = 8
    test = GLAFwdTest(2, steps, 4, 64, 64, torch.bfloat16, has_initial_state=True)
    _skip_unless_kernel_serves(GLADenseDecodeFwdKernel, GLAFwdTest(2, 1, 4, 64, 64, torch.bfloat16))
    q, k, v, g, state = test.gen_inputs()
    ref_o, ref_state = test.ref_program(q, k, v, g, state)
    op = GLAFwdOp()
    outputs = []
    for t in range(steps):
        o, state = op(*(x[:, t : t + 1] for x in (q, k, v, g)), state)
        outputs.append(o)
    compare_outputs((torch.cat(outputs, dim=1), state), (ref_o, ref_state), test.verification(q))


class GLADecodeTest(GLADecodeWorkload, TestBase):
    pass


class GLADecodeFixture(FixtureBase):
    PARAMS = [
        (
            "batch, heads, dim_k, dim_v, dtype, tune",
            [
                pytest.param(1, 4, 64, 64, torch.float32, False, marks=pytest.mark.smoke),
                pytest.param(1, 4, 64, 64, torch.float16, False, marks=pytest.mark.smoke),
                pytest.param(1, 4, 64, 64, torch.bfloat16, False, marks=pytest.mark.smoke),
                pytest.param(2, 8, 64, 64, torch.float32, False, marks=pytest.mark.full),
                pytest.param(2, 4, 128, 128, torch.float32, False, marks=pytest.mark.full),
                pytest.param(2, 8, 64, 64, torch.float16, False, marks=pytest.mark.full),
                pytest.param(2, 8, 64, 64, torch.bfloat16, False, marks=pytest.mark.full),
            ],
        ),
    ]


@GLADecodeFixture
def test_gla_decode(
    batch: int, heads: int, dim_k: int, dim_v: int, dtype: torch.dtype, tune: bool
) -> None:
    torch.manual_seed(42)
    test = GLADecodeTest(batch, heads, dim_k, dim_v, dtype)
    op = GLARecurrentFwdOp()
    if tune:
        op.request_tune()
    test.check(op, *test.gen_inputs())


@GLADecodeFixture
def test_gla_decode_multi_step(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """Test multiple sequential decode steps to verify state propagation."""
    torch.manual_seed(42)
    num_steps = 8
    B, H, DK, DV = batch, heads, dim_k, dim_v

    op = GLARecurrentFwdOp()
    if tune:
        op.request_tune()

    state_op = torch.zeros(B, H, DK, DV, device=run_device(), dtype=dtype)
    state_ref = torch.zeros(B, H, DK, DV, device=run_device(), dtype=dtype)

    for _ in range(num_steps):
        q = torch.randn(B, H, DK, device=run_device(), dtype=dtype) * 0.1
        k = torch.randn(B, H, DK, device=run_device(), dtype=dtype) * 0.1
        v = torch.randn(B, H, DV, device=run_device(), dtype=dtype) * 0.1
        gk = -torch.rand(B, H, DK, device=run_device(), dtype=dtype)

        o_ref, state_ref = gla_decode_torch(q, k, v, gk, state_ref)
        o_ref = o_ref.to(dtype)
        state_ref = state_ref.to(dtype)

        with torch.no_grad():
            o_op, state_op = op(q, k, v, gk, state_op)

        compare_outputs(o_op, o_ref, decode_verification(dtype))
        compare_outputs(state_op, state_ref, decode_verification(dtype))


@pytest.mark.cuda_only
@GLADecodeFixture
def test_gla_decode_vs_fla(
    batch: int,
    heads: int,
    dim_k: int,
    dim_v: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    """Compare TileOPs GLA decode against FLA fused_recurrent_gla with T=1."""
    if fused_recurrent_gla is None:
        pytest.skip("FLA not installed")

    torch.manual_seed(42)
    B, H, DK, DV = batch, heads, dim_k, dim_v
    scale = DK**-0.5

    q = torch.randn(B, H, DK, device=run_device(), dtype=dtype) * 0.1
    k = torch.randn(B, H, DK, device=run_device(), dtype=dtype) * 0.1
    v = torch.randn(B, H, DV, device=run_device(), dtype=dtype) * 0.1
    gk = -torch.rand(B, H, DK, device=run_device(), dtype=dtype)
    state = torch.randn(B, H, DK, DV, device=run_device(), dtype=dtype) * 0.1

    op = GLARecurrentFwdOp(scale=scale)
    if tune:
        op.request_tune()
    with torch.no_grad():
        o_tile, s_tile = op(q, k, v, gk, state)

    # FLA: needs BTHD layout with T=1
    # q [B,H,DK] -> [B,1,H,DK]
    q_fla = q.unsqueeze(1)
    k_fla = k.unsqueeze(1)
    v_fla = v.unsqueeze(1)
    gk_fla = gk.unsqueeze(1)

    o_fla, s_fla = fused_recurrent_gla(
        q_fla,
        k_fla,
        v_fla,
        gk=gk_fla,
        scale=scale,
        initial_state=state.contiguous(),
        output_final_state=True,
    )
    o_fla = o_fla.squeeze(1).to(dtype)

    compare_outputs((o_tile, s_tile), (o_fla, s_fla.to(dtype)), decode_verification(dtype))
