import functools
import math

import pytest
import torch

from tests.cuda_graph_contract import CapturedGraph
from tests.state_pool_guard import (
    check_padded_slots_skipped,
    check_strided_pool_in_place,
    pooled_call,
)
from tests.workload_test_base import TestBase
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention import GDNCall
from tileops.backend import BUILTIN, OpNotAvailableError, TensorSpec
from tileops.backend import _registry as registry
from tileops.linear_attention import GDNFwdOp
from workloads.device import run_device
from workloads.linear_attention.gdn import GDNFwdCall, GDNFwdWorkload
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call

# SM89 and SM90 prefill run different implementations. Each architecture's copy
# carries its marker, so ``-m sm89`` selects the SM89 copies.
PREFILL_ARCHITECTURES = pytest.mark.parametrize(
    "_architecture",
    [pytest.param(89, marks=pytest.mark.sm89), pytest.param(90, marks=pytest.mark.sm90)],
)


# Smoke runs the first state layout; the second is full.
STATE_LAYOUTS = [
    pytest.param(False, marks=pytest.mark.smoke, id="key-major"),
    pytest.param(True, marks=pytest.mark.full, id="value-major"),
]


class GDNFwdTest(GDNFwdWorkload, TestBase):
    pass


@pytest.mark.smoke
@pytest.mark.sm89
@pytest.mark.in_tree_kernels
def test_gdn_split_prefill_preserves_tail_state() -> None:
    """A sparse long scan must exercise the split implementation, including its tail."""
    sm_count = torch.cuda.get_device_properties(run_device()).multi_processor_count
    torch.manual_seed(42)
    test = GDNFwdTest(1, 16385, 1, 128, torch.bfloat16, value_heads=2, has_initial_state=True)
    op = GDNFwdOp()
    test.check(op, *test.gen_inputs())
    expected = "GDNSplitPrefillFwdKernel" if sm_count >= 64 else "GDNChunkPrefillFwdKernel"
    assert [type(kernel).__name__ for kernel in op.iter_kernels()] == [expected]


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@PREFILL_ARCHITECTURES
def test_gdn_dense_prefill_matches_reference(_architecture: int, dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    test = GDNFwdTest(1, 64, 2, 128, dtype)
    inputs = test.gen_inputs()
    op = GDNFwdOp()
    # The FP32 state must not inherit rounding of the chunk's cumulative log-gates.
    test.check(op, *inputs)
    reference = test.ref_program(*inputs)
    evidence = test.verification(*inputs)
    # Small outputs must not make either returned tensor optional to correctness.
    for cleared in (0, 1):
        faulty = list(reference)
        faulty[cleared] = torch.zeros_like(faulty[cleared])
        with pytest.raises(AssertionError):
            compare_outputs(tuple(faulty), reference, evidence)
        faulty[cleared] = reference[cleared] * 1.1
        with pytest.raises(AssertionError):
            compare_outputs(tuple(faulty), reference, evidence)


@pytest.mark.smoke
@PREFILL_ARCHITECTURES
def test_gdn_dense_prefill_continues_an_initial_state(_architecture: int) -> None:
    torch.manual_seed(42)
    test = GDNFwdTest(1, 64, 2, 128, torch.bfloat16, has_initial_state=True)
    test.check(GDNFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@PREFILL_ARCHITECTURES
def test_gdn_dense_prefill_carries_a_value_major_state(_architecture: int) -> None:
    """The caller's state is ``[N, HV, V, K]`` at both ends of the recurrence."""
    torch.manual_seed(42)
    test = GDNFwdTest(1, 64, 2, 128, torch.bfloat16, has_initial_state=True, state_v_first=True)
    op = GDNFwdOp(state_v_first=True)
    test.check(op, *test.gen_inputs())


@pytest.mark.sm89
@pytest.mark.parametrize(
    "length",
    [
        pytest.param(16, marks=pytest.mark.smoke, id="partial-subblock"),
        pytest.param(65, marks=pytest.mark.full, id="cross-chunk"),
    ],
)
def test_gdn_prefill_preserves_finite_strong_decay(length: int) -> None:
    """SM89 head-gate scaling must stay finite across partial blocks and chunks."""
    torch.manual_seed(42)
    log_decay = -4.0
    test = GDNFwdTest(
        batch=1,
        seq_len=length,
        heads=2,
        dim=128,
        dtype=torch.float16,
        has_initial_state=True,
    )
    inputs = test.gen_inputs()
    gate = torch.full_like(inputs[3], log_decay)
    test.check(GDNFwdOp(), *inputs[:3], gate, *inputs[4:])


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
def test_gdn_partitioned_prefill_carries_a_value_major_state() -> None:
    """The partition correction reads and writes the caller's layout, not the recurrence's."""
    torch.manual_seed(42)
    test = GDNFwdTest(1, 512, 2, 128, torch.bfloat16, has_initial_state=True, state_v_first=True)
    q, k, v, g, beta, *state = test.gen_inputs()
    # A gentle decay, so the state carried across partitions still reaches the output.
    _check_partitioned(test, q, k, v, g * 0.01, beta, *state, state_v_first=True)


@pytest.mark.smoke
@PREFILL_ARCHITECTURES
def test_gdn_prefill_packs_ragged_sequences_with_grouped_value_heads(_architecture: int) -> None:
    """Lengths below, across and on a chunk boundary, with four value heads per key head."""
    torch.manual_seed(42)
    test = GDNFwdTest(
        1,
        0,
        2,
        64,
        torch.bfloat16,
        has_initial_state=True,
        value_heads=8,
        sequence_lengths=(1, 63, 100, 192),
    )
    test.check(GDNFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.sm90
def test_gdn_prefill_packs_sequences_of_one_length() -> None:
    """Packed sequences of one length that is not a whole chunk, with grouped value heads."""
    torch.manual_seed(42)
    test = GDNFwdTest(1, 0, 2, 128, torch.bfloat16, value_heads=4, sequence_lengths=(100, 100, 100))
    test.check(GDNFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.sm90
def test_gdn_prefill_packs_many_sequences_shorter_than_a_chunk(monkeypatch) -> None:
    """The warp-group solve over bfloat16 sequences of one length, each a chunk whose last
    rows lie past the sequence, with grouped value heads and the step size from raw logits
    doubled."""
    from tileops._kernels.linear_attention.gdn import prefill

    solves = []
    solve = prefill.prefill_solve_warpgroup
    monkeypatch.setattr(
        prefill, "prefill_solve_warpgroup", lambda *a, **kw: solves.append(1) or solve(*a, **kw)
    )
    torch.manual_seed(42)
    test = GDNFwdTest(
        1,
        0,
        1,
        64,
        torch.bfloat16,
        value_heads=4,
        sequence_lengths=(40,) * 264,
        beta_sigmoid=True,
        allow_neg_eigval=True,
    )
    op = GDNFwdOp(use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True)
    test.check(op, *test.gen_inputs())
    assert solves


@pytest.mark.smoke
@pytest.mark.sm90
def test_gdn_prefill_keeps_busy_packed_sequences_whole(monkeypatch) -> None:
    """Packed sequences whose walks hold more than half the SMs run the one-kernel prefill."""
    from tileops._kernels.linear_attention.gdn import prefill

    calls = []
    fused = prefill.fused_gdr_prefill
    monkeypatch.setattr(
        prefill, "fused_gdr_prefill", lambda *a, **kw: calls.append(1) or fused(*a, **kw)
    )
    torch.manual_seed(42)
    test = GDNFwdTest(
        1, 0, 8, 128, torch.bfloat16, value_heads=32, sequence_lengths=(128, 128, 128, 100)
    )
    test.check(GDNFwdOp(), *test.gen_inputs())
    assert calls


@pytest.mark.smoke
@pytest.mark.sm90
def test_gdn_prefill_leaves_the_previous_sequence_to_an_empty_one() -> None:
    """An empty sequence writes no output, so the sequence before it keeps its last rows.

    The long first sequence's 64 value heads hold the SMs, so the empty sequence's CTAs start
    only after the short sequence's CTAs finish, the order in which a stray store shows."""
    torch.manual_seed(42)
    test = GDNFwdTest(1, 0, 16, 64, torch.bfloat16, value_heads=64, sequence_lengths=(2048, 64, 0))
    test.check(GDNFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@PREFILL_ARCHITECTURES
def test_gdn_prefill_reads_offsets_rewritten_in_place(_architecture: int) -> None:
    """The same tensors, with the boundary between two sequences moved, compute the new split."""
    torch.manual_seed(42)
    test = GDNFwdTest(
        1, 0, 2, 64, torch.bfloat16, has_initial_state=True, sequence_lengths=(1, 63, 100, 192)
    )
    inputs = test.gen_inputs()
    op = GDNFwdOp()
    test.check(op, *inputs)
    # A cache keyed by tensor identity would replay the first call's split here.
    inputs[6][2] = 66
    inputs[7][2] = 66
    test.check(op, *inputs)


def _check_correlated_keys(rows: int, heads: int, dim: int, dtype: torch.dtype) -> None:
    """One 64-token chunk a row of identical keys, step size 0.5 and no decay: every power of
    the chunk's strictly lower matrix reaches the inverse."""
    test = GDNFwdTest(rows, 64, heads, dim, dtype)
    q = torch.zeros(rows, 64, heads, dim, dtype=dtype, device=run_device())
    q[..., 0] = 1
    g = torch.zeros(rows, 64, heads, dtype=dtype, device=run_device())
    test.check(GDNFwdOp(), q, q.clone(), torch.ones_like(q), g, torch.full_like(g, 0.5))


def _fast_decay(test: GDNFwdTest) -> list:
    """*test*'s raw-gate inputs with every row decaying by exp(A_log) * softplus(g + dt_bias)
    = 16 * 0.1 nats."""
    inputs = test.gen_inputs()
    inputs[3].zero_()
    inputs[8].fill_(math.log(16.0))
    inputs[9].fill_(math.log(math.expm1(0.1)))
    return inputs


@pytest.mark.smoke
@PREFILL_ARCHITECTURES
def test_gdn_prefill_inverts_a_chunk_of_correlated_keys(_architecture: int) -> None:
    """Identical keys, step size 0.5 and no decay: every power of the chunk's strictly
    lower matrix reaches the inverse, and every off-diagonal block depends on another."""
    _check_correlated_keys(1, 1, 128, torch.float16)


@pytest.mark.smoke
@pytest.mark.sm90
def test_gdn_prefill_inverts_correlated_keys_in_the_warp_group_solve() -> None:
    """The warp-group solve's inverse on bfloat16 chunks of identical keys, step size 0.5
    and no decay, where every off-diagonal block depends on another. Keys 64 wide keep the
    call off the one-kernel prefill."""
    _check_correlated_keys(66, 16, 64, torch.bfloat16)


# Two rows of 64 value heads: 128 walks, enough to route a 128-wide call to the one-kernel
# prefill on SM90.
FUSED_ROWS, FUSED_HEADS = 2, 64


@pytest.mark.sm90
@pytest.mark.parametrize(
    ("dtype", "params", "flags"),
    [
        pytest.param(
            torch.bfloat16, dict(seq_len=100), {}, marks=pytest.mark.smoke, id="bf16-partial-chunk"
        ),
        pytest.param(
            torch.float16,
            dict(seq_len=128, has_initial_state=True, state_v_first=True),
            dict(state_v_first=True),
            marks=pytest.mark.smoke,
            id="fp16-value-major-state",
        ),
        pytest.param(
            torch.bfloat16,
            dict(seq_len=128, raw_gate=True, beta_sigmoid=True, allow_neg_eigval=True),
            dict(use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True, allow_neg_eigval=True),
            marks=pytest.mark.full,
            id="in-kernel-transforms",
        ),
    ],
)
def test_gdn_fused_prefill_matches_reference(dtype: torch.dtype, params: dict, flags: dict) -> None:
    """The one-kernel prefill: a row that ends inside a chunk, a float16 call continuing a
    value-major state, and the gate and step-size transforms taken in kernel."""
    torch.manual_seed(42)
    seq_len = params.pop("seq_len")
    test = GDNFwdTest(FUSED_ROWS, seq_len, 16, 128, dtype, value_heads=FUSED_HEADS, **params)
    test.check(GDNFwdOp(**flags), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.sm90
def test_gdn_fused_prefill_inverts_correlated_keys_in_float16() -> None:
    """The one-kernel prefill's float16 inverse on chunks of identical keys, step size 0.5 and
    no decay, where every off-diagonal block of the inverse depends on another."""
    _check_correlated_keys(FUSED_ROWS, FUSED_HEADS, 128, torch.float16)


@pytest.mark.smoke
@pytest.mark.sm90
def test_gdn_fused_prefill_keeps_a_fast_decay_finite_in_fp16() -> None:
    """A float16 raw gate decaying 1.6 nats a row: the one-kernel prefill's gate factors and
    e^gc underflow inside a chunk and must stay finite."""
    torch.manual_seed(42)
    test = GDNFwdTest(
        FUSED_ROWS, 64, 16, 128, torch.float16, value_heads=FUSED_HEADS, raw_gate=True
    )
    test.check(GDNFwdOp(use_gate_in_kernel=True), *_fast_decay(test))


@pytest.mark.smoke
@PREFILL_ARCHITECTURES
def test_gdn_prefill_runs_a_row_that_is_not_a_whole_chunk(_architecture: int) -> None:
    torch.manual_seed(42)
    test = GDNFwdTest(2, 100, 2, 64, torch.bfloat16)
    test.check(GDNFwdOp(), *test.gen_inputs())


@PREFILL_ARCHITECTURES
@pytest.mark.parametrize(
    ("l2norm", "raw_gate", "beta_sigmoid", "allow_neg_eigval", "dtype"),
    [
        pytest.param(
            True, True, True, True, torch.bfloat16, marks=pytest.mark.smoke, id="every-transform"
        ),
        pytest.param(
            True,
            True,
            True,
            True,
            torch.float16,
            marks=pytest.mark.smoke,
            id="every-transform-fp16",
        ),
        pytest.param(
            True, False, False, False, torch.bfloat16, marks=pytest.mark.full, id="l2norm"
        ),
        pytest.param(
            False, True, False, False, torch.bfloat16, marks=pytest.mark.full, id="raw-gate"
        ),
        pytest.param(
            False, False, True, False, torch.bfloat16, marks=pytest.mark.full, id="beta-sigmoid"
        ),
        pytest.param(
            False,
            False,
            True,
            True,
            torch.bfloat16,
            marks=pytest.mark.full,
            id="beta-sigmoid-negative",
        ),
    ],
)
def test_gdn_prefill_takes_each_input_transform(
    _architecture: int,
    l2norm: bool,
    raw_gate: bool,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    dtype: torch.dtype,
) -> None:
    """Each transform the op may leave to the kernel, alone and all together."""
    torch.manual_seed(42)
    test = GDNFwdTest(
        1,
        128,
        2,
        64,
        dtype,
        l2norm=l2norm,
        raw_gate=raw_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
    )
    op = GDNFwdOp(
        use_qk_l2norm_in_kernel=l2norm,
        use_gate_in_kernel=raw_gate,
        use_beta_sigmoid_in_kernel=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
    )
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    if l2norm and beta_sigmoid:
        # Correlated keys expose a truncated triangular inverse that small random
        # inner products hide. Reuse the same contract and compiled kernel.
        inputs[1].copy_(inputs[1][:, :1].expand_as(inputs[1]).clone())
        inputs[4].fill_(-2)
        test.check(op, *inputs)


@pytest.mark.sm89
@pytest.mark.parametrize(
    ("seq_len", "dtype"),
    [
        pytest.param(4096, torch.float16, marks=pytest.mark.smoke, id="16-columns-fp16"),
        pytest.param(4096, torch.bfloat16, marks=pytest.mark.smoke, id="16-columns-bf16"),
        pytest.param(16385, torch.bfloat16, marks=pytest.mark.full, id="32-columns-bf16"),
    ],
)
def test_gdn_sm89_prefill_scans_a_long_sequence_in_narrow_value_tiles(
    seq_len: int, dtype: torch.dtype
) -> None:
    """One long sequence over few value heads gives SM89 too few blocks to fill its SMs,
    so the scan splits each head's value columns into narrow tiles."""
    torch.manual_seed(42)
    test = GDNFwdTest(1, seq_len, 16, 128, dtype, has_initial_state=True)
    test.check(GDNFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
@PREFILL_ARCHITECTURES
def test_gdn_prefill_keeps_a_fast_decay_finite_in_fp16(_architecture: int) -> None:
    """A float16 raw gate whose decay leaves float16's range within one chunk.

    Every row decays by 1.6 nats, so a 16-row block decays by more than 2**24 and eight
    rows by more than 2**16: past what a float16 decay or a float16 key tile can hold.
    """
    torch.manual_seed(42)
    test = GDNFwdTest(1, 64, 2, 128, torch.float16, raw_gate=True)
    test.check(GDNFwdOp(use_gate_in_kernel=True), *_fast_decay(test))


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
def test_gdn_partitioned_prefill_normalizes_the_key_it_stages() -> None:
    """The warmup pass stages the key itself, so partitioning normalizes it a second time."""
    torch.manual_seed(42)
    test = GDNFwdTest(1, 512, 2, 128, torch.bfloat16, l2norm=True)
    q, k, v, g, beta = test.gen_inputs()
    _check_partitioned(test, q, k, v, g * 0.01, beta, use_qk_l2norm_in_kernel=True)


def _check_partitioned(test, *inputs, **params) -> None:
    """Check a call the in-tree prefill walks in partitions."""
    test.check(GDNFwdOp(**params, target=BUILTIN), *inputs)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "has_initial_state",
    [
        pytest.param(True, marks=pytest.mark.smoke, id="continued"),
        pytest.param(False, marks=pytest.mark.full, id="from-zero"),
    ],
)
def test_gdn_partitioned_dense_prefill_matches_reference(
    has_initial_state: bool,
) -> None:
    """Exercise warmup, state correction, and partitioned forward together."""
    torch.manual_seed(42)
    test = GDNFwdTest(1, 512, 2, 128, torch.bfloat16, has_initial_state=has_initial_state)
    q, k, v, g, beta, *state = test.gen_inputs()
    # A gentle decay, so the state carried across partitions still reaches the output.
    _check_partitioned(test, q, k, v, g * 0.01, beta, *state)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
def test_gdn_partitioned_prefill_ends_on_a_short_piece() -> None:
    """A width that does not divide the sequence's 45 chunks leaves its last piece short, and
    the last chunk is partial."""
    torch.manual_seed(42)
    test = GDNFwdTest(1, 45 * 64 - 20, 2, 128, torch.bfloat16, value_heads=8)
    q, k, v, g, beta = test.gen_inputs()
    _check_partitioned(test, q, k, v, g * 0.01, beta)


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
def test_gdn_partitioned_prefill_splits_only_the_long_packed_sequences() -> None:
    """Packed sequences of 32, 15 and one chunk: the long two are split and the short ones
    walked whole, under a decay strong enough that warmup stops before a piece's start.
    Without host offsets the same call walks every sequence whole."""
    torch.manual_seed(42)
    test = GDNFwdTest(
        1, 0, 2, 64, torch.float16, value_heads=8, sequence_lengths=(2003, 59, 937, 1)
    )
    inputs = test.gen_inputs()
    _check_partitioned(test, *inputs)
    test.check(GDNFwdOp(target=BUILTIN), *inputs[:7], None)


@pytest.mark.parametrize(
    ("batch", "dtype"),
    [
        pytest.param(8, torch.float16, marks=pytest.mark.smoke, id="b8-fp16"),
        pytest.param(8, torch.bfloat16, marks=pytest.mark.smoke, id="b8-bf16"),
    ],
)
def test_gdn_dense_decode_matches_reference(batch: int, dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    test = GDNFwdTest(batch, 1, 16, 128, dtype, has_initial_state=True)
    test.check(GDNFwdOp(), *test.gen_inputs())


@pytest.mark.parametrize(
    "flags",
    [
        pytest.param({"state_v_first": True}, marks=pytest.mark.smoke, id="v-first"),
        pytest.param({"use_gate_in_kernel": True}, marks=pytest.mark.full, id="gate-fused"),
        pytest.param(
            {"use_beta_sigmoid_in_kernel": True, "allow_neg_eigval": True},
            marks=pytest.mark.full,
            id="beta-sigmoid-neg",
        ),
        pytest.param({"use_qk_l2norm_in_kernel": True}, marks=pytest.mark.full, id="l2norm"),
    ],
)
def test_gdn_decode_runs_each_recurrence_flag(flags: dict) -> None:
    torch.manual_seed(42)
    workload_flags = {
        "state_v_first": "state_v_first",
        "use_gate_in_kernel": "raw_gate",
        "use_beta_sigmoid_in_kernel": "beta_sigmoid",
        "allow_neg_eigval": "allow_neg_eigval",
        "use_qk_l2norm_in_kernel": "l2norm",
    }
    test = GDNFwdTest(
        2,
        1,
        4,
        128,
        torch.bfloat16,
        has_initial_state=True,
        **{workload_flags[name]: value for name, value in flags.items()},
    )
    test.check(GDNFwdOp(**flags), *test.gen_inputs())


@pytest.mark.smoke
def test_gdn_decode_starts_a_value_major_state_from_zero() -> None:
    """A value-major step without a starting state writes the state it forms, k v^T, in the
    16-byte runs a value-major row moves in."""
    torch.manual_seed(42)
    test = GDNFwdTest(2, 1, 4, 128, torch.bfloat16, state_v_first=True)
    test.check(GDNFwdOp(state_v_first=True), *test.gen_inputs())


@pytest.mark.smoke
def test_gdn_decode_groups_value_heads_over_a_64_wide_state() -> None:
    """A batch and head counts that are neither powers of two nor warp multiples."""
    torch.manual_seed(42)
    test = GDNFwdTest(17, 1, 6, 64, torch.bfloat16, has_initial_state=True, value_heads=12)
    test.check(GDNFwdOp(), *test.gen_inputs())


@pytest.mark.smoke
def test_gdn_dense_decode_propagates_fp32_state() -> None:
    torch.manual_seed(42)
    workload = GDNFwdWorkload(
        1,
        1,
        16,
        128,
        torch.bfloat16,
        has_initial_state=True,
    )
    q, k, v, g, beta, state = workload.gen_inputs()
    expected_state = state.clone()
    op = GDNFwdOp()
    for _ in range(4):
        expected_o, expected_state = workload.ref_program(q, k, v, g, beta, expected_state)
        got_o, state = op(q, k, v, g, beta, state)
        compare_outputs(
            (got_o, state),
            (expected_o, expected_state),
            workload.verification(q, k, v, g, beta, state),
        )


@pytest.mark.smoke
def test_gdn_contract_reaches_target_builder(isolated_registry) -> None:
    calls = []

    def build_kernel(*inputs, **params):
        calls.append((inputs, params))

        def kernel(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            cu_seqlens_cpu,
            A_log,
            dt_bias,
            state_indices,
            num_accepted_tokens,
        ):
            del k, g, beta, initial_state, cu_seqlens_cpu, A_log, dt_bias
            del state_indices, num_accepted_tokens
            batch, seq_len, _heads, dim_k = q.shape
            value_heads, dim_v = v.shape[2:]
            state_batch = cu_seqlens.shape[0] - 1
            return (
                torch.empty(batch, seq_len, value_heads, dim_v, dtype=q.dtype),
                torch.empty(state_batch, value_heads, dim_k, dim_v, dtype=torch.float32),
            )

        return kernel

    registry.register_kernel_builder("GDNFwdOp", "gdn_test", build_kernel)

    batch, seq_len, heads, value_heads, dim_k, dim_v = 1, 7, 2, 4, 8, 6
    q = torch.randn(batch, seq_len, heads, dim_k, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(batch, seq_len, value_heads, dim_v, dtype=torch.float16)
    g = torch.randn(batch, seq_len, value_heads, dtype=torch.float16)
    beta = torch.randn_like(g)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int64)
    cu_seqlens_cpu = cu_seqlens.clone()
    initial_state = torch.randn(2, value_heads, dim_k, dim_v, dtype=torch.float32)
    A_log = torch.randn(value_heads, dtype=torch.float32)
    dt_bias = torch.randn(value_heads, dtype=torch.float32)

    op = GDNFwdOp(
        scale=0.125,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        target="gdn_test",
    )
    o, final_state = op(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        cu_seqlens_cpu,
        A_log,
        dt_bias,
    )

    assert o.shape == (batch, seq_len, value_heads, dim_v)
    assert final_state.shape == (2, value_heads, dim_k, dim_v)
    assert calls == [
        (
            tuple(
                TensorSpec.of(tensor)
                for tensor in (
                    q,
                    k,
                    v,
                    g,
                    beta,
                    initial_state,
                    cu_seqlens,
                    cu_seqlens_cpu,
                    A_log,
                    dt_bias,
                )
            )
            + (None, None),
            {
                "scale": 0.125,
                "use_qk_l2norm_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "allow_neg_eigval": True,
                "state_v_first": False,
                "use_gate_in_kernel": True,
                "validate_inputs": False,
            },
        )
    ]


class GDNFwdCallTest(GDNFwdCall, TestBase):
    pass


_pooled = functools.partial(pooled_call, "GDNFwdOp", GDNFwdCallTest)


@pytest.mark.parametrize(
    ("dtype", "dim", "flags"),
    [
        pytest.param(torch.float16, 128, {}, marks=pytest.mark.smoke, id="fp16-key-major"),
        pytest.param(
            torch.bfloat16,
            128,
            {"state_v_first": True, "use_qk_l2norm_in_kernel": True},
            marks=pytest.mark.smoke,
            id="bf16-value-major",
        ),
        pytest.param(
            torch.bfloat16,
            64,
            {"state_v_first": True},
            marks=pytest.mark.full,
            id="bf16-value-major-64",
        ),
    ],
)
def test_gdn_decode_updates_its_slots_of_a_state_pool(
    dtype: torch.dtype, dim: int, flags: dict
) -> None:
    """A decode step reads each sequence's state from its slot and writes it back in place,
    in either state layout; a 64-wide value-major row takes half the lanes of a 128-wide one."""
    torch.manual_seed(42)
    test = _pooled(dtype, B=5, S=1, H=2, HV=4, K=dim, V=dim, P=16, **flags)
    test.check(GDNFwdOp(**flags), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.parametrize("state_v_first", STATE_LAYOUTS)
def test_gdn_prefill_updates_its_slots_of_a_state_pool(state_v_first: bool) -> None:
    """Packed sequences walked whole start from and end in their slots, in either state layout,
    with Q/K normalized in kernel and grouped value heads, the Qwen3.6 form; an empty sequence
    leaves its slot as it was."""
    torch.manual_seed(42)
    flags = {"state_v_first": state_v_first, "use_qk_l2norm_in_kernel": True}
    test = _pooled(
        torch.bfloat16, some=("cu_seqlens",), B=1, S=201, seq_lens=[100, 0, 37, 64], H=2, HV=4,
        K=128, V=128, P=12, **flags,
    )  # fmt: skip
    test.check(GDNFwdOp(**flags), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.parametrize("state_v_first", STATE_LAYOUTS)
def test_gdn_fused_prefill_updates_its_slots_of_a_state_pool(state_v_first: bool) -> None:
    """The one-kernel prefill reads and writes the states through their slots, in either
    layout."""
    torch.manual_seed(42)
    test = _pooled(
        torch.float16, B=FUSED_ROWS, S=100, H=16, HV=FUSED_HEADS, K=128, V=128, P=8,
        state_v_first=state_v_first,
    )  # fmt: skip
    test.check(GDNFwdOp(state_v_first=state_v_first), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("state_v_first", STATE_LAYOUTS)
def test_gdn_partitioned_prefill_updates_its_slots_of_a_state_pool(state_v_first: bool) -> None:
    """A walk split in partitions starts each partition from the slot's state and writes the
    last partition's state back to the slot, in either state layout."""
    torch.manual_seed(42)
    test = _pooled(
        torch.bfloat16, B=1, S=512, H=2, HV=2, K=128, V=128, P=4, state_v_first=state_v_first
    )
    q, k, v, g, beta, *rest = test.gen_inputs()
    # A decay too gentle for the warmup to forget the slot's state, so the correction carries
    # it into every partition rather than only the first.
    _check_partitioned(test, q, k, v, g * 1e-4, beta, *rest, state_v_first=state_v_first)


@pytest.mark.smoke
def test_gdn_decode_updates_a_strided_pool_in_place() -> None:
    """A pool that is a strided view is advanced in place, the call returning that view."""
    torch.manual_seed(42)
    check_strided_pool_in_place(
        _pooled(torch.bfloat16, B=4, S=1, H=2, HV=4, K=128, V=128, P=8), GDNFwdOp()
    )


def _columns(tensors: list[torch.Tensor], pad: int) -> list[torch.Tensor]:
    """*tensors*, each ``(B, S, *row)``, copied into the columns of one ``(B * S, width)``
    tensor, as a fused projection lays them out: each starts on a 16-byte boundary, and the
    rows hold *pad* columns past the last. Returns views of those columns."""
    first = tensors[0]
    widths = [math.prod(t.shape[2:]) for t in tensors]
    starts = [sum(-(-w // 8) * 8 for w in widths[:i]) for i in range(len(widths))]
    wide = torch.randn(
        first.shape[0] * first.shape[1], starts[-1] + widths[-1] + pad, device=first.device
    ).to(first.dtype)
    views = [
        wide[:, a : a + w].view(t.shape) for t, a, w in zip(tensors, starts, widths, strict=True)
    ]
    for view, t in zip(views, tensors, strict=True):
        view.copy_(t)
    return views


_QWEN_FLAGS = {
    "use_qk_l2norm_in_kernel": True,
    "use_beta_sigmoid_in_kernel": True,
    "use_gate_in_kernel": True,
}


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    ("row", "flags", "pads"),
    [
        # Qwen3.8-27B: q, k, v the 10240 columns of the convolution output; beta and g
        # columns 0 and 48 of the 16480-wide input projection; block solve, whole walk.
        pytest.param(
            {"S": 1000, "H": 16, "HV": 48, "P": 2}, _QWEN_FLAGS, (0, 16384),
            marks=pytest.mark.smoke, id="qwen3.8-27b",
        ),
        # A split walk: the warmup and state passes read k, v and beta in place.
        pytest.param(
            {"S": 8192, "H": 16, "HV": 48, "P": 1}, _QWEN_FLAGS, (8, 16),
            marks=pytest.mark.full, id="partitioned",
        ),
        pytest.param(
            {"S": 4096, "H": 16, "HV": 48, "P": 1}, {"use_gate_in_kernel": True}, (8, 16),
            marks=pytest.mark.full, id="warp-group-solve",
        ),
        # The one-kernel prefill reads packed copies.
        pytest.param(
            {"S": 4096, "H": 16, "HV": 128, "P": 1}, {}, (8, 16), marks=pytest.mark.full,
            id="one-kernel-prefill",
        ),
    ],
)  # fmt: skip
def test_gdn_prefill_reads_projection_columns_in_place(row: dict, flags: dict, pads) -> None:
    """q, k and v as columns of one wide tensor, g and beta of another: outputs and states
    match the reference, and neither wide tensor changes."""
    torch.manual_seed(42)
    some = ("A_log", "dt_bias") if flags.get("use_gate_in_kernel") else ()
    test = _pooled(torch.bfloat16, some=some, B=1, K=128, V=128, **row, **flags)
    q, k, v, g, beta, *rest = test.gen_inputs()
    q, k, v = _columns([q, k, v], pads[0])
    beta, g = _columns([beta, g], pads[1])
    wide = (q._base, beta._base)
    before = tuple(t.clone() for t in wide)
    op = GDNFwdOp(**test.call.arguments())

    def run(*args, **keywords):
        out = op(*args, **keywords)
        assert all(torch.equal(t, b) for t, b in zip(wide, before, strict=True))
        return out

    test.check(op, q, k, v, g, beta, *rest, runs=run)


_COLUMN_FLAGS = {"use_qk_l2norm_in_kernel": True, "use_beta_sigmoid_in_kernel": True}


@pytest.mark.parametrize(
    ("row", "flags", "layout"),
    [
        # Tokens 10244 elements apart, off 16-byte alignment.
        pytest.param(
            {"B": 1, "S": 1000, "P": 2}, _QWEN_FLAGS, "unaligned",
            marks=(pytest.mark.smoke, pytest.mark.sm90), id="prefill",
        ),
        # The SM89 chunk prefill copies at its entry.
        pytest.param(
            {"B": 1, "S": 200}, _QWEN_FLAGS, "unaligned",
            marks=(pytest.mark.full, pytest.mark.sm89), id="unpooled-prefill",
        ),
        # q's heads 136 elements apart: a token's heads are not packed.
        pytest.param(
            {"B": 1, "S": 200, "P": 4}, _QWEN_FLAGS, "padded-heads",
            marks=(pytest.mark.full, pytest.mark.sm90), id="padded-heads",
        ),
        # v expanded: its tokens zero elements apart.
        pytest.param(
            {"B": 1, "S": 200, "P": 4}, _QWEN_FLAGS, "expanded",
            marks=(pytest.mark.full, pytest.mark.sm90), id="expanded",
        ),
        # The dense decode and, in batch order, the column decode read packed copies.
        pytest.param(
            {"B": 3, "S": 1, "P": 4}, _QWEN_FLAGS, "unaligned", marks=pytest.mark.full,
            id="decode",
        ),
        pytest.param(
            {"B": 2, "S": 1, "HV": 16}, _COLUMN_FLAGS, "unaligned",
            marks=(pytest.mark.full, pytest.mark.sm90), id="column-decode",
        ),
    ],
)  # fmt: skip
def test_gdn_copies_projection_columns_it_cannot_read_in_place(
    row: dict, flags: dict, layout: str
) -> None:
    """Inputs a kernel does not read in place are copied first: outputs and states match the
    reference."""
    torch.manual_seed(42)
    row = {"H": 16, "HV": 48, "K": 128, "V": 128, **row}
    some = ("A_log", "dt_bias") if flags.get("use_gate_in_kernel") else ()
    if "P" in row:
        test = _pooled(torch.bfloat16, some=some, **row, **flags)
    else:
        call = manifest_call(
            "GDNFwdOp", {"T": "bfloat16"}, some=["initial_state", *some], **row, **flags
        )
        test = GDNFwdCallTest(call)
    q, k, v, g, beta, *rest = test.gen_inputs()
    if layout == "unaligned":
        q, k, v = _columns([q, k, v], 4)
        beta, g = _columns([beta, g], 3)
    elif layout == "padded-heads":
        wide = torch.randn(*q.shape[:-1], q.shape[-1] + 8, device=q.device).to(q.dtype)
        q = wide[..., : q.shape[-1]].copy_(q)
    else:
        v = v[:, :1].clone().expand(v.shape)
    test.check(GDNFwdOp(**test.call.arguments()), q, k, v, g, beta, *rest)


@pytest.mark.sm90
@pytest.mark.cuda_graph(op="GDNFwdOp")
@pytest.mark.smoke
def test_gdn_graph_replay() -> None:
    """Qwen3.8-27B's decode call, one token on a one-slot pool with the gate, step size and Q/K
    norm formed in kernel. Each of two replays advances the state the replay before it left in
    the pool, so the test checks each against the state before it; both hold the one slot
    index."""
    test = _pooled(
        torch.bfloat16, some=("A_log", "dt_bias"), B=1, S=1, H=16, HV=48, K=128, V=128, P=1,
        use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, use_gate_in_kernel=True,
    )  # fmt: skip
    op = GDNFwdOp(**test.call.arguments())
    inputs = test.gen_inputs()
    graph = CapturedGraph((op, inputs))
    graph.renew(test.gen_inputs())
    for _ in range(2):
        before = [*inputs[:5], inputs[5].clone(), *inputs[6:]]
        graph.replay(held=[inputs[10]])
        test.check(op, *before, runs=lambda *args, **kwargs: graph.outputs[0])


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_gdn_refuses_a_pooled_call_no_kernel_serves():
    """Dispatch on SM89 facts refuses the pooled prefill before a build: the SM89 chunk
    prefill reads and writes the states in batch order."""
    (in_tree,) = Kernel.devices
    call = GDNCall(
        arch=89,
        sm_count=128,
        smem_budget=101376,
        batch=1,
        seq_len=256,
        heads=2,
        value_heads=4,
        dim_k=128,
        dim_v=128,
        dtype=torch.bfloat16,
        scale=128**-0.5,
        has_initial_state=True,
        varlen=True,
        num_sequences=2,
        pooled=True,
        device=torch.device(in_tree),
    )
    with pytest.raises(OpNotAvailableError, match="a state pool is not served"):
        GDNFwdOp().key_for("gdn", call)


@pytest.mark.parametrize(
    ("dtype", "dim", "drafts", "flags", "row"),
    [
        pytest.param(
            torch.bfloat16,
            128,
            8,
            {"state_v_first": True, "use_qk_l2norm_in_kernel": True},
            {"B": 1, "S": 21, "seq_lens": [8, 5, 0, 1, 7], "some": ("cu_seqlens",)},
            marks=pytest.mark.smoke,
            id="packed-two-chunks",
        ),
        pytest.param(
            torch.float16,
            64,
            4,
            {},
            {"B": 3, "S": 3},
            marks=pytest.mark.smoke,
            id="fp16-key-major-64",
        ),
        pytest.param(
            torch.bfloat16,
            128,
            4,
            {},
            {"B": 64, "S": 3},
            marks=pytest.mark.full,
            id="key-major-one-warp",
        ),
        pytest.param(
            torch.bfloat16,
            128,
            4,
            {"use_gate_in_kernel": True, "use_beta_sigmoid_in_kernel": True},
            {"B": 2, "S": 4, "some": ("A_log", "dt_bias")},
            marks=pytest.mark.full,
            id="raw-gate-and-step",
        ),
    ],
)
def test_gdn_decode_verifies_drafts_from_the_accepted_slot(
    dtype: torch.dtype, dim: int, drafts: int, flags: dict, row: dict
) -> None:
    """With num_accepted_tokens, each sequence starts from the slot of its last accepted token,
    a random one of its drafts, and leaves the state after each of its tokens in that token's
    slot, every other slot keeping its state: the Qwen3.6 form, packed with full, partial,
    one-token and empty sequences of eight drafts, more than the block stages at once; equal
    length over a key-major 64-wide state; a key-major grid large enough for one-warp blocks,
    which stage in several passes; and the raw gate and step size formed each token."""
    torch.manual_seed(42)
    row = dict(row)
    some = ("num_accepted_tokens", *row.pop("some", ()))
    sequences = len(row["seq_lens"]) if "seq_lens" in row else row["B"]
    test = _pooled(
        dtype, some=some, H=2, HV=4, K=dim, V=dim, P=sequences * drafts, L=drafts, **row, **flags
    )
    inputs = list(test.gen_inputs())
    accepted = inputs[11]
    inputs[11] = torch.randint(
        1, drafts + 1, accepted.shape, dtype=accepted.dtype, device=accepted.device
    )
    test.check(GDNFwdOp(**flags), *inputs)


_VALUE_MAJOR = {"state_v_first": True}
_ROLLBACK = {"B": 3, "S": 3, "P": 12, "L": 4, "some": ("num_accepted_tokens",)}


@pytest.mark.parametrize(
    ("dtype", "flags", "row", "padded"),
    [
        pytest.param(
            torch.float16,
            {},
            {"B": 5, "S": 1, "P": 16},
            [1, 3],
            marks=pytest.mark.smoke,
            id="decode-key-major",
        ),
        pytest.param(
            torch.bfloat16,
            _VALUE_MAJOR,
            {"B": 5, "S": 1, "P": 16},
            [1, 3],
            marks=pytest.mark.smoke,
            id="decode-value-major",
        ),
        pytest.param(
            torch.float16,
            {},
            _ROLLBACK,
            [0, (1, 2)],
            marks=pytest.mark.full,
            id="rollback-key-major",
        ),
        pytest.param(
            torch.bfloat16,
            _VALUE_MAJOR,
            _ROLLBACK,
            [0, (1, 2)],
            marks=pytest.mark.full,
            id="rollback-value-major",
        ),
        pytest.param(
            torch.bfloat16,
            _VALUE_MAJOR,
            {"B": 1, "S": 201, "seq_lens": [100, 0, 37, 64], "P": 12, "some": ("cu_seqlens",)},
            [0, 2],
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="prefill",
        ),
        pytest.param(
            torch.float16,
            {},
            {"B": FUSED_ROWS, "S": 100, "HV": FUSED_HEADS, "P": 8},
            [1],
            marks=[pytest.mark.full, pytest.mark.sm90],
            id="fused-prefill",
        ),
        pytest.param(
            torch.bfloat16,
            {"target": BUILTIN},
            {"B": 1, "S": 512, "HV": 2, "P": 4},
            [0],
            marks=[pytest.mark.full, pytest.mark.sm90, pytest.mark.in_tree_kernels],
            id="partitioned-prefill",
        ),
    ],
)
def test_gdn_skips_the_states_of_negative_slots(
    dtype: torch.dtype, flags: dict, row: dict, padded: list
) -> None:
    """Each program that reads a pool writes no state for a slot of -1: the pool's
    neighbours and the slots of the padded sequences keep their states, every other slot ends as
    in the unpadded call. Under rollback a padded start slot skips the sequence and a padded
    token slot skips that token's store."""
    torch.manual_seed(42)
    row = {"H": 2, "HV": 4, "K": 128, "V": 128, **row}
    some = row.pop("some", ())
    layout = {key: value for key, value in flags.items() if key != "target"}
    inputs = _pooled(dtype, some=some, **row, **layout).gen_inputs()
    check_padded_slots_skipped(GDNFwdOp(**flags, validate_inputs=True), inputs, 5, 10, padded)


@pytest.mark.sm90
@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "batch,heads,allow_neg_eigval",
    [
        pytest.param(1, 8, False, marks=pytest.mark.smoke, id="one-request-eight-heads"),
        pytest.param(4, 2, True, marks=pytest.mark.full, id="four-requests-two-heads"),
    ],
)
def test_gdn_column_decode_uses_the_call_gate_and_beta_shapes(batch, heads, allow_neg_eigval):
    """The Hopper decode reads the given gate and beta for each request and head."""
    from tileops._kernels.linear_attention import GDNColumnDecodeKernel

    test = GDNFwdTest(
        batch,
        1,
        heads,
        128,
        torch.bfloat16,
        has_initial_state=True,
        l2norm=True,
        beta_sigmoid=True,
        allow_neg_eigval=allow_neg_eigval,
    )
    op = GDNFwdOp(
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=allow_neg_eigval,
        target=BUILTIN,
    )
    inputs = test.gen_inputs()
    test.check(op, *inputs)
    [kernel] = op.iter_kernels()
    assert type(kernel) is GDNColumnDecodeKernel
    with pytest.raises(ValueError, match="runs on"):
        kernel.forward(*(tensor.cpu() for tensor in inputs[:6]))
