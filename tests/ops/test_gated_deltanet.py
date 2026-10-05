import pytest
import torch

from tests.test_base import TestBase
from tileops.backend import BUILTIN, TensorSpec, registry
from tileops.kernels.linear_attention import GatedDeltaNetDensePrefillFwdKernel
from tileops.ops import GatedDeltaNetFwdOp
from workloads.linear_attention.gated_deltanet import GatedDeltaNetFwdWorkload
from workloads.numerics import compare_outputs

pytestmark = pytest.mark.smoke


class GatedDeltaNetFwdTest(GatedDeltaNetFwdWorkload, TestBase):
    pass


@pytest.fixture(autouse=True)
def isolated_registry():
    state = registry.snapshot()
    registry.DETECTORS.clear()
    registry.BUILDERS.clear()
    registry.LOAD_FAILURES.clear()
    registry.default_target = None
    registry._loaded = True
    yield
    registry.restore(state)


@pytest.mark.sm90
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_gated_deltanet_dense_prefill_matches_reference(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(1, 64, 2, 128, dtype)
    inputs = test.gen_inputs()
    op = GatedDeltaNetFwdOp()
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


@pytest.mark.sm90
def test_gated_deltanet_dense_prefill_continues_an_initial_state() -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(1, 64, 2, 128, torch.bfloat16, has_initial_state=True)
    test.check(GatedDeltaNetFwdOp(), *test.gen_inputs())


@pytest.mark.sm90
def test_gated_deltanet_dense_prefill_carries_a_value_major_state() -> None:
    """The caller's state is ``[N, HV, V, K]`` at both ends of the recurrence."""
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(
        1, 64, 2, 128, torch.bfloat16, has_initial_state=True, state_v_first=True
    )
    op = GatedDeltaNetFwdOp(state_v_first=True)
    test.check(op, *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.cuda_only
def test_gated_deltanet_partitioned_prefill_carries_a_value_major_state() -> None:
    """The partition correction reads and writes the caller's layout, not the recurrence's."""
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(
        1, 512, 2, 128, torch.bfloat16, has_initial_state=True, state_v_first=True
    )
    q, k, v, g, beta, *state = (tensor.to("cuda") for tensor in test.gen_inputs())
    # A gentle decay, so the state carried across partitions still reaches the output.
    _check_partitioned(test, q, k, v, g * 0.01, beta, *state, state_v_first=True)


@pytest.mark.sm90
def test_gated_deltanet_dense_prefill_runs_a_64_wide_state() -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(1, 64, 2, 64, torch.bfloat16)
    test.check(GatedDeltaNetFwdOp(), *test.gen_inputs())


@pytest.mark.sm90
def test_gated_deltanet_prefill_packs_ragged_sequences_with_grouped_value_heads() -> None:
    """Lengths below, across and on a chunk boundary, with four value heads per key head."""
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(
        1,
        0,
        2,
        64,
        torch.bfloat16,
        has_initial_state=True,
        value_heads=8,
        sequence_lengths=(1, 63, 100, 192),
    )
    test.check(GatedDeltaNetFwdOp(), *test.gen_inputs())


@pytest.mark.sm90
def test_gated_deltanet_prefill_reads_offsets_rewritten_in_place() -> None:
    """The same tensors, with the boundary between two sequences moved, compute the new split."""
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(
        1, 0, 2, 64, torch.bfloat16, has_initial_state=True, sequence_lengths=(1, 63, 100, 192)
    )
    inputs = test.gen_inputs()
    op = GatedDeltaNetFwdOp()
    test.check(op, *inputs)
    # A cache keyed by tensor identity would replay the first call's split here.
    inputs[6][2] = 66
    inputs[7][2] = 66
    test.check(op, *inputs)


@pytest.mark.sm90
def test_gated_deltanet_prefill_runs_a_row_that_is_not_a_whole_chunk() -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(2, 100, 2, 64, torch.bfloat16)
    test.check(GatedDeltaNetFwdOp(), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.parametrize(
    ("l2norm", "raw_gate", "beta_sigmoid", "allow_neg_eigval", "dtype"),
    [
        (True, False, False, False, torch.bfloat16),
        (False, True, False, False, torch.bfloat16),
        (False, False, True, False, torch.bfloat16),
        (False, False, True, True, torch.bfloat16),
        (True, True, True, True, torch.bfloat16),
        (True, True, True, True, torch.float16),
    ],
    ids=[
        "l2norm",
        "raw-gate",
        "beta-sigmoid",
        "beta-sigmoid-negative",
        "every-transform",
        "every-transform-fp16",
    ],
)
def test_gated_deltanet_prefill_takes_each_input_transform(
    l2norm: bool, raw_gate: bool, beta_sigmoid: bool, allow_neg_eigval: bool, dtype: torch.dtype
) -> None:
    """Each transform the op may leave to the kernel, alone and all together."""
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(
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
    op = GatedDeltaNetFwdOp(
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


@pytest.mark.sm90
@pytest.mark.cuda_only
def test_gated_deltanet_partitioned_prefill_normalizes_the_key_it_stages() -> None:
    """The warmup pass stages the key itself, so partitioning normalizes it a second time."""
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(1, 512, 2, 128, torch.bfloat16, l2norm=True)
    q, k, v, g, beta = (tensor.to("cuda") for tensor in test.gen_inputs())
    _check_partitioned(test, q, k, v, g * 0.01, beta, use_qk_l2norm_in_kernel=True)


class _FourChunkPartitionsKernel(GatedDeltaNetDensePrefillFwdKernel):
    """Partition every four chunks, so a short sequence crosses partitions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**kwargs, "config": {"max_local_chunks": 4}})


def _check_partitioned(test, *inputs, **params) -> None:
    """Run the op with four-chunk partitions and check it against the workload."""
    op = GatedDeltaNetFwdOp(
        **params,
        kernel_map={"gated_deltanet_dense_prefill": _FourChunkPartitionsKernel},
        target=BUILTIN,
    )
    test.check(op, *inputs)
    (kernel,) = op.built_kernels("gated_deltanet").values()
    assert type(kernel) is _FourChunkPartitionsKernel


@pytest.mark.sm90
@pytest.mark.cuda_only
@pytest.mark.parametrize("has_initial_state", [False, True], ids=["from-zero", "continued"])
def test_gated_deltanet_partitioned_dense_prefill_matches_reference(
    has_initial_state: bool,
) -> None:
    """Exercise warmup, state correction, and partitioned forward together."""
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(1, 512, 2, 128, torch.bfloat16, has_initial_state=has_initial_state)
    q, k, v, g, beta, *state = (tensor.to("cuda") for tensor in test.gen_inputs())
    # A gentle decay, so the state carried across partitions still reaches the output.
    _check_partitioned(test, q, k, v, g * 0.01, beta, *state)


@pytest.mark.sm90
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("batch", [1, 8], ids=["b1", "b8"])
def test_gated_deltanet_dense_decode_matches_reference(dtype: torch.dtype, batch: int) -> None:
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(batch, 1, 16, 128, dtype, has_initial_state=True)
    test.check(GatedDeltaNetFwdOp(), *test.gen_inputs())


@pytest.mark.sm90
@pytest.mark.parametrize(
    "flags",
    [
        # Each bound is ten times the agreement that flag reaches, so the case fails when
        # the transform it names is dropped. A flag that stiffens the recurrence carries a
        # looser one: doubling beta puts the state at 1.7e-1 rather than 4e-2.
        {"state_v_first": True},
        {"use_gate_in_kernel": True},
        {"use_beta_sigmoid_in_kernel": True, "allow_neg_eigval": True},
        {"use_qk_l2norm_in_kernel": True},
    ],
    ids=["v-first", "gate-fused", "beta-sigmoid-neg", "l2norm"],
)
def test_gated_deltanet_decode_runs_each_recurrence_flag(flags: dict) -> None:
    torch.manual_seed(42)
    workload_flags = {
        "state_v_first": "state_v_first",
        "use_gate_in_kernel": "raw_gate",
        "use_beta_sigmoid_in_kernel": "beta_sigmoid",
        "allow_neg_eigval": "allow_neg_eigval",
        "use_qk_l2norm_in_kernel": "l2norm",
    }
    test = GatedDeltaNetFwdTest(
        2,
        1,
        4,
        128,
        torch.bfloat16,
        has_initial_state=True,
        **{workload_flags[name]: value for name, value in flags.items()},
    )
    test.check(GatedDeltaNetFwdOp(**flags), *test.gen_inputs())


@pytest.mark.sm90
def test_gated_deltanet_decode_groups_value_heads_over_a_64_wide_state() -> None:
    """A batch and head counts that are neither powers of two nor warp multiples."""
    torch.manual_seed(42)
    test = GatedDeltaNetFwdTest(
        17, 1, 6, 64, torch.bfloat16, has_initial_state=True, value_heads=12
    )
    test.check(GatedDeltaNetFwdOp(), *test.gen_inputs())


@pytest.mark.sm90
def test_gated_deltanet_dense_decode_propagates_fp32_state() -> None:
    torch.manual_seed(42)
    workload = GatedDeltaNetFwdWorkload(
        1,
        1,
        16,
        128,
        torch.bfloat16,
        has_initial_state=True,
    )
    q, k, v, g, beta, state = workload.gen_inputs()
    expected_state = state.clone()
    op = GatedDeltaNetFwdOp()
    for _ in range(4):
        expected_o, expected_state = workload.ref_program(q, k, v, g, beta, expected_state)
        got_o, state = op(q, k, v, g, beta, state)
        compare_outputs(
            (got_o, state),
            (expected_o, expected_state),
            workload.verification(q, k, v, g, beta, state),
        )


def test_gated_deltanet_contract_reaches_target_builder() -> None:
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
        ):
            del k, g, beta, initial_state, cu_seqlens_cpu, A_log, dt_bias
            batch, seq_len, _heads, dim_k = q.shape
            value_heads, dim_v = v.shape[2:]
            state_batch = cu_seqlens.shape[0] - 1
            return (
                torch.empty(batch, seq_len, value_heads, dim_v, dtype=q.dtype),
                torch.empty(state_batch, value_heads, dim_k, dim_v, dtype=torch.float32),
            )

        return kernel

    registry.register_kernel_builder("GatedDeltaNetFwdOp", "gdn_test", build_kernel)

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

    op = GatedDeltaNetFwdOp(
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
            ),
            {
                "scale": 0.125,
                "use_qk_l2norm_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "allow_neg_eigval": True,
                "state_v_first": False,
                "use_gate_in_kernel": True,
            },
        )
    ]
