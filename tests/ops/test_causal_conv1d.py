import re

import pytest
import torch

from tests.cuda_graph_contract import check_replay
from tests.state_pool_guard import check_padded_slots_skipped
from tests.workload_test_base import TestBase
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.mamba import CausalConv1dCall, CausalConv1dUpdateCall
from tileops.backend import OpNotAvailableError
from tileops.mamba import CausalConv1dFwdOp, CausalConv1dUpdateFwdOp
from workloads.device import run_device
from workloads.mamba import (
    CausalConv1dFwdCall,
    CausalConv1dUpdateFwdCall,
    causal_conv1d_update_ref,
    conv_state_result,
)
from workloads.numerics import compare_outputs
from workloads.workload_base import manifest_call


class CausalConv1dFwdTest(CausalConv1dFwdCall, TestBase):
    pass


class CausalConv1dUpdateFwdTest(CausalConv1dUpdateFwdCall, TestBase):
    pass


_SMOKE = pytest.mark.smoke
_FULL = pytest.mark.full
_BOTH = ["bias", "has_initial_state"]

# A warp walks 64 tokens (48 at W = 3) of 256 channels. One case per path through it.
_PREFILL = (
    pytest.param(
        [1024, 1024], 5376, 4, "bfloat16", _BOTH, "silu", marks=_SMOKE, id="one-sequence-a-chunk"
    ),
    pytest.param(
        [100, 900], 8192, 4, "float16", _BOTH, "silu", marks=_SMOKE, id="start-mid-chunk-fp16"
    ),
    pytest.param(
        [64, 64, 128], 512, 4, "bfloat16", _BOTH, "silu", marks=_FULL, id="start-on-chunk"
    ),
    pytest.param(
        [62, 70], 256, 4, "bfloat16", _BOTH, "silu", marks=_FULL, id="head-past-chunk-end"
    ),
    pytest.param(
        [1, 2, 0, 3, 5, 17, 0, 64, 33, 1, 0],
        264,
        4,
        "bfloat16",
        _BOTH,
        "silu",
        marks=_FULL,
        id="short-and-empty",
    ),
    pytest.param(
        [300, 200],
        1024,
        4,
        "float16",
        ["has_initial_state"],
        None,
        marks=_FULL,
        id="no-bias-no-act",
    ),
    pytest.param([129, 70], 768, 4, "bfloat16", ["bias"], "silu", marks=_FULL, id="no-state"),
    # An empty sequence without has_initial_state keeps its state.
    pytest.param(
        [0, 4, 0, 70], 256, 4, "bfloat16", ["bias"], "silu", marks=_FULL, id="empty-keeps-state"
    ),
    pytest.param(
        [0, 0, 7, 70, 0],
        136,
        3,
        "float16",
        ["has_initial_state"],
        "silu",
        marks=_FULL,
        id="width-3",
    ),
    pytest.param([5, 3, 2, 0, 90], 40, 2, "bfloat16", ["bias"], None, marks=_FULL, id="width-2"),
    pytest.param([37, 91, 1, 2], 100, 4, "float16", _BOTH, "silu", marks=_FULL, id="vec-4"),
    pytest.param([9, 70], 6, 4, "float16", _BOTH, "silu", marks=_FULL, id="vec-2-async-4-bytes"),
    pytest.param(
        [13, 60], 7, 4, "bfloat16", ["has_initial_state"], "silu", marks=_FULL, id="vec-1-no-async"
    ),
    pytest.param([1] * 300, 256, 4, "bfloat16", _BOTH, "silu", marks=_FULL, id="tokens-a-sequence"),
    # A short sequence starting before the chunk must not suppress the next one's head.
    pytest.param(
        [1, 62, 2, 0, 1, 67],
        136,
        4,
        "float16",
        _BOTH,
        "silu",
        marks=_FULL,
        id="short-sequence-past-chunk-end",
    ),
    pytest.param(
        [2500, 1000, 500, 96],
        8192,
        4,
        "float16",
        ["has_initial_state"],
        "silu",
        marks=_FULL,
        id="ragged",
    ),
)


@pytest.mark.parametrize("seq_lens, dim, width, dtype, some, activation", _PREFILL)
def test_causal_conv1d(seq_lens, dim, width, dtype, some, activation) -> None:
    call = manifest_call(
        "CausalConv1dFwdOp",
        {"T": dtype, "Tw": dtype},
        D=dim,
        W=width,
        L=width - 1,
        Tt=sum(seq_lens),
        seq_lens=seq_lens,
        activation=activation,
        some=some,
        **({} if "bias" in some else {"bias": None}),
    )
    test = CausalConv1dFwdTest(call)
    op = CausalConv1dFwdOp(**call.arguments())
    test.check(op, *test.gen_inputs(), runs=lambda *args, **kw: conv_state_result(op, *args, **kw))


_DECODE = (
    pytest.param(64, 5376, 4, "bfloat16", ["bias"], "silu", marks=_SMOKE, id="bf16-bias"),
    pytest.param(3, 8192, 4, "float16", [], "silu", marks=_SMOKE, id="fp16"),
    pytest.param(5, 100, 3, "bfloat16", ["bias"], None, marks=_FULL, id="width-3-vec-4"),
    pytest.param(4, 6, 4, "bfloat16", [], "silu", marks=_FULL, id="vec-2"),
    pytest.param(2, 7, 2, "float16", ["bias"], "silu", marks=_FULL, id="width-2-vec-1"),
)


@pytest.mark.parametrize("batch, dim, width, dtype, some, activation", _DECODE)
def test_causal_conv1d_update(batch, dim, width, dtype, some, activation) -> None:
    call = manifest_call(
        "CausalConv1dUpdateFwdOp",
        {"T": dtype, "Tw": dtype},
        B=batch,
        D=dim,
        W=width,
        L=width - 1,
        activation=activation,
        some=some,
    )
    test = CausalConv1dUpdateFwdTest(call)
    op = CausalConv1dUpdateFwdOp(**call.arguments())
    test.check(op, *test.gen_inputs(), runs=lambda *args, **kw: conv_state_result(op, *args, **kw))


# A dilated filter spans (W - 1) * dilation tokens; a float32 filter widens the arithmetic.
_SERVED_PREFILL = (
    # GLM-5.3-Flash KDA convolution: float32 filters over bf16 activations.
    pytest.param(
        [1024] * 4, 24576, 4, 1, 3, None, "bfloat16", "float32", ["has_initial_state"],
        marks=_SMOKE, id="glm-fp32-filters",
    ),
    pytest.param(
        [100, 900], 8192, 3, 2, 4, None, "float16", "float16", ["has_initial_state"],
        marks=_SMOKE, id="dilated-width-3",
    ),
    # Qwen3.8-Flash-Next PLE short conv: dilation 3, a 9-token state, SiLU, no bias.
    pytest.param(
        [1024] * 4, 10240, 4, 3, 9, 256, "bfloat16", "bfloat16",
        ["has_initial_state", "cache_indices"], marks=_FULL, id="ple-4k-pool",
    ),
    # A short initialized sequence whose head crosses a chunk boundary.
    pytest.param(
        [1, 77, 5, 0, 2, 161], 136, 4, 3, 12, 8, "float16", "float32",
        ["bias", "has_initial_state", "cache_indices"], marks=_FULL,
        id="dilated-head-past-chunk-end",
    ),
    # A state wider than the span keeps its columns past it.
    pytest.param(
        [100, 900], 256, 4, 3, 12, None, "float16", "float32", ["has_initial_state"],
        marks=_FULL, id="dilated-fp32-wide-state",
    ),
    pytest.param(
        [0, 2, 7, 0, 1, 70], 264, 4, 3, 9, None, "bfloat16", "bfloat16",
        ["bias", "has_initial_state"], marks=_FULL, id="dilated-short-and-empty",
    ),
    pytest.param(
        [5, 3, 2, 0, 90], 40, 2, 4, 4, None, "bfloat16", "bfloat16", ["bias"], marks=_FULL,
        id="dilated-width-2",
    ),
)  # fmt: skip


@pytest.mark.parametrize(
    "seq_lens, dim, width, dilation, state_len, pool, dtype, weight_dtype, some", _SERVED_PREFILL
)
def test_causal_conv1d_dilation_and_float32_filters(
    seq_lens, dim, width, dilation, state_len, pool, dtype, weight_dtype, some
) -> None:
    call = manifest_call(
        "CausalConv1dFwdOp",
        {"T": dtype, "Tw": weight_dtype},
        D=dim,
        W=width,
        L=state_len,
        Tt=sum(seq_lens),
        seq_lens=seq_lens,
        dilation=dilation,
        some=some,
        **({} if "bias" in some else {"bias": None}),
        **({} if pool is None else {"P": pool}),
    )
    test = CausalConv1dFwdTest(call)
    op = CausalConv1dFwdOp(**call.arguments())
    test.check(op, *test.gen_inputs(), runs=lambda *args, **kw: conv_state_result(op, *args, **kw))


_SERVED_DECODE = (
    # Float32 filters and bias over bfloat16 activations, on a state pool.
    pytest.param(
        64, 24576, 4, 1, 3, 256, "bfloat16", "float32", ["bias", "conv_state_indices"],
        marks=_SMOKE, id="glm-fp32-filters-pool",
    ),
    pytest.param(
        3, 8192, 4, 3, 9, None, "float16", "float16", [], marks=_SMOKE, id="dilated-no-pool"
    ),
    # A state wider than the span is advanced in its first span columns, the rest untouched.
    pytest.param(
        64, 10240, 4, 3, 12, 256, "bfloat16", "float32", ["conv_state_indices"], marks=_FULL,
        id="dilated-fp32-wide-state",
    ),
    pytest.param(
        2, 7, 2, 4, 4, None, "float16", "float32", [], marks=_FULL, id="width-2-dilation-4"
    ),
)  # fmt: skip


@pytest.mark.parametrize(
    "batch, dim, width, dilation, state_len, pool, dtype, weight_dtype, some", _SERVED_DECODE
)
def test_causal_conv1d_update_dilation_and_float32_filters(
    batch, dim, width, dilation, state_len, pool, dtype, weight_dtype, some
) -> None:
    call = manifest_call(
        "CausalConv1dUpdateFwdOp",
        {"T": dtype, "Tw": weight_dtype},
        B=batch,
        D=dim,
        W=width,
        L=state_len,
        dilation=dilation,
        some=some,
        **({} if pool is None else {"P": pool}),
    )
    test = CausalConv1dUpdateFwdTest(call)
    op = CausalConv1dUpdateFwdOp(**call.arguments())
    test.check(op, *test.gen_inputs(), runs=lambda *args, **kw: conv_state_result(op, *args, **kw))


@pytest.mark.smoke
def test_causal_conv1d_update_unaligned_state() -> None:
    """A state whose storage starts off a 16-byte boundary is advanced in place all the same."""
    batch, dim, width = 4, 256, 4
    dev, dtype = run_device(), torch.bfloat16
    flat = torch.randn(batch * dim * (width - 1) + 1, device=dev, dtype=dtype)
    state = flat[1:].view(batch, dim, width - 1)
    x = torch.randn(batch, dim, device=dev, dtype=dtype)
    weight = torch.rand(dim, width, device=dev, dtype=dtype) - 0.5
    expected_state = state.clone()
    expected = causal_conv1d_update_ref(x, weight, expected_state)

    got = CausalConv1dUpdateFwdOp()(x, weight, state)

    workload = CausalConv1dUpdateFwdCall(
        manifest_call(
            "CausalConv1dUpdateFwdOp",
            {"T": "bfloat16", "Tw": "bfloat16"},
            B=batch,
            D=dim,
            W=width,
            L=width - 1,
        )
    )
    evidence = workload.verification(x, weight, state, None)
    compare_outputs((got, state), (expected, expected_state), evidence)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype, weight_dtype",
    [("float16", "float16"), ("bfloat16", "bfloat16"), ("bfloat16", "float32")],
)
def test_prefill_bias(dtype: str, weight_dtype: str) -> None:
    """A contiguous bias with an offset address preserves outputs and the final state."""
    call = manifest_call(
        "CausalConv1dFwdOp",
        {"T": dtype, "Tw": weight_dtype},
        D=256,
        W=4,
        L=3,
        Tt=72,
        seq_lens=[7, 65],
        some=["bias", "has_initial_state"],
    )
    test = CausalConv1dFwdTest(call)
    inputs = list(test.gen_inputs())
    bias = inputs[4]
    storage = torch.full((bias.numel() + 2,), 123, device=bias.device, dtype=bias.dtype)
    inputs[4] = storage[1:-1]
    inputs[4].copy_(bias)
    assert inputs[4].is_contiguous() and inputs[4].data_ptr() % 16 != 0
    original = storage.clone()
    op = CausalConv1dFwdOp(**call.arguments())

    def run(*args, **keywords):
        result = conv_state_result(op, *args, **keywords)
        torch.testing.assert_close(storage, original, rtol=0, atol=0)
        return result

    test.check(op, *inputs, runs=run)


_POOLED_PREFILL = (
    pytest.param(
        [1, 2, 0, 3, 5, 17, 0, 64, 33, 1, 0],
        264,
        "bfloat16",
        marks=_SMOKE,
        id="short-and-empty",
    ),
    pytest.param([100, 900], 8192, "float16", marks=_SMOKE, id="start-mid-chunk-fp16"),
)


@pytest.mark.parametrize("seq_lens, dim, dtype", _POOLED_PREFILL)
def test_causal_conv1d_reads_and_writes_its_slots_of_a_state_pool(seq_lens, dim, dtype) -> None:
    """Each sequence reads its history from its slot where it starts on a chunk, where it starts
    inside one and where it leaves its tail; an empty sequence keeps its slot's state."""
    call = manifest_call(
        "CausalConv1dFwdOp",
        {"T": dtype, "Tw": dtype},
        D=dim,
        W=4,
        L=3,
        P=32,
        Tt=sum(seq_lens),
        seq_lens=seq_lens,
        some=["bias", "has_initial_state", "cache_indices"],
    )
    test = CausalConv1dFwdTest(call)
    op = CausalConv1dFwdOp(**call.arguments())
    test.check(op, *test.gen_inputs(), runs=lambda *args, **kw: conv_state_result(op, *args, **kw))


@pytest.mark.smoke
def test_causal_conv1d_update_advances_its_slots_of_a_state_pool() -> None:
    """Each sequence's state is the pool row its slot names, advanced in place; the other rows
    keep their states."""
    call = manifest_call(
        "CausalConv1dUpdateFwdOp",
        {"T": "bfloat16", "Tw": "bfloat16"},
        B=64,
        D=5376,
        W=4,
        L=3,
        P=256,
        some=["bias", "conv_state_indices"],
    )
    test = CausalConv1dUpdateFwdTest(call)
    op = CausalConv1dUpdateFwdOp(**call.arguments())
    test.check(op, *test.gen_inputs(), runs=lambda *args, **kw: conv_state_result(op, *args, **kw))


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.parametrize(
    "op_type, workload_type, shape, serving",
    [
        pytest.param(
            CausalConv1dFwdOp,
            CausalConv1dFwdTest,
            {"Tt": 87, "seq_lens": [2, 85]},
            ["has_initial_state", "cache_indices"],
            marks=pytest.mark.cuda_graph(op="CausalConv1dFwdOp"),
            id="prefill",
        ),
        pytest.param(
            CausalConv1dUpdateFwdOp,
            CausalConv1dUpdateFwdTest,
            {"B": 2},
            ["conv_state_indices"],
            marks=pytest.mark.cuda_graph(op="CausalConv1dUpdateFwdOp"),
            id="decode",
        ),
    ],
)
def test_causal_conv1d_dilated_graph_replay(op_type, workload_type, shape, serving) -> None:
    """Replay reads changed tokens, filters and slots at their captured addresses."""
    call = manifest_call(
        op_type.__name__,
        {"T": "bfloat16", "Tw": "float32"},
        D=100,
        W=4,
        L=12,
        P=4,
        dilation=3,
        activation=None,
        some=["bias", *serving],
        **shape,
    )
    test = workload_type(call)
    inputs = test.gen_inputs()
    slots = inputs[6] if op_type is CausalConv1dFwdOp else inputs[4]
    slots.copy_(slots.roll(1))
    if op_type is CausalConv1dFwdOp:
        bounds, initial = inputs[3], inputs[5]
        bounds.copy_(bounds[-1] - bounds.flip(0))
        initial.logical_not_()
    check_replay(test, op_type(**call.arguments()), inputs, result=conv_state_result)


_ROLLBACK = (
    pytest.param(
        [4, 1, 0, 3, 2], 264, 4, 1, "bfloat16", "bfloat16", ["bias"], marks=_SMOKE,
        id="ragged-with-an-empty-sequence",
    ),
    pytest.param(
        [2, 4, 1], 100, 4, 3, "float16", "float32", [], marks=_SMOKE,
        id="dilated-float32-weight",
    ),
    pytest.param(
        [2, 3, 1], 301, 3, 2, "bfloat16", "float32", [], marks=_FULL,
        id="odd-channels-one-a-thread",
    ),
)  # fmt: skip


@pytest.mark.parametrize("seq_lens, dim, width, dilation, dtype, weight_dtype, some", _ROLLBACK)
def test_causal_conv1d_update_verifies_drafts_from_the_accepted_column(
    seq_lens, dim, width, dilation, dtype, weight_dtype, some
) -> None:
    """With num_accepted_tokens, each packed sequence reads its history from the column of its
    last accepted token, a random count in its range, and its first (W - 1) * dilation + q - 1
    columns take the history and its q tokens; an empty sequence keeps its state, the filter
    taps are dilation tokens apart in the activation dtype or float32, and an odd channel
    count puts one channel on a thread."""
    torch.manual_seed(42)
    span = (width - 1) * dilation
    call = manifest_call(
        "CausalConv1dUpdateFwdOp",
        {"T": dtype, "Tw": weight_dtype},
        D=dim,
        W=width,
        L=span + max(seq_lens) - 1,
        P=16,
        Tt=sum(seq_lens),
        seq_lens=seq_lens,
        max_query_len=max(seq_lens),
        dilation=dilation,
        some=["conv_state_indices", "num_accepted_tokens", "query_start_loc", *some],
        **({} if "bias" in some else {"bias": None}),
    )
    test = CausalConv1dUpdateFwdTest(call)
    op = CausalConv1dUpdateFwdOp(**call.arguments())
    inputs = list(test.gen_inputs())
    inputs[5] = torch.randint(1, max(seq_lens) + 1, inputs[5].shape, dtype=torch.int32).to(
        inputs[5].device
    )
    test.check(op, *inputs, runs=lambda *args, **kw: conv_state_result(op, *args, **kw))


@pytest.mark.smoke
def test_causal_conv1d_update_resumes_one_token_from_the_accepted_column() -> None:
    """With num_accepted_tokens and one token a sequence in batch order, each sequence reads its
    history from the column of its last accepted token and writes the first (W - 1) columns;
    the columns past them keep their tokens."""
    torch.manual_seed(42)
    call = manifest_call(
        "CausalConv1dUpdateFwdOp",
        {"T": "bfloat16", "Tw": "bfloat16"},
        B=5,
        D=512,
        W=4,
        L=5,
        some=["num_accepted_tokens"],
        bias=None,
    )
    test = CausalConv1dUpdateFwdTest(call)
    op = CausalConv1dUpdateFwdOp(**call.arguments())
    inputs = list(test.gen_inputs())
    inputs[5] = torch.randint(1, 4, inputs[5].shape, dtype=torch.int32).to(inputs[5].device)
    test.check(op, *inputs, runs=lambda *args, **kw: conv_state_result(op, *args, **kw))


_PADDED_PREFILL = {"D": 264, "L": 3, "Tt": 92, "seq_lens": [1, 2, 0, 3, 5, 17, 0, 64]}
_PADDED_ROLLBACK = {"D": 264, "L": 6, "Tt": 15, "seq_lens": [4, 1, 0, 3, 2, 4, 1]}


@pytest.mark.parametrize(
    ("op_type", "test_type", "row", "some", "slots_at"),
    [
        pytest.param(
            CausalConv1dFwdOp,
            CausalConv1dFwdTest,
            _PADDED_PREFILL,
            ["bias", "has_initial_state", "cache_indices"],
            6,
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            CausalConv1dUpdateFwdOp,
            CausalConv1dUpdateFwdTest,
            {"B": 8, "D": 264, "L": 3},
            ["conv_state_indices"],
            4,
            marks=_FULL,
        ),
        # A state row longer than one 128-byte L2 line runs the per-thread decode kernel.
        pytest.param(
            CausalConv1dUpdateFwdOp,
            CausalConv1dUpdateFwdTest,
            {"B": 8, "D": 264, "L": 480},
            ["bias", "conv_state_indices"],
            4,
            marks=_FULL,
        ),
        pytest.param(
            CausalConv1dUpdateFwdOp,
            CausalConv1dUpdateFwdTest,
            {**_PADDED_ROLLBACK, "max_query_len": 4},
            ["conv_state_indices", "num_accepted_tokens", "query_start_loc"],
            4,
            marks=_FULL,
        ),
    ],
    ids=["prefill", "update", "update-long-state", "rollback"],
)
def test_causal_conv1d_skips_the_states_of_negative_slots(
    op_type, test_type, row, some, slots_at
) -> None:
    """Each convolution kernel writes no state for a slot of -1: the pool's neighbours
    and the slots of the padded sequences keep their states, every other slot ends as in the
    unpadded call."""
    torch.manual_seed(42)
    call = manifest_call(
        op_type.__name__, {"T": "bfloat16", "Tw": "bfloat16"}, W=4, P=16, some=some, **row
    )
    inputs = test_type(call).gen_inputs()
    op = op_type(**{**call.arguments(), "validate_inputs": True})
    check_padded_slots_skipped(op, inputs, 2, slots_at, [0, 4])


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_causal_conv1d_prefill_refuses_a_ring_past_shared_memory() -> None:
    """A dilated filter lengthens the token ring a block stages in shared memory: dispatch
    refuses a ring past the budget before a build and names the budget."""
    (in_tree,) = Kernel.devices
    call = CausalConv1dCall(
        arch=89, sm_count=142, smem_budget=101376, dim=256, width=4, dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16, state_len=210, dilation=70, device=torch.device(in_tree),
    )  # fmt: skip
    with pytest.raises(OpNotAvailableError, match="bytes of tokens, past the 101376"):
        CausalConv1dFwdOp(dilation=70).key_for("causal_conv1d", call)


# Float16 builds: a dilated build's 832 blocks of one warp, which hold more than 128 registers'
# worth; 256-channel by 256-token blocks: 512 (4 an H200 SM), 480 (4, the same program) and 256
# (2); 640, whose 5 an SM would leave a thread 102 registers, and 40, one an SM, sharing one
# unbounded program.
_RESIDENCY = (
    pytest.param(8192, 3, ([100, 900],), marks=_SMOKE, id="dilated"),
    pytest.param(
        8192, 1, ([2500, 1000, 500, 96], [2000, 1800], [1000, 1048]), marks=_FULL,
        id="one-wave-and-reuse",
    ),
    pytest.param(10240, 1, ([3000, 1000, 96], [200]), marks=_FULL, id="register-floor-one-block"),
)  # fmt: skip


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize("dim, dilation, runs", _RESIDENCY)
def test_causal_conv1d_prefill_holds_a_one_wave_grid_resident(dim, dilation, runs) -> None:
    """An undilated grid asks ptxas to hold ceil(blocks / SMs) blocks an SM, unless that is one,
    leaves a thread fewer than 128 registers or overfills shared memory; a dilated one asks for
    none. One kernel serves
    token counts that ask different residencies, a residency seen before reuses its program,
    and every call's output and state stay right."""
    props = torch.cuda.get_device_properties(run_device())
    sms, regs = props.multi_processor_count, props.regs_per_multiprocessor
    smem = props.shared_memory_per_multiprocessor
    op = None
    expected = set()
    for seq_lens in runs:
        call = manifest_call(
            "CausalConv1dFwdOp", {"T": "float16", "Tw": "float16"}, D=dim, W=4, L=3 * dilation,
            Tt=sum(seq_lens), seq_lens=seq_lens, dilation=dilation, some=["has_initial_state"],
            bias=None,
        )  # fmt: skip
        test = CausalConv1dFwdTest(call)
        op = op or CausalConv1dFwdOp(**call.arguments())
        test.check(
            op,
            *test.gen_inputs(),
            runs=lambda *args, op=op, **kw: conv_state_result(op, *args, **kw),
        )
        held = -(-(dim // 256 * -(-sum(seq_lens) // 256)) // sms)
        # A block's ring: 4 warps x 2 stages x 8 tokens x 256 float16 channels.
        asked = held > 1 and regs // (held * 128) >= 128 and held * 32768 <= smem
        expected.add("32, 1" if dilation > 1 else f"128, {held if asked else 1}")
    (kernel,) = op._entries_built("causal_conv1d").values()
    sources = [program.get_kernel_source() for program in kernel._generation.programs.values()]
    bounds = [b for s in sources for b in re.findall(r"__launch_bounds__\(([\d, ]+)\)", s)]
    assert sorted(bounds) == sorted(expected)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_causal_conv1d_update_refuses_a_state_past_shared_memory() -> None:
    """A draft verification stages its block's state rows in shared memory, which a long
    enough state outgrows: dispatch refuses the call before a build and names the budget."""
    (in_tree,) = Kernel.devices
    call = CausalConv1dUpdateCall(
        arch=90, sm_count=132, smem_budget=232448, dim=256, width=4, dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16, state_len=480, pooled=True, varlen=True, rollback=True,
        device=torch.device(in_tree),
    )  # fmt: skip
    with pytest.raises(OpNotAvailableError, match="bytes of state, past the 232448"):
        CausalConv1dUpdateFwdOp().key_for("causal_conv1d_update", call)


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
def test_causal_conv1d_update_stages_rows_of_one_cache_line() -> None:
    """A decode step whose state rows each fill one 128-byte L2 line (64 bf16 columns) stages
    them through shared memory."""
    (in_tree,) = Kernel.devices
    spec = CausalConv1dUpdateCall(
        arch=90, sm_count=132, smem_budget=232448, dim=256, width=4, dtype=torch.bfloat16,
        weight_dtype=torch.bfloat16, state_len=64, pooled=True, device=torch.device(in_tree),
    )  # fmt: skip
    assert CausalConv1dUpdateFwdOp().key_for("causal_conv1d_update", spec) == (
        "causal_conv1d_update"
    )


_LONG_STATE = (
    # One column past a 128-byte row: width 2, two channels a thread, no bias, no activation.
    pytest.param(2, 6, 2, 1, 65, "float16", "float16", [], None, marks=_SMOKE, id="fp16-width-2-plain"),
    # A state exactly its span; eight channels a thread take two 16-byte bias loads.
    pytest.param(
        3, 8, 4, 152, 456, "bfloat16", "float32", ["bias"], "silu", marks=_SMOKE,
        id="span-fp32-filters",
    ),
)  # fmt: skip


@pytest.mark.in_tree_kernels
@pytest.mark.parametrize(
    "batch, dim, width, dilation, state_len, dtype, weight_dtype, some, activation", _LONG_STATE
)
def test_causal_conv1d_update_holds_a_long_state_in_registers(
    batch, dim, width, dilation, state_len, dtype, weight_dtype, some, activation
) -> None:
    """A decode step whose state rows are longer than one 128-byte L2 line runs the kernel
    that holds each thread's state tokens in registers, and advances its first
    (W - 1) * dilation columns."""
    (in_tree,) = Kernel.devices
    spec = CausalConv1dUpdateCall(
        arch=90, sm_count=132, smem_budget=232448, dim=dim, width=width,
        dtype=getattr(torch, dtype), weight_dtype=getattr(torch, weight_dtype),
        state_len=state_len, dilation=dilation, has_bias="bias" in some,
        silu=activation is not None, device=torch.device(in_tree),
    )  # fmt: skip
    op = CausalConv1dUpdateFwdOp(activation=activation, dilation=dilation)
    assert op.key_for("causal_conv1d_update", spec) == "causal_conv1d_update_register"
    call = manifest_call(
        "CausalConv1dUpdateFwdOp",
        {"T": dtype, "Tw": weight_dtype},
        B=batch,
        D=dim,
        W=width,
        L=state_len,
        dilation=dilation,
        activation=activation,
        some=some,
        **({} if "bias" in some else {"bias": None}),
    )
    test = CausalConv1dUpdateFwdTest(call)
    op = CausalConv1dUpdateFwdOp(**call.arguments())
    test.check(op, *test.gen_inputs(), runs=lambda *args, **kw: conv_state_result(op, *args, **kw))
