import pytest
import torch

from tests.workload_test_base import TestBase
from tileops._kernels.kernel_base import Kernel
from tileops._kernels.mamba import CausalConv1dUpdateCall
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
        [64, 64, 128], 512, 4, "bfloat16", _BOTH, "silu", marks=_SMOKE, id="start-on-chunk"
    ),
    pytest.param(
        [62, 70], 256, 4, "bfloat16", _BOTH, "silu", marks=_SMOKE, id="head-past-chunk-end"
    ),
    pytest.param(
        [1, 2, 0, 3, 5, 17, 0, 64, 33, 1, 0],
        264,
        4,
        "bfloat16",
        _BOTH,
        "silu",
        marks=_SMOKE,
        id="short-and-empty",
    ),
    pytest.param(
        [300, 200],
        1024,
        4,
        "float16",
        ["has_initial_state"],
        None,
        marks=_SMOKE,
        id="no-bias-no-act",
    ),
    pytest.param([129, 70], 768, 4, "bfloat16", ["bias"], "silu", marks=_SMOKE, id="no-state"),
    # Regression: an empty sequence without a state got zeros written over its state.
    pytest.param(
        [0, 4, 0, 70], 256, 4, "bfloat16", ["bias"], "silu", marks=_SMOKE, id="empty-keeps-state"
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
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_prefill_bias(dtype: str) -> None:
    """A contiguous bias with an offset address preserves outputs and the final state."""
    call = manifest_call(
        "CausalConv1dFwdOp",
        {"T": dtype, "Tw": dtype},
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
