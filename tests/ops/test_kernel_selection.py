"""Which kernel an attention call lands on, and when a call is refused.

The tables here are the record that moving the choice out of per-op predicates
changed no dispatch on SM90: one row per capability region the attention ops
used to encode by hand. Selection is asserted through
``select_kernel_key``, which resolves the key without compiling it — the tables would otherwise cost
one kernel compile per row.
"""

import pytest
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.ops import (
    GroupedQueryAttentionDecodePagedWithKVCacheFwdOp,
    GroupedQueryAttentionDecodeWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillDenseFwdOp,
    GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp,
    GroupedQueryAttentionPrefillVarlenFwdOp,
)
from tileops.ops.attention.selection import (
    DECODE_KEYS,
    DENSE_PREFILL_KEYS,
    PAGED_DECODE_KEYS,
    PAGED_PREFILL_KEYS,
    VARLEN_PREFILL_KEYS,
)
from tileops.utils import is_h200

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="attention selection reads the device architecture",
)


_PREFILL_CTOR = {
    "batch": 4,
    "heads": 32,
    "heads_kv": 8,
    "dim": 128,
    "seq_len": 512,
    "seq_len_kv": 512,
    "is_causal": True,
    "dtype": torch.float16,
}


def _op(cls: type, **overrides: object):
    kwargs = dict(_PREFILL_CTOR)
    kwargs.update(overrides)
    return cls(**kwargs)


def _prefill_op(**overrides: object) -> GroupedQueryAttentionPrefillDenseFwdOp:
    return _op(GroupedQueryAttentionPrefillDenseFwdOp, **overrides)


def _prefill_key(
    op: GroupedQueryAttentionPrefillDenseFwdOp,
    dtype: torch.dtype = torch.float16,
) -> str:
    return op.select_kernel_key(DENSE_PREFILL_KEYS, op.attention_call(dtype))


# Region -> dispatch key the pre-refactor selectors landed on. One row per
# capability region the attention ops used to encode: FP8, sliding window, the
# H200 square causal fast path, warp-specialized causal dense, plain dense,
# ragged varlen.
_PREFILL_DISPATCH = [
    ("square-causal-h200", {}, {}, "gqa_prefill_square_fwd_kernel"),
    ("square-causal-bf16", {"dtype": torch.bfloat16}, {}, "gqa_prefill_square_fwd_kernel"),
    (
        "causal-q-lt-kv",
        {"batch": 2, "seq_len": 512, "seq_len_kv": 4096},
        {},
        "gqa_prefill_causal_fwd_kernel",
    ),
    ("causal-dim64", {"dim": 64}, {}, "gqa_prefill_fwd_kernel"),
    ("noncausal-dim128", {"is_causal": False}, {}, "gqa_prefill_fwd_kernel"),
    (
        "small-causal-work",
        {"batch": 1, "heads": 8, "heads_kv": 8, "seq_len": 128, "seq_len_kv": 128},
        {},
        "gqa_prefill_causal_fwd_kernel",
    ),
    ("dense-sliding-window", {"window_size_left": 128}, {}, "gqa_prefill_dense_sliding_fwd_kernel"),
    (
        "fp8-square-noncausal",
        {"is_causal": False},
        {"dtype": torch.float8_e4m3fn},
        "gqa_prefill_fp8_tensor_core_fwd_kernel",
    ),
]


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("ctor", "call", "expected"),
    [pytest.param(c, f, e, id=name) for name, c, f, e in _PREFILL_DISPATCH],
)
def test_packed_prefill_dispatch_is_unchanged(ctor: dict, call: dict, expected: str) -> None:
    """Every Dense-prefill region still lands on the kernel it used to."""
    if not is_h200():
        pytest.skip("the recorded dispatch table is the H200 one")
    assert _prefill_key(_prefill_op(**ctor), **call) == expected


@pytest.mark.smoke
def test_bshd_wrapper_dispatches_like_the_packed_op() -> None:
    """The BSHD wrapper reaches the same dense candidate the packed op does."""
    if not is_h200():
        pytest.skip("the recorded dispatch table is the H200 one")
    op = GroupedQueryAttentionPrefillDenseFwdOp(4, 32, 8, 512, 128, True)
    call = op.attention_call(torch.float16)
    assert op.select_kernel_key(DENSE_PREFILL_KEYS, call) == "gqa_prefill_square_fwd_kernel"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "ctor",
    [
        pytest.param({}, id="fp8-causal"),
        pytest.param({"is_causal": False, "window_size_left": 128}, id="fp8-window"),
        pytest.param({"fuse_rope": True, "rotary_dim": 64}, id="fp8-rope"),
    ],
)
def test_dense_fp8_semantics_use_the_native_tensor_core_family(ctor: dict) -> None:
    """Causal and window policies remain native FP8, not a dequant fallback."""
    assert _prefill_key(_prefill_op(**ctor), torch.float8_e4m3fn) == (
        "gqa_prefill_fp8_tensor_core_fwd_kernel"
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("window_size_left", "expected"),
    [
        pytest.param(-1, "gqa_prefill_varlen_fwd_kernel", id="general"),
        pytest.param(128, "gqa_sliding_window_varlen_fwd_kernel", id="sliding"),
    ],
)
def test_varlen_prefill_dispatch(window_size_left: int, expected: str) -> None:
    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch=4,
        heads=32,
        heads_kv=8,
        dim=128,
        max_seqlen_q=512,
        max_seqlen_kv=512,
        window_size_left=window_size_left,
    )
    call = op.attention_call(torch.float16)
    assert op.select_kernel_key(VARLEN_PREFILL_KEYS, call) == expected


@pytest.mark.smoke
def test_varlen_window_falls_back_to_general_kernel_before_sm90() -> None:
    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch=4,
        heads=32,
        heads_kv=8,
        dim=128,
        max_seqlen_q=512,
        max_seqlen_kv=1024,
        window_size_left=256,
    )
    call = op.attention_call(torch.float16)
    sm80 = type(call)(**{**call.__dict__, "arch": 80})

    assert op.select_kernel_key(VARLEN_PREFILL_KEYS, sm80) == ("gqa_prefill_varlen_fwd_kernel")


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        pytest.param(torch.float16, "gqa_prefill_varlen_fwd_kernel", id="fp16-rope"),
        pytest.param(
            torch.float8_e4m3fn, "gqa_prefill_varlen_fp8_tensor_core_fwd_kernel", id="fp8-rope"
        ),
    ],
)
def test_varlen_rope_dispatch(dtype: torch.dtype, expected: str) -> None:
    op = GroupedQueryAttentionPrefillVarlenFwdOp(
        batch=2,
        heads=8,
        heads_kv=2,
        dim=128,
        max_seqlen_q=64,
        max_seqlen_kv=128,
        dtype=torch.float16 if dtype == torch.float8_e4m3fn else None,
        fuse_rope=True,
        rotary_dim=64,
        rope_layout="interleaved",
    )
    call = op.attention_call(dtype, max_position=128)
    assert op.select_kernel_key(VARLEN_PREFILL_KEYS, call) == expected


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("ctor", "dtype", "expected"),
    [
        pytest.param({}, torch.float16, "gqa_decode_bs1_kernel", id="bs1-fp16"),
        pytest.param({}, torch.bfloat16, "gqa_decode_kernel", id="bf16-falls-back"),
        pytest.param({"batch": 4}, torch.float16, "gqa_decode_kernel", id="batched"),
        pytest.param({"dim": 64}, torch.float16, "gqa_decode_kernel", id="head-dim"),
        pytest.param({"softcap": 2.0}, torch.float16, "gqa_decode_kernel", id="softcap"),
    ],
)
def test_decode_dispatch_is_unchanged(ctor: dict, dtype: torch.dtype, expected: str) -> None:
    """Contiguous decode keeps its batch-1 fast path and its fallbacks."""
    kwargs = {"batch": 1, "heads": 32, "heads_kv": 4, "seqlen_kv": 8192, "dim": 128}
    kwargs.update(ctor)
    op = GroupedQueryAttentionDecodeWithKVCacheFwdOp(**kwargs)
    candidate = op.select_kernel_key(DECODE_KEYS, op.attention_call(dtype))
    assert candidate == expected


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("ctor", "dtype", "expected"),
    [
        pytest.param({}, torch.float16, "gqa_decode_paged_bs1_kernel", id="bs1-fp16"),
        pytest.param({}, torch.bfloat16, "gqa_decode_paged_kernel", id="bf16-falls-back"),
        pytest.param({"batch": 2}, torch.float16, "gqa_decode_paged_kernel", id="batched"),
        pytest.param(
            {"page_size": 192, "seqlen_kv": 8064},
            torch.float16,
            "gqa_decode_paged_kernel",
            id="page-tile",
        ),
    ],
)
def test_paged_decode_dispatch_is_unchanged(ctor: dict, dtype: torch.dtype, expected: str) -> None:
    """Paged decode keeps its batch-1 fast path and its page-tile guard."""
    kwargs = {
        "batch": 1,
        "heads": 32,
        "heads_kv": 4,
        "seqlen_kv": 8192,
        "dim": 128,
        "page_size": 256,
    }
    kwargs.update(ctor)
    op = GroupedQueryAttentionDecodePagedWithKVCacheFwdOp(**kwargs)
    candidate = op.select_kernel_key(PAGED_DECODE_KEYS, op.attention_call(dtype))
    assert candidate == expected


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("ctor", "expected"),
    [
        pytest.param({}, "gqa_prefill_paged_with_kv_cache_fwd_kernel", id="plain-cache"),
        pytest.param(
            {"fuse_rope": True},
            "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel",
            id="fused-rope-neox",
        ),
        pytest.param(
            {"fuse_rope": True, "rope_layout": "interleaved"},
            "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel",
            id="fused-rope-interleaved",
        ),
    ],
)
def test_paged_prefill_dispatch_is_unchanged(ctor: dict, expected: str) -> None:
    """Paged prefill keeps its plain and fused-RoPE regions."""
    kwargs = {
        "batch": 2,
        "heads": 32,
        "heads_kv": 8,
        "max_pages_per_req": 8,
        "page_size": 256,
        "dim": 128,
        "max_seqlen_q": 128,
    }
    kwargs.update(ctor)
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(**kwargs)
    max_position = 4096 if op.fuse_rope else None
    candidate = op.select_kernel_key(
        PAGED_PREFILL_KEYS, op.attention_call(torch.float16, max_position)
    )
    assert candidate == expected


@pytest.mark.smoke
@pytest.mark.parametrize("fuse_rope", [False, True], ids=["plain", "rope"])
def test_paged_prefill_fp8_cache_dispatch_is_unchanged(fuse_rope: bool) -> None:
    """An FP8 KV cache still selects the FP8-cache kernel."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("this torch build has no float8_e4m3fn")
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        batch=2,
        heads=32,
        heads_kv=8,
        max_pages_per_req=8,
        page_size=256,
        dim=128,
        max_seqlen_q=128,
        fuse_rope=fuse_rope,
        rotary_dim=64 if fuse_rope else None,
        cache_dtype=torch.float8_e4m3fn,
    )
    candidate = op.select_kernel_key(
        PAGED_PREFILL_KEYS, op.attention_call(torch.float16, 4096 if fuse_rope else None)
    )
    assert candidate == "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel"


@pytest.mark.smoke
def test_paged_prefill_native_fp8_rope_dispatch() -> None:
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        batch=2,
        heads=32,
        heads_kv=8,
        max_pages_per_req=8,
        page_size=256,
        dim=128,
        max_seqlen_q=128,
        cache_dtype=torch.float8_e4m3fn,
        dtype=torch.float16,
        fuse_rope=True,
        rotary_dim=64,
    )
    candidate = op.select_kernel_key(
        PAGED_PREFILL_KEYS, op.attention_call(torch.float8_e4m3fn, 4096)
    )
    assert candidate == "gqa_prefill_paged_native_fp8_tensor_core_fwd_kernel"


@pytest.mark.smoke
def test_paged_prefill_semantic_policies_reach_the_kernel_call() -> None:
    """Window and append are compile-time kernel policies, not new public Ops."""
    op = GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp(
        batch=2,
        heads=32,
        heads_kv=8,
        max_pages_per_req=8,
        page_size=256,
        dim=128,
        max_seqlen_q=128,
        is_causal=False,
        window_size_left=64,
        window_size_right=8,
        append_kv=False,
    )
    call = op.attention_call(torch.float16)

    assert call.window_size_left == 64
    assert call.window_size_right == 8
    assert call.append_kv is False
    assert op.select_kernel_key(PAGED_PREFILL_KEYS, call) == (
        "gqa_prefill_paged_with_kv_cache_fwd_kernel"
    )


# Architecture is one capability filter among the others, applied per call.


@pytest.mark.smoke
def test_a_call_on_an_older_arch_lands_on_the_candidate_that_supports_it() -> None:
    """A Hopper-only candidate is skipped, not fatal, when an older arch asks.

    Every warp-specialized packed-prefill candidate is SM90-only while
    ``GQAPrefillFwdKernel`` covers SM80. The op constructs either way — it never
    probed the device — and the SM80 call lands on the candidate that applies.
    """
    op = _prefill_op()
    call = op.attention_call(torch.float16)
    sm80 = type(call)(**{**call.__dict__, "arch": 80})

    assert op.select_kernel_key(DENSE_PREFILL_KEYS, sm80) == ("gqa_prefill_fwd_kernel")


@pytest.mark.smoke
def test_a_call_no_candidate_supports_on_this_arch_is_refused() -> None:
    """When the whole table misses the architecture the call is refused, not silently run."""
    op = _prefill_op()
    call = op.attention_call(torch.float16)
    ancient = type(call)(**{**call.__dict__, "arch": 70})

    with pytest.raises(ValueError, match="no implementation serves this call"):
        op.select_kernel_key(DENSE_PREFILL_KEYS, ancient)


# The rule itself, stated over fake implementations so the assertions are about
# selection rather than about any shipped kernel's region.


class _Everything(Kernel):
    """A specialised implementation that forgot to state a region."""

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _General(Kernel):
    """The implementation behind the specialised ones."""

    general = True

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


class _NeverApplies(Kernel):
    """A specialised implementation whose region excludes every call."""

    @classmethod
    def applies(cls, call: object) -> bool:
        return False

    def forward(self, *args: object, **kwargs: object) -> None:
        return None


# Two real dispatch keys of the packed-prefill slot: _SPECIAL stands for a
# specialised slot, _GENERAL for the one behind it. An implementation supplied
# through ``kernel_map=`` counts as a replacement the caller owns; one an op
# declares itself does not.
_SPECIAL = "gqa_prefill_causal_fwd_kernel"
_GENERAL = "gqa_prefill_fwd_kernel"
_RULE_KEYS = (_GENERAL, _SPECIAL)


def _op_declaring(**declared: type) -> GroupedQueryAttentionPrefillDenseFwdOp:
    """An op whose own implementations are *declared*, so none is an override."""

    class DeclaringOp(GroupedQueryAttentionPrefillDenseFwdOp):
        @property
        def default_kernel_map(self) -> dict:
            return dict(declared)

    return _op(DeclaringOp)


@pytest.mark.smoke
def test_two_implementations_claiming_one_call_is_an_error() -> None:
    """Overlap is reported, never resolved by the order the keys are written in."""
    op = _op_declaring(**{_SPECIAL: _Everything, _GENERAL: _Everything})
    call = op.attention_call(torch.float16)

    with pytest.raises(ValueError, match="dispatch is ambiguous"):
        op.select_kernel_key(_RULE_KEYS, call)


@pytest.mark.smoke
def test_a_general_implementation_yields_to_a_specialised_one() -> None:
    """The general implementation runs only where no specialised one serves."""
    serving = _op_declaring(**{_SPECIAL: _Everything, _GENERAL: _General})
    call = serving.attention_call(torch.float16)
    assert serving.select_kernel_key(_RULE_KEYS, call) == _SPECIAL

    declining = _op_declaring(**{_SPECIAL: _NeverApplies, _GENERAL: _General})
    assert declining.select_kernel_key(_RULE_KEYS, call) == _GENERAL


@pytest.mark.smoke
def test_a_replacement_is_never_silently_stood_in_for() -> None:
    """A shipped kernel must not take a refused replacement's place.

    Only the specialised key is replaced, so the general one behind it is the
    op's own and would otherwise stand in.
    """
    op = _prefill_op(kernel_map={_SPECIAL: _NeverApplies})
    call = op.attention_call(torch.float16)

    with pytest.raises(ValueError, match="the kernel supplied for"):
        op.select_kernel_key(_RULE_KEYS, call)


@pytest.mark.smoke
def test_one_replacement_winning_while_another_is_refused_is_the_callers_own_doing() -> None:
    """Both candidates are the caller's, so no shipped kernel stands in for either.

    The only guard on the second half of that rule. The test above pins that a
    refused replacement is not stood in for; this one pins that the refusal is
    about *standing in*, not about the refusal itself — weakening the rule to
    "any refused replacement is an error" passes every other test in the repo
    and fails only here.
    """
    op = _prefill_op(kernel_map={_SPECIAL: _NeverApplies, _GENERAL: _Everything})
    call = op.attention_call(torch.float16)

    assert op.select_kernel_key(_RULE_KEYS, call) == _GENERAL
