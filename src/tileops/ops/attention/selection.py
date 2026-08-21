"""Which dispatch key an attention call takes.

The keys implementing each attention slot, and the check that a request does not
contradict the user-visible ``backend`` parameter. Choosing among the keys is
``Op.select_kernel_key``; see docs/design/ops-design.md § Kernel selection.
"""

from tileops.kernels.attention.call_spec import AttentionCall, fp8_dtype, uses_sliding_window

__all__ = [
    "DENSE_FWD_DECODE_KEYS",
    "DENSE_FWD_PREFILL_KEYS",
    "PACKED_PREFILL_KEYS",
    "PAGED_DECODE_KEYS",
    "PAGED_PREFILL_KEYS",
    "AttentionCall",
    "check_packed_prefill_request",
    "fp8_dtype",
]

#: Implementations of packed GQA prefill, as ``kernel_map`` keys.
PACKED_PREFILL_KEYS = (
    "gqa_sliding_window_varlen_fwd_kernel",
    "gqa_prefill_varlen_fwd_kernel",
)

#: The subset serving a uniform dense request, for the fixed-shape wrapper.
DENSE_FWD_PREFILL_KEYS = (
    "gqa_prefill_fp8_tensor_core_fwd_kernel",
    "gqa_prefill_dense_sliding_fwd_kernel",
    "gqa_prefill_square_fwd_kernel",
    "gqa_prefill_causal_fwd_kernel",
    "gqa_prefill_dense_fwd_kernel",
)

#: Implementations of paged GQA prefill.
PAGED_PREFILL_KEYS = (
    "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel",
    "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel",
    "gqa_prefill_paged_with_kv_cache_fwd_kernel",
)

#: Implementations of contiguous GQA decode.
DENSE_FWD_DECODE_KEYS = ("gqa_decode_bs1_kernel", "gqa_decode_kernel")

#: Implementations of paged GQA decode.
PAGED_DECODE_KEYS = ("gqa_decode_paged_bs1_kernel", "gqa_decode_paged_kernel")

#: Implementations of paged MHA decode.
MHA_PAGED_DECODE_KEYS = ("mha_decode_paged_ws_kernel", "mha_decode_paged_kernel")


def check_packed_prefill_request(call: AttentionCall) -> None:
    """Reject a packed prefill request its ``backend`` knob cannot describe.

    This is the user-visible contract of the parameter, not a statement about
    any implementation: it says what the caller asked for is not what the
    request is. Which implementation then runs is decided by capability.

    Raises:
        ValueError: When ``backend`` contradicts the request.
    """
    if call.is_fp8:
        raise ValueError(
            "Packed FP8 prefill moved to GroupedQueryAttentionDenseFwdOp; "
            "Varlen FP8 support is tracked by #1917."
        )
    if call.backend == "fp8":
        raise ValueError("backend='fp8' moved to GroupedQueryAttentionDenseFwdOp.")
    if call.backend == "dense":
        raise ValueError("backend='dense' moved to GroupedQueryAttentionDenseFwdOp.")
    if uses_sliding_window(call):
        if call.backend not in ("auto", "sliding_window"):
            raise ValueError(
                "sliding-window prefill requires backend='auto' or backend='sliding_window'."
            )
        return
    if call.backend == "sliding_window":
        raise ValueError("backend='sliding_window' requires window_size_left or window_size_right.")
