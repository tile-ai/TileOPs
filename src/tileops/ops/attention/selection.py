"""Which in-tree dispatch key an attention call takes.

The keys implementing each attention slot are consumed by
``Op.select_kernel_key``; see the Kernel selection section of
``docs/design/ops-design.md``. External targets own their own candidate lists
and never use these keys.
"""

from tileops.kernels.attention.call_spec import AttentionCall, fp8_dtype

__all__ = [
    "DECODE_KEYS",
    "PACKED_PREFILL_KEYS",
    "PAGED_DECODE_KEYS",
    "PAGED_PREFILL_KEYS",
    "AttentionCall",
    "fp8_dtype",
]

#: Implementations of packed GQA prefill, as ``kernel_map`` keys.
PACKED_PREFILL_KEYS = (
    "gqa_prefill_fp8_tensor_core_fwd_kernel",
    "gqa_sliding_window_varlen_fwd_kernel",
    "gqa_prefill_square_fwd_kernel",
    "gqa_prefill_causal_fwd_kernel",
    "gqa_prefill_fwd_kernel",
    "gqa_prefill_varlen_fwd_kernel",
)

#: The subset serving a uniform dense request, for the fixed-shape wrapper.
DENSE_PREFILL_KEYS = (
    "gqa_prefill_square_fwd_kernel",
    "gqa_prefill_causal_fwd_kernel",
    "gqa_prefill_fwd_kernel",
)

#: Implementations of paged GQA prefill.
PAGED_PREFILL_KEYS = (
    "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel",
    "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel",
    "gqa_prefill_paged_with_kv_cache_fwd_kernel",
)

#: Implementations of contiguous GQA decode.
DECODE_KEYS = ("gqa_decode_bs1_kernel", "gqa_decode_kernel")

#: Implementations of paged GQA decode.
PAGED_DECODE_KEYS = ("gqa_decode_paged_bs1_kernel", "gqa_decode_paged_kernel")
