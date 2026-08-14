"""Which dispatch key an attention call takes.

The tuples below list the implementations of each topology-specific attention
slot. Choosing among them is capability based and order independent; see
``Op.select_kernel_key`` and docs/design/ops-design.md § Kernel selection.
"""

from tileops.kernels.attention.call_spec import AttentionCall, fp8_dtype

__all__ = [
    "DECODE_KEYS",
    "DENSE_PREFILL_KEYS",
    "PAGED_DECODE_KEYS",
    "PAGED_PREFILL_KEYS",
    "VARLEN_PREFILL_KEYS",
    "AttentionCall",
    "fp8_dtype",
]

#: The subset serving a uniform dense request, for the fixed-shape wrapper.
DENSE_PREFILL_KEYS = (
    "gqa_prefill_fp8_tensor_core_fwd_kernel",
    "gqa_prefill_dense_sliding_fwd_kernel",
    "gqa_prefill_square_fwd_kernel",
    "gqa_prefill_causal_fwd_kernel",
    "gqa_prefill_fwd_kernel",
)

#: Implementations of packed variable-length GQA prefill.
VARLEN_PREFILL_KEYS = (
    "gqa_prefill_varlen_fp8_tensor_core_fwd_kernel",
    "gqa_sliding_window_varlen_fwd_kernel",
    "gqa_prefill_varlen_fwd_kernel",
)

#: Implementations of paged GQA prefill.
PAGED_PREFILL_KEYS = (
    "gqa_prefill_paged_native_fp8_tensor_core_fwd_kernel",
    "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel",
    "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel",
    "gqa_prefill_paged_with_kv_cache_fwd_kernel",
)

#: Implementations of contiguous GQA decode.
DECODE_KEYS = ("gqa_decode_bs1_kernel", "gqa_decode_kernel")

#: Implementations of paged GQA decode.
PAGED_DECODE_KEYS = ("gqa_decode_paged_bs1_kernel", "gqa_decode_paged_kernel")
