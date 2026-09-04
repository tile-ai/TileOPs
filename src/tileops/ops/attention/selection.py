"""Which dispatch key an attention call takes.

The keys implementing each attention slot, and the check that a request does not
contradict the user-visible ``backend`` parameter. Choosing among the keys is
``Op.select_kernel_key``; see docs/design/ops-design.md § Kernel selection.
"""

from tileops.kernels.attention.call_spec import AttentionCall, fp8_dtype

__all__ = [
    "PAGED_DECODE_KEYS",
    "PAGED_PREFILL_KEYS",
    "AttentionCall",
    "fp8_dtype",
]

# Implementations of paged GQA prefill.
PAGED_PREFILL_KEYS = (
    "gqa_prefill_paged_with_kv_cache_rope_fwd_kernel",
    "gqa_prefill_paged_with_fp8_kv_cache_fwd_kernel",
    "gqa_prefill_paged_with_kv_cache_fwd_kernel",
)

# Implementations of contiguous GQA decode.
# Implementations of paged GQA decode.
PAGED_DECODE_KEYS = ("gqa_decode_paged_bs1_kernel", "gqa_decode_paged_kernel")

# Implementations of paged MHA decode.
MHA_PAGED_DECODE_KEYS = ("mha_decode_paged_ws_kernel", "mha_decode_paged_kernel")
