"""Compile-time mask policy shared by dense and ragged GQA prefill."""

import tilelang.language as T

__all__ = ["make_bottom_right_attention_mask"]


def make_bottom_right_attention_mask(
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    accum_dtype: str,
    block_m: int,
    block_n: int,
    preserve_valid: bool = False,
):
    """Build a bottom-right-aligned causal/window mask macro.

    ``preserve_valid=False`` initializes an MMA accumulator with zero for
    visible entries. ``preserve_valid=True`` applies the same mask after a
    score transform such as softcap without overwriting visible scores.
    """
    has_left_window = window_size_left >= 0
    has_right_window = window_size_right >= 0

    @T.macro
    def apply_mask(acc_s, k_idx, bx, q_len, kv_len, offset):
        if is_causal and has_left_window:
            for i, j in T.Parallel(block_m, block_n):
                q_pos = bx * block_m + i
                kv_pos = k_idx * block_n + j
                center = q_pos + offset
                masked = (
                    (q_pos >= q_len)
                    or (kv_pos >= kv_len)
                    or (kv_pos > center)
                    or (kv_pos < center - window_size_left)
                )
                acc_s[i, j] = T.if_then_else(
                    masked,
                    -T.infinity(accum_dtype),
                    acc_s[i, j] if preserve_valid else 0,
                )
        elif is_causal:
            # A non-negative right window cannot narrow the causal upper bound.
            for i, j in T.Parallel(block_m, block_n):
                q_pos = bx * block_m + i
                kv_pos = k_idx * block_n + j
                center = q_pos + offset
                masked = (
                    (q_pos >= q_len) or (kv_pos >= kv_len) or (kv_pos > center)
                )
                acc_s[i, j] = T.if_then_else(
                    masked,
                    -T.infinity(accum_dtype),
                    acc_s[i, j] if preserve_valid else 0,
                )
        elif has_left_window and has_right_window:
            for i, j in T.Parallel(block_m, block_n):
                q_pos = bx * block_m + i
                kv_pos = k_idx * block_n + j
                center = q_pos + offset
                masked = (
                    (q_pos >= q_len)
                    or (kv_pos >= kv_len)
                    or (kv_pos < center - window_size_left)
                    or (kv_pos > center + window_size_right)
                )
                acc_s[i, j] = T.if_then_else(
                    masked,
                    -T.infinity(accum_dtype),
                    acc_s[i, j] if preserve_valid else 0,
                )
        elif has_left_window:
            for i, j in T.Parallel(block_m, block_n):
                q_pos = bx * block_m + i
                kv_pos = k_idx * block_n + j
                center = q_pos + offset
                masked = (
                    (q_pos >= q_len)
                    or (kv_pos >= kv_len)
                    or (kv_pos < center - window_size_left)
                )
                acc_s[i, j] = T.if_then_else(
                    masked,
                    -T.infinity(accum_dtype),
                    acc_s[i, j] if preserve_valid else 0,
                )
        elif has_right_window:
            for i, j in T.Parallel(block_m, block_n):
                q_pos = bx * block_m + i
                kv_pos = k_idx * block_n + j
                center = q_pos + offset
                masked = (
                    (q_pos >= q_len)
                    or (kv_pos >= kv_len)
                    or (kv_pos > center + window_size_right)
                )
                acc_s[i, j] = T.if_then_else(
                    masked,
                    -T.infinity(accum_dtype),
                    acc_s[i, j] if preserve_valid else 0,
                )
        else:
            for i, j in T.Parallel(block_m, block_n):
                q_pos = bx * block_m + i
                kv_pos = k_idx * block_n + j
                masked = (q_pos >= q_len) or (kv_pos >= kv_len)
                acc_s[i, j] = T.if_then_else(
                    masked,
                    -T.infinity(accum_dtype),
                    acc_s[i, j] if preserve_valid else 0,
                )

    return apply_mask
