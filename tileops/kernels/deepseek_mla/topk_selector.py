import itertools
from typing import Optional

import torch
import tilelang
import tilelang.language as T

from tileops.kernels.kernel import Kernel

# pass_configs = {
#     tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
# }

__all__ = ["TopkSelectorKernel"]


def convert_to_uint16(x):
    hval = T.Cast(T.float16, x)
    bits_uint = T.reinterpret(T.uint16, hval)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret(T.uint32, T.Cast(T.float32, x))
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast(T.uint32, (0xFFFFFFFF)),
        bits_uint | T.Cast(T.uint32, (0x80000000)),
    )
    return bits_uint


def _topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype):

    @tilelang.jit(
        out_idx=[1], pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        })
    def topk_selector_fwd_func(RADIX=1 << 8, BLOCK_SIZE=1024, SMEM_INPUT_SIZE=4096, block_m=32):
        batch = T.dynamic("batch")
        seq_len_kv = T.dynamic("seq_len_kv")

        @T.prim_func
        def _topk_selector_kernel_main(
            index_score: T.Tensor[(batch, seq_len, seq_len_kv, kv_group), in_dtype],
            index: T.Tensor[(batch, seq_len, kv_group, topk), out_dtype],
            starts: T.Tensor[(batch, seq_len), out_dtype],
            ends: T.Tensor[(batch, seq_len), out_dtype],
        ):
            with T.Kernel(
                    batch, T.ceildiv(seq_len, block_m), kv_group,
                    threads=BLOCK_SIZE) as (bx, by, g):
                tx = T.get_thread_binding()
                for m_i in T.Parallel(block_m):

                    s_threshold_bin_id = T.alloc_shared([1], T.int32)
                    s_histogram = T.alloc_shared([RADIX + 1], T.int32)
                    s_num_input = T.alloc_shared([2], T.int32)
                    s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

                    l_threshold_bin_id = T.alloc_var(T.int32)
                    l_new_topk = T.alloc_var(T.int32)
                    l_num_input = T.alloc_var(T.int32)
                    l_bin_id32 = T.alloc_var(T.int32)
                    l_val = T.alloc_var(T.int32)
                    l_start_pos = T.alloc_var(T.int32)
                    l_start_idx = T.alloc_var(T.int32)
                    l_end_idx = T.alloc_var(T.int32)
                    l_out_pos = T.alloc_var(T.int32)
                    l_pos = T.alloc_var(T.int32)

                    l_new_topk = topk
                    l_start_idx = starts[bx, by * block_m + m_i]
                    l_end_idx = ends[bx, by * block_m + m_i]

                    # stage 1: use 8bit to do quick topk
                    # T.fill(s_histogram, 0)
                    # T.fill(s_num_input[0], 0)

                    for j in T.serial(RADIX + 1):
                        s_histogram[j] = 0
                    s_num_input[0] = 0

                    T.sync_threads()

                    for s in T.serial(T.ceildiv(seq_len_kv, BLOCK_SIZE)):
                        input_idx = s * BLOCK_SIZE + tx
                        if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len_kv:
                            inval_int16 = convert_to_uint16(index_score[bx, by * block_m + m_i,
                                                                        input_idx, g])
                            T.atomic_add(s_histogram[inval_int16], 1)
                    T.sync_threads()

                    # cumsum
                    if tx < RADIX:
                        for i in T.serial(8):
                            offset = 1 << i
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                l_val = s_histogram[tx] + s_histogram[tx + offset]
                            T.sync_threads(3, RADIX)
                            if tx < RADIX - offset:
                                s_histogram[tx] = l_val

                        # find threshold bin id
                        T.sync_threads(3, RADIX)
                        if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                            s_threshold_bin_id[0] = tx
                    T.sync_threads()
                    l_threshold_bin_id = s_threshold_bin_id[0]
                    l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                    T.sync_threads()

                    # collect all elements with exponent ≥ threshold
                    for s in T.serial(T.ceildiv(seq_len_kv, BLOCK_SIZE)):
                        T.sync_threads()
                        input_idx = s * BLOCK_SIZE + tx
                        if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len_kv:
                            bin_id = convert_to_uint16(index_score[bx, by * block_m + m_i,
                                                                   input_idx, g])
                            l_bin_id32 = T.Cast(T.int32, bin_id)
                            if l_bin_id32 > l_threshold_bin_id:
                                # need a pos = T.atomic_add(s_histogram[bin_id32+1], 1)
                                l_pos = T.atomic_add(
                                    s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                                index[bx, by * block_m + m_i, g, l_pos] = input_idx

                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                # pos = s_num_input[0]
                                l_pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                                s_input_idx[0, l_pos] = input_idx

                    # stage 2: tail pass
                    for round in T.serial(4):
                        if l_new_topk <= 0:
                            T.loop_break()

                        r_idx = round % 2
                        l_start_pos = topk - l_new_topk

                        T.sync_threads()
                        for j in T.serial(RADIX + 1):
                            s_histogram[j] = 0
                        # T.fill(s_histogram, 0)
                        if tx == 0:
                            s_num_input[r_idx ^ 1] = 0
                        T.sync_threads()

                        l_num_input = s_num_input[r_idx]
                        for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                            if s * BLOCK_SIZE + tx < l_num_input:
                                l_bin_id32 = T.Cast(T.int32, ((convert_to_uint32(
                                    index_score[bx, by * block_m + m_i,
                                                s_input_idx[r_idx, s * BLOCK_SIZE + tx], g]) >>
                                                               (24 - round * 8)) & 0xFF))
                                T.atomic_add(s_histogram[l_bin_id32], 1)
                        T.sync_threads()
                        # cumsum
                        if tx < RADIX:
                            for i in T.serial(8):
                                offset = 1 << i
                                T.sync_threads(3, RADIX)
                                if tx < RADIX - offset:
                                    l_val = s_histogram[tx] + s_histogram[tx + offset]
                                T.sync_threads(3, RADIX)
                                if tx < RADIX - offset:
                                    s_histogram[tx] = l_val

                            # find threshold bin id
                            T.sync_threads(3, RADIX)
                            if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                                s_threshold_bin_id[0] = tx
                        T.sync_threads()

                        l_threshold_bin_id = s_threshold_bin_id[0]
                        l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                        T.sync_threads()

                        for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                            T.sync_threads()
                            if s * BLOCK_SIZE + tx < l_num_input:
                                l_bin_id32 = T.Cast(T.int32, ((convert_to_uint32(
                                    index_score[bx, by * block_m + m_i,
                                                s_input_idx[r_idx, s * BLOCK_SIZE + tx], g]) >>
                                                               (24 - round * 8)) & 0xFF))
                                if l_bin_id32 > l_threshold_bin_id:
                                    l_pos = T.atomic_add(
                                        s_histogram[l_bin_id32 + 1], 1,
                                        return_prev=True) + l_start_pos
                                    index[bx, by * block_m + m_i, g,
                                          l_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                                elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                    if round == 3:
                                        l_out_pos = T.atomic_add(
                                            s_histogram[l_bin_id32 + 1], 1,
                                            return_prev=True) + l_start_pos
                                        if l_out_pos < topk:
                                            index[bx, by * block_m + m_i, g,
                                                  l_out_pos] = s_input_idx[r_idx,
                                                                           s * BLOCK_SIZE + tx]
                                    else:
                                        l_pos = T.atomic_add(
                                            s_num_input[r_idx ^ 1], 1, return_prev=True)
                                        s_input_idx[r_idx ^ 1,
                                                    l_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

        return _topk_selector_kernel_main

    return topk_selector_fwd_func


@torch.library.custom_op("top::topk_selector_wrapped_kernel", mutates_args=())
def _topk_selector_wrapped_kernel(
    batch: int,
    seq_len: int,
    seq_len_kv: int,
    kv_group: int,
    topk: int,
    in_dtype: str,
    out_dtype: str,
    RADIX: int,
    BLOCK_SIZE: int,
    SMEM_INPUT_SIZE: int,
    block_m: int,
    index_score: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    return _topk_selector_kernel(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype,
                                 out_dtype)(RADIX, BLOCK_SIZE, SMEM_INPUT_SIZE,
                                            block_m)(index_score, starts, ends)


@_topk_selector_wrapped_kernel.register_fake
def _(batch, seq_len, seq_len_kv, kv_group, topk, in_dtype, out_dtype, *inputs) -> None:
    return torch.empty([batch, seq_len, kv_group, topk], device=inputs[0].device, dtype=torch.int32)


class TopkSelectorKernel(Kernel):

    supported_archs: list[int] = [90]

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 seq_len_kv: int,
                 kv_group: int,
                 topk: int,
                 in_dtype: torch.dtype,
                 out_dtype: torch.dtype,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.topk = topk
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.in_dtype_str = self.dtype_to_str(self.in_dtype)
        self.out_dtype_str = self.dtype_to_str(self.out_dtype)

        self.kernel = _topk_selector_kernel(self.batch, self.seq_len, self.seq_len_kv,
                                            self.kv_group, self.topk, self.in_dtype_str,
                                            self.out_dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "RADIX": 1 << 8,
            "BLOCK_SIZE": 1024,
            "SMEM_INPUT_SIZE": 4096,
            "block_m": 32,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        """
        Generates a list of autotuning configurations for the kernel.

        Returns:
            list[dict]: A list of dictionaries containing 'block_i' and 'threads' combinations.
        """
        RADIX = [1 << 8]
        BLOCK_SIZE = [1024]
        SMEM_INPUT_SIZE = [4096]
        block_m = [32]
        _configs = list(itertools.product(RADIX, BLOCK_SIZE, SMEM_INPUT_SIZE, block_m))

        return [{'RADIX': c[0], 'BLOCK_SIZE': c[1], 'SMEM_INPUT_SIZE': c[2]} for c in _configs]

    def forward(self, index_score: torch.Tensor, starts: torch.Tensor,
                ends: torch.Tensor) -> torch.Tensor:
        return _topk_selector_wrapped_kernel(self.batch, self.seq_len, self.seq_len_kv,
                                             self.kv_group, self.topk, self.in_dtype_str,
                                             self.out_dtype_str, self.config["RADIX"],
                                             self.config["BLOCK_SIZE"],
                                             self.config["SMEM_INPUT_SIZE"], self.config["block_m"],
                                             index_score, starts, ends)
