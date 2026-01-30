import itertools
from typing import Optional

import torch
import tilelang
import tilelang.language as T

from top.kernels.kernel import Kernel

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
    bits_uint = T.reinterpret(T.uint32, x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast(T.uint32, (0xFFFFFFFF)),
        bits_uint | T.Cast(T.uint32, (0x80000000)),
    )
    return bits_uint


def _topk_selector_kernel(batch, seq_len, topk, in_dtype, out_dtype):

    @tilelang.jit(
        out_idx=[1], pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        })
    def topk_selector_fwd_func(RADIX=1 << 8, BLOCK_SIZE=1024, SMEM_INPUT_SIZE=4096):
        batch = T.dynamic("batch")
        seq_len = T.dynamic("seq_len")

        @T.prim_func
        def _topk_selector_kernel_main(
            index_score: T.Tensor[(batch, seq_len), in_dtype],
            index: T.Tensor[(batch, topk), out_dtype],
            starts: T.Tensor[(batch), out_dtype],
            ends: T.Tensor[(batch), out_dtype],
        ):
            with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
                tx = T.get_thread_binding()

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

                l_new_topk = topk
                l_start_idx = starts[bx]
                l_end_idx = ends[bx]

                # stage 1: use 8bit to do quick topk
                T.fill(s_histogram, 0)
                T.fill(s_num_input[0], 0)

                T.sync_threads()
                for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                    input_idx = s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                        inval_int16 = convert_to_uint16(index_score[bx, input_idx])
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
                for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                    T.sync_threads()
                    input_idx = s * BLOCK_SIZE + tx
                    if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                        bin_id = convert_to_uint16(index_score[bx, input_idx])
                        l_bin_id32 = T.Cast(T.int32, bin_id)
                        if l_bin_id32 > l_threshold_bin_id:
                            # need a pos = T.atomic_add(s_histogram[bin_id32+1], 1)
                            pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                            index[bx, pos] = input_idx

                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            # pos = s_num_input[0]
                            pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                            s_input_idx[0, pos] = input_idx

                # stage 2: tail pass
                for round in T.serial(4):
                    if l_new_topk <= 0:
                        T.loop_break()

                    r_idx = round % 2
                    l_start_pos = topk - l_new_topk

                    T.sync_threads()
                    T.fill(s_histogram, 0)
                    if tx == 0:
                        s_num_input[r_idx ^ 1] = 0
                    T.sync_threads()

                    l_num_input = s_num_input[r_idx]
                    for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                        if s * BLOCK_SIZE + tx < l_num_input:
                            l_bin_id32 = T.Cast(T.int32, ((convert_to_uint32(
                                index_score[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
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
                                index_score[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]]) >>
                                                           (24 - round * 8)) & 0xFF))
                            if l_bin_id32 > l_threshold_bin_id:
                                pos = T.atomic_add(
                                    s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                                if round == 3:
                                    l_out_pos = T.atomic_add(
                                        s_histogram[l_bin_id32 + 1], 1,
                                        return_prev=True) + l_start_pos
                                    if l_out_pos < topk:
                                        index[bx, l_out_pos] = s_input_idx[r_idx,
                                                                           s * BLOCK_SIZE + tx]
                                else:
                                    pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                    s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx,
                                                                              s * BLOCK_SIZE + tx]

        return _topk_selector_kernel_main

    return topk_selector_fwd_func


@torch.library.custom_op("top::topk_selector_wrapped_kernel", mutates_args=())
def _topk_selector_wrapped_kernel(
    batch: int,
    seq_len: int,
    topk: int,
    in_dtype: str,
    out_dtype: str,
    RADIX: int,
    BLOCK_SIZE: int,
    SMEM_INPUT_SIZE: int,
    index_score: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> torch.Tensor:
    return _topk_selector_kernel(batch, seq_len, topk, in_dtype,
                                 out_dtype)(RADIX, BLOCK_SIZE, SMEM_INPUT_SIZE)(index_score, starts,
                                                                                ends)


@_topk_selector_wrapped_kernel.register_fake
def _(batch, seq_len, topk, in_dtype, out_dtype, *inputs) -> None:
    return torch.empty([batch, topk], device=inputs[0].device, dtype=torch.int32)


class TopkSelectorKernel(Kernel):

    supported_archs: list[int] = [90]

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 topk: int,
                 in_dtype: str,
                 out_dtype: str,
                 config: Optional[dict] = None,
                 tune: bool = False):
        super().__init__()
        self.batch = batch
        self.seq_len = seq_len
        self.topk = topk
        self.out_dtype = out_dtype
        self.in_dtype_str = str(in_dtype).split('.')[-1]
        self.out_dtype_str = str(out_dtype).split('.')[-1]

        self.kernel = _topk_selector_kernel(self.batch, self.seq_len, self.topk, self.in_dtype_str,
                                            self.out_dtype_str)
        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "RADIX": 1 << 8,
            "BLOCK_SIZE": 1024,
            "SMEM_INPUT_SIZE": 4096,
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
        _configs = list(itertools.product(RADIX, BLOCK_SIZE, SMEM_INPUT_SIZE))

        return [{'RADIX': c[0], 'BLOCK_SIZE': c[1], 'SMEM_INPUT_SIZE': c[2]} for c in _configs]

    def forward(self, index_score: torch.Tensor, starts: torch.Tensor,
                ends: torch.Tensor) -> torch.Tensor:
        return _topk_selector_wrapped_kernel(self.batch, self.seq_len, self.topk, self.in_dtype_str,
                                             self.out_dtype_str, self.config["RADIX"],
                                             self.config["BLOCK_SIZE"],
                                             self.config["SMEM_INPUT_SIZE"], index_score, starts,
                                             ends)
