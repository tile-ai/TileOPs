import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from top.kernels.kernel import Kernel

__all__ = ["gqa_decode_paged_kernel"]


def _gqa_decode_kernel(batch, heads, groups, seqlen_kv, dim, page_size, dtype):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    accum_dtype = "float"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _gqa_decode_func(block_H, block_N, num_split, num_stages, threads):

        shape_q = [batch, heads, dim]
        shape_kv = [seqlen_kv, groups, dim]
        shape_o = [batch, heads, dim]
        kv_group_num = heads // groups

        part_shape = [batch, heads, num_split, dim]
        valid_block_H = min(block_H, kv_group_num)

        @T.macro
        def _gqa_decode_no_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_kv, dtype),
                V: T.Tensor(shape_kv, dtype),
                real_seqlen_kv: T.Tensor([batch], T.int32),
                block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
                Output: T.Tensor([batch, heads, dim], dtype),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):

                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                bid = bx
                hid = by
                cur_kv_head = hid // (kv_group_num // valid_block_H)
                seqlen_kv_b = real_seqlen_kv[bid]

                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv(seqlen_kv_b, block_N)
                num_blockn_in_page = page_size // block_N

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    page_idx = k // num_blockn_in_page
                    block_idx_in_page = k % num_blockn_in_page
                    blockn_num_offset = block_table[
                        bid, page_idx] * num_blockn_in_page + block_idx_in_page

                    T.copy(
                        K[blockn_num_offset * block_N:(blockn_num_offset + 1) * block_N,
                          cur_kv_head, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else((k * block_N + j < seqlen_kv_b), acc_s[i, j],
                                                     -T.infinity(accum_dtype))
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(
                        V[blockn_num_offset * block_N:(blockn_num_offset + 1) * block_N,
                          cur_kv_head, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                for i in T.Parallel(block_H):
                    logsum_safe = T.if_then_else(logsum[i] == 0, 1, logsum[i])
                    logsum[i] = T.log2(logsum_safe) + scores_max[i] * scale

                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output[bid, hid * valid_block_H:(hid + 1) * valid_block_H, :])

        @T.macro
        def _gqa_decode_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_kv, dtype),
                V: T.Tensor(shape_kv, dtype),
                real_seqlen_kv: T.Tensor([batch], T.int32),
                block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                split_length: T.Tensor([batch, num_split], "int32"),
        ):
            with T.Kernel(
                    batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)

                split_length_shared = T.alloc_shared([num_split], "int32")
                bid = bx
                hid = by
                sid = bz
                T.copy(split_length[bid, :], split_length_shared, disable_tma=True)
                cur_kv_head = hid // (kv_group_num // valid_block_H)
                seqlen_kv_b = real_seqlen_kv[bid]

                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # Per-batch loop_range: only iterate blocks that are within real_seqlen_kv[bid],
                # so shorter batches (e.g. batch1 with 2048) don't run empty splits -> all -inf -> NaN
                start_block_sid = T.if_then_else(sid > 0, split_length_shared[sid - 1] // block_N,
                                                 0)
                end_block_valid = T.ceildiv(seqlen_kv_b, block_N)
                blocks_valid_this_split = end_block_valid - start_block_sid
                blocks_in_split = T.if_then_else(
                    sid > 0,
                    T.ceildiv(split_length_shared[sid] - split_length_shared[sid - 1], block_N),
                    T.ceildiv(split_length_shared[0], block_N),
                )
                loop_range = T.if_then_else(
                    blocks_valid_this_split <= 0,
                    0,
                    T.if_then_else(
                        blocks_valid_this_split <= blocks_in_split,
                        blocks_valid_this_split,
                        blocks_in_split,
                    ),
                )

                num_blockn_in_page = page_size // block_N
                offset = 0 if sid == 0 else split_length_shared[sid - 1] // block_N
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    k_global = k
                    k_global += offset
                    page_idx = k_global // num_blockn_in_page
                    block_idx_in_page = k_global % num_blockn_in_page
                    blockn_num_offset = block_table[
                        bid, page_idx] * num_blockn_in_page + block_idx_in_page

                    T.copy(
                        K[blockn_num_offset * block_N:(blockn_num_offset + 1) * block_N,
                          cur_kv_head, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    start_sid = T.if_then_else(sid > 0, split_length_shared[sid - 1], 0)
                    for i, j in T.Parallel(block_H, block_N):
                        logical_pos = start_sid + k * block_N + j
                        acc_s[i, j] = T.if_then_else(logical_pos < seqlen_kv_b, acc_s[i, j],
                                                     -T.infinity(accum_dtype))
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_H):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_H, dim):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(
                        V[blockn_num_offset * block_N:(blockn_num_offset + 1) * block_N,
                          cur_kv_head, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    # When loop_range was 0 (split entirely beyond real_seqlen_kv), logsum=0 -> avoid 0/0
                    acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                for i in T.Parallel(block_H):
                    # Avoid log2(0)=-inf when logsum=0; glse=-inf is ok in combine (weight 0)
                    logsum_safe = T.if_then_else(logsum[i] == 0, 1, logsum[i])
                    logsum[i] = T.log2(logsum_safe) + scores_max[i] * scale

                for i in T.Parallel(block_H):
                    if i < valid_block_H:
                        glse[bid, hid * valid_block_H + i, sid] = logsum[i]
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output_partial[bid, hid * valid_block_H:(hid + 1) * valid_block_H,
                                                sid, :])

        @T.macro
        def combine(
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, batch, threads=128) as (by, bz):
                #
                glse_vec = T.alloc_fragment([num_split], dtype)
                for k in T.Parallel(num_split):
                    glse_vec[k] = glse[bz, by, k]
                lse_max = T.alloc_fragment([1], accum_dtype)
                T.fill(lse_max, -T.infinity(accum_dtype))
                T.reduce_max(glse_vec, lse_max, dim=0, clear=False)

                #
                lse_logsum = T.alloc_local([1], accum_dtype)
                lse_logsum[0] = 0
                for k in T.serial(num_split):
                    lse_logsum[0] += T.exp2(glse[bz, by, k] - lse_max[0])
                lse_logsum[0] = T.log2(lse_logsum[0]) + lse_max[0]

                #
                o_accum = T.alloc_fragment([dim], accum_dtype)
                T.clear(o_accum)
                for k in T.serial(num_split):
                    w = T.exp2(glse[bz, by, k] - lse_logsum[0])
                    for i in T.Parallel(dim):
                        o_accum[i] += Output_partial[bz, by, k, i] * w
                for i in T.Parallel(dim):
                    Output[bz, by, i] = o_accum[i]

        @T.prim_func
        def gqa_decode_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_kv, dtype),
                V: T.Tensor(shape_kv, dtype),
                real_seqlen_kv: T.Tensor([batch], T.int32),
                block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
                glse: T.Tensor([batch, heads, num_split], dtype),
                Output_partial: T.Tensor(part_shape, dtype),
                split_length: T.Tensor([batch, num_split], "int32"),
                Output: T.Tensor(shape_o, dtype),
        ):
            _gqa_decode_split(Q, K, V, real_seqlen_kv, block_table, glse, Output_partial,
                              split_length)
            combine(glse, Output_partial, Output)

        @T.prim_func
        def gqa_decode_no_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_kv, dtype),
                V: T.Tensor(shape_kv, dtype),
                real_seqlen_kv: T.Tensor([batch], T.int32),
                block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32),
                Output: T.Tensor(shape_o, dtype),
        ):
            _gqa_decode_no_split(Q, K, V, real_seqlen_kv, block_table, Output)

        if num_split > 1:
            return gqa_decode_split
        else:
            return gqa_decode_no_split

    return _gqa_decode_func


# Use a distinct op name so paged and non-paged (gqa_decode.py) do not overwrite each other.
@torch.library.custom_op("top::gqa_decode_paged_wrapped_kernel", mutates_args=())
def _gqa_decode_wrapped_kernel(batch: int, heads: int, groups: int, seqlen_kv: int, dim: int,
                               page_size: int, dtype: str, block_H: int, block_N: int,
                               num_stages: int, threads: int, num_split: int, Q: torch.Tensor,
                               K: torch.Tensor, V: torch.Tensor, real_seqlen_kv: torch.Tensor,
                               block_table: torch.Tensor, glse: torch.Tensor,
                               Output_partial: torch.Tensor) -> torch.Tensor:

    assert K.shape[0] == V.shape[0] == seqlen_kv, "error: dimension mismatch!"
    assert K.shape[1] == V.shape[1] == groups, "error: groups mismatch!"
    real_max = real_seqlen_kv.max().item() if real_seqlen_kv.dim() > 0 else real_seqlen_kv.item()
    split_length = torch.zeros(batch, num_split, dtype=torch.int32, device=Q.device)
    for i in range(batch):
        for k in range(num_split):
            split_length[i, k] = real_max // (num_split * block_N) * block_N
        split_length[i, -1] = real_max - (num_split - 1) * (
            real_max // (num_split * block_N) * block_N)
    acc_split_length = torch.zeros(batch, num_split, dtype=torch.int32, device=Q.device)
    for i in range(batch):
        acc_split_length[i, 0] = split_length[i, 0]
    for i in range(batch):
        for k in range(1, num_split):
            acc_split_length[i, k] = acc_split_length[i, k - 1] + split_length[i, k]

    if num_split == 1:
        return _gqa_decode_kernel(batch, heads, groups, seqlen_kv, dim, page_size,
                                  dtype)(block_H, block_N, num_split, num_stages,
                                         threads)(Q, K, V, real_seqlen_kv, block_table)

    return _gqa_decode_kernel(batch, heads, groups, seqlen_kv, dim, page_size,
                              dtype)(block_H, block_N, num_split, num_stages,
                                     threads)(Q, K, V, real_seqlen_kv, block_table, glse,
                                              Output_partial, acc_split_length)


@_gqa_decode_wrapped_kernel.register_fake
def _(
        batch: int,
        heads: int,
        groups: int,
        seqlen_kv: int,
        dim: int,
        page_size: int,
        block_H: int,
        block_N: int,
        num_stages: int,
        threads: int,
        num_split: int,
        *inputs
) -> torch.Tensor:
    return torch.empty_like(inputs[0])


class gqa_decode_paged_kernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 heads,
                 groups,
                 seqlen_kv,
                 dim,
                 page_size,
                 dtype="float16",
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.heads = heads
        self.groups = groups
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.dtype = dtype

        self.kernel = _gqa_decode_kernel(self.batch, self.heads, self.groups, self.seqlen_kv,
                                         self.dim, self.page_size, self.dtype_str)

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        # block_N must be <= page_size so num_blockn_in_page = page_size // block_N >= 1 (no div by zero)
        block_N = min(128, self.page_size)
        return {"block_H": 64, "block_N": block_N, "num_split": 16, "num_stages": 2, "threads": 128}

    @property
    def autotune_configs(self) -> list[dict]:
        block_N = [64, 128]
        block_H = [64]
        num_split = [2, 4, 8]
        num_stages = [1, 2, 3]
        threads = [128]
        _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads))

        configs = [{
            'block_N': c[0],
            'block_H': c[1],
            'num_split': c[2],
            'num_stages': c[3],
            'threads': c[4]
        } for c in _configs if c[0] <= self.page_size]
        return configs

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                real_seqlen_kv: torch.Tensor, block_table: torch.Tensor):
        glse = torch.empty((self.batch, self.heads, self.config["num_split"]),
                           dtype=self.dtype,
                           device=Q.device)
        Output_partial = torch.empty((self.batch, self.heads, self.config["num_split"], self.dim),
                                     dtype=self.dtype,
                                     device=Q.device)
        return _gqa_decode_wrapped_kernel(self.batch, self.heads, self.groups, self.seqlen_kv,
                                          self.dim, self.page_size, self.dtype_str,
                                          self.config["block_H"], self.config["block_N"],
                                          self.config["num_stages"], self.config["threads"],
                                          self.config["num_split"], Q, K, V, real_seqlen_kv,
                                          block_table, glse, Output_partial)


def main():
    import math
    from torch.nn import functional as F

    batch = 2
    heads = 16
    groups = 8
    seqlen_kv = 10240
    dim = 128
    page_size = 1024
    dtype = torch.float16

    Q = torch.randn(batch, heads, dim, dtype=dtype, device="cuda")
    K = torch.randn(seqlen_kv, groups, dim, dtype=dtype, device="cuda")
    V = torch.randn(seqlen_kv, groups, dim, dtype=dtype, device="cuda")
    real_seqlen_kv = torch.tensor([4096, 2048], dtype=torch.int32, device="cuda")
    block_table = torch.randint(
        0,
        seqlen_kv // page_size, (batch, seqlen_kv // page_size),
        dtype=torch.int32,
        device="cuda")

    kernel = gqa_decode_paged_kernel(batch, heads, groups, seqlen_kv, dim, page_size, dtype)
    output = kernel.forward(Q, K, V, real_seqlen_kv, block_table)
    print("kernel output shape:", output.shape)
    print("kernel output:", output)

    # torch reference: reassemble paged K/V to logical layout per batch, then SDPA with GQA
    kv_group_num = heads // groups
    for i_b in range(batch):
        q = Q[i_b:i_b + 1, :, :]  # [1, heads, dim]
        k_logical = torch.zeros(seqlen_kv, groups, dim, dtype=dtype, device="cuda")
        v_logical = torch.zeros(seqlen_kv, groups, dim, dtype=dtype, device="cuda")
        num_pages = math.ceil(real_seqlen_kv[i_b].item() / page_size)
        for i_paged in range(num_pages):
            start_pos = block_table[i_b, i_paged].item() * page_size
            end_pos = min(start_pos + page_size, seqlen_kv)
            page_len = end_pos - start_pos
            k_logical[i_paged * page_size:i_paged * page_size +
                      page_len, :, :] = K[start_pos:end_pos, :, :]
            v_logical[i_paged * page_size:i_paged * page_size +
                      page_len, :, :] = V[start_pos:end_pos, :, :]
        k_logical = k_logical[:real_seqlen_kv[i_b].item(), :, :]  # [real_len, groups, dim]
        v_logical = v_logical[:real_seqlen_kv[i_b].item(), :, :]

        # GQA: expand K/V from groups to heads (each group shared by kv_group_num heads)
        group_id = torch.arange(heads, dtype=torch.long, device="cuda") // kv_group_num  # [heads]
        k_bhsd = k_logical[:, group_id, :].unsqueeze(0).transpose(1, 2)  # [1, heads, S_kv, dim]
        v_bhsd = v_logical[:, group_id, :].unsqueeze(0).transpose(1, 2)
        q_bhsd = q.unsqueeze(2)  # [1, heads, 1, dim]

        output_ref = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
        output_ref = output_ref.squeeze(2)  # [1, heads, dim]
        print("torch ref output batch", i_b, "shape:", output_ref.shape)
        print("torch ref output batch", i_b, ":", output_ref)
        out_slice = output[i_b:i_b + 1, :, :]
        diff = (output_ref - out_slice).abs().max().item()
        print("max diff batch", i_b, ":", diff)


if __name__ == "__main__":
    main()
