from typing import Optional, Tuple

import torch

from workloads.base import WorkloadBase


class Fp8LightingIndexerTest(WorkloadBase):

    def __init__(self,
                 batch: int,
                 seq_len: int,
                 heads: int,
                 index_dim: int,
                 seq_len_kv: int,
                 kv_group: int,
                 clean_logits: bool = True,
                 config: Optional[dict] = None):
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.clean_logits = clean_logits
        self.config = config
        self.dtype = torch.float8_e4m3fn
        self.accum_dtype = torch.float32
        self.index_dtype = torch.int32

    def cal_seq_idx_for_q(self, cu_seqlens_qs: torch.LongTensor, cu_seqlens_qe: torch.LongTensor,
                          seq_len: int) -> torch.IntTensor:
        seq_idx_for_q = torch.zeros(seq_len, dtype=torch.int32, device=cu_seqlens_qs.device)
        if len(cu_seqlens_qs) > 1:
            seq_idx_for_q[cu_seqlens_qs[1:]] = 1
        return torch.cumsum(seq_idx_for_q, dim=0, dtype=torch.int32)

    def cal_cu_seqlen_ke_for_q(
        self,
        cu_seqlens_qs: torch.LongTensor,
        cu_seqlens_qe: torch.LongTensor,
        cu_seqlens_ks: torch.LongTensor,
        cu_seqlens_ke: torch.LongTensor,
        q_start_idxs: torch.LongTensor,
        seq_len: int,
        kv_stride: int,
    ) -> torch.IntTensor:
        cu_seqlen_ke_for_each_q = torch.gather(
            input=torch.cat(
                [cu_seqlens_ke,
                 torch.zeros(1, dtype=torch.int32, device=cu_seqlens_qs.device)]),
            dim=0,
            index=self.cal_seq_idx_for_q(
                cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long(),
        )
        casual_cu_seqlen_ke_for_each_q = torch.zeros((seq_len,),
                                                     dtype=torch.int32,
                                                     device=cu_seqlens_qs.device)
        for i in range(len(cu_seqlens_qs)):
            casual_cu_seqlen_ke_for_each_q[cu_seqlens_qs[i]:cu_seqlens_qe[i]] = (torch.arange(
                q_start_idxs[i],
                q_start_idxs[i] + cu_seqlens_qe[i] - cu_seqlens_qs[i],
                dtype=torch.int32,
                device=cu_seqlens_qs.device) + 1) // kv_stride + cu_seqlens_ks[i]
        cu_seqlen_ke_for_each_q = torch.minimum(casual_cu_seqlen_ke_for_each_q,
                                                cu_seqlen_ke_for_each_q)
        return cu_seqlen_ke_for_each_q.int()

    def cal_cu_seqlen_ks_for_q(
        self,
        cu_seqlens_qs: torch.LongTensor,
        cu_seqlens_qe: torch.LongTensor,
        cu_seqlens_ks: torch.LongTensor,
        seq_len: int,
    ) -> torch.IntTensor:
        cu_seqlen_ks_for_each_q = torch.gather(
            input=torch.cat([
                cu_seqlens_ks,
                torch.full((1,),
                           torch.iinfo(torch.int32).max,
                           dtype=torch.int32,
                           device=cu_seqlens_qs.device)
            ]),
            dim=0,
            index=self.cal_seq_idx_for_q(
                cu_seqlens_qs=cu_seqlens_qs, cu_seqlens_qe=cu_seqlens_qe, seq_len=seq_len).long(),
        )
        return cu_seqlen_ks_for_each_q.int()

    def generate_random_cu_seqlens(self,
                                   cp_size: int = 4,
                                   cp_rank: int = 3,
                                   kv_stride: int = 1,
                                   average_q_len: int = 512):
        total_seqlen = self.seq_len * cp_size

        cu_seqlens = torch.randint(0, average_q_len * 2,
                                   (total_seqlen // average_q_len * 2,)).cuda()
        last_seq_id = torch.where(cu_seqlens.cumsum(0) >= total_seqlen)[0][0]
        cu_seqlens = cu_seqlens[:last_seq_id]

        if cu_seqlens.sum() < total_seqlen:
            cu_seqlens = torch.cat(
                [cu_seqlens, torch.tensor([total_seqlen - cu_seqlens.sum()]).cuda()])

        cu_seqlens_cumsum = torch.cumsum(cu_seqlens, dim=0)
        cu_seqlens_k_cumsum = torch.cumsum(cu_seqlens // kv_stride, dim=0)
        cu_seqlens_qs = torch.cat([torch.tensor([0]).cuda(), cu_seqlens_cumsum[:-1]])
        cu_seqlens_ks = torch.cat([torch.tensor([0]).cuda(), cu_seqlens_k_cumsum[:-1]])
        cu_seqlens_qe = cu_seqlens_cumsum.clone()
        cu_seqlens_ke = cu_seqlens_k_cumsum.clone()

        cu_seqlens_ks_for_each_q = self.cal_cu_seqlen_ks_for_q(
            cu_seqlens_qs,
            cu_seqlens_qe,
            cu_seqlens_ks,
            total_seqlen,
        )

        cu_seqlens_ke_for_each_q = self.cal_cu_seqlen_ke_for_q(
            cu_seqlens_qs=cu_seqlens_qs,
            cu_seqlens_qe=cu_seqlens_qe,
            cu_seqlens_ks=cu_seqlens_ks,
            cu_seqlens_ke=cu_seqlens_ke,
            q_start_idxs=torch.zeros_like(cu_seqlens_qs),
            seq_len=total_seqlen,
            kv_stride=kv_stride,
        )

        assert self.seq_len % 2 == 0
        per_chunk_seqlen = self.seq_len // 2
        slice_short = slice(cp_rank * per_chunk_seqlen, (cp_rank + 1) * per_chunk_seqlen)
        slice_long = slice(
            total_seqlen - (cp_rank + 1) * per_chunk_seqlen,
            total_seqlen - cp_rank * per_chunk_seqlen,
        )

        ks = torch.cat([
            cu_seqlens_ks_for_each_q[slice_short],
            cu_seqlens_ks_for_each_q[slice_long],
        ])
        ke = torch.cat([
            cu_seqlens_ke_for_each_q[slice_short],
            cu_seqlens_ke_for_each_q[slice_long],
        ])
        assert len(ks) == len(ke) == self.seq_len
        return ks, ke

    def gen_inputs(
            self,
            params=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        IndexQ = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.index_dim,
            device='cuda',
            dtype=torch.bfloat16)
        IndexK = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            device='cuda',
            dtype=torch.bfloat16)
        Weights = torch.randn(self.seq_len, self.heads, device='cuda', dtype=self.accum_dtype)
        CuSeqLenKS, CuSeqLenKE = self.generate_random_cu_seqlens(
            cp_size=4, cp_rank=3, kv_stride=1, average_q_len=2048)
        return IndexQ, IndexK, Weights, CuSeqLenKS, CuSeqLenKE
