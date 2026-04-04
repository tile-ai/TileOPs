import math
from typing import Tuple

import torch
import torch.nn.functional as F

from workloads.base import WorkloadBase


class MhaDecodePagedTest(WorkloadBase):

    def __init__(self, batch: int, heads: int, seqlen_q: int, seqlen_kv: int, dim: int,
                 page_size: int, is_causal: bool, dtype: torch.dtype) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
            self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pages = self.seqlen_kv // self.page_size
        real_seqlen_kv = torch.ones(
            (self.batch,), dtype=torch.int32, device="cuda") * self.seqlen_kv
        q = torch.randn(
            self.batch, self.seqlen_q, self.heads, self.dim, device="cuda", dtype=self.dtype)
        k = torch.randn(self.seqlen_kv, self.heads, self.dim, device="cuda", dtype=self.dtype)
        v = torch.randn(self.seqlen_kv, self.heads, self.dim, device="cuda", dtype=self.dtype)
        # Identity block_table: logical page i -> physical page i (contiguous layout)
        block_table = torch.arange(
            num_pages, dtype=torch.int32, device="cuda").unsqueeze(0).expand(self.batch, -1)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        block_table = block_table.contiguous()
        real_seqlen_kv = real_seqlen_kv.contiguous()

        return q, k, v, real_seqlen_kv, block_table

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    real_seqlen_kv: torch.Tensor, block_table: torch.Tensor) -> torch.Tensor:
        """Reassemble paged K/V to logical layout per batch, then SDPA."""
        batch, seqlen_q, heads, dim = q.shape
        seqlen_kv = k.shape[0]
        out_list = []
        for i_b in range(batch):
            q_b = q[i_b:i_b + 1, :, :, :]
            k_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            v_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            num_pages = math.ceil(real_seqlen_kv[i_b].item() / self.page_size)
            for i_paged in range(num_pages):
                start_pos = block_table[i_b, i_paged].item() * self.page_size
                end_pos = min(start_pos + self.page_size, seqlen_kv)
                page_len = end_pos - start_pos
                k_logical[i_paged * self.page_size:i_paged * self.page_size +
                          page_len, :, :] = k[start_pos:end_pos, :, :]
                v_logical[i_paged * self.page_size:i_paged * self.page_size +
                          page_len, :, :] = v[start_pos:end_pos, :, :]
            k_logical = k_logical[:real_seqlen_kv[i_b].item(), :, :]
            v_logical = v_logical[:real_seqlen_kv[i_b].item(), :, :]
            k_b = k_logical.unsqueeze(0)
            v_b = v_logical.unsqueeze(0)
            q_bhsd = q_b.transpose(1, 2)
            k_bhsd = k_b.transpose(1, 2)
            v_bhsd = v_b.transpose(1, 2)
            out_b = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
            out_b = out_b.transpose(1, 2).contiguous()
            out_list.append(out_b)
        return torch.cat(out_list, dim=0)
