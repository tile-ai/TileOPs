"""Workload definitions for the MHA attention ops."""

import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from workloads.attention.gqa import _compute_gqa_square_lse
from workloads.workload_base import WorkloadBase


class MhaBwdWorkload(WorkloadBase):
    def __init__(
        self, batch: int, heads: int, seq_len: int, dim: int, is_causal: bool, dtype: torch.dtype
    ):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device="cuda",
            requires_grad=True,
        )
        k = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device="cuda",
            requires_grad=True,
        )
        v = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device="cuda",
            requires_grad=True,
        )
        grad_output = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device="cuda"
        )

        with torch.no_grad():
            o = (
                F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    is_causal=self.is_causal,
                )
                .transpose(1, 2)
                .contiguous()
            )
            lse = _compute_gqa_square_lse(
                q,
                k,
                heads=self.heads,
                heads_kv=self.heads,
                dim=self.dim,
                is_causal=self.is_causal,
            )

        return q, k, v, o, grad_output, lse

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        grad_output: torch.Tensor,
        lse: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_bhsd = q.transpose(1, 2)  # [B, H, S, D]
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            output_bhsd = F.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=self.is_causal
            )
        output = output_bhsd.transpose(1, 2).contiguous()

        output.backward(grad_output)
        return q.grad, k.grad, v.grad


class MhaDecodePagedWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        seqlen_q: int,
        seqlen_kv: int,
        dim: int,
        page_size: int,
        is_causal: bool,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.dim = dim
        self.page_size = page_size
        self.is_causal = is_causal
        self.dtype = dtype

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_pages = self.seqlen_kv // self.page_size
        real_seqlen_kv = (
            torch.ones((self.batch,), dtype=torch.int32, device="cuda") * self.seqlen_kv
        )
        q = torch.randn(
            self.batch, self.seqlen_q, self.heads, self.dim, device="cuda", dtype=self.dtype
        )
        k = torch.randn(self.seqlen_kv, self.heads, self.dim, device="cuda", dtype=self.dtype)
        v = torch.randn(self.seqlen_kv, self.heads, self.dim, device="cuda", dtype=self.dtype)
        # Identity block_table: logical page i -> physical page i (contiguous layout)
        block_table = (
            torch.arange(num_pages, dtype=torch.int32, device="cuda")
            .unsqueeze(0)
            .expand(self.batch, -1)
        )

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        block_table = block_table.contiguous()
        real_seqlen_kv = real_seqlen_kv.contiguous()

        return q, k, v, real_seqlen_kv, block_table

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        real_seqlen_kv: torch.Tensor,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Reassemble paged K/V to logical layout per batch, then SDPA."""
        batch, seqlen_q, heads, dim = q.shape
        seqlen_kv = k.shape[0]
        out_list = []
        for i_b in range(batch):
            q_b = q[i_b : i_b + 1, :, :, :]
            k_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            v_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            num_pages = math.ceil(real_seqlen_kv[i_b].item() / self.page_size)
            for i_paged in range(num_pages):
                start_pos = block_table[i_b, i_paged].item() * self.page_size
                end_pos = min(start_pos + self.page_size, seqlen_kv)
                page_len = end_pos - start_pos
                k_logical[i_paged * self.page_size : i_paged * self.page_size + page_len, :, :] = k[
                    start_pos:end_pos, :, :
                ]
                v_logical[i_paged * self.page_size : i_paged * self.page_size + page_len, :, :] = v[
                    start_pos:end_pos, :, :
                ]
            k_logical = k_logical[: real_seqlen_kv[i_b].item(), :, :]
            v_logical = v_logical[: real_seqlen_kv[i_b].item(), :, :]
            k_b = k_logical.unsqueeze(0)
            v_b = v_logical.unsqueeze(0)
            q_bhsd = q_b.transpose(1, 2)
            k_bhsd = k_b.transpose(1, 2)
            v_bhsd = v_b.transpose(1, 2)
            out_b = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd)
            out_b = out_b.transpose(1, 2).contiguous()
            out_list.append(out_b)
        return torch.cat(out_list, dim=0)
