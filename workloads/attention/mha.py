"""Workload definitions for the MHA paged decode op."""

import math

import torch

from workloads.attention.paged_kv_cache import make_fragmented_block_table
from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase


class MHADecodePagedWorkload(WorkloadBase):
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
            torch.ones((self.batch,), dtype=torch.int32, device=run_device()) * self.seqlen_kv
        )
        q = torch.randn(
            self.batch, self.seqlen_q, self.heads, self.dim, device=run_device(), dtype=self.dtype
        )
        k = torch.randn(self.seqlen_kv, self.heads, self.dim, device=run_device(), dtype=self.dtype)
        v = torch.randn(self.seqlen_kv, self.heads, self.dim, device=run_device(), dtype=self.dtype)
        block_table = make_fragmented_block_table(self.batch, num_pages, num_pages)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
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
        """Reassemble paged K/V to logical layout per batch, then attend.

        A causal query ``i`` sees the keys up to position ``i + kv_len - seqlen_q``; a query
        that sees no key outputs zeros.
        """
        batch, seqlen_q, heads, dim = q.shape
        seqlen_kv = k.shape[0]
        out_list = []
        for i_b in range(batch):
            kv_len = real_seqlen_kv[i_b].item()
            k_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            v_logical = torch.zeros(seqlen_kv, heads, dim, dtype=q.dtype, device=q.device)
            num_pages = math.ceil(kv_len / self.page_size)
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
            q_b = q[i_b].float().transpose(0, 1)  # [H, S_q, D]
            k_b = k_logical[:kv_len].float().transpose(0, 1)  # [H, kv_len, D]
            v_b = v_logical[:kv_len].float().transpose(0, 1)
            scores = q_b @ k_b.transpose(1, 2) * dim**-0.5
            if self.is_causal:
                last_key = torch.arange(seqlen_q, device=q.device)[:, None] + kv_len - seqlen_q
                visible = torch.arange(kv_len, device=q.device) <= last_key
                scores = scores.masked_fill(~visible, float("-inf"))
            probs = scores.softmax(-1).nan_to_num(0.0)
            out_list.append((probs @ v_b).transpose(0, 1).unsqueeze(0).to(q.dtype))
        return torch.cat(out_list, dim=0)

    def verification(self, *inputs):
        from workloads.numerics import Custom

        atol = {torch.float16: 1e-3, torch.bfloat16: 5e-3}[inputs[0].dtype]

        def validate(got, expected):
            torch.testing.assert_close(got, expected, atol=atol, rtol=atol)
            cosine = torch.nn.functional.cosine_similarity(
                got.reshape(self.batch, -1), expected.reshape(self.batch, -1), dim=-1
            )
            assert cosine.min() > 0.99

        return Custom(validate, "attention values and cosine agreement")


class MHADecodePagedCall(CallWorkload, MHADecodePagedWorkload):
    """A manifest call of MHADecodePagedWithKVCacheFwdOp."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        MHADecodePagedWorkload.__init__(
            self,
            ix["B"],
            ix["H"],
            ix["S_q"],
            ix["N_kv"],
            ix["D"],
            params["page_size"],
            params["is_causal"],
            getattr(torch, ix["T"]),
        )

    gen_inputs = CallWorkload.gen_inputs
