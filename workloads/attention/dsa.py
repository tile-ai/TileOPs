import torch

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = ["DsaDecodeCall", "DsaDecodeWorkload"]


class DsaDecodeWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        seq_len: int,
        seq_len_kv: int,
        dim: int,
        dim_tail: int,
        topk: int,
        stride_kv: int,
        heads_kv: int,
        q_start_index_s: int,
        sm_scale: float = None,
        is_causal: bool = True,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_tail = dim_tail
        self.topk = topk
        self.stride_kv = stride_kv
        self.heads_kv = heads_kv
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.dtype = dtype
        self.q_start_index_s = q_start_index_s

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim + self.dim_tail,
            device=run_device(),
            dtype=self.dtype,
        )
        kv = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim + self.dim_tail,
            device=run_device(),
            dtype=self.dtype,
        )
        indices = torch.full(
            (self.batch, self.seq_len, self.heads_kv, self.topk),
            self.seq_len_kv,
            dtype=torch.int32,
            device=run_device(),
        )
        for b in range(self.batch):
            for t in range(self.seq_len):
                for h in range(self.heads_kv):
                    i_i = torch.randperm(
                        min(
                            max(1, ((t + int(self.q_start_index_s)) // self.stride_kv)),
                            self.seq_len_kv,
                        )
                    )[: self.topk]
                    indices[b, t, h, : len(i_i)] = i_i
        return q, kv, indices

    def selection_mask(self, indices: torch.Tensor) -> torch.Tensor:
        """Return the ``[batch, kv heads, q, kv]`` mask this workload's top-k selection implies.

        The mask combines the selected positions with the compressed causal limit.
        ``ref_program`` and any baseline attending over the same selection both read it,
        so the two cannot drift apart.
        """
        idx = indices.transpose(1, 2)
        b, g, sq, _ = idx.shape
        sk = self.seq_len_kv
        q_start_index_s = self.q_start_index_s
        if q_start_index_s is None:
            q_start_index_s = sk * self.stride_kv - sq
        device = indices.device
        compressed_causal_mask = torch.arange(
            q_start_index_s, sq + q_start_index_s, dtype=torch.int32, device=device
        ).view(-1, 1) >= torch.arange(
            self.stride_kv - 1,
            sk * self.stride_kv,
            self.stride_kv,
            dtype=torch.int32,
            device=device,
        ).view(1, -1)

        mask = torch.zeros(b, g, sq, sk + 1, dtype=torch.bool, device=device).scatter(
            3, idx.long(), 1
        )[..., :-1]
        mask = mask & compressed_causal_mask.view(1, 1, sq, sk)
        mask[:, :, : self.stride_kv - 1, 0] = True
        return mask

    def ref_program(self, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        b, sq, h, dim_q = q.shape
        _, sk, g, _ = kv.shape
        assert dim_q == self.dim + self.dim_tail, "you should assign dim otherwise"
        mask = self.selection_mask(indices).view(b, g, 1, sq, sk)
        k = kv.float()
        v = k[..., : self.dim]
        sm_scale = dim_q**-0.5 if self.sm_scale is None else self.sm_scale
        # Bound each FP32 score tensor to 256 MiB for long-context workloads.
        chunk = max(1, (256 * 1024**2) // (b * h * sk * 4))
        output = torch.empty((b, sq, h, self.dim), dtype=q.dtype, device=q.device)
        for start in range(0, sq, chunk):
            stop = min(start + chunk, sq)
            query = q[:, start:stop].float().reshape(b, stop - start, g, h // g, dim_q)
            score = torch.einsum("bmghd,bngd->bghmn", query, k)
            score = score.mul(sm_scale).masked_fill(~mask[..., start:stop, :], float("-inf"))
            out = torch.einsum("bghmn,bngd->bmghd", score.softmax(-1), v)
            output[:, start:stop] = out.reshape(b, stop - start, h, self.dim)
        return output


class DsaDecodeCall(CallWorkload, DsaDecodeWorkload):
    """A manifest call of DeepSeekSparseAttentionDecodeWithKVCacheFwdOp; the row's generator
    selects each query's keys."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix, params = call.ix, call.params
        DsaDecodeWorkload.__init__(
            self,
            ix["B"],
            ix["H"],
            ix["S"],
            ix["S_kv"],
            ix["D"],
            params["dim_tail"],
            ix["K"],
            params["stride_kv"],
            ix["H_kv"],
            params["q_start_index_s"],
            sm_scale=params["sm_scale"],
            is_causal=params["is_causal"],
            dtype=getattr(torch, ix["T"]),
        )

    gen_inputs = CallWorkload.gen_inputs
