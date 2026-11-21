import torch

from workloads.device import run_device
from workloads.numerics import Custom, assert_normalized_error, zeroed_input
from workloads.workload_base import CallWorkload, WorkloadBase


class FP8LightningIndexerWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        index_dim: int,
        seq_len_kv: int,
        kv_group: int,
        clean_logits: bool = True,
    ):
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.clean_logits = clean_logits
        self.dtype = torch.float8_e4m3fn

    def _key_windows(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Each query row's key window ``[ks, ke)`` on this rank of a context-parallel sequence.

        Four ranks split a packed sequence of random request lengths, about 2048 tokens each;
        each rank holds two halves of ``seq_len // 2`` rows, rank 3's being the fourth from the
        start and the fourth from the end. A row's window runs from its request's first token
        through itself.
        """
        cp_size, cp_rank, average_q_len = 4, 3, 2048
        total = self.seq_len * cp_size
        lengths = torch.randint(0, average_q_len * 2, (total // average_q_len * 2,)).to(
            run_device()
        )
        lengths = lengths[: torch.where(lengths.cumsum(0) >= total)[0][0]]
        if lengths.sum() < total:
            lengths = torch.cat([lengths, torch.tensor([total - lengths.sum()]).to(run_device())])
        ends = lengths.cumsum(0)
        starts = torch.cat([torch.tensor([0]).to(run_device()), ends[:-1]])
        # Row t's request index: how many requests after the first start at or before t.
        request = torch.zeros(total, dtype=torch.int32, device=run_device())
        if len(starts) > 1:
            request[starts[1:]] = 1
        request = torch.cumsum(request, dim=0, dtype=torch.int32).long()
        int_max = torch.full((1,), torch.iinfo(torch.int32).max, device=run_device())
        ks_all = torch.gather(torch.cat([starts, int_max]), 0, request).int()
        causal = torch.zeros(total, dtype=torch.int32, device=run_device())
        for i in range(len(starts)):
            causal[starts[i] : ends[i]] = (
                torch.arange(1, ends[i] - starts[i] + 1, dtype=torch.int32, device=run_device())
                + starts[i]
            )
        zero = torch.zeros(1, dtype=torch.int32, device=run_device())
        ke_all = torch.minimum(causal, torch.gather(torch.cat([ends, zero]), 0, request)).int()
        assert self.seq_len % 2 == 0
        half = self.seq_len // 2
        short = slice(cp_rank * half, (cp_rank + 1) * half)
        long = slice(total - (cp_rank + 1) * half, total - cp_rank * half)
        ks = torch.cat([ks_all[short], ks_all[long]])
        ke = torch.cat([ke_all[short], ke_all[long]])
        if int(ks.min()) >= self.seq_len_kv:
            raise ValueError(
                f"no query row reaches a key: ranges start at {int(ks.min())} and "
                f"seq_len_kv is {self.seq_len_kv}. These offsets span the "
                f"{total} keys of a context-parallel sequence"
            )
        return ks, ke

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        index_q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.index_dim,
            device=run_device(),
            dtype=torch.bfloat16,
        )
        index_k = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.index_dim,
            device=run_device(),
            dtype=torch.bfloat16,
        )
        weights = torch.randn(self.seq_len, self.heads, device=run_device(), dtype=torch.float32)
        ks, ke = self._key_windows()
        return index_q, index_k, weights, ks, ke

    def gen_weight_boundary_inputs(self) -> tuple:
        """Normal FP32 weights whose halves are subnormal, with a compensating key scale."""
        device = run_device()
        q = torch.full(
            (self.batch, self.seq_len, self.heads, self.index_dim), 448.0, device=device
        ).to(self.dtype)
        k = torch.full(
            (self.batch, self.seq_len_kv, self.kv_group, self.index_dim), 448.0, device=device
        ).to(self.dtype)
        weights = torch.full(
            (self.seq_len, self.heads), torch.finfo(torch.float32).tiny * 1.5, device=device
        )
        ks = torch.zeros(self.seq_len, dtype=torch.int32, device=device)
        ke = torch.full((self.seq_len,), self.seq_len_kv, dtype=torch.int32, device=device)
        scales = torch.full((self.batch, self.seq_len_kv, self.kv_group), 1e30, device=device)
        return q, k, weights, ks, ke, scales

    def ref_program(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        weights: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        index_k_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        q = q.float()
        k = kv.float()
        if index_k_scale is not None:
            k = k * index_k_scale.unsqueeze(-1)
        batch, seq_len, heads, index_dim = q.shape
        seq_len_kv = self.seq_len_kv
        kv_group = self.kv_group
        heads_per_group = heads // kv_group

        k = k.view(batch, seq_len_kv, kv_group, index_dim)
        q = q.view(batch, seq_len, kv_group, heads_per_group, index_dim)

        keys = torch.arange(seq_len_kv, device=q.device)[None, :]
        mask = (keys >= starts[:, None]) & (keys < ends[:, None])

        # Bound the FP32 head-score workspace while checking the complete output.
        score_elements = 64 * 1024 * 1024
        query_chunk = max(1, score_elements // (batch * heads * seq_len_kv))
        logits = torch.empty(
            (batch, seq_len, seq_len_kv, kv_group), device=q.device, dtype=torch.float32
        )
        for start in range(0, seq_len, query_chunk):
            end = min(start + query_chunk, seq_len)
            score = torch.einsum("bsghd,bngd->bghsn", q[:, start:end], k)
            chunk_weights = weights[start:end].view(end - start, kv_group, heads_per_group)
            chunk_weights = chunk_weights.permute(1, 2, 0).unsqueeze(0).unsqueeze(-1)
            chunk_logits = (score.relu() * chunk_weights).sum(dim=2).permute(0, 2, 3, 1)
            logits[:, start:end] = chunk_logits.masked_fill(
                ~mask[start:end].unsqueeze(0).unsqueeze(-1), float("-inf")
            )
        return (logits,)

    def verification(self, *inputs):
        return Custom(
            assert_normalized_error,
            "normalized squared error <= 1e-3",
            controls=(zeroed_input(0, "query-zeroed"),),
        )


class FP8LightningIndexerCall(CallWorkload, FP8LightningIndexerWorkload):
    """A manifest call of FP8LightningIndexerFwdOp; the row's generator gives the windows.

    FP8 index tensors are bf16 draws cast down; a passed ``index_k_scale`` is the per-key
    absolute maximum the keys were quantized by, so the keys it scales are the bf16 draw.
    """

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        indices = call.indices
        FP8LightningIndexerWorkload.__init__(
            self,
            indices["B"],
            indices["S"],
            indices["H"],
            indices["D"],
            indices["S_kv"],
            indices["G"],
            indices["clean_logits"],
        )
        self.index_dtype = getattr(torch, indices["T"])

    def gen_inputs(self) -> tuple[torch.Tensor | None, ...]:
        index_q, index_k, weights, ks, ke, index_k_scale = CallWorkload.gen_inputs(self)
        index_q = torch.randn(index_q.shape, device=index_q.device).to(self.index_dtype)
        keys = torch.randn(index_k.shape, device=index_k.device)
        if index_k_scale is not None:
            index_k_scale = keys.abs().amax(dim=-1).clamp(min=1e-4) / 448.0
            keys = keys / index_k_scale.unsqueeze(-1)
        return index_q, keys.to(self.index_dtype), weights, ks, ke, index_k_scale
