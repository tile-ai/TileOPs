import torch

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = ["DSADecodeCall", "DSADecodeWorkload", "DSAPagedCall"]


class DSADecodeWorkload(WorkloadBase):
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

    def verification(self, *inputs):
        from workloads.numerics import Exact, reference_tolerance

        tolerance = {"atol": 3e-4, "rtol": 1e-5}
        if inputs[0].dtype == torch.bfloat16 or self.dim_tail == 0:
            tolerance = reference_tolerance(inputs[0].dtype)
        # Ignored/padded slots change normalization and fp16 rounding at the boundary.
        if inputs[0].dtype == torch.float16 and (inputs[2] >= self.seq_len_kv).any():
            tolerance = {"atol": 2e-3, "rtol": 2e-3}
        return Exact(**tolerance)


class DSADecodeCall(CallWorkload, DSADecodeWorkload):
    """A manifest call of DSADecodeWithKVCacheFwdOp; the row's generator
    selects each query's keys."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        indices, params = call.indices, call.params
        DSADecodeWorkload.__init__(
            self,
            indices["B"],
            indices["H"],
            indices["S"],
            indices["S_kv"],
            indices["D"],
            params["dim_tail"],
            indices["K"],
            params["stride_kv"],
            indices["H_kv"],
            params["q_start_index_s"],
            sm_scale=params["sm_scale"],
            is_causal=params["is_causal"],
            dtype=getattr(torch, indices["T"]),
        )

    gen_inputs = CallWorkload.gen_inputs


# The FlashMLA DeepSeek-V3.2 FP8 cache row: 512 float8_e4m3fn latent values, one float32
# scale per 128 of them, then 64 bfloat16 rope values. The cache writer, the row reader
# and the negative controls must agree on it.
_LATENT, _GROUP = 512, 128
_SCALES = _LATENT + 4 * (_LATENT // _GROUP)


def _fp8_latent_rows(rows: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
    """The float32 dequantized latent ``[..., 512]`` and rope ``[..., 64]`` of FP8 cache rows."""
    latent = rows[..., :_LATENT].contiguous().view(torch.float8_e4m3fn).float()
    scales = rows[..., _LATENT:_SCALES].contiguous().view(torch.float32)
    value = (latent.unflatten(-1, (-1, _GROUP)) * scales[..., None]).flatten(-2)
    rope = rows[..., _SCALES:].contiguous().view(torch.bfloat16).float()
    return value, rope


class DSAPagedCall(CallWorkload):
    """A manifest call of DSAPagedFwdOp over a serving-style quantized FP8 latent cache."""

    def gen_inputs(self):
        q, kv_cache, block_table, cache_seqlens, indices = super().gen_inputs()
        device = kv_cache.device
        gen = self.rng("cache", device=device)
        q = torch.randn(q.shape, device=device, generator=gen).to(q.dtype)
        pages, page_size, _ = kv_cache.shape
        # Pages are quantized a chunk at a time to bound the float32 staging tensor.
        chunk = max(1, (1 << 26) // (page_size * _LATENT))
        for first in range(0, pages, chunk):
            rows = kv_cache[first : first + chunk]
            latent = torch.randn(
                (*rows.shape[:2], _LATENT // _GROUP, _GROUP), device=device, generator=gen
            )
            # 448 is float8_e4m3fn's largest value; the floor keeps a zero group finite.
            scales = latent.abs().amax(-1).clamp_min(1e-4) / 448.0
            codes = (latent / scales[..., None]).to(torch.float8_e4m3fn)
            rope = torch.randn((*rows.shape[:2], 64), device=device, generator=gen)
            rows[..., :_LATENT] = codes.flatten(-2).view(torch.uint8)
            rows[..., _LATENT:_SCALES] = scales.view(torch.uint8)
            rows[..., _SCALES:] = rope.to(torch.bfloat16).view(torch.uint8)
        return q, kv_cache, block_table, cache_seqlens, indices

    @staticmethod
    def selected(kv_cache, block_table, cache_seqlens, indices):
        """Dequantized latent and rope of each slot's cache row, and whether the slot is live."""
        page_size = kv_cache.shape[1]
        pos = indices.long()
        valid = (pos >= 0) & (pos < cache_seqlens.long()[:, None, None])
        slot = pos.masked_fill(~valid, 0)
        page = torch.gather(block_table.long(), 1, (slot // page_size).flatten(1)).view_as(slot)
        value, rope = _fp8_latent_rows(kv_cache.flatten(0, 1)[page * page_size + slot % page_size])
        return value, rope, valid

    def _scale(self, q):
        sm_scale = self.call.params["sm_scale"]
        return q.shape[-1] ** -0.5 if sm_scale is None else sm_scale

    @staticmethod
    def _attend(q, value, rope, valid, sm_scale):
        """Float32 output, log-sum-exp and softmax weights over each row's valid slots."""
        key = torch.cat([value, rope], -1)
        scores = torch.einsum("bshd,bskd->bshk", q.float(), key) * sm_scale
        scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
        weights = scores.softmax(-1).nan_to_num(0.0)
        return torch.einsum("bshk,bskd->bshd", weights, value), scores.logsumexp(-1), weights

    def ref_program(self, q, kv_cache, block_table, cache_seqlens, indices):
        """Float32 attention of each query row over the dequantized rows its valid slots name."""
        value, rope, valid = self.selected(kv_cache, block_table, cache_seqlens, indices)
        out, lse, _ = self._attend(q, value, rope, valid, self._scale(q))
        return out.to(q.dtype), lse

    def verification(self, q, kv_cache, block_table, cache_seqlens, indices):
        from workloads.numerics import Custom, NegativeControl

        # First-order element-wise bounds; a bfloat16 rounding errs by up to 2^-8:
        # - key/value rounding: measured drift against attention over the rounded rows;
        # - weight rounding 2^-8, exp2 2^-20, float32 sums K * 2^-23 each (three sums);
        # - a score's 576-product sum errs by delta_k, moving weight k by
        #   w_k * (delta_k + sum_j w_j delta_j) and the lse by sum_j w_j delta_j.
        # Weight errors reach dim d through the weights times |value_d|.
        sm_scale = self._scale(q)
        topk = indices.shape[-1]
        value, rope, valid = self.selected(kv_cache, block_table, cache_seqlens, indices)
        exact = self._attend(q, value, rope, valid, sm_scale)
        rounded_value = value.bfloat16().float()
        rounded = self._attend(q, rounded_value, rope, valid, sm_scale)
        weights, magnitude = rounded[2], rounded_value.abs()
        key = torch.cat([rounded_value, rope], -1).abs()
        dot = torch.einsum("bshd,bskd->bshk", q.float().abs(), key)
        weighted_delta = weights * (abs(sm_scale) * q.shape[-1] * 2**-23 * dot)
        shift = weighted_delta.sum(-1)
        spread = torch.einsum("bshk,bskd->bshd", weights, magnitude)
        scored = torch.einsum("bshk,bskd->bshd", weighted_delta, magnitude)
        relative = 2**-8 + 2**-20 + 3 * topk * 2**-23
        drift = (rounded[0] - exact[0]).abs() + spread * (relative + shift[..., None]) + scored
        # Both outputs then round to bfloat16.
        out_bound = drift * (1 + 2**-8) + 2**-7 * exact[0].abs()
        lse_bound = (rounded[1] - exact[1]).abs() + shift + topk * 2**-23 + 2**-20

        def validate(got, expected):
            for name, mine, ref, bound in (
                ("o", got[0].float(), expected[0].float(), out_bound),
                ("lse", got[1], expected[1], lse_bound),
            ):
                both_dead = torch.isneginf(mine) & torch.isneginf(ref)
                error = (mine - ref).abs().masked_fill(both_dead, 0)
                worst = (error - bound).argmax()
                assert (error <= bound).all(), (
                    f"{name}: error {error.flatten()[worst].item()} exceeds its bound "
                    f"{bound.flatten()[worst].item()} at flat index {worst.item()}"
                )

        def first_block_dropped(reference, inputs):
            changed = list(inputs)
            changed[4] = inputs[4].clone()
            changed[4][..., :64] = -1  # one key block
            return reference(*changed)

        def rope_zeroed(reference, inputs):
            changed = list(inputs)
            changed[1] = inputs[1].clone()
            changed[1][..., _SCALES:] = 0
            return reference(*changed)

        return Custom(
            validate,
            "output and lse within first-order rounding bounds",
            controls=(
                NegativeControl("first-key-block-dropped", first_block_dropped),
                NegativeControl("rope-zeroed", rope_zeroed),
            ),
        )
