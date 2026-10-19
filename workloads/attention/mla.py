import torch
import torch.nn.functional as F
from einops import einsum, rearrange

from workloads.attention.gqa.call_metadata import _dtype, _segments
from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "MLADecodeCall",
    "MLADecodeWorkload",
    "MLAPagedCall",
    "MLAPagedWorkload",
    "MLAVarlenCall",
    "MLAVarlenWorkload",
    "mla_varlen_inputs",
    "mla_varlen_reference",
]


class MLADecodeWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len_kv: int,
        dim: int,
        dim_pe: int,
        dtype: torch.dtype,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.dim_pe = dim_pe
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = torch.randn(self.batch, self.heads, self.dim, device=run_device(), dtype=self.dtype)
        Q_pe = torch.randn(
            self.batch, self.heads, self.dim_pe, device=run_device(), dtype=self.dtype
        )
        K = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim,
            device=run_device(),
            dtype=self.dtype,
        )
        K_pe = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.heads_kv,
            self.dim_pe,
            device=run_device(),
            dtype=self.dtype,
        )
        return Q, Q_pe, K, K_pe

    def ref_program(
        self, q: torch.Tensor, q_pe: torch.Tensor, kv: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """
        Inputs:
        - q (Tensor): [batch, heads, dim]
        - q_pe (Tensor): [batch, heads, dim_pe]
        - kv (Tensor): [batch, seqlen_kv, heads_kv, dim]
        - k_pe (Tensor): [batch, seqlen_kv, heads_kv, dim_pe]
        Outputs:
        - output (Tensor): [batch, heads, dim]

        Computed in float32 and rounded once to the input dtype.
        """
        dtype = q.dtype
        q, q_pe, kv, k_pe = (t.float() for t in (q, q_pe, kv, k_pe))
        dim = q.shape[-1]
        dim_pe = q_pe.shape[-1]
        num_head_groups = q.shape[1] // kv.shape[2]
        scale = (dim + dim_pe) ** 0.5
        Q = rearrange(
            q, "b (h g) d -> b g h d", g=num_head_groups
        )  # [batch_size, num_head_groups, groups, dim]

        Q_pe = rearrange(
            q_pe, "b (h g) d -> b g h d", g=num_head_groups
        )  # [batch_size, num_head_groups, groups, dim_pe]

        KV = rearrange(kv, "b n h d -> b h n d")  # [batch_size, groups, seqlen_kv, dim]

        K_pe = rearrange(
            k_pe, "b n h d -> b h n d"
        )  # [batch_size, num_head_groups, groups, dim_pe]

        query = torch.concat([Q, Q_pe], dim=-1)
        key = torch.concat([KV, K_pe], dim=-1)

        scores = einsum(
            query, key, "b g h d, b h s d -> b g h s"
        )  # [batch_size, num_head_groups, groups, seqlen_kv]

        attention = F.softmax(
            scores / scale, dim=-1
        )  # [batch_size, num_head_groups, groups, seqlen_kv]

        out = einsum(
            attention, KV, "b g h s, b h s d -> b g h d"
        )  # [batch_size, num_head_groups, groups, dim]
        out = rearrange(out, "b g h d -> b (h g) d")  # [batch_size, heads, dim]
        return out.to(dtype)

    def verification(self, *inputs):
        from workloads.numerics import Exact, zeroed_input

        # The kernel rounds the probabilities to the input dtype before the PV GEMM,
        # one relative eps on outputs below 1; split-K tail tiles add a second.
        eps = torch.finfo(self.dtype).eps
        tol = 2 * eps if self.seq_len_kv % 64 else eps
        # With zero or one key, attention does not depend on the query.
        controls = (zeroed_input(0, "first-input-zeroed"),) if self.seq_len_kv > 1 else ()
        return Exact(controls=controls, atol=tol, rtol=tol)


class MLADecodeCall(CallWorkload, MLADecodeWorkload):
    """A manifest call of MLADecodeWithKVCacheFwdOp."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        indices = call.indices
        MLADecodeWorkload.__init__(
            self,
            indices["B"],
            indices["H"],
            indices["H_kv"],
            indices["N_kv"],
            indices["D"],
            indices["PE"],
            getattr(torch, indices["T"]),
        )

    gen_inputs = CallWorkload.gen_inputs


class MLAPagedWorkload(WorkloadBase):
    """Absorbed-form MLA decode over a paged latent cache.

    Request ``b`` holds ``cache_lens[b]`` keys on the pages ``block_table[b]`` names; its
    ``s_q`` query tokens are its last ``s_q`` positions. A key is a whole cache row and a
    value its first ``kv_lora_rank`` columns. Omitted, the block table is a random,
    fragmented choice of distinct pages.
    """

    def __init__(
        self,
        heads: int,
        s_q: int,
        dim: int,
        kv_lora_rank: int,
        cache_lens: list[int],
        page_size: int,
        width: int,
        num_pages: int,
        dtype: torch.dtype,
        cache_dtype: torch.dtype | None = None,
        is_causal: bool = True,
        sm_scale: float | None = None,
        block_table: list[list[int]] | None = None,
    ) -> None:
        self.heads, self.s_q, self.dim, self.kv_lora_rank = heads, s_q, dim, kv_lora_rank
        self.cache_lens, self.page_size = cache_lens, page_size
        self.width, self.num_pages = width, num_pages
        self.dtype = dtype
        self.cache_dtype = dtype if cache_dtype is None else cache_dtype
        self.is_causal, self.sm_scale = is_causal, sm_scale
        self.block_table = block_table

    @property
    def quantized(self) -> bool:
        return self.cache_dtype == torch.float8_e4m3fn

    def gen_inputs(self):
        device = run_device()
        batch = len(self.cache_lens)
        q = torch.randn(batch, self.s_q, self.heads, self.dim, device=device).to(self.dtype)
        kv_cache = torch.randn(self.num_pages, self.page_size, self.dim, device=device)
        if self.block_table is None:
            if batch * self.width > self.num_pages:
                raise ValueError(f"{batch} x {self.width} pages from a pool of {self.num_pages}")
            pages = torch.randperm(self.num_pages, generator=self.rng("pages"))[
                : batch * self.width
            ]
            table = pages.view(batch, self.width).to(device=device, dtype=torch.int32)
        else:
            table = torch.tensor(self.block_table, dtype=torch.int32, device=device)
        lens = torch.tensor(self.cache_lens, dtype=torch.int32, device=device)
        return q, kv_cache.to(self.cache_dtype), table, lens, self.cache_scale(device)

    def cache_scale(self, device):
        """A non-unit scale, so a kernel that drops it does not pass; None for a 16-bit cache."""
        if not self.quantized:
            return None
        return torch.tensor([0.5], dtype=torch.float32, device=device)

    def ref_program(self, q, kv_cache, block_table, cache_seqlens, kv_scale=None):
        """Gather each request's pages into a dense cache, then attend in float32."""
        batch, s_q, heads, dim = q.shape
        dim_v = self.kv_lora_rank
        scale = dim**-0.5 if self.sm_scale is None else self.sm_scale
        o = torch.empty(batch, s_q, heads, dim_v, dtype=q.dtype, device=q.device)
        lse = torch.empty(batch, s_q, heads, dtype=torch.float32, device=q.device)
        for b in range(batch):
            length = int(cache_seqlens[b])
            pages = block_table[b, : -(-length // self.page_size)].long()
            keys = kv_cache[pages].reshape(-1, dim)[:length].float()
            if kv_scale is not None:
                keys = keys * kv_scale[0]
            scores = torch.einsum("shd,nd->shn", q[b].float(), keys) * scale
            if self.is_causal:
                position = length - s_q + torch.arange(s_q, device=q.device)
                hidden = torch.arange(length, device=q.device)[None, :] > position[:, None]
                scores = scores.masked_fill(hidden[:, None, :], float("-inf"))
            lse[b] = scores.logsumexp(-1)
            o[b] = torch.einsum("shn,nd->shd", scores.softmax(-1), keys[:, :dim_v]).to(q.dtype)
        return o, lse

    def verification(self, *inputs):
        from workloads.numerics import Custom, assert_close, assert_normalized_error

        # The probabilities are rounded to the query dtype before the value product, one
        # relative step on outputs below 1, and the output lands once in that dtype.
        eps = torch.finfo(self.dtype).eps

        # The log-sum-exp is float32 throughout: 2e-4 admits the comparators' reductions and
        # rejects one 64-key page left out of a 32K-key request, which moves it by 2e-3.
        # Over the whole output the roundings average, so its normalized squared error
        # stays below eps^2 / 10, under the 1.2e-5 of that page's values left out, which
        # no element-wise eps bound sees.
        def validate(got, expected):
            assert_close(got[0], expected[0], atol=eps, rtol=eps)
            assert_normalized_error(got[0], expected[0], eps * eps / 10)
            assert_close(got[1], expected[1], atol=2e-4, rtol=1e-6)

        return Custom(validate, "storage-dtype attention output and float32 log-sum-exp")


class MLAPagedCall(CallWorkload, MLAPagedWorkload):
    """A manifest call of MLAPagedFwdOp, over the block table and lengths it carries."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        indices, params = call.indices, call.params
        MLAPagedWorkload.__init__(
            self,
            indices["H"],
            indices["S_q"],
            indices["DK"],
            params["kv_lora_rank"],
            call.metadata_values("cache_seqlens"),
            indices["PS"],
            indices["W"],
            indices["NP"],
            _dtype(call, "q"),
            _dtype(call, "kv_cache"),
            is_causal=params["is_causal"],
            sm_scale=params.get("sm_scale"),
            block_table=call.metadata_values("block_table"),
        )

    def gen_inputs(self):
        """The manifest's tensors, with a positive cache scale in place of a random draw."""
        q, kv_cache, table, lens, kv_scale = CallWorkload.gen_inputs(self)
        return q, kv_cache, table, lens, self.cache_scale(q.device)


def mla_varlen_inputs(seq_lens, heads, dim_nope, dim_pe, dim_v, dtype):
    """The packed tensors one call takes, for ``seq_lens`` requests."""
    total = sum(seq_lens)
    device = run_device()
    return (
        torch.randn(total, heads, dim_nope + dim_pe, dtype=dtype, device=device),
        torch.randn(total, heads, dim_nope, dtype=dtype, device=device),
        torch.randn(total, dim_pe, dtype=dtype, device=device),
        torch.randn(total, heads, dim_v, dtype=dtype, device=device),
        torch.tensor(
            [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        ),
    )


def mla_varlen_reference(q, k_nope, k_pe, v, cu_seqlens, *, is_causal, dim_v, sm_scale=None):
    """Expand the shared rope half to every head, then attend per request in float32.

    The score block of a whole request is quadratic in its length, so query rows
    and heads are taken a slab at a time; each row still sees its whole key
    axis, so the numbers are those of the unsplit form.
    """
    # Bound the temporary score tensor's memory without changing the reference result.
    heads_per_chunk = 8
    rows_per_chunk = 1024
    heads = q.shape[1]
    scale = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
    bounds = cu_seqlens.tolist()
    out = torch.empty_like(q[..., :dim_v])
    lse = torch.empty(q.shape[0], heads, dtype=torch.float32, device=q.device)
    for start, end in zip(bounds, bounds[1:], strict=False):
        span = end - start
        key = torch.cat(
            [k_nope[start:end], k_pe[start:end, None].expand(-1, heads, -1)], dim=-1
        ).float()
        value = v[start:end].float()
        rows = torch.arange(span, device=q.device)
        for h0 in range(0, heads, heads_per_chunk):
            h1 = min(h0 + heads_per_chunk, heads)
            for r0 in range(0, span, rows_per_chunk):
                r1 = min(r0 + rows_per_chunk, span)
                scores = (
                    torch.einsum(
                        "shd,nhd->hsn",
                        q[start + r0 : start + r1, h0:h1].float(),
                        key[:, h0:h1],
                    )
                    * scale
                )
                if is_causal:
                    visible = rows[None, :] <= rows[r0:r1, None]
                    scores = scores.masked_fill(~visible[None], float("-inf"))
                out[start + r0 : start + r1, h0:h1] = torch.einsum(
                    "hsn,nhd->shd", scores.softmax(-1), value[:, h0:h1]
                ).to(q.dtype)
                lse[start + r0 : start + r1, h0:h1] = scores.logsumexp(-1).T
    return out, lse


class MLAVarlenWorkload(WorkloadBase):
    """Packed MLA prefill, including its FP32 log-sum-exp output."""

    def __init__(
        self, seq_lens, heads, dim_nope, dim_pe, dim_v, dtype, is_causal=True, sm_scale=None
    ):
        self.seq_lens, self.heads = seq_lens, heads
        self.dim_nope, self.dim_pe, self.dim_v = dim_nope, dim_pe, dim_v
        self.dtype, self.is_causal, self.sm_scale = dtype, is_causal, sm_scale

    def gen_inputs(self):
        return mla_varlen_inputs(
            self.seq_lens, self.heads, self.dim_nope, self.dim_pe, self.dim_v, self.dtype
        )

    def ref_program(self, *inputs):
        return mla_varlen_reference(
            *inputs, is_causal=self.is_causal, dim_v=self.dim_v, sm_scale=self.sm_scale
        )

    def verification(self, *inputs):
        from workloads.numerics import Custom, assert_close

        tolerance = 2e-2 if inputs[0].dtype == torch.bfloat16 else 4e-3

        def validate(got, expected):
            assert_close(got[0], expected[0], atol=tolerance, rtol=tolerance)
            assert_close(got[1], expected[1], atol=2e-3, rtol=2e-3)

        return Custom(validate, "storage-dtype attention output and FP32 log-sum-exp")


class MLAVarlenCall(CallWorkload, MLAVarlenWorkload):
    """A manifest call of MLAVarlenFwdOp, over the request lengths its offsets carry."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        indices, params = call.indices, call.params
        MLAVarlenWorkload.__init__(
            self,
            _segments(call.metadata_values("cu_seqlens")),
            indices["H"],
            indices["DN"],
            indices["PE"],
            indices["DV"],
            _dtype(call, "q"),
            is_causal=params["is_causal"],
            sm_scale=params.get("sm_scale"),
        )

    gen_inputs = CallWorkload.gen_inputs
