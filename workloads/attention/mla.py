import torch
import torch.nn.functional as F
from einops import einsum, rearrange

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "MlaDecodeCall",
    "MlaDecodeWorkload",
    "MlaVarlenWorkload",
    "mla_varlen_inputs",
    "mla_varlen_reference",
]


class MlaDecodeWorkload(WorkloadBase):
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
        """
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
        return out

    def verification(self, *inputs):
        from workloads.numerics import Exact, zeroed_input

        # Split-K tail tiles change the accumulation order.
        tol = 2e-3 if self.seq_len_kv % 64 else 1e-3
        # With zero or one key, attention does not depend on the query.
        controls = (zeroed_input(0, "first-input-zeroed"),) if self.seq_len_kv > 1 else ()
        return Exact(controls=controls, atol=tol, rtol=tol)


class MlaDecodeCall(CallWorkload, MlaDecodeWorkload):
    """A manifest call of MultiHeadLatentAttentionDecodeWithKVCacheFwdOp."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        ix = call.ix
        MlaDecodeWorkload.__init__(
            self,
            ix["B"],
            ix["H"],
            ix["H_kv"],
            ix["N_kv"],
            ix["D"],
            ix["PE"],
            getattr(torch, ix["T"]),
        )

    gen_inputs = CallWorkload.gen_inputs


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


class MlaVarlenWorkload(WorkloadBase):
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
