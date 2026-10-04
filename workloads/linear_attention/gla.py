import torch

from workloads.device import run_device
from workloads.linear_attention.input_values import _log_gates, _small
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "GLAChunkwiseCall",
    "GLAChunkwiseWorkload",
    "GLADecodeCall",
    "GLADecodeWorkload",
    "GLAInferenceCall",
    "GLAInferenceWorkload",
    "gla_autograd_bwd_torch",
    "gla_decode_torch",
    "gla_fwd_chunked_torch",
]


class GLADecodeWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        scale: float = -1.0,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dtype = dtype
        self.scale = scale

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        B, H, DK, DV = self.batch, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, H, DK, device=run_device(), dtype=self.dtype) * 0.1
        k = torch.randn(B, H, DK, device=run_device(), dtype=self.dtype) * 0.1
        v = torch.randn(B, H, DV, device=run_device(), dtype=self.dtype) * 0.1
        gk = -torch.rand(B, H, DK, device=run_device(), dtype=self.dtype)
        state = torch.randn(B, H, DK, DV, device=run_device(), dtype=self.dtype) * 0.1
        return q, k, v, gk, state

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gk: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        o, new_state = gla_decode_torch(q, k, v, gk, state, self.scale)
        return o.to(self.dtype), new_state.to(self.dtype)

    def verification(self, *inputs):
        return decode_verification(inputs[0].dtype)


class GLAChunkwiseWorkload(WorkloadBase):
    def __init__(
        self,
        batch,
        seq_len,
        heads,
        dim_k,
        dim_v,
        chunk_size,
        dtype,
        has_initial_state: bool = False,
    ):
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.has_initial_state = has_initial_state

    def gen_inputs(self):
        B, T, H, K, V = self.batch, self.seq_len, self.heads, self.dim_k, self.dim_v
        q = torch.randn(B, T, H, K, device=run_device(), dtype=self.dtype) * 0.1
        k = torch.randn(B, T, H, K, device=run_device(), dtype=self.dtype) * 0.1
        v = torch.randn(B, T, H, V, device=run_device(), dtype=self.dtype) * 0.1
        g = -torch.rand(B, T, H, K, device=run_device(), dtype=self.dtype)
        # Absent means None: the recurrence then starts from zeros. Present, it is
        # fp32, the dtype the recurrence carries the state in.
        initial_state = (
            torch.randn(B, H, K, V, device=run_device(), dtype=torch.float32) * 0.1
            if self.has_initial_state
            else None
        )
        return q, k, v, g, initial_state


class GLAInferenceWorkload(GLAChunkwiseWorkload):
    """Inference GLA prefill with a caller-owned optional recurrent state."""

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim_k: int,
        dim_v: int,
        dtype: torch.dtype,
        has_initial_state: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__(batch, seq_len, heads, dim_k, dim_v, 64, dtype, has_initial_state)
        self.scale = scale

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from fla.ops.gla import chunk_gla, fused_recurrent_gla

        reference = fused_recurrent_gla if q.shape[1] == 1 else chunk_gla
        return reference(
            q,
            k,
            v,
            g,
            scale=self.scale if self.scale is not None else self.dim_k**-0.5,
            initial_state=initial_state,
            output_final_state=True,
        )

    def verification(self, *inputs):
        return inference_verification(inputs[0].dtype, decode=inputs[0].shape[1] == 1)


def gla_fwd_chunked_torch(q, k, v, g, chunk_size, scale=None, initial_state=None):
    """Fully differentiable chunked GLA forward in float32.

    Returns the output and the state the last chunk leaves, which is what the op declares.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    BC = chunk_size
    NC = T // BC

    if scale is None:
        scale = K**-0.5

    q = q.float() * scale
    k = k.float()
    v = v.float()
    g = g.float()

    g_cum = g.reshape(B, NC, BC, H, K).cumsum(dim=2).reshape(B, T, H, K)

    h = q.new_zeros(B, H, K, V) if initial_state is None else initial_state.float().clone()
    mask = torch.tril(torch.ones(BC, BC, device=q.device, dtype=torch.float32))

    o_chunks = []
    for c in range(NC):
        sl = slice(c * BC, (c + 1) * BC)
        qc = q[:, sl, :, :]
        kc = k[:, sl, :, :]
        vc = v[:, sl, :, :]
        gc = g_cum[:, sl, :, :]
        g_last = gc[:, -1:, :, :]

        q_gated = qc * torch.exp(gc)
        o_inter = torch.einsum("bthk,bhkv->bthv", q_gated, h)

        k_ungated = kc * torch.exp(-gc)
        A = torch.einsum("bihk,bjhk->bhij", q_gated, k_ungated)
        A = A * mask.unsqueeze(0).unsqueeze(0)
        o_intra = torch.einsum("bhij,bjhv->bihv", A, vc)

        o_chunks.append(o_inter + o_intra)

        k_adj = kc * torch.exp(g_last - gc)
        h = h * torch.exp(g_last).permute(0, 2, 3, 1).squeeze(-1).unsqueeze(-1)
        h = h + torch.einsum("bthk,bthv->bhkv", k_adj, vc)

    return torch.cat(o_chunks, dim=1), h


def gla_autograd_bwd_torch(do, q, k, v, g, chunk_size, scale=-1.0, *, initial_state=None, dht=None):
    """Compute GLA backward gradients via autograd on the differentiable forward."""
    sc = (q.shape[-1] ** -0.5) if scale <= 0 else scale

    q_ = q.float().detach().requires_grad_(True)
    k_ = k.float().detach().requires_grad_(True)
    v_ = v.float().detach().requires_grad_(True)
    g_ = g.float().detach().requires_grad_(True)

    o, final = gla_fwd_chunked_torch(
        q_, k_, v_, g_, chunk_size, scale=sc, initial_state=initial_state
    )
    loss = (o * do.float()).sum()
    if dht is not None:
        loss = loss + (final * dht.float()).sum()
    dq, dk, dv, dg = torch.autograd.grad(loss, [q_, k_, v_, g_])
    return dq, dk, dv, dg


def gla_decode_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    state: torch.Tensor,
    scale: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for single-step GLA recurrence."""
    DK = q.shape[-1]
    if scale <= 0:
        scale = DK**-0.5

    q, k, v = q.float(), k.float(), v.float()
    gk = gk.float()
    state = state.float()

    alpha = torch.exp(gk)
    new_state = alpha.unsqueeze(-1) * state + k.unsqueeze(-1) * v.unsqueeze(-2)
    o = scale * torch.einsum("bhk,bhkv->bhv", q, new_state)

    return o, new_state


class GLADecodeCall(CallWorkload):
    """A manifest call of GLARecurrentFwdOp."""

    def gen_inputs(self):
        q, k, v, gk, state = super().gen_inputs()
        return _small(q), _small(k), _small(v), _log_gates(gk), _small(state)

    def ref_program(self, q, k, v, gk, state):
        o, new_state = gla_decode_torch(q, k, v, gk, state, self.call.ix["scale"])
        return o.to(q.dtype), new_state.to(q.dtype)

    def verification(self, *inputs):
        return decode_verification(inputs[0].dtype)


class GLAChunkwiseCall(CallWorkload):
    """A manifest call of GLAChunkFwdOp or GLAChunkBwdOp."""

    def gen_inputs(self):
        tensors = dict(zip(self.call.signature.inputs, super().gen_inputs(), strict=True))
        return tuple(_log_gates(t) if name == "g" else _small(t) for name, t in tensors.items())

    def ref_program(self, *inputs):
        a = self.arguments()
        if self.call.signature.name == "GLAChunkFwdOp":
            q, k, v, g, initial = inputs
            out, state = gla_fwd_chunked_torch(
                q, k, v, g, a["chunk_size"], scale=a["scale"], initial_state=initial
            )
            return out.to(q.dtype), state.float()
        q, k, v, g, h, do, dht = inputs
        return tuple(
            t.float()
            for t in gla_autograd_bwd_torch(
                do, q, k, v, g, a["chunk_size"], scale=a["scale"], initial_state=h[:, 0], dht=dht
            )
        )

    def verification(self, *inputs):
        return chunkwise_verification(
            inputs[0].dtype, backward=self.call.signature.name == "GLAChunkBwdOp"
        )


class GLAInferenceCall(CallWorkload):
    """A manifest call of GLAInferenceFwdOp; FLA's chunk_gla is the reference."""

    def gen_inputs(self):
        q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu = super().gen_inputs()
        return (
            _small(q),
            _small(k),
            _small(v),
            _log_gates(g),
            _small(initial_state),
            cu_seqlens,
            cu_seqlens_cpu,
        )

    def ref_program(self, q, k, v, g, initial_state, cu_seqlens, cu_seqlens_cpu):
        from fla.ops.gla import chunk_gla, fused_recurrent_gla

        scale = self.call.ix["scale"]
        arguments = dict(
            scale=q.shape[-1] ** -0.5 if scale is None else scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        if q.shape[1] == 1:
            return fused_recurrent_gla(q, k, v, g, **arguments)
        # chunk_gla builds its chunk index on the host, and the host copy of the offsets
        # is what spares it a device-to-host synchronization for them.
        return chunk_gla(q, k, v, g, cu_seqlens_cpu=cu_seqlens_cpu, **arguments)

    def verification(self, *inputs):
        return inference_verification(inputs[0].dtype, decode=inputs[0].shape[1] == 1)


def decode_verification(dtype):
    from workloads.numerics import Exact

    tol = {torch.float32: 3e-06, torch.float16: 0.0007, torch.bfloat16: 0.003}[dtype]
    return Exact(atol=tol, rtol=tol)


def inference_verification(dtype, *, decode=False):
    from workloads.numerics import Custom, assert_close

    if decode:

        def validate_decode(got, expected):
            # A single step keeps its state in FP32: prefill's accumulation bound
            # does not apply. The stored output may straddle a half/bfloat rounding
            # boundary after two FP32 reduction orders; allow one adjacent value,
            # then apply the original absolute bound to the remaining error.
            actual, target = got[0], expected[0]
            adjacent = torch.nextafter(target, actual)
            residual = (actual.float() - adjacent.float()).abs()
            assert_close(residual, torch.zeros_like(residual), atol=3e-7, rtol=0)
            assert_close(got[1], expected[1], atol=3e-7, rtol=3e-7)

        return Custom(validate_decode, "single-step FP32 state and one-rounding-unit output")

    tol = {torch.float32: 3e-07, torch.float16: 1e-3, torch.bfloat16: 1.6e-2}[dtype]
    state_tol = 0.0025 if dtype == torch.float16 else tol

    def validate(got, expected):
        assert_close(got[0], expected[0], atol=tol, rtol=tol)
        assert_close(got[1], expected[1], atol=state_tol, rtol=tol)

    return Custom(validate, "output and accumulated FP32 state")


def chunkwise_verification(dtype, *, backward=False):
    from workloads.numerics import Custom, Exact, compare_outputs

    if not backward:
        tol = {torch.float32: 1e-2, torch.float16: 5e-2, torch.bfloat16: 1e-1}[dtype]

        def validate_forward(got, expected):
            compare_outputs(got, expected, Exact(atol=tol, rtol=tol))
            actuals = (got,) if isinstance(got, torch.Tensor) else got
            references = (expected,) if isinstance(expected, torch.Tensor) else expected
            for actual, reference in zip(actuals, references, strict=True):
                a, b = actual.float().flatten(), reference.float().flatten()
                if torch.count_nonzero(b):
                    # Absolute slack alone must not accept zero for low-amplitude outputs.
                    cosine = torch.dot(a, b) / (
                        torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
                    )
                    assert cosine > 0.99

        return Custom(validate_forward, "forward error bound and nonzero signal alignment")

    def validate(got, expected):
        # Gradients are about 1e-2; bound each by 1% of its largest reference entry.
        for actual, reference in zip(got, expected, strict=True):
            torch.testing.assert_close(
                actual, reference, atol=1e-2 * reference.abs().max().item(), rtol=0
            )

    return Custom(validate, "gradient error bounded by 1% of each gradient's amplitude")
