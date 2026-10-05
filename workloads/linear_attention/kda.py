import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.linear_attention.input_values import _small, _step_sizes
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = ["KDAFwdCall", "KDAFwdWorkload", "kda_ref"]

# The reference's chunk: the exact per-channel decay gap it materializes is
# [sequences, value heads, chunk, chunk, K] in FP64.
_REF_CHUNK = 16


def _channel_log_gates(like: torch.Tensor) -> torch.Tensor:
    """Per-channel log decays in ``(log sigmoid(0), log sigmoid(1))``, as FLA's tests draw them."""
    return F.logsigmoid(torch.rand(like.shape, device=like.device)).to(like.dtype)


class KDAFwdWorkload(WorkloadBase):
    """BTHD Kimi Delta Attention (KDA) prefill or decode, equal-length or packed.

    ``sequence_lengths`` packs the rows into one ``B = 1`` token axis and makes the call
    carry ``cu_seqlens``; ``value_heads`` gives the recurrence more heads than the key
    carries. ``g`` holds one precomputed log-space decay per key channel and ``beta`` the
    post-sigmoid step size, the variant the in-tree kernels serve.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        has_initial_state: bool = True,
        value_heads: int | None = None,
        sequence_lengths: tuple[int, ...] | None = None,
        l2norm: bool = True,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim = dim
        self.dtype = dtype
        self.has_initial_state = has_initial_state
        self.value_heads = heads if value_heads is None else value_heads
        self.sequence_lengths = sequence_lengths
        self.l2norm = l2norm

    def gen_inputs(self) -> tuple[torch.Tensor | None, ...]:
        packed = self.sequence_lengths is not None
        rows = (1, sum(self.sequence_lengths)) if packed else (self.batch, self.seq_len)
        sequences = len(self.sequence_lengths) if packed else self.batch
        device = run_device()
        q = torch.randn((*rows, self.heads, self.dim), device=device, dtype=self.dtype) * 0.1
        k = torch.randn_like(q)
        shape = (*rows, self.value_heads, self.dim)
        v = torch.randn(shape, device=device, dtype=self.dtype) * 0.1
        g = _channel_log_gates(v)
        beta = _step_sizes(v[..., 0])
        initial_state = (
            torch.randn(
                sequences, self.value_heads, self.dim, self.dim, device=device, dtype=torch.float32
            )
            * 0.01
            if self.has_initial_state
            else None
        )
        if not packed:
            return q, k, v, g, beta, initial_state
        offsets = [0]
        for length in self.sequence_lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.int64)
        return q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens.cpu()

    def ref_program(
        self, q, k, v, g, beta, initial_state=None, cu_seqlens=None, cu_seqlens_cpu=None
    ):
        return kda_ref(
            q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu, l2norm=self.l2norm
        )

    def verification(self, *inputs):
        return kda_verification(inputs)


class KDAFwdCall(CallWorkload):
    """A manifest call sharing the focused workload's FP64 recurrence and verification."""

    def gen_inputs(self):
        q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu, a_log, dt_bias = (
            super().gen_inputs()
        )
        raw_gate = self.call.ix["use_gate_in_kernel"]
        if raw_gate:
            # Log decay rates in [1, 16]; dt_bias around zero, as a trained layer holds it.
            a_log = torch.empty_like(a_log).uniform_(1.0, 16.0).log()
            dt_bias = None if dt_bias is None else torch.randn_like(dt_bias) * 0.1
        return (
            _small(q),
            _small(k),
            _small(v),
            g if raw_gate else _channel_log_gates(g),
            beta if self.call.ix["use_beta_sigmoid_in_kernel"] else _step_sizes(beta),
            _small(initial_state, 0.01),
            cu_seqlens,
            cu_seqlens_cpu,
            a_log,
            dt_bias,
        )

    def ref_program(
        self, q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu, a_log, dt_bias
    ):
        p = self.call.ix
        return kda_ref(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            cu_seqlens_cpu,
            a_log,
            dt_bias,
            scale=p["scale"],
            raw_gate=p["use_gate_in_kernel"],
            lower_bound=p["lower_bound"],
            beta_sigmoid=p["use_beta_sigmoid_in_kernel"],
            allow_neg_eigval=p["allow_neg_eigval"],
            l2norm=p["use_qk_l2norm_in_kernel"],
            state_v_first=p["state_v_first"],
        )

    def verification(self, *inputs):
        return kda_verification(inputs)


def kda_ref(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    a_log=None,
    dt_bias=None,
    *,
    scale=None,
    raw_gate=False,
    lower_bound=None,
    beta_sigmoid=False,
    allow_neg_eigval=False,
    l2norm=False,
    state_v_first=False,
):
    """The KDA recurrence evaluated independently of the in-tree kernels and of FLA.

    Per token: the state decays by ``exp(g)`` along its key axis, the delta rule writes
    ``beta * (v - k^T S)`` along ``k``, and the output reads ``q^T S``. The recurrence runs
    chunk by chunk in FP64: within a chunk the written values solve the unit
    lower-triangular system exactly, with every per-channel decay gap materialized, so no
    gap is exponentiated with a positive argument. The sequence boundaries come from the
    call's offsets, and a sequence with no token ends on the state it started from.
    """
    dtype = torch.float64
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    group = v.shape[2] // q.shape[2]
    if cu_seqlens is None:
        rows = list(range(q.shape[0]))
        starts, lengths = [0] * q.shape[0], [q.shape[1]] * q.shape[0]
    else:
        bounds = (cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu).tolist()
        rows, starts = [0] * (len(bounds) - 1), bounds[:-1]
        lengths = [end - start for start, end in zip(bounds[:-1], bounds[1:], strict=True)]
    g = g.to(dtype)
    if raw_gate:
        heads, dim = v.shape[2], q.shape[-1]
        x = g + (0.0 if dt_bias is None else dt_bias.to(dtype).view(heads, dim))
        rate = a_log.to(dtype).exp()[:, None]
        g = (
            lower_bound * torch.sigmoid(rate * x)
            if lower_bound is not None
            else -rate * F.softplus(x)
        )
    beta = beta.to(dtype)
    if beta_sigmoid:
        beta = torch.sigmoid(beta) * (2.0 if allow_neg_eigval else 1.0)
    longest = max(lengths, default=0)
    # A decode call runs one token per step, and needs no gap tensor.
    chunk = 1 if longest <= 1 else _REF_CHUNK
    # Longest first, so the sequences still running at a step are a prefix.
    order = sorted(range(len(lengths)), key=lambda sequence: -lengths[sequence])
    row = torch.tensor([rows[s] for s in order], dtype=torch.long, device=q.device)
    start = torch.tensor([starts[s] for s in order], dtype=torch.long, device=q.device)
    length = torch.tensor([lengths[s] for s in order], dtype=torch.long, device=q.device)
    if initial_state is None:
        state = torch.zeros(
            len(order), v.shape[2], q.shape[-1], v.shape[-1], dtype=dtype, device=q.device
        )
    else:
        state = initial_state[order].to(dtype)
        state = state.transpose(-1, -2).contiguous() if state_v_first else state
    output = torch.zeros_like(v)
    offset = torch.arange(chunk, device=q.device)
    inclusive = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril()
    strict = inclusive.tril(-1)
    running = len(order)
    for first in range(0, longest, chunk):
        while lengths[order[running - 1]] <= first:
            running -= 1
        r = row[:running, None]
        valid = first + offset < length[:running, None]
        t = (start[:running, None] + first + offset).clamp(max=q.shape[1] - 1)
        mask = valid.to(dtype)[..., None]
        # [running, chunk, value heads, K]; value head h reads the key head its group shares.
        q_c = q[r, t].to(dtype).repeat_interleave(group, dim=2)
        k_c = k[r, t].to(dtype).repeat_interleave(group, dim=2)
        if l2norm:
            q_c = q_c * torch.rsqrt(q_c.square().sum(-1, keepdim=True) + 1e-6)
            k_c = k_c * torch.rsqrt(k_c.square().sum(-1, keepdim=True) + 1e-6)
        # [running, value heads, chunk, ...]. Padding carries no key, value, step or decay.
        q_c = (q_c * scale).transpose(1, 2)
        k_c = (k_c * mask[..., None]).transpose(1, 2)
        v_c = (v[r, t].to(dtype) * mask[..., None]).transpose(1, 2)
        step = (beta[r, t] * mask).transpose(1, 2)
        log_decay = (g[r, t] * mask[..., None]).transpose(1, 2).cumsum(2)
        live = state[:running]
        # gap[i, j, c] = log_decay[i, c] - log_decay[j, c], used only where j <= i.
        gap = log_decay[..., :, None, :] - log_decay[..., None, :, :]
        weight = gap.masked_fill(~inclusive[..., None], -torch.inf).exp()
        system = step[..., None] * torch.einsum("nhic,nhijc,nhjc->nhij", k_c, weight, k_c)
        system = system.masked_fill(~strict, 0.0)
        decayed = log_decay.exp()
        rhs = step[..., None] * (v_c - (k_c * decayed) @ live)
        written = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
        attention = torch.einsum("nhic,nhijc,nhjc->nhij", q_c, weight, k_c)
        out = ((q_c * decayed) @ live + attention @ written).transpose(1, 2)
        carry = (log_decay[..., -1:, :] - log_decay).exp()
        live = decayed[..., -1, :, None] * live + (k_c * carry).transpose(-1, -2) @ written
        state[:running] = live
        output[r.expand_as(t)[valid], t[valid]] = out[valid].to(v.dtype)
    final = torch.empty_like(state)
    final[order] = state
    final = final.transpose(-1, -2) if state_v_first else final
    return output, final.float()


def kda_verification(inputs):
    from workloads.numerics import Custom, assert_close, assert_normalized_error, zeroed_input

    q = inputs[0]
    atol = rtol = 2e-3 if q.dtype == torch.float16 else 1.6e-2

    def validate(got, expected):
        assert_close(got[0], expected[0], atol=atol, rtol=rtol)
        assert_close(got[1], expected[1], atol=atol, rtol=rtol)
        # Absolute bounds can exceed small outputs; the shared normalized-error budget
        # (1e-3 of combined energy) rejects a cleared or scaled tensor whatever its size.
        assert_normalized_error(got, expected)

    return Custom(
        validate,
        "output and FP32 recurrence state, including input transforms",
        controls=(zeroed_input(0, "query-zeroed"),),
    )
