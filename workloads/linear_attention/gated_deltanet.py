import torch

from workloads.device import run_device
from workloads.linear_attention.input_values import _log_gates, _small, _step_sizes
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = ["GatedDeltaNetFwdCall", "GatedDeltaNetFwdWorkload"]


class GatedDeltaNetFwdWorkload(WorkloadBase):
    """BTHD Gated DeltaNet inference prefill or decode, equal-length or packed.

    ``sequence_lengths`` packs the rows into one ``B = 1`` token axis and makes the call
    carry ``cu_seqlens``; ``value_heads`` gives the recurrence more heads than the key
    carries, with key head ``h // (value_heads // heads)`` serving value head ``h``.
    ``l2norm``, ``raw_gate`` and ``beta_sigmoid`` hand each of Q and K, the gate and the
    step size over untransformed, and ``raw_gate`` adds ``A_log`` and ``dt_bias`` to the
    call.
    """

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        dim: int,
        dtype: torch.dtype,
        scale: float | None = None,
        has_initial_state: bool = False,
        value_heads: int | None = None,
        sequence_lengths: tuple[int, ...] | None = None,
        l2norm: bool = False,
        raw_gate: bool = False,
        beta_sigmoid: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
    ) -> None:
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.dim = dim
        self.dtype = dtype
        self.scale = scale
        self.has_initial_state = has_initial_state
        self.value_heads = heads if value_heads is None else value_heads
        self.sequence_lengths = sequence_lengths
        self.l2norm = l2norm
        self.raw_gate = raw_gate
        self.beta_sigmoid = beta_sigmoid
        self.allow_neg_eigval = allow_neg_eigval
        self.state_v_first = state_v_first

    @property
    def _spans(self) -> tuple[tuple[int, int], ...]:
        """The ``(start, end)`` token span of every sequence in the call."""
        lengths = (
            (self.seq_len,) * self.batch if self.sequence_lengths is None else self.sequence_lengths
        )
        spans, start = [], 0
        for length in lengths:
            spans.append((start, start + length))
            start += length
        return tuple(spans)

    def gen_inputs(self) -> tuple[torch.Tensor | None, ...]:
        packed = self.sequence_lengths is not None
        rows = (1, sum(self.sequence_lengths)) if packed else (self.batch, self.seq_len)
        q = torch.randn((*rows, self.heads, self.dim), device=run_device(), dtype=self.dtype) * 0.1
        k = torch.randn((*rows, self.heads, self.dim), device=run_device(), dtype=self.dtype) * 0.1
        shape = (*rows, self.value_heads, self.dim)
        v = torch.randn(shape, device=run_device(), dtype=self.dtype) * 0.1
        g = (
            torch.randn(shape[:3], device=run_device(), dtype=self.dtype)
            if self.raw_gate
            else -torch.rand(shape[:3], device=run_device(), dtype=self.dtype)
        )
        beta = (
            torch.randn(shape[:3], device=run_device(), dtype=self.dtype)
            if self.beta_sigmoid
            else torch.rand(shape[:3], device=run_device(), dtype=self.dtype) * 0.5
        )
        initial_state = (
            torch.randn(
                len(self._spans),
                self.value_heads,
                self.dim,
                self.dim,
                device=run_device(),
                dtype=torch.float32,
            )
            * 0.01
            if self.has_initial_state
            else None
        )
        offsets = (
            torch.tensor(
                [0, *(end for _, end in self._spans)], dtype=torch.int64, device=run_device()
            )
            if packed
            else None
        )
        if self.raw_gate:
            rates = torch.empty(
                self.value_heads, device=run_device(), dtype=torch.float32
            ).uniform_(1.0, 16.0)
            steps = torch.empty(
                self.value_heads, device=run_device(), dtype=torch.float32
            ).uniform_(1e-3, 0.1)
            return (
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                offsets,
                None if offsets is None else offsets.cpu(),
                rates.log(),
                steps.expm1().log(),
            )
        if not packed:
            return (
                (q, k, v, g, beta) if initial_state is None else (q, k, v, g, beta, initial_state)
            )
        return q, k, v, g, beta, initial_state, offsets, offsets.cpu()

    def ref_program(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        A_log: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return gated_deltanet_ref(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            cu_seqlens_cpu,
            A_log,
            dt_bias,
            scale=self.scale,
            raw_gate=self.raw_gate,
            beta_sigmoid=self.beta_sigmoid,
            allow_neg_eigval=self.allow_neg_eigval,
            l2norm=self.l2norm,
            state_v_first=self.state_v_first,
        )

    def verification(self, *inputs):
        return gated_verification(
            inputs,
            l2norm=self.l2norm,
            state_v_first=self.state_v_first,
        )


class GatedDeltaNetFwdCall(CallWorkload):
    """A manifest call sharing the focused workload's FP32 recurrence and verification."""

    def gen_inputs(self):
        q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu, a_log, dt_bias = (
            super().gen_inputs()
        )
        raw_gate = self.call.ix["use_gate_in_kernel"]
        if raw_gate:
            # Log decay rates in [1, 16] and inverse-softplus time steps in [1e-3, 0.1].
            a_log = torch.empty_like(a_log).uniform_(1.0, 16.0).log()
            dt_bias = torch.empty_like(dt_bias).uniform_(1e-3, 0.1).expm1().log()
        return (
            _small(q),
            _small(k),
            _small(v),
            g if raw_gate else _log_gates(g),
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
        return gated_deltanet_ref(
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
            beta_sigmoid=p["use_beta_sigmoid_in_kernel"],
            allow_neg_eigval=p["allow_neg_eigval"],
            l2norm=p["use_qk_l2norm_in_kernel"],
            state_v_first=p["state_v_first"],
        )

    def verification(self, *inputs):
        p = self.call.ix
        return gated_verification(
            inputs,
            l2norm=p["use_qk_l2norm_in_kernel"],
            state_v_first=p["state_v_first"],
        )


def gated_deltanet_ref(
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
    beta_sigmoid=False,
    allow_neg_eigval=False,
    l2norm=False,
    state_v_first=False,
):
    """A token-by-token FP32 recurrence, independent of the chunked kernels and of FLA.

    The sequence boundaries come from the call's offsets, never from the fixture's
    lengths, so a call whose offsets were rewritten in place is checked as it ran.
    """
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    group = v.shape[2] // q.shape[2]
    if cu_seqlens is None:
        spans = [(0, q.shape[1])] * q.shape[0]
    else:
        bounds = (cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu).tolist()
        spans = list(zip(bounds[:-1], bounds[1:], strict=True))
    if raw_gate:
        g = -torch.exp(a_log) * torch.nn.functional.softplus(g.float() + dt_bias)
    if beta_sigmoid:
        beta = torch.sigmoid(beta.float()) * (2.0 if allow_neg_eigval else 1.0)
    states, output = [], torch.empty_like(v)
    for sequence, (first, last) in enumerate(spans):
        state = (
            torch.zeros(v.shape[2], q.shape[-1], v.shape[-1], device=q.device, dtype=torch.float32)
            if initial_state is None
            else (
                initial_state[sequence].float().transpose(-1, -2)
                if state_v_first
                else initial_state[sequence].float()
            )
        )
        for token in range(last - first):
            index = (0, first + token) if cu_seqlens is not None else (sequence, token)
            # Value head h reads the key head its group shares.
            q_t = q[index].float().repeat_interleave(group, dim=0)
            k_t = k[index].float().repeat_interleave(group, dim=0)
            if l2norm:
                q_t = q_t * torch.rsqrt(q_t.square().sum(-1, keepdim=True) + 1e-6)
                k_t = k_t * torch.rsqrt(k_t.square().sum(-1, keepdim=True) + 1e-6)
            q_t = q_t * scale
            decay = g[index].float().exp()
            old_value = torch.einsum("hkv,hk->hv", state, k_t)
            value = beta[index].float().unsqueeze(-1) * (
                v[index].float() - decay.unsqueeze(-1) * old_value
            )
            state = decay[:, None, None] * state + k_t.unsqueeze(-1) * value.unsqueeze(-2)
            output[index] = torch.einsum("hk,hkv->hv", q_t, state).to(q.dtype)
        states.append(state.transpose(-1, -2) if state_v_first else state)
    return output, torch.stack(states)


def gated_verification(
    inputs,
    *,
    l2norm=False,
    state_v_first=False,
):
    from workloads.numerics import (
        Custom,
        assert_close,
        assert_normalized_error,
        assert_rounded,
        zeroed_input,
    )

    q = inputs[0]
    if q.shape[1] == 1:
        tol = 6e-7 if q.dtype == torch.float16 else 1e-7
        if q.shape[-1] == 64:
            tol = 1e-5
        if state_v_first or l2norm:
            tol = 4e-8
        output_atol, state_atol, rtol = tol, tol, tol
    else:
        output_atol = rtol = 1e-3 if q.dtype == torch.float16 else 1.6e-2
        state_atol = (
            1e-4
            if q.shape[1] == 64
            and q.shape[-1] == 128
            and all(value is None for value in inputs[5:])
            else output_atol
        )

    def validate(got, expected):
        compare_output = assert_rounded if q.shape[1] == 1 else assert_close
        compare_output(got[0], expected[0], atol=output_atol, rtol=rtol)
        assert_close(got[1], expected[1], atol=state_atol, rtol=rtol)
        if q.shape[1] != 1:
            # Absolute bounds cover cancellation in the chunked recurrence, but can
            # exceed small outputs. Also apply the shared normalized-error budget
            # (1e-3 of combined energy, ~4.5% relative RMS) to each tensor. Clearing
            # a nonzero tensor has ratio 1 and cannot pass, regardless of magnitude.
            assert_normalized_error(got, expected)

    controls = ()
    if q.shape[1] == 1:
        controls = (zeroed_input(0, "query-zeroed"),)
        # A single token from zero state does not depend on the gate.
        if len(inputs) > 5 and inputs[5] is not None:
            controls += (zeroed_input(3, "gate-zeroed"),)
    return Custom(
        validate,
        "output and FP32 recurrence state, including input transforms",
        controls=controls,
    )
