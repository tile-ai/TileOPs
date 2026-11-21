import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from workloads.attention.gqa.call_metadata import _dtype
from workloads.attention.gqa.dense import dense_gqa_logits
from workloads.device import run_device
from workloads.numerics import Exact, reference_tolerance, zeroed_input
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = ["GQABwdCall", "GQABwdWorkload"]


class GQABwdWorkload(WorkloadBase):
    def __init__(
        self,
        batch: int,
        heads: int,
        heads_kv: int,
        seq_len: int,
        dim: int,
        is_causal: bool,
        dtype: torch.dtype,
        sm_scale: float | None = None,
    ) -> None:
        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv
        self.seq_len = seq_len
        self.dim = dim
        self.is_causal = is_causal
        self.dtype = dtype
        self.sm_scale = sm_scale

    def gen_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim,
            dtype=self.dtype,
            device=run_device(),
            requires_grad=True,
        )
        k = torch.randn(
            self.batch,
            self.seq_len,
            self.heads_kv,
            self.dim,
            dtype=self.dtype,
            device=run_device(),
            requires_grad=True,
        )
        v = torch.randn(
            self.batch,
            self.seq_len,
            self.heads_kv,
            self.dim,
            dtype=self.dtype,
            device=run_device(),
            requires_grad=True,
        )
        grad_output = torch.randn(
            self.batch, self.seq_len, self.heads, self.dim, dtype=self.dtype, device=run_device()
        )

        with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
            o = (
                F.scaled_dot_product_attention(
                    q.float().transpose(1, 2),
                    k.float().transpose(1, 2),
                    v.float().transpose(1, 2),
                    is_causal=self.is_causal,
                    scale=self.sm_scale,
                    enable_gqa=True,
                )
                .transpose(1, 2)
                .to(self.dtype)
                .contiguous()
            )
            lse = dense_gqa_logits(
                q,
                k,
                heads=self.heads,
                heads_kv=self.heads_kv,
                is_causal=self.is_causal,
                sm_scale=self.sm_scale,
            ).logsumexp(-1)

        return grad_output, q, k, v, o, lse

    def ref_program(self, grad_output, q, k, v, o, lse):
        """FP32 SDPA gradients in BSHD layout, rounded once to the input dtype.

        A row whose ``lse`` is ``-inf`` saw no key and contributes no gradient.
        """
        dtype = q.dtype
        grad_output = grad_output.masked_fill(torch.isneginf(lse).transpose(1, 2)[..., None], 0)
        with torch.enable_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
            q, k, v = (t.detach().float().requires_grad_(True) for t in (q, k, v))
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=self.is_causal,
                scale=self.sm_scale,
                enable_gqa=True,
            ).transpose(1, 2)
            gradients = torch.autograd.grad(out, (q, k, v), grad_output.float())
            return tuple(grad.to(dtype) for grad in gradients)

    def verification(self, *inputs):
        # Gradients reach |g| ~ 8, where one stored step exceeds any fixed atol: the dtype
        # rtol covers that step, the 5e-3 floor the FP32 accumulation error near zero.
        tolerance = reference_tolerance(inputs[0].dtype)
        return Exact(
            controls=(zeroed_input(0, "first-input-zeroed"),),
            atol=max(5e-3, tolerance["atol"]),
            rtol=tolerance["rtol"],
        )


class GQABwdCall(CallWorkload, GQABwdWorkload):
    """A manifest call of GQADenseBwdOp; ``o`` and ``lse`` are the forward's."""

    def __init__(self, call) -> None:
        CallWorkload.__init__(self, call)
        indices = call.indices
        GQABwdWorkload.__init__(
            self,
            indices["B"],
            indices["H"],
            indices["H_kv"],
            indices["S"],
            indices["D"],
            indices["is_causal"],
            _dtype(call, "q"),
            call.params["sm_scale"],
        )

    gen_inputs = GQABwdWorkload.gen_inputs

    def ref_program(self, grad_output, q, k, v, o, lse):
        """The FP32 SDPA gradients; a call with a window or a softcap has no reference."""
        params = self.call.params
        if (params["window_size_left"], params["window_size_right"], params["softcap"]) != (
            -1,
            -1,
            None,
        ):
            raise NotImplementedError("the reference computes no window and no softcap")
        return GQABwdWorkload.ref_program(self, grad_output, q, k, v, o, lse)
