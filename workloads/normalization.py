"""Workload definitions for the normalization op family."""

import torch
import torch.nn.functional as F

from workloads.workload_base import CallWorkload, WorkloadBase


class RMSNormWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-6):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.n, dtype=self.dtype, device="cuda")
        return x, weight

    def ref_program(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        rms = torch.sqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x_f32 / rms) * weight.float()).to(x.dtype)


class LayerNormWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.n, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.n, dtype=self.dtype, device="cuda")
        return x, weight, bias

    def ref_program(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        # Reference uses torch.nn.functional.layer_norm
        return F.layer_norm(
            x.float(),
            (self.n,),
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)


class FusedAddRMSNormWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-6):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        residual = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.n, dtype=self.dtype, device="cuda")
        return x, residual, weight

    def ref_program(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        add_result = (x.float() + residual.float()).to(x.dtype)
        add_f32 = add_result.float()
        rms = torch.sqrt(add_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = ((add_f32 / rms) * weight.float()).to(x.dtype)
        return y, add_result


class FusedAddLayerNormWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        residual = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.n, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.n, dtype=self.dtype, device="cuda")
        return x, residual, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        add_result = (x.float() + residual.float()).to(x.dtype)
        y = F.layer_norm(
            add_result.float(),
            (self.n,),
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)
        return y, add_result


class AdaLayerNormWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        scale = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        shift = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return x, scale, shift

    def ref_program(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        # AdaLN: y = scale * LayerNorm(x) + shift
        normed = F.layer_norm(
            x.float(),
            (self.n,),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        y = scale.float() * normed + shift.float()
        return y.to(x.dtype)


class AdaLayerNormZeroWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        scale = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        shift = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        gate = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return x, scale, shift, gate

    def ref_program(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        # AdaLN-Zero: y = gate * (scale * LayerNorm(x) + shift)
        normed = F.layer_norm(
            x.float(),
            (self.n,),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        y = gate.float() * (scale.float() * normed + shift.float())
        return y.to(x.dtype)


class GroupNormWorkload(WorkloadBase):
    def __init__(
        self, n: int, c: int, spatial: tuple, g: int, dtype: torch.dtype, eps: float = 1e-5
    ):
        self.n = n
        self.c = c
        self.spatial = spatial
        self.g = g
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (self.n, self.c, *self.spatial)
        x = torch.randn(shape, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.c, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.c, dtype=self.dtype, device="cuda")
        return x, weight, bias

    def ref_program(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return F.group_norm(
            x.float(),
            self.g,
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)


class InstanceNormWorkload(WorkloadBase):
    def __init__(self, n: int, c: int, spatial: tuple, dtype: torch.dtype, eps: float = 1e-5):
        self.n = n
        self.c = c
        self.spatial = spatial
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple:
        """Inputs in ``torch.nn.functional.instance_norm`` order.

        The running stats sit between ``x`` and the affine pair, and this
        workload exercises the affine call, so they are ``None``.
        """
        shape = (self.n, self.c, *self.spatial)
        x = torch.randn(shape, dtype=self.dtype, device="cuda")
        weight = torch.randn(self.c, dtype=self.dtype, device="cuda")
        bias = torch.randn(self.c, dtype=self.dtype, device="cuda")
        return x, None, None, weight, bias

    def ref_program(
        self, x: torch.Tensor, running_mean, running_var, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return F.instance_norm(
            x.float(),
            running_mean=running_mean,
            running_var=running_var,
            weight=weight.float(),
            bias=bias.float(),
            eps=self.eps,
        ).to(x.dtype)


def _make_tensors(N, C, spatial, dtype, device="cuda"):
    shape = (N, C, *spatial)
    x = torch.randn(*shape, device=device, dtype=dtype)
    weight = torch.randn(C, device=device, dtype=torch.float32)
    bias = torch.randn(C, device=device, dtype=torch.float32)
    running_mean = torch.zeros(C, device=device, dtype=torch.float32)
    running_var = torch.ones(C, device=device, dtype=torch.float32)
    return x, weight, bias, running_mean, running_var


def batch_norm_fwd_ref(
    x, weight, bias, running_mean, running_var, training, momentum=0.1, eps=1e-5
):
    """Reference: torch.nn.functional.batch_norm (float32 upcast)."""
    x32 = x.float()
    rm = running_mean.clone()
    rv = running_var.clone()
    y32 = torch.nn.functional.batch_norm(
        x32, rm, rv, weight.float(), bias.float(), training=training, momentum=momentum, eps=eps
    )
    return y32.to(x.dtype), rm, rv


class BatchNormBwdWorkload(WorkloadBase):
    def __init__(self, N, C, spatial, dtype):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x, weight, bias, running_mean, running_var = _make_tensors(
            self.N, self.C, self.spatial, self.dtype
        )
        grad_out = torch.randn_like(x)
        # Need mean/rstd from a forward pass.
        x32 = x.float()
        # Compute mean and rstd via native batch norm internals.
        C = self.C
        L = x32.numel() // C
        x_cl = x32.permute(1, 0, *range(2, x32.ndim)).reshape(C, L).contiguous()
        mean = x_cl.mean(dim=1)
        var = x_cl.var(dim=1, unbiased=False)
        rstd = 1.0 / torch.sqrt(var + 1e-5)
        return grad_out, x, weight, mean, rstd

    def ref_program(self, grad_out, x, weight, mean, rstd):
        """Reference via torch.autograd on a float32 graph."""
        x32 = x.float().requires_grad_(True)
        w32 = weight.float().requires_grad_(True)
        b32 = torch.zeros(self.C, device=x.device, dtype=torch.float32, requires_grad=True)
        rm = torch.zeros(self.C, device=x.device, dtype=torch.float32)
        rv = torch.ones(self.C, device=x.device, dtype=torch.float32)
        y32 = torch.nn.functional.batch_norm(
            x32, rm, rv, w32, b32, training=True, momentum=0.1, eps=1e-5
        )
        y32.backward(grad_out.float())
        return x32.grad.to(x.dtype), w32.grad, b32.grad


class BatchNormFwdWorkload(WorkloadBase):
    def __init__(self, N, C, spatial, dtype, training):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype
        self.training = training

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        return _make_tensors(self.N, self.C, self.spatial, self.dtype)

    def ref_program(self, x, weight, bias, running_mean, running_var):
        y, rm, rv = batch_norm_fwd_ref(
            x, weight, bias, running_mean, running_var, training=self.training
        )
        return (y,)


class NormCall(CallWorkload):
    """One manifest call of a norm op, with its first input's shape and dtype for the report."""

    def __init__(self, call, device: "torch.device | str" = "cuda"):
        super().__init__(call, device)
        spec = call.specs[next(iter(call.signature.inputs))]
        self.shape, self.dtype = spec.shape, spec.dtype


class RunningStatsCall(NormCall):
    """A variance is positive: a passed ``running_var`` holds values in [0.5, 1.5)."""

    def gen_inputs(self) -> tuple:
        inputs = dict(zip(self.call.signature.inputs, super().gen_inputs(), strict=True))
        if inputs.get("running_var") is not None:
            inputs["running_var"] = torch.rand_like(inputs["running_var"]) + 0.5
        return tuple(inputs.values())


class BatchNormBwdCall(NormCall):
    """``mean`` and ``rstd`` are the batch statistics of ``x``, as the forward saved them."""

    def gen_inputs(self) -> tuple:
        grad_out, x, weight, _mean, _rstd = super().gen_inputs()
        axes = [0, *range(2, x.ndim)]
        var, mean = torch.var_mean(x.float(), dim=axes, correction=0)
        return grad_out, x, weight, mean, torch.rsqrt(var + 1e-5)

    def ref_program(self, grad_out, x, weight, mean, rstd):
        """The gradients of the training forward, from the saved statistics."""
        shape = (1, -1, *[1] * (x.ndim - 2))
        axes = [0, *range(2, x.ndim)]
        x_hat = (x.float() - mean.view(shape)) * rstd.view(shape)
        grad_bias = grad_out.float().sum(axes)
        grad_weight = (grad_out.float() * x_hat).sum(axes)
        length = x.numel() // x.shape[1]
        grad_x = (
            (weight * rstd).view(shape)
            / length
            * (length * grad_out.float() - grad_bias.view(shape) - x_hat * grad_weight.view(shape))
        )
        return grad_x.to(x.dtype), grad_weight, grad_bias
