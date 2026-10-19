"""Workload definitions for the normalization op family."""

import torch
import torch.nn.functional as F

from workloads.device import run_device
from workloads.workload_base import CallWorkload, WorkloadBase


class RMSNormWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-6):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        weight = torch.randn(self.n, dtype=self.dtype, device=run_device())
        return x, weight

    def ref_program(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _norm_reference(
            "RMSNormFwdOp", (x, weight), dict(eps=self.eps, normalized_shape=(self.n,))
        )

    def verification(self, *inputs):
        return norm_verification(inputs[0].dtype)


class LayerNormWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        weight = torch.randn(self.n, dtype=self.dtype, device=run_device())
        bias = torch.randn(self.n, dtype=self.dtype, device=run_device())
        return x, weight, bias

    def ref_program(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        # Reference uses torch.nn.functional.layer_norm
        return _norm_reference(
            "LayerNormFwdOp", (x, weight, bias), dict(eps=self.eps, normalized_shape=(self.n,))
        )

    def verification(self, *inputs):
        return layer_norm_verification(inputs[0].dtype)


class LayerNormLargeOffsetWorkload(LayerNormWorkload):
    """Adversarial mean 1e4 / standard deviation 1e-2 tests variance stability."""

    def gen_inputs(self):
        x = (10000.0 + 0.01 * torch.randn(self.m, self.n, device=run_device())).to(self.dtype)
        weight = torch.ones(self.n, dtype=self.dtype, device=x.device)
        return x, weight, torch.zeros_like(weight)

    def verification(self, *inputs):
        from workloads.numerics import Custom, Exact, compare_outputs

        # Cancellation amplifies FP32 reduction-order differences by about 1-2%.
        evidence = (
            Exact(atol=1e-1, rtol=5e-2)
            if inputs[0].dtype == torch.float32
            else super().verification(*inputs)
        )

        def validate(got, expected):
            result = compare_outputs(got, expected, evidence)
            assert result.max_abs_err < 1.0, "catastrophic cancellation"

        return Custom(validate, "large-offset normalization without catastrophic cancellation")


class FusedAddRMSNormWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-6):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        residual = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        weight = torch.randn(self.n, dtype=self.dtype, device=run_device())
        return x, residual, weight

    def ref_program(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _norm_reference(
            "FusedAddRMSNormFwdOp",
            (x, residual, weight),
            dict(eps=self.eps, normalized_shape=(self.n,)),
        )

    def verification(self, *inputs):
        return norm_verification(inputs[0].dtype)


class FusedAddLayerNormWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        residual = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        weight = torch.randn(self.n, dtype=self.dtype, device=run_device())
        bias = torch.randn(self.n, dtype=self.dtype, device=run_device())
        return x, residual, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _norm_reference(
            "FusedAddLayerNormFwdOp",
            (x, residual, weight, bias),
            dict(eps=self.eps, normalized_shape=(self.n,)),
        )


class AdaLayerNormWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        scale = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        shift = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        return x, scale, shift

    def ref_program(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        # AdaLN: y = scale * LayerNorm(x) + shift
        return _norm_reference(
            "AdaLayerNormFwdOp", (x, scale, shift), dict(eps=self.eps, normalized_shape=(self.n,))
        )


class AdaLayerNormZeroWorkload(WorkloadBase):
    def __init__(self, m: int, n: int, dtype: torch.dtype, eps: float = 1e-5):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.eps = eps

    def gen_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        scale = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        shift = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        gate = torch.randn(self.m, self.n, dtype=self.dtype, device=run_device())
        return x, scale, shift, gate

    def ref_program(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        # AdaLN-Zero: y = gate * (scale * LayerNorm(x) + shift)
        return _norm_reference(
            "AdaLayerNormZeroFwdOp",
            (x, scale, shift, gate),
            dict(eps=self.eps, normalized_shape=(self.n,)),
        )


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
        x = torch.randn(shape, dtype=self.dtype, device=run_device())
        weight = torch.randn(self.c, dtype=self.dtype, device=run_device())
        bias = torch.randn(self.c, dtype=self.dtype, device=run_device())
        return x, weight, bias

    def ref_program(
        self,
        x: torch.Tensor,
        weight: "torch.Tensor | None" = None,
        bias: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        return _norm_reference(
            "GroupNormFwdOp", (x, weight, bias), dict(eps=self.eps, num_groups=self.g)
        )


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
        x = torch.randn(shape, dtype=self.dtype, device=run_device())
        weight = torch.randn(self.c, dtype=self.dtype, device=run_device())
        bias = torch.randn(self.c, dtype=self.dtype, device=run_device())
        return x, None, None, weight, bias

    def ref_program(
        self, x: torch.Tensor, running_mean, running_var, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return _norm_reference(
            "InstanceNormFwdOp",
            (x, running_mean, running_var, weight, bias),
            dict(eps=self.eps, use_input_stats=True, momentum=0.1),
        )


def _make_tensors(N, C, spatial, dtype, device=None):
    device = device or run_device()
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
    rm = None if running_mean is None else running_mean.clone()
    rv = None if running_var is None else running_var.clone()
    y32 = torch.nn.functional.batch_norm(
        x32,
        rm,
        rv,
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        training=training,
        momentum=momentum,
        eps=eps,
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
        return batch_norm_backward(grad_out, x, weight, mean, rstd)

    def verification(self, *inputs):
        return batch_norm_backward_verification(inputs[0].dtype)


class BatchNormFwdWorkload(WorkloadBase):
    def __init__(self, N, C, spatial, dtype, training):
        self.N = N
        self.C = C
        self.spatial = spatial
        self.dtype = dtype
        self.training = training

    def gen_inputs(self) -> tuple[torch.Tensor, ...]:
        x, weight, bias, mean, var = _make_tensors(self.N, self.C, self.spatial, self.dtype)
        return x, mean, var, weight, bias

    def ref_program(self, x, running_mean, running_var, weight, bias):
        return batch_norm_fwd_ref(
            x, weight, bias, running_mean, running_var, training=self.training
        )

    def verification(self, *inputs):
        return batch_norm_forward_verification(inputs[0].dtype)


class NormCall(CallWorkload):
    """One manifest call of a norm op, with its first input's shape and dtype for the report."""

    def __init__(self, call, device: "torch.device | str | None" = None):
        super().__init__(call, device)
        spec = call.specs[next(iter(call.signature.inputs))]
        self.shape, self.dtype = spec.shape, spec.dtype

    def ref_program(self, *inputs):
        return _norm_reference(self.call.signature.name, inputs, self.call.params)

    def verification(self, *inputs):
        return normalization_verification(self.call.signature.name, inputs[0].dtype)


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
        return batch_norm_backward(grad_out, x, weight, mean, rstd)

    def verification(self, *inputs):
        return batch_norm_backward_verification(inputs[0].dtype)


def batch_norm_backward_verification(dtype):
    from workloads.numerics import Custom

    atol = rtol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 2e-2}[dtype]
    # Channel gradients reduce N * spatial values; cancellation needs an absolute bound.
    sum_atol = 1e-4 if dtype == torch.float32 else atol

    def validate(got, expected):
        for actual, reference, bound in zip(got, expected, (atol, sum_atol, sum_atol), strict=True):
            torch.testing.assert_close(actual, reference, atol=bound, rtol=rtol)

    return Custom(validate, "input gradient and accumulated channel gradients")


def batch_norm_backward(grad_out, x, weight, mean, rstd):
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


def norm_verification(dtype):
    from workloads.numerics import Exact

    if dtype == torch.float16:
        return Exact(atol=1e-2, rtol=1e-2)
    return Exact()


def batch_norm_forward_result(subject, *inputs):
    """Expose the returned tensor and the two declared running-stat side effects."""
    output = subject(*inputs)
    return output, inputs[1], inputs[2]


def batch_norm_forward_verification(dtype):
    from workloads.numerics import Exact

    # Output narrowing and running-stat reductions share the original forward bound.
    tol = {torch.float32: 1e-5, torch.float16: 1e-2, torch.bfloat16: 2e-2}[dtype]
    return Exact(atol=tol, rtol=tol)


def layer_norm_verification(dtype):
    from workloads.numerics import Exact

    tol = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1e-2}[dtype]
    return Exact(atol=tol, rtol=tol)


def normalization_verification(name, dtype):
    """Select the normalization contract by operation, independently of its consumer."""
    from workloads.numerics import Exact

    if name == "BatchNormFwdOp":
        return batch_norm_forward_verification(dtype)
    if name == "BatchNormBwdOp":
        return batch_norm_backward_verification(dtype)
    if name == "LayerNormFwdOp":
        return layer_norm_verification(dtype)
    if name in ("RMSNormFwdOp", "FusedAddRMSNormFwdOp"):
        return norm_verification(dtype)
    return Exact()


def _norm_reference(name, inputs, p):
    """One normalization oracle for focused fixtures and manifest calls."""
    x = inputs[0]
    eps = p.get("eps")
    eps = torch.finfo(torch.float32).eps if eps is None else eps
    if name == "GroupNormFwdOp":
        _, weight, bias = inputs
        return F.group_norm(x, p["num_groups"], weight, bias, eps)
    if name in ("InstanceNormFwdOp", "BatchNormFwdOp"):
        _, rm, rv, weight, bias = inputs
        if name == "InstanceNormFwdOp":
            if rm is not None:
                rm, rv = rm.clone(), rv.clone()
            return F.instance_norm(
                x, rm, rv, weight, bias, p["use_input_stats"], p["momentum"], eps
            )
        return batch_norm_fwd_ref(x, weight, bias, rm, rv, p["training"], p["momentum"], eps)
    if name in ("AdaLayerNormFwdOp", "AdaLayerNormZeroFwdOp"):
        _, scale, shift, *gate = inputs
        normed = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
        out = scale.float() * normed + shift.float()
        return (out * gate[0].float() if gate else out).to(x.dtype)
    fused = name in ("FusedAddRMSNormFwdOp", "FusedAddLayerNormFwdOp")
    if fused:
        x = (x.float() + inputs[1].float()).to(x.dtype)
        weight, *bias = inputs[2:]
    else:
        weight, *bias = inputs[1:]
    shape = tuple(p.get("normalized_shape", (x.shape[-1],)))
    w = None if weight is None else weight.float()
    if name in ("RMSNormFwdOp", "FusedAddRMSNormFwdOp"):
        out = F.rms_norm(x.float(), shape, w, eps).to(x.dtype)
    else:
        b = None if not bias or bias[0] is None else bias[0].float()
        out = F.layer_norm(x.float(), shape, w, b, eps).to(x.dtype)
    return (out, x) if fused else out


def fused_qk_norm_rope_result(subject, *inputs):
    """Expose the declared write of the in-place op, qkv, as its numeric result."""
    subject(*inputs)
    return inputs[0]


def fused_qk_norm_rope_heads(
    qkv, q_weight, k_weight, cos_sin_cache, positions, num_heads, num_kv_heads, eps, rope_layout
):
    """The q and k columns of ``qkv`` after the op, ``(N, (num_heads + num_kv_heads) * D)``.

    Float32 throughout; each value rounds once to the ``qkv`` dtype.
    """
    dim = q_weight.shape[0]
    tokens = qkv.shape[0]
    half = cos_sin_cache.shape[1] // 2
    cos, sin = cos_sin_cache[positions].float().chunk(2, -1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    heads = []
    for first, count, weight in (
        (0, num_heads, q_weight),
        (num_heads, num_kv_heads, k_weight),
    ):
        x = qkv[:, first * dim : (first + count) * dim].float().view(tokens, count, dim)
        y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()
        rot, rest = y[..., : 2 * half], y[..., 2 * half :]
        if rope_layout == "neox":
            a, b = rot[..., :half], rot[..., half:]
            rotated = torch.cat([a * cos - b * sin, b * cos + a * sin], -1)
        else:
            a, b = rot[..., 0::2], rot[..., 1::2]
            rotated = torch.stack([a * cos - b * sin, b * cos + a * sin], -1).flatten(-2)
        heads.append(torch.cat([rotated, rest], -1).reshape(tokens, count * dim))
    return torch.cat(heads, -1).to(qkv.dtype)


class FusedQKNormRoPEWorkload(WorkloadBase):
    """A packed projection of one token per entry of ``positions``, normalized and rotated in
    place.

    ``qkv`` is random; the weights lie in ``[0.5, 1.5)`` and differ between q and k, so a
    kernel that swaps them cannot pass. The cache holds the cosines and sines of the
    standard RoPE frequencies at base 1e6 for every one of its ``positions_max`` rows.
    """

    def __init__(
        self,
        positions: list[int],
        num_heads: int,
        num_kv_heads: int,
        dim: int,
        rot: int,
        positions_max: int,
        dtype: torch.dtype,
        cache_dtype: torch.dtype,
        rope_layout: str = "neox",
        eps: float = 1e-6,
    ) -> None:
        self.positions = positions
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.dim, self.rot, self.positions_max = dim, rot, positions_max
        self.dtype, self.cache_dtype = dtype, cache_dtype
        self.rope_layout, self.eps = rope_layout, eps

    def gen_inputs(self):
        device = run_device()
        width = (self.num_heads + 2 * self.num_kv_heads) * self.dim
        qkv = torch.randn(
            (len(self.positions), width),
            dtype=torch.float32,
            device=device,
            generator=self.rng("qkv", device=device),
        ).to(self.dtype)
        weights = torch.rand(
            (2, self.dim),
            dtype=torch.float32,
            device=device,
            generator=self.rng("w", device=device),
        )
        q_weight, k_weight = (weights + 0.5).to(self.dtype).unbind(0)
        inv = 1.0 / 1e6 ** (
            torch.arange(0, self.rot, 2, dtype=torch.float32, device=device) / self.rot
        )
        angles = torch.arange(self.positions_max, dtype=torch.float32, device=device)[:, None] * inv
        cos_sin_cache = torch.cat([angles.cos(), angles.sin()], -1).to(self.cache_dtype)
        positions = torch.tensor(self.positions, dtype=torch.int64, device=device)
        return qkv, q_weight, k_weight, cos_sin_cache, positions

    def ref_program(self, qkv, q_weight, k_weight, cos_sin_cache, positions):
        """``qkv`` with its q and k columns replaced, on a clone: the shared inputs stay as
        they were, and the v columns of the result are the ones it was handed."""
        out = qkv.clone()
        out[:, : (self.num_heads + self.num_kv_heads) * self.dim] = fused_qk_norm_rope_heads(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.eps,
            self.rope_layout,
        )
        return out


class FusedQKNormRoPECall(CallWorkload, FusedQKNormRoPEWorkload):
    """The manifest call; its positions are the row's packed sequence positions."""

    def __init__(self, call):
        CallWorkload.__init__(self, call)
        FusedQKNormRoPEWorkload.__init__(
            self,
            positions=call.metadata_values("positions"),
            num_heads=call.params["num_heads"],
            num_kv_heads=call.params["num_kv_heads"],
            dim=call.indices["D"],
            rot=call.indices["R"],
            positions_max=call.indices["P"],
            dtype=getattr(torch, call.tensors["qkv"][1]),
            cache_dtype=getattr(torch, call.tensors["cos_sin_cache"][1]),
            rope_layout=call.params["rope_layout"],
            eps=call.params["eps"],
        )

    gen_inputs = FusedQKNormRoPEWorkload.gen_inputs
    ref_program = FusedQKNormRoPEWorkload.ref_program

    def arguments(self):
        """The op is built from its parameters alone; nothing is materialized for it."""
        return self.call.arguments({})
