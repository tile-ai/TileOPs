import pytest
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp


def _to_cl(x: torch.Tensor) -> torch.Tensor:
    return x.permute(1, 0, *range(2, x.ndim)).reshape(x.shape[1], -1).contiguous()


def _from_cl(x_cl: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    n = orig_shape[0]
    c = orig_shape[1]
    spatial = orig_shape[2:]
    return x_cl.reshape(c, n, *spatial).permute(1, 0, *range(2, len(orig_shape))).contiguous()


class _FakeBatchNormFwdInferKernel(Kernel):
    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype,
        eps: float,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.dtype = dtype
        self.eps = eps
        self.tune = tune

    def forward(
        self,
        x_cl: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> torch.Tensor:
        y = (x_cl.float() - running_mean[:, None]) * torch.rsqrt(
            running_var[:, None] + self.eps)
        y = y * weight[:, None] + bias[:, None]
        return y.to(self.dtype)


class _FakeBatchNormFwdTrainKernel(_FakeBatchNormFwdInferKernel):
    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype,
        eps: float,
        momentum: float,
        tune: bool = False,
    ) -> None:
        super().__init__(C, L, dtype, eps, tune=tune)
        self.momentum = momentum

    def forward(
        self,
        x_cl: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x_cl.float().mean(dim=1)
        var = x_cl.float().var(dim=1, unbiased=False)
        rstd = torch.rsqrt(var + self.eps)
        running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
        running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        y = (x_cl.float() - mean[:, None]) * rstd[:, None]
        y = y * weight[:, None] + bias[:, None]
        return y.to(self.dtype), mean, rstd


class _FakeBatchNormBwdKernel(Kernel):
    def __init__(
        self,
        C: int,
        L: int,
        dtype: torch.dtype,
        tune: bool = False,
    ) -> None:
        super().__init__()
        self.C = C
        self.L = L
        self.dtype = dtype
        self.tune = tune

    def forward(
        self,
        grad_out_cl: torch.Tensor,
        x_cl: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_out_f = grad_out_cl.float()
        x_hat = (x_cl.float() - mean[:, None]) * rstd[:, None]
        grad_bias = grad_out_f.sum(dim=1)
        grad_weight = (grad_out_f * x_hat).sum(dim=1)
        grad_x = (
            weight[:, None]
            * rstd[:, None]
            * (self.L * grad_out_f - grad_bias[:, None] - x_hat * grad_weight[:, None])
            / self.L
        )
        return grad_x.to(self.dtype), grad_weight, grad_bias


def _batch_norm_infer_ref(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    affine_shape = (1, x.shape[1]) + (1,) * (x.ndim - 2)
    y = (x.float() - running_mean.reshape(affine_shape)) * torch.rsqrt(
        running_var.reshape(affine_shape) + eps)
    y = y * weight.reshape(affine_shape) + bias.reshape(affine_shape)
    return y.to(x.dtype)


def _batch_norm_train_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_cl = _to_cl(x)
    mean = x_cl.float().mean(dim=1)
    var = x_cl.float().var(dim=1, unbiased=False)
    rstd = torch.rsqrt(var + eps)
    y_cl = (x_cl.float() - mean[:, None]) * rstd[:, None]
    y_cl = y_cl * weight[:, None] + bias[:, None]
    return _from_cl(y_cl.to(x.dtype), x.shape), mean, rstd


def _batch_norm_bwd_ref(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_out_cl = _to_cl(grad_out)
    x_cl = _to_cl(x)
    kernel = _FakeBatchNormBwdKernel(x.shape[1], grad_out.numel() // x.shape[1], x.dtype)
    grad_x_cl, grad_weight, grad_bias = kernel(grad_out_cl, x_cl, weight, mean, rstd)
    return _from_cl(grad_x_cl, x.shape), grad_weight, grad_bias


@pytest.mark.smoke
def test_batch_norm_fwd_lazy_cache_reuse_and_respecialization() -> None:
    """BatchNorm op-layer cache reuses identical specs and caches changed specs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for forward call")

    op = BatchNormFwdOp(
        training=False,
        kernel_map={
            "fwd_infer_kernel": _FakeBatchNormFwdInferKernel,
            "fwd_train_kernel": _FakeBatchNormFwdTrainKernel,
        },
    )

    def run_case(N: int, C: int, spatial: tuple[int, ...], dtype: torch.dtype) -> None:
        x = torch.randn((N, C, *spatial), device="cuda", dtype=dtype)
        weight = torch.randn(C, device="cuda", dtype=torch.float32)
        bias = torch.randn(C, device="cuda", dtype=torch.float32)
        running_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
        running_var = torch.ones(C, device="cuda", dtype=torch.float32)

        y = op(x, running_mean, running_var, weight, bias)
        ref_y = _batch_norm_infer_ref(
            x, running_mean, running_var, weight, bias, op.eps)
        assert torch.allclose(y.float(), ref_y.float(), atol=0.0, rtol=0.0)

    run_case(2, 8, (4, 4), torch.float16)
    assert len(op._kernel_cache) == 1
    first_kernel = op.kernel
    assert op.eval_roofline() == (
        10 * 8 * 32,
        2 * 8 * 32 * torch.float16.itemsize + 4 * 8 * 4,
    )

    run_case(2, 8, (4, 4), torch.float16)
    assert len(op._kernel_cache) == 1
    assert op.kernel is first_kernel

    run_case(3, 12, (2, 8), torch.bfloat16)
    assert len(op._kernel_cache) == 2
    assert op.kernel is not first_kernel
    assert op.eval_roofline() == (
        10 * 12 * 48,
        2 * 12 * 48 * torch.bfloat16.itemsize + 4 * 12 * 4,
    )


@pytest.mark.smoke
def test_batch_norm_training_fwd_lazy_cache_reuse_and_respecialization() -> None:
    """Training BatchNorm forward cache path is executable under fake kernels."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for forward call")

    op = BatchNormFwdOp(
        training=True,
        kernel_map={
            "fwd_infer_kernel": _FakeBatchNormFwdInferKernel,
            "fwd_train_kernel": _FakeBatchNormFwdTrainKernel,
        },
    )

    def run_case(N: int, C: int, spatial: tuple[int, ...], dtype: torch.dtype) -> None:
        x = torch.randn((N, C, *spatial), device="cuda", dtype=dtype)
        weight = torch.randn(C, device="cuda", dtype=torch.float32)
        bias = torch.randn(C, device="cuda", dtype=torch.float32)
        running_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
        running_var = torch.ones(C, device="cuda", dtype=torch.float32)

        y = op(x, running_mean, running_var, weight, bias)
        ref_y, _, _ = _batch_norm_train_ref(x, weight, bias, op.eps)
        assert torch.allclose(y.float(), ref_y.float(), atol=0.0, rtol=0.0)

    run_case(2, 8, (4, 4), torch.float16)
    assert len(op._kernel_cache) == 1
    first_kernel = op.kernel
    assert op.eval_roofline() == (
        10 * 8 * 32,
        2 * 8 * 32 * torch.float16.itemsize + 4 * 8 * 4,
    )

    run_case(2, 8, (4, 4), torch.float16)
    assert len(op._kernel_cache) == 1
    assert op.kernel is first_kernel

    run_case(3, 12, (2, 8), torch.bfloat16)
    assert len(op._kernel_cache) == 2
    assert op.kernel is not first_kernel
    assert op.eval_roofline() == (
        10 * 12 * 48,
        2 * 12 * 48 * torch.bfloat16.itemsize + 4 * 12 * 4,
    )


@pytest.mark.smoke
def test_batch_norm_bwd_lazy_cache_reuse_and_respecialization() -> None:
    """BatchNorm backward cache path is executable under fake kernels."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for backward call")

    eps = 1e-5
    op = BatchNormBwdOp(kernel_map={"bwd_kernel": _FakeBatchNormBwdKernel})

    def run_case(N: int, C: int, spatial: tuple[int, ...], dtype: torch.dtype) -> None:
        x = torch.randn((N, C, *spatial), device="cuda", dtype=dtype)
        grad_out = torch.randn((N, C, *spatial), device="cuda", dtype=dtype)
        weight = torch.randn(C, device="cuda", dtype=torch.float32)
        _y, mean, rstd = _batch_norm_train_ref(
            x, torch.ones_like(weight), torch.zeros_like(weight), eps)

        grad_x, grad_weight, grad_bias = op(grad_out, x, weight, mean, rstd)
        ref_grad_x, ref_grad_weight, ref_grad_bias = _batch_norm_bwd_ref(
            grad_out, x, weight, mean, rstd)
        assert torch.allclose(grad_x.float(), ref_grad_x.float(), atol=0.0, rtol=0.0)
        assert torch.allclose(grad_weight, ref_grad_weight, atol=0.0, rtol=0.0)
        assert torch.allclose(grad_bias, ref_grad_bias, atol=0.0, rtol=0.0)

    run_case(2, 8, (4, 4), torch.float16)
    assert len(op._kernel_cache) == 1
    first_kernel = op.kernel
    assert op.eval_roofline() == (
        8 * 8 * 32,
        3 * 8 * 32 * torch.float16.itemsize + 3 * 8 * 4,
    )

    run_case(2, 8, (4, 4), torch.float16)
    assert len(op._kernel_cache) == 1
    assert op.kernel is first_kernel

    run_case(3, 12, (2, 8), torch.bfloat16)
    assert len(op._kernel_cache) == 2
    assert op.kernel is not first_kernel
    assert op.eval_roofline() == (
        8 * 12 * 48,
        3 * 12 * 48 * torch.bfloat16.itemsize + 3 * 12 * 4,
    )
