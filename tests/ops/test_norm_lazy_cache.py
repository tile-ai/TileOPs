import pytest
import torch

from tileops.kernels.kernel_base import Kernel
from tileops.ops.norm.batch_norm import BatchNormFwdOp


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
    assert op.eval_roofline() == (
        10 * 8 * 32,
        2 * 8 * 32 * torch.float16.itemsize + 4 * 8 * 4,
    )

    run_case(2, 8, (4, 4), torch.float16)
    assert len(op._kernel_cache) == 1

    run_case(3, 12, (2, 8), torch.bfloat16)
    assert len(op._kernel_cache) == 2
    assert op.eval_roofline() == (
        10 * 12 * 48,
        2 * 12 * 48 * torch.bfloat16.itemsize + 4 * 12 * 4,
    )
