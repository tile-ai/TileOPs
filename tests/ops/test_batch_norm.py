"""Tests for BatchNormFwdOp and BatchNormBwdOp, against torch.nn.functional.batch_norm and the analytical
gradient from torch.autograd.
"""

import re

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.backend import BUILTIN
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.norm.call_spec import (
    BatchNormBwdInterface,
    BatchNormCall,
    BatchNormInferFwdInterface,
    BatchNormTrainFwdInterface,
)
from tileops.ops.norm.batch_norm import BatchNormBwdOp, BatchNormFwdOp
from workloads.device import run_device, run_device_available
from workloads.norm import (
    BatchNormBwdWorkload,
    BatchNormFwdWorkload,
    batch_norm_backward,
    batch_norm_backward_verification,
    batch_norm_forward_result,
    batch_norm_forward_verification,
    batch_norm_fwd_ref,
)
from workloads.numerics import compare_outputs


class BatchNormBwdTest(BatchNormBwdWorkload, TestBase):
    pass


class BatchNormFwdTest(BatchNormFwdWorkload, TestBase):
    pass


class BatchNormFwdFixture(FixtureBase):
    """(N, C, *spatial, dtype, training)"""

    PARAMS = [
        (
            "N, C, spatial, dtype, training",
            [
                # BatchNorm1d – (N, C)
                pytest.param(32, 64, (), torch.float16, True, marks=pytest.mark.smoke),
                pytest.param(32, 64, (), torch.bfloat16, True, marks=pytest.mark.smoke),
                pytest.param(4, 256, (28, 28), torch.float32, True, marks=pytest.mark.smoke),
                pytest.param(32, 64, (), torch.float16, False, marks=pytest.mark.full),
                pytest.param(32, 256, (), torch.bfloat16, True, marks=pytest.mark.full),
                # BatchNorm1d – (N, C, L)
                pytest.param(16, 64, (512,), torch.float16, True, marks=pytest.mark.full),
                # L > 8192, so the tiled kernel this shape falls back to reads global
                # memory twice rather than holding the tile in shared memory.
                pytest.param(4, 64, (64, 64), torch.float16, True, marks=pytest.mark.full),
                # BatchNorm2d – (N, C, H, W)
                pytest.param(8, 64, (1024, 1024), torch.float16, True, marks=pytest.mark.full),
                # N*C*H*W = 2**31, one past INT32_MAX: a flat index over these extents does
                # not fit a signed 32-bit integer. Costs 60 GiB, the largest case in the suite.
                pytest.param(8, 64, (2048, 2048), torch.float16, False, marks=pytest.mark.full),
                pytest.param(4, 128, (32, 32), torch.bfloat16, True, marks=pytest.mark.full),
                # Non-aligned spatial: H*W=900, exercises partial-tile path
                pytest.param(8, 64, (30, 30), torch.float16, True, marks=pytest.mark.full),
                pytest.param(8, 64, (30, 30), torch.bfloat16, True, marks=pytest.mark.full),
                # High channel count oversubscribes the SMs, exposing the running-stat update race.
                pytest.param(16, 1024, (512,), torch.float16, True, marks=pytest.mark.full),
                # The streamed path, which no other case reaches: a channel too long to
                # hold in registers (L = 131072) and too many channels to split (C >= 1024).
                pytest.param(2048, 1024, (64,), torch.float16, True, marks=pytest.mark.full),
                # The split path with a chunk no vector step divides: a block must stop at
                # its own chunk rather than sum into the next one's.
                pytest.param(3, 5, (300, 301), torch.float16, True, marks=pytest.mark.full),
            ],
        ),
    ]


class BatchNormBwdFixture(FixtureBase):
    """(N, C, *spatial, dtype)"""

    PARAMS = [
        (
            "N, C, spatial, dtype",
            [
                pytest.param(32, 64, (), torch.float16, marks=pytest.mark.smoke),
                pytest.param(32, 64, (), torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4, 256, (28, 28), torch.float32, marks=pytest.mark.smoke),
                pytest.param(8, 64, (32, 32), torch.float16, marks=pytest.mark.full),
                pytest.param(4, 128, (32, 32), torch.bfloat16, marks=pytest.mark.full),
                # A register-held channel taking several steps per thread.
                pytest.param(4, 64, (64, 64), torch.float16, marks=pytest.mark.full),
                # Non-aligned spatial: H*W=900, a vector narrower than the dtype allows.
                pytest.param(8, 64, (30, 30), torch.float16, marks=pytest.mark.full),
                pytest.param(8, 64, (30, 30), torch.bfloat16, marks=pytest.mark.full),
                # The split path with a chunk no vector step divides: a block must stop at
                # its own chunk rather than sum into the next one's.
                pytest.param(3, 5, (300, 301), torch.bfloat16, marks=pytest.mark.full),
                # The streamed tiled path: a channel too long for registers (L = 73728, odd
                # spatial run) and too many channels to split (C >= 1024).
                pytest.param(8192, 1024, (9,), torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@BatchNormFwdFixture
def test_batch_norm_fwd(N, C, spatial, dtype, training):
    test = BatchNormFwdTest(N, C, spatial, dtype, training)
    inputs = test.gen_inputs()
    op = BatchNormFwdOp(training=training)
    test.check(op, *inputs, runs=lambda *args: batch_norm_forward_result(op, *args))

    if training:
        # Repeat from the same initial statistics to detect update races separately.
        x, mean, var, weight, bias = inputs
        mean2, var2 = mean.clone(), var.clone()
        op(x, mean, var, weight, bias)
        op(x, mean2, var2, weight, bias)
        assert torch.equal(mean, mean2) and torch.equal(var, var2)


@BatchNormBwdFixture
def test_batch_norm_bwd(N, C, spatial, dtype):
    test = BatchNormBwdTest(N, C, spatial, dtype)
    test.check(BatchNormBwdOp(), *test.gen_inputs())


@pytest.mark.smoke
def test_batch_norm_fwd_returns_single_tensor() -> None:
    """BatchNormFwdOp forward must produce one tensor — manifest declares
    a single output. ``training`` is bound at ctor; the runtime kwarg is
    no longer accepted."""
    if not run_device_available():
        pytest.skip("the run device is not available")

    N, C, H, W = 4, 8, 4, 4
    op = BatchNormFwdOp(training=False)
    x = torch.randn(N, C, H, W, device=run_device(), dtype=torch.float16)
    weight = torch.randn(C, device=run_device(), dtype=torch.float32)
    bias = torch.randn(C, device=run_device(), dtype=torch.float32)
    rm = torch.zeros(C, device=run_device(), dtype=torch.float32)
    rv = torch.ones(C, device=run_device(), dtype=torch.float32)

    y = op(x, rm, rv, weight, bias)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape


@pytest.mark.smoke
def test_training_updates_a_non_contiguous_running_stat() -> None:
    """Contiguity normalization must not swallow the write a mutated input promises."""
    if not run_device_available():
        pytest.skip("the run device is not available")

    N, C, H, W = 4, 8, 4, 4
    op = BatchNormFwdOp(training=True)
    x = torch.randn(N, C, H, W, device=run_device(), dtype=torch.float16)
    weight = torch.ones(C, device=run_device(), dtype=torch.float32)
    bias = torch.zeros(C, device=run_device(), dtype=torch.float32)
    # Every other element of a wider buffer: a view the kernel cannot be handed as is.
    rm = torch.zeros(2 * C, device=run_device(), dtype=torch.float32)[::2]
    rv = torch.ones(2 * C, device=run_device(), dtype=torch.float32)[::2]
    assert not rm.is_contiguous()

    workload = BatchNormFwdTest(N, C, (H, W), x.dtype, training=True)
    workload.check(
        op,
        x,
        rm,
        rv,
        weight,
        bias,
        runs=lambda *args: batch_norm_forward_result(op, *args),
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_training_rejects_one_value_per_channel() -> None:
    """Bessel's correction divides by L - 1; torch refuses the same call."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for forward call")

    C = 4
    x = torch.randn(1, C, 1, 1, device="cuda", dtype=torch.float32)
    weight = torch.ones(C, device="cuda", dtype=torch.float32)
    bias = torch.zeros(C, device="cuda", dtype=torch.float32)
    rm = torch.zeros(C, device="cuda", dtype=torch.float32)
    rv = torch.ones(C, device="cuda", dtype=torch.float32)

    torch_message = ""
    try:
        torch.nn.functional.batch_norm(x, rm, rv, weight, bias, training=True)
    except ValueError as exc:
        torch_message = str(exc)
    assert torch_message, "torch accepted L=1 in training; the guard's premise is gone"

    with pytest.raises(ValueError, match=re.escape("B * prod(L) != 1")):
        BatchNormFwdOp(training=True)(x, rm, rv, weight, bias)

    # Inference applies no correction; torch normalizes the same shape.
    infer = BatchNormFwdOp(training=False)
    expected = batch_norm_fwd_ref(x, weight, bias, rm, rv, training=False, eps=infer.eps)
    compare_outputs(
        batch_norm_forward_result(infer, x, rm, rv, weight, bias),
        expected,
        batch_norm_forward_verification(x.dtype),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("shape", [(2, 8, 5, 6), (3, 4, 97, 101)], ids=["short", "long"])
def test_a_channel_length_no_tile_divides_matches_torch(shape) -> None:
    """The training forward (no statistics, no affine) and the backward cover a channel
    length no tile divides."""
    x = torch.randn(shape, dtype=torch.float16, device=run_device())
    c = shape[1]
    op = BatchNormFwdOp(training=True)
    compare_outputs(
        batch_norm_forward_result(op, x, None, None, None, None),
        batch_norm_fwd_ref(x, None, None, None, None, training=True),
        batch_norm_forward_verification(x.dtype),
    )
    grad_out, weight = torch.randn_like(x), torch.randn(c, device=run_device())
    var, mean = torch.var_mean(x.float(), dim=[0, 2, 3], correction=0)
    rstd = torch.rsqrt(var + 1e-5)
    inputs = grad_out, x, weight, mean, rstd
    compare_outputs(
        BatchNormBwdOp()(*inputs),
        batch_norm_backward(*inputs),
        batch_norm_backward_verification(x.dtype),
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, interface, n, c, spatial, dtype, key",
    [
        (BatchNormFwdOp, "batch_norm_fwd_train", 32, 64, 1, torch.float16, "fwd_train_whole"),
        (BatchNormFwdOp, "batch_norm_fwd_train", 4, 256, 784, torch.float32, "fwd_train_wide"),
        (BatchNormFwdOp, "batch_norm_fwd_train", 3, 5, 90300, torch.float16, "fwd_train_split"),
        (BatchNormFwdOp, "batch_norm_fwd_train", 4096, 1024, 64, torch.float16, "fwd_train_kernel"),
        (BatchNormBwdOp, "batch_norm_bwd", 4, 256, 784, torch.float32, "bwd_wide"),
        (BatchNormBwdOp, "batch_norm_bwd", 3, 5, 90300, torch.bfloat16, "bwd_split"),
        (BatchNormBwdOp, "batch_norm_bwd", 8192, 1024, 9, torch.float16, "bwd_kernel"),
    ],
)
def test_each_region_selects_its_one_implementation(
    op_cls, interface, n, c, spatial, dtype, key
) -> None:
    """Exactly one non-general implementation, or else the general one, serves each shape."""
    op = op_cls()
    call = BatchNormCall(arch=90, sm_count=132, n=n, c=c, spatial=spatial, dtype=dtype)
    assert op.select_implementation(interface, call) == key


# Input validation and torch.compile.
@pytest.mark.smoke
class TestBatchNormFwdValidation:
    def _make_op(self):
        from tileops.ops.norm.batch_norm import BatchNormFwdOp

        return BatchNormFwdOp()

    def _make_inputs(self, device=None, dtype=torch.float16):
        device = device or run_device()
        x = torch.randn(4, 8, 4, 4, device=device, dtype=dtype)
        weight = torch.randn(8, device=device, dtype=torch.float32)
        bias = torch.randn(8, device=device, dtype=torch.float32)
        rm = torch.zeros(8, device=device, dtype=torch.float32)
        rv = torch.ones(8, device=device, dtype=torch.float32)
        return x, weight, bias, rm, rv

    def test_rejects_wrong_dtype(self):
        from tileops.ops.norm.batch_norm import BatchNormFwdOp

        op = BatchNormFwdOp()
        x_wrong = torch.randn(4, 8, 4, 4, device=run_device(), dtype=torch.float64)
        _, weight, bias, rm, rv = self._make_inputs()
        with pytest.raises(ValueError, match="dtype"):
            op(x_wrong, rm, rv, weight, bias)

    def test_rejects_wrong_shape(self):
        op = self._make_op()
        x_wrong = torch.randn(4, 16, 4, 4, device=run_device(), dtype=torch.float16)
        _, weight, bias, rm, rv = self._make_inputs()
        with pytest.raises(ValueError, match="shape|channel|Channel"):
            op(x_wrong, rm, rv, weight, bias)


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
class TestBatchNormCustomOp:
    def test_fwd_torch_compile_smoke(self):
        from tileops.ops.norm.batch_norm import BatchNormFwdOp

        op = BatchNormFwdOp(training=True)
        x = torch.randn(4, 8, 4, 4, device=run_device(), dtype=torch.float16)
        weight = torch.randn(8, device=run_device(), dtype=torch.float32)
        bias = torch.randn(8, device=run_device(), dtype=torch.float32)
        rm = torch.zeros(8, device=run_device(), dtype=torch.float32)
        rv = torch.ones(8, device=run_device(), dtype=torch.float32)

        compiled = torch.compile(op, fullgraph=False)
        # Manifest input order: (x, running_mean, running_var, weight, bias).
        y = compiled(x, rm, rv, weight, bias)
        assert y.shape == x.shape

    def test_bwd_torch_compile_smoke(self):
        from tileops.ops.norm.batch_norm import BatchNormBwdOp

        N, C, H, W = 4, 8, 4, 4
        op = BatchNormBwdOp()
        grad_out = torch.randn(N, C, H, W, device=run_device(), dtype=torch.float16)
        x = torch.randn(N, C, H, W, device=run_device(), dtype=torch.float16)
        weight = torch.randn(C, device=run_device(), dtype=torch.float32)
        x32 = x.float()
        x_cl = x32.permute(1, 0, 2, 3).reshape(C, -1).contiguous()
        mean = x_cl.mean(dim=1)
        rstd = 1.0 / torch.sqrt(x_cl.var(dim=1, unbiased=False) + 1e-5)

        compiled = torch.compile(op, fullgraph=False)
        gx, gw, gb = compiled(grad_out, x, weight, mean, rstd)
        assert gx.shape == x.shape


def _to_cl(x: torch.Tensor) -> torch.Tensor:
    return x.permute(1, 0, *range(2, x.ndim)).reshape(x.shape[1], -1).contiguous()


def _from_cl(x_cl: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    n = orig_shape[0]
    c = orig_shape[1]
    spatial = orig_shape[2:]
    return x_cl.reshape(c, n, *spatial).permute(1, 0, *range(2, len(orig_shape))).contiguous()


class _FakeKernel(Kernel):
    general = True

    def __init__(self, call: BatchNormCall) -> None:
        super().__init__()
        self.L = call.n * call.spatial
        self.dtype = call.dtype
        self.eps = call.eps
        self.momentum = call.momentum


class _FakeBatchNormFwdInferKernel(_FakeKernel, BatchNormInferFwdInterface):
    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        x_cl = _to_cl(x)
        y = (x_cl.float() - running_mean[:, None]) * torch.rsqrt(running_var[:, None] + self.eps)
        y = y * weight[:, None] + bias[:, None]
        return _from_cl(y.to(self.dtype), x.shape)


class _FakeBatchNormFwdTrainKernel(_FakeKernel, BatchNormTrainFwdInterface):
    def forward(
        self,
        x: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_cl = _to_cl(x)
        mean = x_cl.float().mean(dim=1)
        var = x_cl.float().var(dim=1, unbiased=False)
        rstd = torch.rsqrt(var + self.eps)
        running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
        running_var.mul_(1 - self.momentum).add_(self.momentum * var)
        y = (x_cl.float() - mean[:, None]) * rstd[:, None]
        y = y * weight[:, None] + bias[:, None]
        return _from_cl(y.to(self.dtype), x.shape), mean, rstd


class _FakeBatchNormBwdKernel(_FakeKernel, BatchNormBwdInterface):
    def forward(
        self,
        grad_out: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_out_cl = _to_cl(grad_out)
        x_cl = _to_cl(x)
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
        return _from_cl(grad_x.to(self.dtype), x.shape), grad_weight, grad_bias


# ``kernel_map=`` replaces what runs under a key, never which key is selected: every
# training key runs the fake, so whichever one a shape selects serves it.
_FAKE_TRAIN_MAP = {
    "fwd_train_whole": _FakeBatchNormFwdTrainKernel,
    "fwd_train_wide": _FakeBatchNormFwdTrainKernel,
    "fwd_train_split": _FakeBatchNormFwdTrainKernel,
    "fwd_train_kernel": _FakeBatchNormFwdTrainKernel,
    "fwd_infer_kernel": _FakeBatchNormFwdInferKernel,
}


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
        running_var.reshape(affine_shape) + eps
    )
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
    n, c = x.shape[0], x.shape[1]
    call = BatchNormCall(n=n, c=c, spatial=x.numel() // (n * c), dtype=x.dtype)
    return _FakeBatchNormBwdKernel(call)(grad_out, x, weight, mean, rstd)


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_batch_norm_fwd_lazy_cache_reuse_and_respecialization() -> None:
    """BatchNorm op-layer cache reuses identical specs and caches changed specs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for forward call")

    op = BatchNormFwdOp(
        training=False,
        kernel_map=_FAKE_TRAIN_MAP,
        target=BUILTIN,
    )

    def run_case(N: int, C: int, spatial: tuple[int, ...], dtype: torch.dtype) -> None:
        x = torch.randn((N, C, *spatial), device="cuda", dtype=dtype)
        weight = torch.randn(C, device="cuda", dtype=torch.float32)
        bias = torch.randn(C, device="cuda", dtype=torch.float32)
        running_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
        running_var = torch.ones(C, device="cuda", dtype=torch.float32)

        y = op(x, running_mean, running_var, weight, bias)
        ref_y = _batch_norm_infer_ref(x, running_mean, running_var, weight, bias, op.eps)
        assert torch.equal(y.float(), ref_y.float())

    run_case(2, 8, (4, 4), torch.float16)
    assert len(list(op.iter_kernels())) == 1
    first_kernel = op.kernel
    assert op.eval_roofline() == (
        4 * 8 * 32,
        2 * 8 * 32 * torch.float16.itemsize + 4 * 8 * 4,
    )

    run_case(2, 8, (4, 4), torch.float16)
    assert len(list(op.iter_kernels())) == 1
    assert op.kernel is first_kernel

    run_case(3, 12, (2, 8), torch.bfloat16)
    assert len(list(op.iter_kernels())) == 2
    assert op.kernel is not first_kernel
    assert op.eval_roofline() == (
        4 * 12 * 48,
        2 * 12 * 48 * torch.bfloat16.itemsize + 4 * 12 * 4,
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_batch_norm_training_fwd_lazy_cache_reuse_and_respecialization() -> None:
    """Training BatchNorm forward cache path is executable under fake kernels."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for forward call")

    op = BatchNormFwdOp(
        training=True,
        kernel_map=_FAKE_TRAIN_MAP,
        target=BUILTIN,
    )

    def run_case(N: int, C: int, spatial: tuple[int, ...], dtype: torch.dtype) -> None:
        x = torch.randn((N, C, *spatial), device="cuda", dtype=dtype)
        weight = torch.randn(C, device="cuda", dtype=torch.float32)
        bias = torch.randn(C, device="cuda", dtype=torch.float32)
        running_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
        running_var = torch.ones(C, device="cuda", dtype=torch.float32)

        y = op(x, running_mean, running_var, weight, bias)
        ref_y, _, _ = _batch_norm_train_ref(x, weight, bias, op.eps)
        assert torch.equal(y.float(), ref_y.float())

    run_case(2, 8, (4, 4), torch.float16)
    assert len(list(op.iter_kernels())) == 1
    first_kernel = op.kernel
    assert op.eval_roofline() == (
        7 * 8 * 32,
        # training writes running_mean and running_var back
        2 * 8 * 32 * torch.float16.itemsize + 4 * 8 * 4 + 2 * 8 * 4,
    )

    run_case(2, 8, (4, 4), torch.float16)
    assert len(list(op.iter_kernels())) == 1
    assert op.kernel is first_kernel

    run_case(3, 12, (2, 8), torch.bfloat16)
    assert len(list(op.iter_kernels())) == 2
    assert op.kernel is not first_kernel
    assert op.eval_roofline() == (
        7 * 12 * 48,
        2 * 12 * 48 * torch.bfloat16.itemsize + 4 * 12 * 4 + 2 * 12 * 4,
    )


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_batch_norm_bwd_lazy_cache_reuse_and_respecialization() -> None:
    """BatchNorm backward cache path is executable under fake kernels."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for backward call")

    eps = 1e-5
    op = BatchNormBwdOp(
        kernel_map={
            "bwd_wide": _FakeBatchNormBwdKernel,
            "bwd_split": _FakeBatchNormBwdKernel,
            "bwd_kernel": _FakeBatchNormBwdKernel,
        },
        target=BUILTIN,
    )

    def run_case(N: int, C: int, spatial: tuple[int, ...], dtype: torch.dtype) -> None:
        x = torch.randn((N, C, *spatial), device="cuda", dtype=dtype)
        grad_out = torch.randn((N, C, *spatial), device="cuda", dtype=dtype)
        weight = torch.randn(C, device="cuda", dtype=torch.float32)
        _y, mean, rstd = _batch_norm_train_ref(
            x, torch.ones_like(weight), torch.zeros_like(weight), eps
        )

        grad_x, grad_weight, grad_bias = op(grad_out, x, weight, mean, rstd)
        ref_grad_x, ref_grad_weight, ref_grad_bias = _batch_norm_bwd_ref(
            grad_out, x, weight, mean, rstd
        )
        assert torch.equal(grad_x.float(), ref_grad_x.float())
        assert torch.equal(grad_weight, ref_grad_weight)
        assert torch.equal(grad_bias, ref_grad_bias)

    run_case(2, 8, (4, 4), torch.float16)
    assert len(list(op.iter_kernels())) == 1
    first_kernel = op.kernel
    assert op.eval_roofline() == (
        9 * 8 * 32,
        # weight, mean and rstd read; grad_weight and grad_bias written
        3 * 8 * 32 * torch.float16.itemsize + 3 * 8 * 4 + 2 * 8 * 4,
    )

    run_case(2, 8, (4, 4), torch.float16)
    assert len(list(op.iter_kernels())) == 1
    assert op.kernel is first_kernel

    run_case(3, 12, (2, 8), torch.bfloat16)
    assert len(list(op.iter_kernels())) == 2
    assert op.kernel is not first_kernel
    assert op.eval_roofline() == (
        9 * 12 * 48,
        3 * 12 * 48 * torch.bfloat16.itemsize + 3 * 12 * 4 + 2 * 12 * 4,
    )
