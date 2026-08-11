"""Contract tests for the native framework performance baselines."""

import pytest
import torch

from benchmarks.ops.bench_batch_norm import _torch_bn_bwd, _torch_bn_fwd
from benchmarks.ops.bench_cumulative import _torch_cumprod, _torch_cumsum
from benchmarks.ops.bench_norm import _torch_rms_norm
from benchmarks.ops.bench_reduce import (
    _torch_mean,
    _torch_prod,
    _torch_std,
    _torch_sum,
    _torch_var,
    _torch_var_mean,
)
from benchmarks.ops.bench_vector_norm import _torch_vector_norm
from workloads.normalization import (
    BatchNormBwdWorkload,
    batch_norm_fwd_ref,
)


def _assert_contract_output(actual, expected, dtype):
    assert actual.dtype == dtype
    assert actual.shape == expected.shape
    assert torch.allclose(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduction_baselines_preserve_contract_dtype(dtype):
    x = torch.randn(8, 64, device="cuda", dtype=dtype)
    dim = -1
    keepdim = True

    cases = [
        (_torch_sum(x, dim=dim, keepdim=keepdim), x.float().sum(dim, keepdim=True)),
        (_torch_mean(x, dim=dim, keepdim=keepdim), x.float().mean(dim, keepdim=True)),
        (_torch_std(x, dim=dim, keepdim=keepdim),
         x.float().std(dim, keepdim=True, correction=1)),
        (_torch_var(x, dim=dim, keepdim=keepdim),
         x.float().var(dim, keepdim=True, correction=1)),
    ]
    for actual, expected in cases:
        _assert_contract_output(actual, expected.to(dtype), dtype)

    actual_var, actual_mean = _torch_var_mean(x, dim=dim, keepdim=keepdim)
    expected_var, expected_mean = torch.var_mean(
        x.float(), dim=dim, keepdim=keepdim, correction=1,
    )
    _assert_contract_output(actual_var, expected_var.to(dtype), dtype)
    _assert_contract_output(actual_mean, expected_mean.to(dtype), dtype)

    prod_input = torch.rand(8, 64, device="cuda", dtype=dtype) * 0.01 + 0.99
    _assert_contract_output(
        _torch_prod(prod_input),
        prod_input.float().prod(dim=-1).to(dtype),
        dtype,
    )


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_norm_and_cumulative_baselines_preserve_contract_dtype(dtype):
    x = torch.randn(8, 64, device="cuda", dtype=dtype)
    weight = torch.randn(64, device="cuda", dtype=dtype)

    for order in (1, 2, float("inf")):
        actual = _torch_vector_norm(x, ord=order, dim=-1, keepdim=False)
        expected = torch.linalg.vector_norm(x.float(), ord=order, dim=-1).to(dtype)
        _assert_contract_output(actual, expected, dtype)

    actual_rms = _torch_rms_norm(x, weight, normalized_shape=(64,), eps=1e-5)
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    expected_rms = (x.float() / rms * weight.float()).to(dtype)
    _assert_contract_output(actual_rms, expected_rms, dtype)

    cumulative_input = torch.randn(8, 64, device="cuda", dtype=dtype) * 0.01
    _assert_contract_output(
        _torch_cumsum(cumulative_input),
        cumulative_input.float().cumsum(dim=-1).to(dtype),
        dtype,
    )

    product_input = cumulative_input + 0.99
    _assert_contract_output(
        _torch_cumprod(product_input),
        product_input.float().cumprod(dim=-1).to(dtype),
        dtype,
    )


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batch_norm_forward_uses_contract_inputs_and_updates_state(dtype):
    n, c, spatial = 4, 8, (4, 4)
    x = torch.randn(n, c, *spatial, device="cuda", dtype=dtype)
    weight = torch.randn(c, device="cuda", dtype=torch.float32)
    bias = torch.randn(c, device="cuda", dtype=torch.float32)
    running_mean = torch.zeros(c, device="cuda", dtype=torch.float32)
    running_var = torch.ones(c, device="cuda", dtype=torch.float32)
    ref_mean = running_mean.clone()
    ref_var = running_var.clone()

    actual = _torch_bn_fwd(
        x, running_mean, running_var, weight, bias, training=True,
    )
    expected, expected_mean, expected_var = batch_norm_fwd_ref(
        x, weight, bias, ref_mean, ref_var, training=True,
    )

    _assert_contract_output(actual, expected, dtype)
    assert torch.allclose(running_mean, expected_mean, atol=2e-2, rtol=2e-2)
    assert torch.allclose(running_var, expected_var, atol=2e-2, rtol=2e-2)


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batch_norm_backward_consumes_saved_statistics(dtype):
    workload = BatchNormBwdWorkload(4, 8, (4, 4), dtype)
    inputs = workload.gen_inputs()

    actual = _torch_bn_bwd(*inputs)
    expected = workload.ref_program(*inputs)

    assert len(actual) == len(expected) == 3
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.dtype == expected_tensor.dtype
        assert torch.allclose(
            actual_tensor.float(), expected_tensor.float(), atol=2e-2, rtol=2e-2,
        )
