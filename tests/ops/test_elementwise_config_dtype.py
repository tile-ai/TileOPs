"""Regression tests for elementwise dtype/config consistency."""

from unittest.mock import patch

import pytest
import torch

from tileops.kernels.elementwise import (
    AddKernel,
    EluKernel,
    EqKernel,
    GeKernel,
    GtKernel,
    HardtanhKernel,
    # Independent kernels (custom-signature)
    LeakyReluKernel,
    LeKernel,
    LogicalAndKernel,
    LogicalOrKernel,
    LtKernel,
    NeKernel,
    PreluKernel,
    ReluKernel,
    SiluAndMulKernel,
)

# ---------------------------------------------------------------------------
# Fix 1: npt 3-way check in default_config
# ---------------------------------------------------------------------------


INDEPENDENT_KERNELS_SIMPLE = [LeakyReluKernel, EluKernel, HardtanhKernel]


@pytest.mark.full
@pytest.mark.parametrize(
    ("dtype", "expected_npt"),
    [
        (torch.float32, 4),
        (torch.float16, 8),
        (torch.bfloat16, 8),
        (torch.float8_e4m3fn, 16),
        (torch.float8_e5m2, 16),
    ],
)
@pytest.mark.parametrize("kernel_cls", INDEPENDENT_KERNELS_SIMPLE)
def test_independent_kernels_use_expected_default_npt(kernel_cls, dtype, expected_npt):
    """Representative independent kernels should preserve dtype-driven npt defaults."""
    kernel = kernel_cls.__new__(kernel_cls)
    kernel.dtype = dtype
    assert kernel.default_config["num_per_thread"] == expected_npt


@pytest.mark.full
@pytest.mark.parametrize(
    ("dtype", "expected_npt"),
    [
        (torch.float32, 4),
        (torch.float16, 8),
        (torch.bfloat16, 8),
        (torch.float8_e4m3fn, 16),
        (torch.float8_e5m2, 16),
    ],
)
def test_prelu_preserves_dtype_driven_default_npt(dtype, expected_npt):
    """Prelu is the custom-signature outlier and should keep the same dtype mapping."""
    kernel = PreluKernel.__new__(PreluKernel)
    kernel.dtype = dtype
    assert kernel.default_config["num_per_thread"] == expected_npt


# ---------------------------------------------------------------------------
# Fix 2: OUTPUT_DTYPE consistency (all should be torch.dtype, not string)
# ---------------------------------------------------------------------------


COMPARISON_KERNELS = [EqKernel, NeKernel, GtKernel, LtKernel, GeKernel, LeKernel]
LOGICAL_BINARY_KERNELS = [LogicalAndKernel, LogicalOrKernel]


@pytest.mark.full
@pytest.mark.parametrize("kernel_cls", COMPARISON_KERNELS + LOGICAL_BINARY_KERNELS)
def test_bool_like_elementwise_kernels_expose_torch_dtype_output(kernel_cls):
    """Comparison and logical kernels should declare `OUTPUT_DTYPE` as `torch.int8`."""
    assert isinstance(kernel_cls.OUTPUT_DTYPE, torch.dtype), (
        f"{kernel_cls.__name__}.OUTPUT_DTYPE is {type(kernel_cls.OUTPUT_DTYPE).__name__} "
        f"({kernel_cls.OUTPUT_DTYPE!r}), expected torch.dtype"
    )
    assert torch.int8 == kernel_cls.OUTPUT_DTYPE


# ---------------------------------------------------------------------------
# Fix 3: output_dtype attribute on all three base kernel types
# ---------------------------------------------------------------------------


@pytest.mark.full
def test_unary_kernel_sets_output_dtype_in_init():
    """Unary kernels should initialize `output_dtype` during construction."""
    with (
        patch.object(ReluKernel, "_build_kernel", return_value=None),
        patch.object(ReluKernel, "init_config"),
    ):
        kernel = ReluKernel(N_total=1024, dtype=torch.float16)
    assert kernel.output_dtype == torch.float16


@pytest.mark.full
def test_binary_kernel_sets_output_dtype_in_init():
    """Binary kernels should initialize `output_dtype` during construction."""
    with (
        patch.object(AddKernel, "_build_kernel", return_value=None),
        patch.object(AddKernel, "init_config"),
    ):
        kernel = AddKernel(
            N_total=1024, dtype=torch.float16,
            coalesced_shape=(1024,), a_strides=(1,), b_strides=(1,),
            a_numel=1024, b_numel=1024,
        )
    assert kernel.output_dtype == torch.float16


@pytest.mark.full
def test_fused_gated_kernel_sets_output_dtype_in_init():
    """Fused-gated kernels should initialize `output_dtype` during construction."""
    with (
        patch.object(SiluAndMulKernel, "_build_kernel", return_value=None),
        patch.object(SiluAndMulKernel, "init_config"),
    ):
        kernel = SiluAndMulKernel(M=32, N=1024, dtype=torch.float16)
    assert kernel.output_dtype == torch.float16
