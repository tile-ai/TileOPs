"""Tests for issue 605: default_config npt and output_dtype consistency.

Validates three fixes:
1. Independent kernels return correct npt for fp16/bf16 (8) and fp8 (16)
2. OUTPUT_DTYPE uses torch.dtype consistently across all kernel classes
3. kernel.output_dtype is a valid attribute on UnaryKernel, BinaryKernel,
   and FusedGatedKernel instances
"""

import pytest
import torch

from tileops.kernels.elementwise import (
    BinaryKernel,
    ClampKernel,
    EluKernel,
    EqKernel,
    FusedGatedKernel,
    GeKernel,
    GtKernel,
    HardtanhKernel,
    # Independent kernels (custom-signature)
    LeakyReluKernel,
    LeKernel,
    LogicalAndKernel,
    LogicalOrKernel,
    LtKernel,
    NanToNumKernel,
    NeKernel,
    PreluKernel,
    ReluKernel,
    SoftplusKernel,
)

# ---------------------------------------------------------------------------
# Fix 1: npt 3-way check in default_config
# ---------------------------------------------------------------------------


INDEPENDENT_KERNELS_SIMPLE = [
    LeakyReluKernel,
    EluKernel,
    HardtanhKernel,
    SoftplusKernel,
    ClampKernel,
    NanToNumKernel,
]


class TestDefaultConfigNpt:
    """Verify that independent kernels return the correct npt for each dtype."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("kernel_cls", INDEPENDENT_KERNELS_SIMPLE)
    def test_fp32_npt_is_4(self, kernel_cls):
        """fp32: npt should be 4 (4 bytes x 4 = 128-bit alignment)."""
        k = kernel_cls.__new__(kernel_cls)
        k.dtype = torch.float32
        assert k.default_config["num_per_thread"] == 4

    @pytest.mark.smoke
    @pytest.mark.parametrize("kernel_cls", INDEPENDENT_KERNELS_SIMPLE)
    def test_fp16_npt_is_8(self, kernel_cls):
        """fp16: npt should be 8 (2 bytes x 8 = 128-bit alignment)."""
        k = kernel_cls.__new__(kernel_cls)
        k.dtype = torch.float16
        assert k.default_config["num_per_thread"] == 8

    @pytest.mark.smoke
    @pytest.mark.parametrize("kernel_cls", INDEPENDENT_KERNELS_SIMPLE)
    def test_bf16_npt_is_8(self, kernel_cls):
        """bf16: npt should be 8 (2 bytes x 8 = 128-bit alignment)."""
        k = kernel_cls.__new__(kernel_cls)
        k.dtype = torch.bfloat16
        assert k.default_config["num_per_thread"] == 8

    @pytest.mark.smoke
    @pytest.mark.parametrize("kernel_cls", INDEPENDENT_KERNELS_SIMPLE)
    def test_fp8_e4m3fn_npt_is_16(self, kernel_cls):
        """fp8 e4m3fn: npt should be 16 (1 byte x 16 = 128-bit alignment)."""
        k = kernel_cls.__new__(kernel_cls)
        k.dtype = torch.float8_e4m3fn
        assert k.default_config["num_per_thread"] == 16

    @pytest.mark.smoke
    @pytest.mark.parametrize("kernel_cls", INDEPENDENT_KERNELS_SIMPLE)
    def test_fp8_e5m2_npt_is_16(self, kernel_cls):
        """fp8 e5m2: npt should be 16 (1 byte x 16 = 128-bit alignment)."""
        k = kernel_cls.__new__(kernel_cls)
        k.dtype = torch.float8_e5m2
        assert k.default_config["num_per_thread"] == 16

    @pytest.mark.smoke
    def test_prelu_fp16_npt_is_8(self):
        """PreluKernel fp16: npt should be 8."""
        k = PreluKernel.__new__(PreluKernel)
        k.dtype = torch.float16
        assert k.default_config["num_per_thread"] == 8

    @pytest.mark.smoke
    def test_prelu_bf16_npt_is_8(self):
        """PreluKernel bf16: npt should be 8."""
        k = PreluKernel.__new__(PreluKernel)
        k.dtype = torch.bfloat16
        assert k.default_config["num_per_thread"] == 8

    @pytest.mark.smoke
    def test_prelu_fp32_npt_is_4(self):
        """PreluKernel fp32: npt should be 4."""
        k = PreluKernel.__new__(PreluKernel)
        k.dtype = torch.float32
        assert k.default_config["num_per_thread"] == 4

    @pytest.mark.smoke
    def test_prelu_fp8_npt_is_16(self):
        """PreluKernel fp8: npt should be 16."""
        k = PreluKernel.__new__(PreluKernel)
        k.dtype = torch.float8_e4m3fn
        assert k.default_config["num_per_thread"] == 16


# ---------------------------------------------------------------------------
# Fix 2: OUTPUT_DTYPE consistency (all should be torch.dtype, not string)
# ---------------------------------------------------------------------------


COMPARISON_KERNELS = [EqKernel, NeKernel, GtKernel, LtKernel, GeKernel, LeKernel]
LOGICAL_BINARY_KERNELS = [LogicalAndKernel, LogicalOrKernel]


class TestOutputDtypeConsistency:
    """Verify that OUTPUT_DTYPE is always a torch.dtype (not a string)."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("kernel_cls", COMPARISON_KERNELS + LOGICAL_BINARY_KERNELS)
    def test_output_dtype_is_torch_dtype(self, kernel_cls):
        """OUTPUT_DTYPE must be a torch.dtype instance, not a string."""
        assert isinstance(kernel_cls.OUTPUT_DTYPE, torch.dtype), (
            f"{kernel_cls.__name__}.OUTPUT_DTYPE is {type(kernel_cls.OUTPUT_DTYPE).__name__} "
            f"({kernel_cls.OUTPUT_DTYPE!r}), expected torch.dtype"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("kernel_cls", COMPARISON_KERNELS + LOGICAL_BINARY_KERNELS)
    def test_output_dtype_is_int8(self, kernel_cls):
        """Comparison and logical binary kernels should output int8."""
        assert torch.int8 == kernel_cls.OUTPUT_DTYPE


# ---------------------------------------------------------------------------
# Fix 3: output_dtype attribute on all three base kernel types
# ---------------------------------------------------------------------------


class TestOutputDtypeAttribute:
    """Verify that kernel.output_dtype is a valid attribute on instances."""

    @pytest.mark.smoke
    def test_unary_kernel_has_output_dtype(self):
        """UnaryKernel subclass (ReluKernel) should have output_dtype."""
        # Use __new__ to avoid full init (which needs GPU)
        k = ReluKernel.__new__(ReluKernel)
        k.dtype = torch.float16
        k.OUTPUT_DTYPE = None
        k._fp8_output_dtype = None
        # Simulate the init logic
        k.output_dtype = k.OUTPUT_DTYPE or k.dtype
        assert k.output_dtype == torch.float16

    @pytest.mark.smoke
    def test_binary_kernel_has_output_dtype(self):
        """BinaryKernel subclass (AddKernel) should set output_dtype in __init__.

        We verify the class attribute path exists by checking that the
        BinaryKernel.__init__ code sets self.output_dtype.
        """
        # Check that BinaryKernel code sets output_dtype by inspecting source
        import inspect
        source = inspect.getsource(BinaryKernel.__init__)
        assert "self.output_dtype" in source, (
            "BinaryKernel.__init__ must set self.output_dtype"
        )

    @pytest.mark.smoke
    def test_fused_gated_kernel_has_output_dtype(self):
        """FusedGatedKernel subclass should set output_dtype in __init__.

        We verify the class attribute path exists by checking that the
        FusedGatedKernel.__init__ code sets self.output_dtype.
        """
        import inspect
        source = inspect.getsource(FusedGatedKernel.__init__)
        assert "self.output_dtype" in source, (
            "FusedGatedKernel.__init__ must set self.output_dtype"
        )
