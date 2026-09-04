"""Contract tests for ``Op.compute_roof`` (docs/design/roofline.md §1.4)."""

import pytest
import torch

from tileops.perf.profile import tensor_core_roof

pytestmark = pytest.mark.smoke


class TestTensorCoreRoof:
    def test_maps_torch_dtypes_to_profile_keys(self):
        assert tensor_core_roof(torch.float16) == "tensor_core.fp16"
        assert tensor_core_roof(torch.bfloat16) == "tensor_core.bf16"
        assert tensor_core_roof(torch.float32) == "tensor_core.tf32"
        assert tensor_core_roof(torch.float8_e4m3fn) == "tensor_core.fp8"
        assert tensor_core_roof(torch.float8_e5m2) == "tensor_core.fp8"

    def test_unbound_or_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="no tensor-core roof"):
            tensor_core_roof(None)
        with pytest.raises(ValueError, match="no tensor-core roof"):
            tensor_core_roof(torch.int32)


class TestComputeRoofContract:
    # __new__ bypasses kernel construction so the smoke stays CUDA-free;
    # only the state each override reads is bound.

    def test_base_default_is_cuda_core_fp32(self):
        from tileops.ops.elementwise.arithmetic import AddFwdOp

        op = AddFwdOp.__new__(AddFwdOp)
        assert op.compute_roof() == "cuda_core.fp32"

    def test_matmul_op_prices_on_the_bound_dtype(self):
        from tileops.ops.gemm.gemm import GemmFwdOp

        op = GemmFwdOp.__new__(GemmFwdOp)
        op.dtype = torch.bfloat16
        assert op.compute_roof() == "tensor_core.bf16"
