import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.quantization import (
    DequantizeCall,
    INT8DequantPerBlockFwdKernel,
    INT8DequantPerChannelFwdKernel,
    INT8DequantPerTensorFwdKernel,
)
from tileops.quantization import (
    INT8DequantPerBlockFwdOp,
    INT8DequantPerChannelFwdOp,
    INT8DequantPerTensorFwdOp,
)
from workloads.device import run_device
from workloads.quantization.int8_dequant import (
    INT8DequantPerBlockWorkload,
    INT8DequantPerChannelWorkload,
    INT8DequantPerTensorWorkload,
)


class INT8DequantPerTensorTest(INT8DequantPerTensorWorkload, TestBase):
    pass


class INT8DequantPerChannelTest(INT8DequantPerChannelWorkload, TestBase):
    pass


class INT8DequantPerBlockTest(INT8DequantPerBlockWorkload, TestBase):
    pass


_TESTS = {
    INT8DequantPerTensorFwdOp: INT8DequantPerTensorTest,
    INT8DequantPerChannelFwdOp: INT8DequantPerChannelTest,
    INT8DequantPerBlockFwdOp: INT8DequantPerBlockTest,
}


class INT8DequantFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        # One typical shape per op; the per-block K is not a multiple of 128, so the last block of
        # each row is partial.
        shapes = {
            INT8DequantPerTensorFwdOp: (1024, 1024),
            INT8DequantPerChannelFwdOp: (256, 1024),
            INT8DequantPerBlockFwdOp: (256, 1000),
        }
        return [
            (
                "op_cls, m, k, out_dtype",
                [
                    pytest.param(op_cls, *shape, dtype, marks=pytest.mark.smoke)
                    for op_cls, shape in shapes.items()
                    for dtype in (torch.float16, torch.bfloat16, torch.float32)
                ]
                # Per-channel rows shorter than a thread's eight codes, over whole blocks: no
                # block takes the vector path.
                + [
                    pytest.param(
                        INT8DequantPerChannelFwdOp, 512, 5, torch.bfloat16, marks=pytest.mark.smoke
                    ),
                    # Per-block rows shorter than a vector: every block converts code by code.
                    pytest.param(
                        INT8DequantPerBlockFwdOp, 512, 5, torch.bfloat16, marks=pytest.mark.smoke
                    ),
                    # Per-block, K % 128 = 3: a vector crosses a row's short last group and the
                    # row end, so its codes take three scales.
                    pytest.param(
                        INT8DequantPerBlockFwdOp, 16, 131, torch.bfloat16, marks=pytest.mark.smoke
                    ),
                    # Per-tensor, past the small-matrix kernel's region, with a tail whose length
                    # is not a multiple of any vector width.
                    pytest.param(
                        INT8DequantPerTensorFwdOp,
                        2051,
                        1025,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                    # Per-block, past the small-matrix kernel's region, float32 with K % 128 = 1:
                    # the staged kernel with three scales per vector and a tail.
                    pytest.param(
                        INT8DequantPerBlockFwdOp, 4100, 129, torch.float32, marks=pytest.mark.full
                    ),
                ],
            ),
        ]


@INT8DequantFixture
def test_int8_dequant_op(op_cls: type, m: int, k: int, out_dtype: torch.dtype) -> None:
    test = _TESTS[op_cls](m, k, out_dtype)
    # One float32 multiply and one cast: a conforming kernel is bit-exact.
    test.check(op_cls(out_dtype), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, scale_shape",
    [
        (INT8DequantPerTensorFwdOp, (2,)),
        (INT8DequantPerChannelFwdOp, (1,)),
        (INT8DequantPerBlockFwdOp, (64, 2)),
    ],
)
def test_int8_dequant_rejects_wrong_scale_shape(op_cls: type, scale_shape: tuple) -> None:
    q = torch.zeros(64, 1000, dtype=torch.int8, device=run_device())
    scale = torch.ones(scale_shape, dtype=torch.float32, device=run_device())
    with pytest.raises(ValueError, match="scale"):
        op_cls(torch.bfloat16)(q, scale)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls", [INT8DequantPerChannelFwdOp, INT8DequantPerTensorFwdOp, INT8DequantPerBlockFwdOp]
)
def test_int8_dequant_misaligned_input(op_cls: type) -> None:
    """A ``q`` whose storage is off the vector boundary, rows that split a thread's codes, a tail."""
    test = _TESTS[op_cls](5, 1001, torch.bfloat16)
    q, scale = test.gen_inputs()
    q = torch.cat([q.new_zeros(1, 1001), q])[1:]
    assert q.is_contiguous() and q.data_ptr() % 16
    test.check(op_cls(torch.bfloat16), q, scale)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "kernel_cls",
    [INT8DequantPerChannelFwdKernel, INT8DequantPerTensorFwdKernel, INT8DequantPerBlockFwdKernel],
)
def test_int8_dequant_refuses_a_last_block_past_int32(kernel_cls: type) -> None:
    """The last block indexes up to one block past ``M * K``, so ``M * K = 2^31 - 1`` is refused."""
    call = DequantizeCall(m=1, k=2**31 - 1, out_dtype=torch.bfloat16, device=run_device())
    assert "int32" in (kernel_cls.refusal(call) or "")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, m, key",
    [
        (INT8DequantPerTensorFwdOp, 127, "int8_dequant_per_tensor_small"),
        (INT8DequantPerTensorFwdOp, 128, "int8_dequant_per_tensor"),
        (INT8DequantPerChannelFwdOp, 1, "int8_dequant_per_channel"),
        (INT8DequantPerBlockFwdOp, 127, "int8_dequant_per_block_small"),
        (INT8DequantPerBlockFwdOp, 128, "int8_dequant_per_block"),
    ],
)
def test_each_region_selects_its_one_implementation(op_cls: type, m: int, key: str) -> None:
    """Below 2^19 codes the small-matrix kernel serves a call, the general one from there."""
    call = DequantizeCall(arch=90, sm_count=132, m=m, k=4096, out_dtype=torch.bfloat16)
    assert op_cls(torch.bfloat16).select_implementation("dequant", call) == key
