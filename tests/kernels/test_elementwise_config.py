"""Elementwise default config policy: the launch config a shape and dtype select."""

from unittest.mock import patch

import pytest
import torch

from tileops.kernels.elementwise import (
    AddFwdKernel,
    BitwiseAndFwdKernel,
    EluFwdKernel,
    HardtanhFwdKernel,
    LeakyReluFwdKernel,
    PowFwdKernel,
    SiluAndMulFwdKernel,
)
from tileops.kernels.elementwise._base import MultiInputElementwiseKernel

INDEPENDENT_KERNELS_SIMPLE = [LeakyReluFwdKernel, EluFwdKernel, HardtanhFwdKernel]


@pytest.mark.cuda_only
@pytest.mark.full
@pytest.mark.parametrize(
    ("dtype", "expected_npt"),
    [
        (torch.float32, 4),
        (torch.float16, 8),
        (torch.bfloat16, 8),
    ],
)
@pytest.mark.parametrize("kernel_cls", INDEPENDENT_KERNELS_SIMPLE)
def test_independent_kernels_use_expected_default_npt(kernel_cls, dtype, expected_npt):
    """Representative independent kernels should preserve dtype-driven npt defaults."""
    # Keep enough work that grid filling does not shrink the dtype-driven width.
    wide_n = 1 << 24
    with (
        patch.object(kernel_cls, "_build_kernel", return_value=None),
        patch.object(kernel_cls, "init_config"),
    ):
        kernel = kernel_cls(wide_n, dtype)
    assert kernel.default_config["num_per_thread"] == expected_npt
    assert kernel.default_config["threads"] == 256


@pytest.mark.full
@pytest.mark.parametrize(
    ("kernel_cls", "dtype", "expected_npt"),
    [
        (AddFwdKernel, torch.float32, 8),
        (BitwiseAndFwdKernel, torch.int64, 4),
        (PowFwdKernel, torch.float32, 4),
    ],
)
def test_same_shape_binary_default_npt(kernel_cls, dtype, expected_npt):
    """Same-shape binary threads carry eight elements, at most two vectors; heavy bodies keep one."""
    # Keep enough work that grid filling does not shrink the dtype-driven width.
    wide_n = 1 << 24
    with (
        patch.object(kernel_cls, "_build_kernel", return_value=None),
        patch.object(kernel_cls, "init_config"),
    ):
        kernel = kernel_cls((wide_n,), (wide_n,), dtype)
    cfg = kernel.default_config
    assert (cfg["strategy"], cfg["num_per_thread"]) == ("register_copy", expected_npt)


@pytest.mark.full
@pytest.mark.parametrize(
    ("dtype", "expected_npt"),
    [
        (torch.int8, 16),
        (torch.float16, 8),
        (torch.float32, 4),
        (torch.int64, 4),
    ],
)
def test_multi_input_kernels_take_the_shared_launch_config(dtype, expected_npt):
    """A several-input thread takes one 16-byte vector, floored at four elements.

    They all stage their per-element inputs the way ``register_copy`` does, so none of
    them states a thread count of its own.
    """
    # Keep enough work that grid filling does not shrink the dtype-driven width.
    wide_n = 1 << 24
    subclasses = MultiInputElementwiseKernel.__subclasses__()
    assert subclasses, "no several-input kernel was imported"
    for kernel_cls in subclasses:
        if kernel_cls.SUPPORTED_DTYPES is not None and dtype not in kernel_cls.SUPPORTED_DTYPES:
            continue
        kernel = kernel_cls.__new__(kernel_cls)
        kernel.dtype = dtype
        kernel.output_dtype = dtype
        kernel.N_total = wide_n
        assert kernel.default_config == {"threads": 128, "num_per_thread": expected_npt}, (
            kernel_cls.__name__
        )


@pytest.mark.cuda_only
@pytest.mark.full
def test_fused_gated_explicit_config_follows_the_work():
    """Fused-gated explicit_parallel sizes its block from the work, not the dtype.

    A row that fills the device keeps the widest thread its dtype allows; one
    that does not gives width back until the grid reaches the device, and silu
    stops at two.
    """
    with (
        patch.object(SiluAndMulFwdKernel, "_build_kernel", return_value=None),
        patch.object(SiluAndMulFwdKernel, "init_config"),
    ):
        wide_fp16 = SiluAndMulFwdKernel(
            M=4096,
            N=14336,
            dtype=torch.float16,
            config={"strategy": "explicit_parallel"},
        )
        wide_bf16 = SiluAndMulFwdKernel(
            M=4096,
            N=14336,
            dtype=torch.bfloat16,
            config={"strategy": "explicit_parallel"},
        )
        decode_bf16 = SiluAndMulFwdKernel(
            M=1,
            N=14336,
            dtype=torch.bfloat16,
            config={"strategy": "explicit_parallel"},
        )
        wide_fp32 = SiluAndMulFwdKernel(
            M=4096,
            N=14336,
            dtype=torch.float32,
            config={"strategy": "explicit_parallel"},
        )
    assert wide_fp16.default_config == {
        "strategy": "explicit_parallel",
        "threads": 128,
        "num_per_thread": 8,
    }
    assert wide_bf16.default_config == {
        "strategy": "explicit_parallel",
        "threads": 128,
        "num_per_thread": 8,
    }
    assert decode_bf16.default_config == {
        "strategy": "explicit_parallel",
        "threads": 128,
        "num_per_thread": 2,
    }
    # float32 takes the shared thread count, not the 256 this family stated.
    assert wide_fp32.default_config == {
        "strategy": "explicit_parallel",
        "threads": 128,
        "num_per_thread": 4,
    }
    # A tuned kernel must be able to land back on its shipped config.
    for kernel in (wide_fp16, wide_bf16, decode_bf16, wide_fp32):
        cfg = kernel.default_config
        assert any(
            c["num_per_thread"] == cfg["num_per_thread"] and c["threads"] == cfg["threads"]
            for c in kernel.autotune_configs
        )


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_fused_gated_kernel_rejects_unknown_strategy() -> None:
    """FusedGatedKernel must reject unknown strategy names."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        SiluAndMulFwdKernel(
            M=16,
            N=16,
            dtype=torch.float16,
            config={"strategy": "nonexistent"},
        )
