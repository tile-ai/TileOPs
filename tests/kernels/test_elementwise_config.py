"""Elementwise kernel config policy: an explicit config, and the default a shape and dtype pick.

The Op takes no config and builds on its first call, so a config it did not choose, and the
default for a shape too large to allocate cheaply, are reachable only on the kernel.
"""

from unittest.mock import patch

import pytest
import torch

from tileops.kernels.elementwise import (
    AddFwdKernel,
    BitwiseAndFwdKernel,
    EluFwdKernel,
    FloorDivideFwdKernel,
    HardtanhFwdKernel,
    LeakyReluFwdKernel,
    PowFwdKernel,
    RemainderFwdKernel,
    SiluAndMulFwdKernel,
)
from tileops.kernels.elementwise._base import MultiInputElementwiseKernel

# Regression: a parametric kernel's block extent has to follow the config it is given


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    ("threads", "npt"),
    [(128, 4), (256, 4), (512, 8), (1024, 8)],
)
def test_parametric_unary_honours_a_non_default_config(threads: int, npt: int) -> None:
    """Every element is written whatever ``threads * npt`` the config asks for.

    The builder is handed the default config and the JIT the actual one, so a block
    extent taken from the builder's arguments leaves part of each block untouched.
    """
    n = 4096 * 7 + 13
    x = torch.randn(n, device="cuda", dtype=torch.float16)
    kernel = LeakyReluFwdKernel(
        n, torch.float16, 0.01, config={"threads": threads, "num_per_thread": npt}
    )
    torch.testing.assert_close(
        kernel.forward(x), torch.nn.functional.leaky_relu(x, 0.01), rtol=1e-3, atol=1e-3
    )


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
    stops at two. The direct strategy keeps 256 threads: one element a thread
    makes the thread count the whole block.
    """
    with (
        patch.object(SiluAndMulFwdKernel, "_build_kernel", return_value=None),
        patch.object(SiluAndMulFwdKernel, "init_config"),
    ):
        direct = SiluAndMulFwdKernel(
            M=32,
            N=1024,
            dtype=torch.float16,
            config={"strategy": "direct"},
        )
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
    assert direct.default_config["num_per_thread"] == 8
    assert direct.default_config["threads"] == 256
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
@pytest.mark.parametrize(
    ("kernel_cls", "ref_fn"),
    [
        pytest.param(RemainderFwdKernel, torch.remainder, id="remainder"),
        pytest.param(FloorDivideFwdKernel, torch.floor_divide, id="floor_divide"),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "npt"),
    [
        # The top of each dtype's sweep: twice the elements its bytes a thread give.
        pytest.param(torch.float16, 16, id="float16"),
        pytest.param(torch.bfloat16, 16, id="bfloat16"),
        pytest.param(torch.float32, 8, id="float32"),
    ],
)
def test_floored_kernels_build_at_the_widest_tuned_fold(kernel_cls, ref_fn, dtype, npt) -> None:
    """The floored bodies build and stay exact where a thread holds more than one vector."""
    threads = 128
    n = threads * npt
    a = torch.rand(n, device="cuda", dtype=dtype) + 0.5
    b = torch.rand(n, device="cuda", dtype=dtype) + 0.5
    kernel = kernel_cls(a.shape, b.shape, dtype, config={"threads": threads, "num_per_thread": npt})
    torch.testing.assert_close(kernel.forward(a, b), ref_fn(a, b), atol=0.0, rtol=0.0)
