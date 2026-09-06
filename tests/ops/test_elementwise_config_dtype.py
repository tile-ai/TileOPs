"""Regression tests for elementwise dtype/config consistency."""

import inspect
from unittest.mock import patch

import pytest
import torch

from tileops.kernels.elementwise import (
    AddFwdKernel,
    EluFwdKernel,
    EqFwdKernel,
    GeFwdKernel,
    GtFwdKernel,
    HardtanhFwdKernel,
    # Independent kernels (custom-signature)
    LeakyReluFwdKernel,
    LeFwdKernel,
    LogicalAndFwdKernel,
    LogicalOrFwdKernel,
    LtFwdKernel,
    NeFwdKernel,
    PreluFwdKernel,
    ReluFwdKernel,
    SiluAndMulFwdKernel,
)

# Regression: a parametric kernel's block extent has to follow the config it is given


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


# Fix 1: npt 3-way check in default_config


INDEPENDENT_KERNELS_SIMPLE = [LeakyReluFwdKernel, EluFwdKernel, HardtanhFwdKernel]


# Big enough that the grid-filling shrink leaves the dtype-driven width alone.
_WIDE_N = 1 << 24


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
    with (
        patch.object(kernel_cls, "_build_kernel", return_value=None),
        patch.object(kernel_cls, "init_config"),
    ):
        kernel = kernel_cls(_WIDE_N, dtype)
    assert kernel.default_config["num_per_thread"] == expected_npt
    assert kernel.default_config["threads"] == 256


@pytest.mark.full
@pytest.mark.parametrize(
    ("dtype", "expected_npt"),
    [
        (torch.float32, 4),
        (torch.float16, 8),
        (torch.bfloat16, 8),
    ],
)
def test_prelu_preserves_dtype_driven_default_npt(dtype, expected_npt):
    """Prelu is the custom-signature outlier and should keep the same dtype mapping."""
    kernel = PreluFwdKernel.__new__(PreluFwdKernel)
    kernel.dtype = dtype
    assert kernel.default_config["num_per_thread"] == expected_npt


# Fix 2: OUTPUT_DTYPE consistency (all should be torch.dtype, not string)


COMPARISON_KERNELS = [EqFwdKernel, NeFwdKernel, GtFwdKernel, LtFwdKernel, GeFwdKernel, LeFwdKernel]
LOGICAL_BINARY_KERNELS = [LogicalAndFwdKernel, LogicalOrFwdKernel]


@pytest.mark.full
@pytest.mark.parametrize("kernel_cls", COMPARISON_KERNELS + LOGICAL_BINARY_KERNELS)
def test_bool_like_elementwise_kernels_expose_torch_dtype_output(kernel_cls):
    """Comparison and logical kernels should declare public bool output."""
    assert isinstance(kernel_cls.OUTPUT_DTYPE, torch.dtype), (
        f"{kernel_cls.__name__}.OUTPUT_DTYPE is {type(kernel_cls.OUTPUT_DTYPE).__name__} "
        f"({kernel_cls.OUTPUT_DTYPE!r}), expected torch.dtype"
    )
    # Bool-storage specializations may use uint8 internally, but the public
    # kernel contract is a torch.bool output dtype.
    assert torch.bool == kernel_cls.OUTPUT_DTYPE


# Fix 3: output_dtype attribute on all three base kernel types


@pytest.mark.full
def test_unary_kernel_sets_output_dtype_in_init():
    """Unary kernels should initialize `output_dtype` during construction."""
    with (
        patch.object(ReluFwdKernel, "_build_kernel", return_value=None),
        patch.object(ReluFwdKernel, "init_config"),
    ):
        kernel = ReluFwdKernel(N_total=1024, dtype=torch.float16)
    assert kernel.output_dtype == torch.float16


@pytest.mark.full
def test_binary_kernel_sets_output_dtype_in_init():
    """Binary kernels should initialize `output_dtype` during construction."""
    with (
        patch.object(AddFwdKernel, "_build_kernel", return_value=None),
        patch.object(AddFwdKernel, "init_config"),
    ):
        kernel = AddFwdKernel(a_shape=(1024,), b_shape=(1024,), dtype=torch.float16)
    assert kernel.output_dtype == torch.float16


@pytest.mark.full
def test_fused_gated_kernel_sets_output_dtype_in_init():
    """Fused-gated kernels should initialize `output_dtype` during construction."""
    with (
        patch.object(SiluAndMulFwdKernel, "_build_kernel", return_value=None),
        patch.object(SiluAndMulFwdKernel, "init_config"),
    ):
        kernel = SiluAndMulFwdKernel(M=32, N=1024, dtype=torch.float16)
    assert kernel.output_dtype == torch.float16


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


# Strategy lives in the kernel config dict, not in op/kernel ctor kwargs


@pytest.mark.smoke
def test_elementwise_ops_do_not_expose_strategy_kwarg():
    """No elementwise Op constructor exposes a ``strategy`` parameter.

    Strategy selection is a kernel-config concern (``config={"strategy": ...}``
    on the kernel); the op layer must not re-expose it as ctor plumbing.
    """
    import tileops.ops.elementwise as ew
    from tileops.ops.op_base import Op

    checked = 0
    for name in dir(ew):
        obj = getattr(ew, name)
        if not (isinstance(obj, type) and issubclass(obj, Op)):
            continue
        params = inspect.signature(obj.__init__).parameters
        assert "strategy" not in params, f"{name}.__init__ still exposes a strategy kwarg"
        checked += 1
    assert checked > 0, "Test bug: no Op classes discovered"


@pytest.mark.smoke
def test_elementwise_kernels_do_not_expose_strategy_kwarg():
    """Kernel ctors take strategy via config, not as a dedicated kwarg."""
    for kernel_cls in (ReluFwdKernel, AddFwdKernel, SiluAndMulFwdKernel):
        params = inspect.signature(kernel_cls.__init__).parameters
        assert "strategy" not in params, (
            f"{kernel_cls.__name__}.__init__ still exposes a strategy kwarg"
        )
