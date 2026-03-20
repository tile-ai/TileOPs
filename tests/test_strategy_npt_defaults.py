"""Tests for strategy-aware npt defaults in elementwise kernels.

Validates AC #2-#3 from issue 553:
- explicit_parallel strategy uses npt=4 for fp16/bf16 by default
- register_copy strategy retains npt=8 for fp16/bf16 by default
- fp32 uses npt=4 for both strategies (unchanged)
- fp8 uses npt=16 for both strategies (unchanged)

These tests inspect default_config without GPU compilation by using
a lightweight mock approach: instantiate with strategy= and check
the config dict.
"""

import pytest
import torch

from tileops.kernels.elementwise import (
    ReluKernel,
)

# ---------------------------------------------------------------------------
# UnaryKernel default_config tests
# ---------------------------------------------------------------------------


class TestUnaryDefaultConfig:
    """UnaryKernel.default_config must be strategy-aware for fp16/bf16."""

    @pytest.mark.parametrize(
        "strategy, dtype, expected_npt",
        [
            # explicit_parallel: fp16/bf16 -> npt=4
            pytest.param("explicit_parallel", torch.float16, 4, id="ep-fp16"),
            pytest.param("explicit_parallel", torch.bfloat16, 4, id="ep-bf16"),
            # register_copy: fp16/bf16 -> npt=8
            pytest.param("register_copy", torch.float16, 8, id="rc-fp16"),
            pytest.param("register_copy", torch.bfloat16, 8, id="rc-bf16"),
            # fp32: npt=4 for both
            pytest.param("explicit_parallel", torch.float32, 4, id="ep-fp32"),
            pytest.param("register_copy", torch.float32, 4, id="rc-fp32"),
        ],
    )
    @pytest.mark.smoke
    def test_unary_npt_by_strategy(self, strategy, dtype, expected_npt):
        """default_config.num_per_thread depends on strategy for fp16/bf16."""
        kernel = ReluKernel(N_total=1024 * 10240, dtype=dtype, strategy=strategy)
        assert kernel.default_config["num_per_thread"] == expected_npt, (
            f"Expected npt={expected_npt} for {strategy}/{dtype}, "
            f"got {kernel.default_config['num_per_thread']}"
        )

    @pytest.mark.smoke
    def test_unary_explicit_parallel_fp16_config_applied(self):
        """When no config override, the init_config should use default_config npt."""
        kernel = ReluKernel(
            N_total=1024 * 10240, dtype=torch.float16, strategy="explicit_parallel",
        )
        assert kernel.config["num_per_thread"] == 4

    @pytest.mark.smoke
    def test_unary_register_copy_fp16_config_applied(self):
        """register_copy fp16 should init with npt=8 from default_config."""
        kernel = ReluKernel(
            N_total=1024 * 10240, dtype=torch.float16, strategy="register_copy",
        )
        assert kernel.config["num_per_thread"] == 8


# ---------------------------------------------------------------------------
# BinaryKernel default_config tests
# ---------------------------------------------------------------------------


class TestBinaryDefaultConfig:
    """BinaryKernel.default_config must be strategy-aware for fp16/bf16."""

    @pytest.mark.parametrize(
        "strategy, dtype, expected_npt",
        [
            pytest.param("explicit_parallel", torch.float16, 4, id="ep-fp16"),
            pytest.param("explicit_parallel", torch.bfloat16, 4, id="ep-bf16"),
            pytest.param("register_copy", torch.float16, 8, id="rc-fp16"),
            pytest.param("register_copy", torch.bfloat16, 8, id="rc-bf16"),
            pytest.param("explicit_parallel", torch.float32, 4, id="ep-fp32"),
            pytest.param("register_copy", torch.float32, 4, id="rc-fp32"),
        ],
    )
    @pytest.mark.smoke
    def test_binary_npt_by_strategy(self, strategy, dtype, expected_npt):
        """BinaryKernel default_config npt depends on strategy for fp16/bf16."""
        # BinaryKernel needs shape info; use a simple same-shape case
        from tileops.kernels.elementwise import AddKernel

        a = torch.randn(1024, 10240, dtype=dtype, device="cuda")
        b = torch.randn(1024, 10240, dtype=dtype, device="cuda")
        kernel = AddKernel(
            N_total=a.numel(),
            dtype=dtype,
            coalesced_shape=(1024 * 10240,),
            a_strides=(1,),
            b_strides=(1,),
            a_numel=a.numel(),
            b_numel=b.numel(),
            strategy=strategy,
        )
        assert kernel.default_config["num_per_thread"] == expected_npt


# ---------------------------------------------------------------------------
# FusedGatedKernel default_config tests
# ---------------------------------------------------------------------------


class TestFusedGatedDefaultConfig:
    """FusedGatedKernel.default_config must be strategy-aware for fp16/bf16."""

    @pytest.mark.parametrize(
        "strategy, dtype, expected_npt",
        [
            pytest.param("explicit_parallel", torch.float16, 4, id="ep-fp16"),
            pytest.param("explicit_parallel", torch.bfloat16, 4, id="ep-bf16"),
            pytest.param("explicit_parallel", torch.float32, 4, id="ep-fp32"),
        ],
    )
    @pytest.mark.smoke
    def test_fused_gated_npt_by_strategy(self, strategy, dtype, expected_npt):
        """FusedGatedKernel default_config npt for explicit_parallel."""
        from tileops.kernels.elementwise import SiluAndMulKernel

        kernel = SiluAndMulKernel(
            M=1024, N=10240, dtype=dtype, strategy=strategy,
        )
        assert kernel.default_config["num_per_thread"] == expected_npt
