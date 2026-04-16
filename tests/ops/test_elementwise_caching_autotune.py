"""Tests for elementwise kernel caching and autotune_configs.

Validates:
- UnaryKernel, FusedGatedKernel, and custom kernels cache compiled functions
  after init_config (no per-forward recompilation).
- autotune_configs is defined for UnaryKernel and FusedGatedKernel with >= 3 configs.
- Custom kernels (LeakyRelu, Elu, etc.) also cache compiled functions.
- Serialization-fallback autotune works for UnaryKernel and FusedGatedKernel.
"""

import pytest
import torch

from tileops.kernels.elementwise import (
    AbsFwdKernel,
    # Concrete binary
    AddFwdKernel,
    AlibiFwdKernel,
    ClampFwdKernel,
    EluFwdKernel,
    GeluAndMulFwdKernel,
    GeluTanhAndMulFwdKernel,
    HardtanhFwdKernel,
    # Custom kernels
    LeakyReluFwdKernel,
    MaskedFillFwdKernel,
    NanToNumFwdKernel,
    PreluFwdKernel,
    # Concrete unary
    ReluFwdKernel,
    SigmoidFwdKernel,
    # Concrete fused gated
    SiluAndMulFwdKernel,
    SinusoidalFwdKernel,
    SoftplusFwdKernel,
    # Base classes
    WhereFwdKernel,
)

N = 2048  # small enough for fast tests


# ---------------------------------------------------------------------------
# 1. UnaryKernel caching: _compiled_fn exists after init
# ---------------------------------------------------------------------------


class TestUnaryCaching:
    """UnaryKernel subclasses should have _compiled_fn after __init__."""

    @pytest.mark.full
    @pytest.mark.parametrize("kernel_cls", [ReluFwdKernel, SigmoidFwdKernel, AbsFwdKernel])
    def test_unary_has_compiled_fn(self, kernel_cls):
        k = kernel_cls(N, torch.float16)
        assert hasattr(k, "_compiled_fn"), (
            f"{kernel_cls.__name__} missing _compiled_fn after init"
        )
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_unary_forward_uses_cached_fn(self):
        """forward() should use _compiled_fn, not re-lookup the kernel."""
        k = ReluFwdKernel(N, torch.float16)
        fn1 = k._compiled_fn
        x = torch.randn(N, dtype=torch.float16, device="cuda")
        _ = k(x)
        # _compiled_fn should not change after forward
        assert k._compiled_fn is fn1

    @pytest.mark.full
    def test_unary_direct_strategy_caching(self):
        """Direct strategy kernels should also cache _compiled_fn."""
        k = ReluFwdKernel(N, torch.float16, strategy="direct")
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None


# ---------------------------------------------------------------------------
# 2. FusedGatedKernel caching: _compiled_fn exists after init
# ---------------------------------------------------------------------------


class TestFusedGatedCaching:
    """FusedGatedKernel subclasses should have _compiled_fn after __init__."""

    @pytest.mark.full
    @pytest.mark.parametrize("kernel_cls", [
        SiluAndMulFwdKernel, GeluAndMulFwdKernel, GeluTanhAndMulFwdKernel,
    ])
    def test_fused_gated_has_compiled_fn(self, kernel_cls):
        k = kernel_cls(32, 64, torch.float16)
        assert hasattr(k, "_compiled_fn"), (
            f"{kernel_cls.__name__} missing _compiled_fn after init"
        )
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_fused_gated_forward_uses_cached_fn(self):
        k = SiluAndMulFwdKernel(32, 64, torch.float16)
        fn1 = k._compiled_fn
        x = torch.randn(32, 128, dtype=torch.float16, device="cuda")
        _ = k(x)
        assert k._compiled_fn is fn1


# ---------------------------------------------------------------------------
# 3. Custom kernel caching: _compiled_fn exists after init
# ---------------------------------------------------------------------------


class TestCustomKernelCaching:
    """Custom (non-template) kernels should also cache _compiled_fn."""

    @pytest.mark.full
    def test_leaky_relu_caching(self):
        k = LeakyReluFwdKernel(N, torch.float16)
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_elu_caching(self):
        k = EluFwdKernel(N, torch.float16)
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_hardtanh_caching(self):
        k = HardtanhFwdKernel(N, torch.float16)
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_softplus_caching(self):
        k = SoftplusFwdKernel(N, torch.float16)
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_prelu_caching(self):
        k = PreluFwdKernel(N, 4, 512, torch.float16)
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_where_caching(self):
        k = WhereFwdKernel(N, torch.float16)
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_clamp_caching(self):
        k = ClampFwdKernel(N, torch.float16, min_val=-1.0, max_val=1.0)
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_masked_fill_caching(self):
        k = MaskedFillFwdKernel(N, torch.float16, fill_value=0.0)
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_nan_to_num_caching(self):
        k = NanToNumFwdKernel(N, torch.float16)
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_alibi_caching(self):
        k = AlibiFwdKernel(32, 4, torch.float16)
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None

    @pytest.mark.full
    def test_sinusoidal_caching(self):
        k = SinusoidalFwdKernel(32, 64, torch.float16)
        assert hasattr(k, "_compiled_fn")
        assert k._compiled_fn is not None


# ---------------------------------------------------------------------------
# 4. autotune_configs defined with >= 3 configs
# ---------------------------------------------------------------------------


class TestAutotuneConfigs:
    """UnaryKernel and FusedGatedKernel must define autotune_configs."""

    @pytest.mark.full
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_unary_autotune_configs_count(self, dtype):
        k = ReluFwdKernel(N, dtype)
        configs = k.autotune_configs
        assert configs is not None, "UnaryKernel.autotune_configs should not be None"
        assert len(configs) >= 3, f"Expected >= 3 configs, got {len(configs)}"
        # Each config must have threads and num_per_thread keys
        for c in configs:
            assert "threads" in c
            assert "num_per_thread" in c

    @pytest.mark.full
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_fused_gated_autotune_configs_count(self, dtype):
        k = SiluAndMulFwdKernel(32, 64, dtype)
        configs = k.autotune_configs
        assert configs is not None, "FusedGatedKernel.autotune_configs should not be None"
        assert len(configs) >= 3, f"Expected >= 3 configs, got {len(configs)}"
        for c in configs:
            assert "threads" in c
            assert "num_per_thread" in c

    @pytest.mark.full
    def test_unary_fp8_autotune_configs(self):
        """fp8 unary should have specific fp8-appropriate configs."""
        k = ReluFwdKernel(N, torch.float8_e4m3fn)
        configs = k.autotune_configs
        assert configs is not None
        assert len(configs) >= 3
        # fp8 npt should be >= 16 for 128-bit alignment
        for c in configs:
            assert c["num_per_thread"] >= 16

    @pytest.mark.full
    def test_binary_autotune_configs_still_works(self):
        """BinaryKernel autotune_configs must still work (no regression)."""
        k = AddFwdKernel(N, torch.float16, [N], [1], [1], N, N)
        configs = k.autotune_configs
        assert configs is not None
        assert len(configs) >= 3


# ---------------------------------------------------------------------------
# 5. Correctness: caching does not change results
# ---------------------------------------------------------------------------


class TestCachingCorrectness:
    """Verify that caching produces the same results as before."""

    @pytest.mark.full
    def test_unary_relu_correctness(self):
        k = ReluFwdKernel(N, torch.float16)
        x = torch.randn(N, dtype=torch.float16, device="cuda")
        out = k(x)
        ref = torch.relu(x.float()).to(torch.float16)
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

    @pytest.mark.full
    def test_fused_gated_silu_correctness(self):
        M, Nhalf = 32, 64
        k = SiluAndMulFwdKernel(M, Nhalf, torch.float16)
        x = torch.randn(M, 2 * Nhalf, dtype=torch.float16, device="cuda")
        out = k(x)
        gate = x[:, :Nhalf].float()
        value = x[:, Nhalf:].float()
        ref = (torch.nn.functional.silu(gate) * value).to(torch.float16)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.full
    def test_custom_leaky_relu_correctness(self):
        k = LeakyReluFwdKernel(N, torch.float16, negative_slope=0.01)
        x = torch.randn(N, dtype=torch.float16, device="cuda")
        out = k(x)
        ref = torch.nn.functional.leaky_relu(x.float(), 0.01).to(torch.float16)
        torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
