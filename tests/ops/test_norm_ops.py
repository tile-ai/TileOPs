"""Tests for BatchNorm validation and torch.compile compatibility.

Covers:
- BatchNormFwdOp.forward() validates CUDA device, dtype, and shape
- BatchNormFwdOp and BatchNormBwdOp pass torch.compile smoke tests
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# BatchNormFwdOp input validation
# ---------------------------------------------------------------------------

@pytest.mark.smoke


class TestBatchNormFwdValidation:

    def _make_op(self):
        from tileops.ops.norm.batch_norm import BatchNormFwdOp
        return BatchNormFwdOp(4, 8, 4, 4, dtype=torch.float16)

    def _make_inputs(self, device="cuda", dtype=torch.float16):
        x = torch.randn(4, 8, 4, 4, device=device, dtype=dtype)
        weight = torch.randn(8, device=device, dtype=torch.float32)
        bias = torch.randn(8, device=device, dtype=torch.float32)
        rm = torch.zeros(8, device=device, dtype=torch.float32)
        rv = torch.ones(8, device=device, dtype=torch.float32)
        return x, weight, bias, rm, rv

    def test_rejects_cpu_tensor(self):
        op = self._make_op()
        x_cpu = torch.randn(4, 8, 4, 4, dtype=torch.float16)
        _, weight, bias, rm, rv = self._make_inputs()
        with pytest.raises(ValueError, match="CUDA"):
            op(x_cpu, rm, rv, weight, bias)

    def test_rejects_wrong_dtype(self):
        op = self._make_op()
        x_wrong = torch.randn(4, 8, 4, 4, device="cuda", dtype=torch.float32)
        _, weight, bias, rm, rv = self._make_inputs()
        with pytest.raises(ValueError, match="dtype"):
            op(x_wrong, rm, rv, weight, bias)

    def test_rejects_wrong_shape(self):
        op = self._make_op()
        x_wrong = torch.randn(4, 16, 4, 4, device="cuda", dtype=torch.float16)
        _, weight, bias, rm, rv = self._make_inputs()
        with pytest.raises(ValueError, match="shape|channel|Channel"):
            op(x_wrong, rm, rv, weight, bias)


# ---------------------------------------------------------------------------
# BatchNorm torch.compile smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.smoke


class TestBatchNormCustomOp:

    def test_fwd_torch_compile_smoke(self):
        from tileops.ops.norm.batch_norm import BatchNormFwdOp
        op = BatchNormFwdOp(4, 8, 4, 4, dtype=torch.float16, training=True)
        x = torch.randn(4, 8, 4, 4, device="cuda", dtype=torch.float16)
        weight = torch.randn(8, device="cuda", dtype=torch.float32)
        bias = torch.randn(8, device="cuda", dtype=torch.float32)
        rm = torch.zeros(8, device="cuda", dtype=torch.float32)
        rv = torch.ones(8, device="cuda", dtype=torch.float32)

        compiled = torch.compile(op, fullgraph=False)
        # Manifest input order: (x, running_mean, running_var, weight, bias).
        y = compiled(x, rm, rv, weight, bias, training=True)
        assert y.shape == x.shape

    def test_bwd_torch_compile_smoke(self):
        from tileops.ops.norm.batch_norm import BatchNormBwdOp
        N, C, H, W = 4, 8, 4, 4
        op = BatchNormBwdOp(N, C, H, W, dtype=torch.float16)
        grad_out = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
        x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
        weight = torch.randn(C, device="cuda", dtype=torch.float32)
        x32 = x.float()
        x_cl = x32.permute(1, 0, 2, 3).reshape(C, -1).contiguous()
        mean = x_cl.mean(dim=1)
        rstd = 1.0 / torch.sqrt(x_cl.var(dim=1, unbiased=False) + 1e-5)

        compiled = torch.compile(op, fullgraph=False)
        gx, gw, gb = compiled(grad_out, x, weight, mean, rstd)
        assert gx.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
