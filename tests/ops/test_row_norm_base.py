"""Tests for RowNormOp base class and BatchNorm validation / custom_op.

Covers:
- AC-1: RowNormOp base class exists in tileops/ops/norm/base.py
- AC-2: RmsNormOp inherits from RowNormOp and class body is <=20 lines
- AC-4: BatchNormFwdOp.forward() validates CUDA device, dtype, and shape
- AC-5: BatchNormFwdOp and BatchNormBwdOp have torch.library.custom_op
        registration and pass a torch.compile smoke test
"""

import inspect
import textwrap

import pytest
import torch

# ---------------------------------------------------------------------------
# AC-1: RowNormOp base class exists
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestRowNormOpExists:
    """Verify the base class can be imported and has expected structure."""

    def test_import(self):
        from tileops.ops.norm.base import RowNormOp  # noqa: F401

    def test_is_subclass_of_op(self):
        from tileops.ops.norm.base import RowNormOp
        from tileops.ops.op import Op
        assert issubclass(RowNormOp, Op)

    def test_has_abstract_hooks(self):
        from tileops.ops.norm.base import RowNormOp
        # Should not be instantiable directly (abstract)
        with pytest.raises(TypeError):
            RowNormOp.__new__(RowNormOp)  # type: ignore[misc]

    def test_has_alignment_constant(self):
        from tileops.ops.norm.base import ALIGNMENT
        assert ALIGNMENT == 256

    def test_has_align_up(self):
        from tileops.ops.norm.base import RowNormOp
        # RowNormOp should have _align_up as a static or module-level helper
        assert hasattr(RowNormOp, '_align_up') or callable(
            getattr(RowNormOp, '_align_up', None)
        )


# ---------------------------------------------------------------------------
# AC-2: RmsNormOp inherits from RowNormOp, <=20 body lines
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestRmsNormMigration:

    def test_inherits_from_row_norm_op(self):
        from tileops.ops.norm.base import RowNormOp
        from tileops.ops.norm.rms_norm import RmsNormOp
        assert issubclass(RmsNormOp, RowNormOp)

    def test_class_body_20_lines_or_less(self):
        from tileops.ops.norm.rms_norm import RmsNormOp
        source = inspect.getsource(RmsNormOp)
        # Remove leading whitespace / dedent, then count non-empty,
        # non-import, non-comment lines that are part of the class body
        lines = textwrap.dedent(source).strip().splitlines()
        # Skip the class definition line itself
        body_lines = [
            ln for ln in lines[1:]
            if ln.strip() and not ln.strip().startswith('#')
        ]
        assert len(body_lines) <= 20, (
            f"RmsNormOp class body has {len(body_lines)} lines, expected <=20"
        )

    def test_api_unchanged(self):
        """RmsNormOp(M, N, dtype) and op(x, weight) must still work."""
        from tileops.ops.norm.rms_norm import RmsNormOp
        sig = inspect.signature(RmsNormOp.__init__)
        params = list(sig.parameters.keys())
        # Must accept M, N, dtype positionally
        assert 'M' in params
        assert 'N' in params
        assert 'dtype' in params


# ---------------------------------------------------------------------------
# AC-4: BatchNormFwdOp input validation
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
            op(x_cpu, weight, bias, rm, rv)

    def test_rejects_wrong_dtype(self):
        op = self._make_op()
        x_wrong = torch.randn(4, 8, 4, 4, device="cuda", dtype=torch.float32)
        _, weight, bias, rm, rv = self._make_inputs()
        with pytest.raises(ValueError, match="dtype"):
            op(x_wrong, weight, bias, rm, rv)

    def test_rejects_wrong_shape(self):
        op = self._make_op()
        x_wrong = torch.randn(4, 16, 4, 4, device="cuda", dtype=torch.float16)
        _, weight, bias, rm, rv = self._make_inputs()
        with pytest.raises(ValueError, match="shape|channel|Channel"):
            op(x_wrong, weight, bias, rm, rv)


# ---------------------------------------------------------------------------
# AC-5: BatchNorm torch.library.custom_op + torch.compile smoke test
# ---------------------------------------------------------------------------

@pytest.mark.smoke
class TestBatchNormCustomOp:

    def test_fwd_has_custom_op(self):
        from tileops.ops.norm.batch_norm import BatchNormFwdOp
        assert hasattr(BatchNormFwdOp, '_wrapped'), \
            "BatchNormFwdOp missing _wrapped (custom_op not registered)"

    def test_bwd_has_custom_op(self):
        from tileops.ops.norm.batch_norm import BatchNormBwdOp
        assert hasattr(BatchNormBwdOp, '_wrapped'), \
            "BatchNormBwdOp missing _wrapped (custom_op not registered)"

    def test_fwd_torch_compile_smoke(self):
        from tileops.ops.norm.batch_norm import BatchNormFwdOp
        op = BatchNormFwdOp(4, 8, 4, 4, dtype=torch.float16)
        x = torch.randn(4, 8, 4, 4, device="cuda", dtype=torch.float16)
        weight = torch.randn(8, device="cuda", dtype=torch.float32)
        bias = torch.randn(8, device="cuda", dtype=torch.float32)
        rm = torch.zeros(8, device="cuda", dtype=torch.float32)
        rv = torch.ones(8, device="cuda", dtype=torch.float32)

        compiled = torch.compile(op, fullgraph=False)
        y, mean, rstd = compiled(x, weight, bias, rm, rv, training=True)
        assert y.shape == x.shape

    def test_bwd_torch_compile_smoke(self):
        from tileops.ops.norm.batch_norm import BatchNormBwdOp
        N, C, H, W = 4, 8, 4, 4
        op = BatchNormBwdOp(N, C, H, W, dtype=torch.float16)
        grad_out = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
        x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
        weight = torch.randn(C, device="cuda", dtype=torch.float32)
        # Compute mean/rstd
        x32 = x.float()
        x_cl = x32.permute(1, 0, 2, 3).reshape(C, -1).contiguous()
        mean = x_cl.mean(dim=1)
        rstd = 1.0 / torch.sqrt(x_cl.var(dim=1, unbiased=False) + 1e-5)

        compiled = torch.compile(op, fullgraph=False)
        gx, gw, gb = compiled(grad_out, x, weight, mean, rstd)
        assert gx.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
