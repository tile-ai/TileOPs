import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.norm.ada_layer_norm import (
    AdaLayerNormKernel,
    _should_use_cp_async,
)
from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
from workloads.normalization import AdaLayerNormTest as _AdaLayerNormTestWorkload


class AdaLayerNormTest(_AdaLayerNormTestWorkload, TestBase):
    def ref_program(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        # AdaLN: y = scale * LayerNorm(x) + shift
        normed = F.layer_norm(
            x.float(),
            (self.n,),
            weight=None,
            bias=None,
            eps=self.eps,
        )
        y = scale.float() * normed + shift.float()
        return y.to(x.dtype)


class AdaLayerNormFixture(FixtureBase):
    PARAMS = [
        ("m, n, dtype", [
            # Standard aligned shapes -- fp32
            pytest.param(1024, 4096, torch.float32, marks=pytest.mark.smoke),
            # Standard aligned shapes -- fp16
            pytest.param(1024, 4096, torch.float16, marks=pytest.mark.smoke),
            # Standard aligned shapes -- bf16
            pytest.param(1024, 4096, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(4096, 4096, torch.float32, marks=pytest.mark.full),
            pytest.param(4096, 4096, torch.float16, marks=pytest.mark.full),
            pytest.param(4096, 4096, torch.bfloat16, marks=pytest.mark.full),
            # Non-power-of-two hidden dims
            pytest.param(1024, 3000, torch.float32, marks=pytest.mark.full),
            pytest.param(1024, 3000, torch.float16, marks=pytest.mark.full),
            pytest.param(1024, 3000, torch.bfloat16, marks=pytest.mark.full),
            # Tail-M: M not divisible by block_m
            pytest.param(1025, 4096, torch.float16, marks=pytest.mark.full),
            pytest.param(1025, 4096, torch.bfloat16, marks=pytest.mark.full),
        ]),
    ]


def _get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-3
    else:  # bfloat16
        return 1.6e-2, 1.6e-2


@AdaLayerNormFixture
def test_ada_layer_norm_op(m: int, n: int, dtype: torch.dtype) -> None:
    test = AdaLayerNormTest(m, n, dtype)
    op = AdaLayerNormFwdOp(dtype=dtype)
    atol, rtol = _get_tolerances(dtype)
    test.check(op, *test.gen_inputs(), atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_ada_layer_norm_kernel_handles_natural_unaligned_shape(
    dtype: torch.dtype,
) -> None:
    m, n = 16, 1152
    test = AdaLayerNormTest(m, n, dtype)
    inputs = test.gen_inputs()
    kernel = AdaLayerNormKernel(m, n, test.eps, dtype, has_gate=False)
    actual = kernel(*inputs)
    expected = test.ref_program(*inputs)
    assert actual.shape == (m, n)
    atol, rtol = _get_tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.smoke
@pytest.mark.parametrize("block_m", [2, 4])
@pytest.mark.parametrize("n", [514, 1152])
def test_ada_layer_norm_async_copy_handles_row_tail(
    block_m: int, n: int,
) -> None:
    """Regression: the async 2-D tile must support block_m > 1 and tail rows."""
    m = 17
    dtype = torch.float16
    test = AdaLayerNormTest(m, n, dtype)
    inputs = test.gen_inputs()
    kernel = AdaLayerNormKernel(
        m,
        n,
        test.eps,
        dtype,
        has_gate=False,
        config={"block_m": block_m, "threads": 128},
    )
    assert kernel.use_cp_async
    actual = kernel(*inputs)
    expected = test.ref_program(*inputs)
    atol, rtol = _get_tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.smoke
def test_ada_layer_norm_async_policy_edges() -> None:
    cases = [
        (511, torch.float16, False),
        (512, torch.float16, False),
        (513, torch.float16, False),
        (514, torch.float16, True),
        (1918, torch.float16, True),
        (1919, torch.float16, False),
        (1920, torch.float16, True),
        (513, torch.float32, True),
        (1919, torch.float32, True),
    ]
    for n, dtype, expected_async in cases:
        assert (
            _should_use_cp_async(n, dtype, has_gate=False)
            is expected_async
        )


@pytest.mark.smoke
def test_ada_layer_norm_async_policy_shared_memory_limit() -> None:
    cases = [
        (8190, torch.float16, False, True),
        (8194, torch.float16, False, False),
        (6142, torch.float16, True, True),
        (6146, torch.float16, True, False),
        (4094, torch.float32, False, True),
        (4098, torch.float32, False, False),
    ]
    for n, dtype, has_gate, expected_async in cases:
        assert (
            _should_use_cp_async(n, dtype, has_gate)
            is expected_async
        )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "n, dtype",
    [
        pytest.param(511, torch.float16, id="fp16-below"),
        pytest.param(514, torch.float16, id="fp16-lower-inside"),
        pytest.param(1919, torch.float16, id="fp16-row-bytes-outside"),
        pytest.param(1920, torch.float16, id="fp16-former-upper-bound"),
        pytest.param(513, torch.float32, id="fp32-lower-inside"),
    ],
)
def test_ada_layer_norm_async_policy_edge_correctness(
    n: int, dtype: torch.dtype,
) -> None:
    m = 4
    test = AdaLayerNormTest(m, n, dtype)
    inputs = test.gen_inputs()
    kernel = AdaLayerNormKernel(m, n, test.eps, dtype, has_gate=False)
    if n == 514:
        assert kernel.use_cp_async
    actual = kernel(*inputs)
    expected = test.ref_program(*inputs)
    atol, rtol = _get_tolerances(dtype)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


class AdaLayerNorm3DFixture(FixtureBase):
    PARAMS = [
        ("batch, seq, hidden, dtype", [
            pytest.param(2, 512, 4096, torch.float32, marks=pytest.mark.smoke),
            pytest.param(2, 512, 4096, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 512, 4096, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@AdaLayerNorm3DFixture
def test_ada_layer_norm_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """Test with 3D input (batch, seq, hidden)."""
    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    scale = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    shift = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")

    op = AdaLayerNormFwdOp(dtype=dtype)

    # Reference: scale * LayerNorm(x) + shift
    eps = 1e-5
    normed = F.layer_norm(
        x.float(), (hidden,), weight=None, bias=None, eps=eps,
    )
    y_ref = (scale.float() * normed + shift.float()).to(dtype)

    y = op(x, scale, shift)
    atol, rtol = _get_tolerances(dtype)
    assert torch.allclose(y, y_ref, atol=atol, rtol=rtol), \
        f"3D test failed, max err: {(y - y_ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
