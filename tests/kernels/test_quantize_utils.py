"""Tests for the vendored int4-unpack helper in tileops.kernels.quantize_utils."""

import pytest
import torch

from tileops.kernels.quantize_utils import _tir_packed_to_unsigned_convert

pytestmark = pytest.mark.smoke


def test_unpacks_both_nibbles_of_every_byte() -> None:
    import tilelang
    import tilelang.language as T

    decode = _tir_packed_to_unsigned_convert("uint", 8)

    @tilelang.jit(out_idx=[-1])
    def build():

        @T.prim_func
        def main(
            packed: T.Tensor((256,), "uint8"),  # type: ignore
            out: T.Tensor((256, 2), "float16"),  # type: ignore
        ) -> None:
            with T.Kernel(1, threads=128) as _:
                for i, j in T.Parallel(256, 2):
                    out[i, j] = decode(4, packed[i], j, "float16")

        return main

    kernel = build()
    packed = torch.arange(256, dtype=torch.uint8, device="cuda")
    out = kernel(packed)
    positions = torch.arange(2, dtype=torch.int32, device="cuda")
    expected = ((packed.to(torch.int32).unsqueeze(1) >> (4 * positions)) & 0xF).half()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
