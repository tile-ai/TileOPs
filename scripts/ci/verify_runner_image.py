"""Verify a built runner image on a GPU host.

Complements ``verify_runtime_stack.py``, which the build runs without a GPU:
this one needs one, and checks the baselines and cuBLAS actually work.

Run inside the image:

    docker run --rm --gpus all <image> python /src/scripts/ci/verify_runner_image.py
"""

import sys

import torch


def main() -> int:
    print(f"torch {torch.__version__} cuda {torch.version.cuda}")
    if not torch.__version__.endswith("+cu132"):
        print(f"FAIL: expected a +cu132 torch build, got {torch.__version__}")
        return 1

    import tilelang

    print(f"tilelang {tilelang.__version__}")

    import flashinfer

    print(f"flashinfer {flashinfer.__version__}")

    # A missing baseline costs the column, not the run, so nothing else notices.
    import flag_gems  # noqa: F401 - import is the check
    import flash_attn
    import flash_attn_interface

    assert flash_attn_interface.flash_attn_func is not None
    print(f"flash-attn {flash_attn.__version__} | flash-attn-3 | flag_gems")

    import mamba_ssm
    import selective_scan_cuda  # noqa: F401 - import is the check
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    assert mamba_chunk_scan_combined is not None
    print(f"mamba-ssm {mamba_ssm.__version__}")

    import deep_gemm

    print(f"deep_gemm {deep_gemm.__version__}")

    import deepspeed
    from deepspeed.ops.quantizer import quantizer_op

    # Row 0 is constant, where a degenerate scale would show up.
    w = torch.arange(-8, 8, device="cuda", dtype=torch.float16).repeat(4, 8)
    w[0] = 0
    packed, params = quantizer_op.quantize(w, 4, 4, quantizer_op.Asymmetric)
    codes = torch.stack((packed >> 4, (packed << 4) >> 4), dim=-1).reshape_as(w)
    restored = codes.float() * params[:, :1] + params[:, 1:]
    assert packed.dtype == torch.int8 and params.dtype == torch.float32
    assert packed.shape == (4, 64) and params.shape == (4, 2)
    assert torch.isfinite(params).all() and (params[:, 0] > 0).all()
    assert ((restored - w.float()).abs() <= params[:, :1] * (1 + 1e-5)).all()
    torch.cuda.synchronize()
    print(f"deepspeed {deepspeed.__version__} INT4 quantizer OK")

    # cuBLAS: a broken install shows up here rather than in the first benchmark.
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    assert torch.matmul(a, a).isfinite().all()
    ab = torch.randn(8, 128, 128, device="cuda", dtype=torch.float16)
    assert torch.bmm(ab, ab).isfinite().all()
    assert torch.einsum("bik,bkj->bij", ab, ab).isfinite().all()
    print("cuBLAS matmul / bmm / einsum OK")

    # Whether nvmath resolves torch's libcublasLt through cuda-pathfinder needs a GPU to answer.
    from nvmath.linalg.advanced import Matmul, MatmulComputeType, MatmulOptions

    with Matmul(a, a, options=MatmulOptions(compute_type=MatmulComputeType.COMPUTE_32F)) as mm:
        algorithms = mm.plan()
        assert algorithms, "cuBLASLt returned no algorithm"
        assert mm.execute(algorithm=algorithms[0]).isfinite().all()
    print(f"nvmath cuBLASLt plan OK ({len(algorithms)} algorithms)")

    print("image OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
