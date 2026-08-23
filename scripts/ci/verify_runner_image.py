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

    # cuBLAS: a broken install shows up here rather than in the first benchmark.
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    assert torch.matmul(a, a).isfinite().all()
    ab = torch.randn(8, 128, 128, device="cuda", dtype=torch.float16)
    assert torch.bmm(ab, ab).isfinite().all()
    assert torch.einsum("bik,bkj->bij", ab, ab).isfinite().all()
    print("cuBLAS matmul / bmm / einsum OK")

    print("image OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
