"""Verify a built runner image on a GPU host.

Complements ``verify_runtime_stack.py``, which the build runs without a GPU:
this one needs one, and checks the baselines and cuBLAS actually work.

Run inside the image:

    docker run --rm --gpus all <image> python /src/scripts/ci/verify_runner_image.py
"""

import sys
from pathlib import Path

import torch


def main() -> int:
    print(f"torch {torch.__version__} cuda {torch.version.cuda}")
    if not torch.__version__.endswith("+cu129"):
        print(f"FAIL: expected a +cu129 torch build, got {torch.__version__}")
        return 1

    import tilelang

    print(f"tilelang {tilelang.__version__}")

    import flashinfer
    import flashinfer_cubin

    print(f"flashinfer {flashinfer.__version__}")
    # The cubin dir must be writable: flashinfer downloads kernels into it at runtime.
    probe = Path(flashinfer_cubin.get_cubin_dir()) / "flashinfer" / "_tileops_write_probe"
    probe.mkdir(parents=True, exist_ok=True)
    (probe / "probe.txt").write_text("ok")
    print("flashinfer cubin dir writable")

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
