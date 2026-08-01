from typing import Any, Callable, Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from tileops.manifest import load_workloads
from tileops.ops import GemmFp8Op, GemmOp, GemmW4A16Op
from workloads.gemm import GemmFp8Workload, GemmW4A16Workload, GemmWorkload

_OP_NAME = "GemmOp"
_FP8_OP_NAME = "GemmFp8Op"
_W4A16_OP_NAME = "GemmW4A16Op"
_W4A16_GROUP_SIZE = 128

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


class GemmBenchmarkWorkload(GemmWorkload):
    def torch_matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)


class GemmFp8BenchmarkWorkload(GemmFp8Workload):
    def _expand_scale(self, scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        if tuple(scale.shape) == (1, 1):
            return scale.expand(rows, cols)
        scale_cols = (cols + 127) // 128
        if tuple(scale.shape) != (rows, scale_cols):
            raise ValueError(
                f"unsupported FP8 scale shape {tuple(scale.shape)} for {(rows, cols)}")
        return scale.repeat_interleave(128, dim=1)[:, :cols]

    def torch_scaled_matmul(self, *inputs: torch.Tensor) -> torch.Tensor:
        a, b, scale_a, scale_b = inputs[:4]
        bias = inputs[4] if len(inputs) == 5 else None
        a_f = a.float() * self._expand_scale(scale_a, self.m, self.k)
        b_f = b.float() * self._expand_scale(scale_b, self.n, self.k)
        out = torch.matmul(a_f, b_f.T)
        if bias is not None:
            out = out + bias.float()
        return out.to(self.out_dtype)


class GemmW4A16BenchmarkWorkload(GemmW4A16Workload):
    def torch_dequantized_matmul(
        self,
        activation: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zero: torch.Tensor,
    ) -> torch.Tensor:
        del packed_weight, weight_scale, weight_zero
        return torch.matmul(activation, self.dequantized_weight.T)


def _flashinfer_fp8_blockscale_ref(
    workload: GemmFp8BenchmarkWorkload, *inputs: torch.Tensor
) -> torch.Tensor:
    from flashinfer.gemm import fp8_blockscale_gemm_sm90

    a, b, scale_a, scale_b = inputs[:4]
    if len(inputs) == 5:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline does not support bias.")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline requires float8_e4m3fn.")
    if workload.out_dtype != torch.bfloat16:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline requires bfloat16 output.")
    if workload.k % 128 != 0:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline requires k divisible by 128.")
    if scale_a.shape != (workload.m, workload.k // 128) or scale_b.shape != (
        workload.n, workload.k // 128
    ):
        raise ValueError(
            "FlashInfer FP8 blockscale GEMM baseline requires exact "
            f"scale shapes {(workload.m, workload.k // 128)} "
            f"and {(workload.n, workload.k // 128)}, "
            f"got {tuple(scale_a.shape)} and {tuple(scale_b.shape)}"
        )
    return fp8_blockscale_gemm_sm90(a, b, scale_a, scale_b, out_dtype=workload.out_dtype)


def _prepare_flashinfer_fp8_per_tensor(
    workload: GemmFp8BenchmarkWorkload, *inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    import flashinfer

    a, b, scale_a, scale_b = inputs[:4]
    if len(inputs) == 5:
        raise ValueError("FlashInfer FP8 per-tensor GEMM baseline does not support bias.")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("FlashInfer FP8 per-tensor GEMM baseline requires float8_e4m3fn.")
    if workload.out_dtype != torch.bfloat16:
        raise ValueError("FlashInfer FP8 per-tensor GEMM baseline requires bfloat16 output.")
    if scale_a.shape != (1, 1) or scale_b.shape != (1, 1):
        raise ValueError(
            "FlashInfer FP8 per-tensor GEMM baseline requires (1, 1) scales, "
            f"got {tuple(scale_a.shape)} and {tuple(scale_b.shape)}"
        )
    prepared_b = flashinfer.prepare_low_latency_gemm_weights(b, {})
    alpha = (scale_a * scale_b).reshape(())
    return prepared_b, alpha


def _flashinfer_fp8_per_tensor_unsupported_reason(device: torch.device) -> Optional[str]:
    major, minor = torch.cuda.get_device_capability(device)
    if major < 10:
        return (
            "TRTLLM low-latency GEMM requires Blackwell (sm100+), "
            f"but the current device is sm{major}{minor}"
        )
    return None


def _prepare_marlin_w4a16_baseline(
    m: int,
    n: int,
    k: int,
    use_fp32_reduce: bool,
) -> tuple[Callable[..., torch.Tensor], tuple[Any, ...]]:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.scalar_type import scalar_types

    if k % 16 or k % _W4A16_GROUP_SIZE or n % 64:
        raise ValueError("Marlin W4A16 benchmark requires K % 128 == 0 and N % 64 == 0")

    device = torch.device("cuda")
    activation = torch.randn((m, k), dtype=torch.float16, device=device)
    qweight = torch.randint(
        -(2**31),
        2**31 - 1,
        (k // 16, n * 2),
        dtype=torch.int32,
        device=device,
    )
    scales = torch.rand((k // _W4A16_GROUP_SIZE, n), dtype=torch.float16, device=device)
    zeros = torch.randint(
        -(2**31),
        2**31 - 1,
        (k // _W4A16_GROUP_SIZE, n // 8),
        dtype=torch.int32,
        device=device,
    )
    workspace = marlin_make_workspace_new(device)

    def _run_marlin(
        a: torch.Tensor,
        packed: torch.Tensor,
        weight_scales: torch.Tensor,
        weight_zeros: torch.Tensor,
        locks: torch.Tensor,
    ) -> torch.Tensor:
        return ops.marlin_gemm(
            a=a,
            c=None,
            b_q_weight=packed,
            b_bias=None,
            b_scales=weight_scales,
            a_scales=None,
            global_scale=None,
            b_zeros=weight_zeros,
            g_idx=None,
            perm=None,
            workspace=locks,
            b_q_type=scalar_types.uint4,
            size_m=m,
            size_n=n,
            size_k=k,
            is_k_full=True,
            use_atomic_add=False,
            use_fp32_reduce=use_fp32_reduce,
            is_zp_float=False,
        )

    return _run_marlin, (activation, qweight, scales, zeros, workspace)


def _manifest_params() -> list:
    """Convert manifest workloads to pytest params (m, n, k, trans_a, trans_b, dtype)."""
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        trans_a = bool(w.get("trans_a", False))
        trans_b = bool(w.get("trans_b", True))
        for dtype_str in w["dtypes"]:
            params.append(
                pytest.param(
                    w["m"],
                    w["n"],
                    w["k"],
                    trans_a,
                    trans_b,
                    dtype_str,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


def _manifest_fp8_params() -> list:
    params = []
    for w in load_workloads(_FP8_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(
                pytest.param(
                    w["m"],
                    w["n"],
                    w["k"],
                    w["scale_mode"],
                    dtype_str,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


def _manifest_w4a16_params() -> list:
    params = []
    for w in load_workloads(_W4A16_OP_NAME):
        label = w.get("label", "unlabeled")
        group_size = int(w.get("group_size", 128))
        for dtype_str in w["dtypes"]:
            params.append(
                pytest.param(
                    w["m"],
                    w["n"],
                    w["k"],
                    group_size,
                    dtype_str,
                    id=f"{label}-{dtype_str}",
                )
            )
    return params


@pytest.mark.parametrize("m, n, k, trans_a, trans_b, dtype_str", _manifest_params())
def test_gemm_bench(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    workload = GemmBenchmarkWorkload(m, n, k, dtype, trans_a, trans_b)
    a, b = workload.gen_inputs()

    op = GemmOp(trans_a=trans_a, trans_b=trans_b)
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # The benchmark framework warms up internally; eval_roofline() is read
    # lazily after profiling, by which point forward() has bound the dims.
    result = bm.profile(op, a, b)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(workload.torch_matmul, a, b)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-cublas")


@pytest.mark.parametrize("m, n, k, scale_mode, dtype_str", _manifest_fp8_params())
def test_gemm_fp8_bench(
    m: int,
    n: int,
    k: int,
    scale_mode: str,
    dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    out_dtype = torch.bfloat16
    workload = GemmFp8BenchmarkWorkload(m, n, k, dtype, scale_mode, out_dtype=out_dtype)
    inputs = workload.gen_inputs()

    op = GemmFp8Op(out_dtype=out_dtype)
    bm = ManifestBenchmark(_FP8_OP_NAME, op, workload)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(workload.torch_scaled_matmul, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-scaled-mm")

    if scale_mode == "per_tensor":
        unsupported_reason = _flashinfer_fp8_per_tensor_unsupported_reason(inputs[0].device)
        if unsupported_reason is not None:
            print(f"  [skip] flashinfer-mm-fp8: {unsupported_reason}")
            return
        flashinfer = pytest.importorskip("flashinfer")
        prepared_b, alpha = _prepare_flashinfer_fp8_per_tensor(workload, *inputs)
        try:
            result_flashinfer = bm.profile(
                lambda a: flashinfer.mm_fp8(a, prepared_b, alpha, out_dtype=out_dtype),
                inputs[0],
            )
        except RuntimeError as exc:
            reason = str(exc).splitlines()[0]
            print(f"  [skip] flashinfer-mm-fp8: {reason}")
            return
        BenchmarkReport.record(op, locals(), result_flashinfer, tag="flashinfer-mm-fp8")
        return

    if scale_mode == "block128":
        pytest.importorskip("flashinfer")
        result_flashinfer = bm.profile(
            lambda *args: _flashinfer_fp8_blockscale_ref(workload, *args), *inputs
        )
        BenchmarkReport.record(
            op, locals(), result_flashinfer, tag="flashinfer-fp8-blockscale-sm90"
        )
        return

    raise ValueError(f"unsupported FP8 GEMM scale_mode for benchmark: {scale_mode!r}")


@pytest.mark.parametrize("m, n, k, group_size, dtype_str", _manifest_w4a16_params())
def test_gemm_w4a16_bench(
    m: int,
    n: int,
    k: int,
    group_size: int,
    dtype_str: str,
) -> None:
    dtype = _DTYPE_MAP[dtype_str]
    workload = GemmW4A16BenchmarkWorkload(m, n, k, dtype, group_size=group_size)
    inputs = workload.gen_inputs()

    op = GemmW4A16Op(group_size=group_size)
    bm = ManifestBenchmark(_W4A16_OP_NAME, op, workload)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(workload.torch_dequantized_matmul, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-dequantized-matmul")

    if m == 1:
        for reduce_mode, use_fp32_reduce in (("fp32", True), ("fp16", False)):
            try:
                marlin, marlin_inputs = _prepare_marlin_w4a16_baseline(
                    m, n, k, use_fp32_reduce=use_fp32_reduce
                )
            except (ImportError, ModuleNotFoundError) as exc:
                print(f"  [skip] marlin-{reduce_mode}: {exc}")
                continue
            actual = marlin(*marlin_inputs)
            if actual.shape != (m, n) or not torch.isfinite(actual).all():
                raise RuntimeError("Marlin W4A16 baseline smoke check failed")
            torch.cuda.synchronize()
            result_marlin = bm.profile(marlin, *marlin_inputs)
            BenchmarkReport.record(op, locals(), result_marlin, tag=f"marlin-{reduce_mode}")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
