from dataclasses import dataclass
from typing import Any, Callable, Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport, ManifestBenchmark
from tests.ops.test_gemm import GemmFp8Test, GemmTest, GemmW4A16Test
from tileops.manifest import load_workloads
from tileops.ops import GemmFp8Op, GemmOp, GemmW4A16Op

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


@dataclass(frozen=True)
class _W4A16BenchmarkCase:
    m: int
    n: int
    k: int
    label: str
    scenario: str
    purpose: str
    weight_metadata_mib: float | None = None


_W4A16_DECODE_CASES = (
    _W4A16BenchmarkCase(
        1,
        8192,
        8192,
        "decode-l2-resident-ish",
        "decode-medium-k",
        "medium K; packed W4 weight + metadata fits within H200 L2, exposing launch/depack/sync overhead",
        34.5,
    ),
    _W4A16BenchmarkCase(
        1,
        8192,
        16384,
        "decode-hbm-streaming-threshold",
        "decode-over-l2",
        "just over H200 L2, entering real HBM streaming and validating TMA/buffering behavior",
        69.0,
    ),
    _W4A16BenchmarkCase(
        1,
        7168,
        20480,
        "decode-non-power2-low-cta",
        "decode-non-power2-n",
        "non-power-of-two N with only 112 N64 CTAs, exposing occupancy and scheduling limits",
        75.5,
    ),
    _W4A16BenchmarkCase(
        1,
        8192,
        81920,
        "decode-long-k-pressure",
        "decode-long-k-stress",
        "very long K far beyond L2, amplifying HBM streaming, depack overlap, activation reuse, and split-K effects",
        345.0,
    ),
)


class _StaticGemmW4A16Benchmark(BenchmarkBase[GemmW4A16Test]):
    def __init__(self, test: GemmW4A16Test, memory_bytes: int) -> None:
        super().__init__(test)
        self._memory_bytes = memory_bytes

    def calculate_flops(self) -> float:
        return float(2 * self.workload.m * self.workload.n * self.workload.k)

    def calculate_memory(self) -> float:
        return float(self._memory_bytes)


def _w4a16_memory_bytes(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype = torch.float16,
    group_size: int = _W4A16_GROUP_SIZE,
) -> int:
    groups = k // group_size
    elem_bytes = dtype.itemsize
    return m * k * elem_bytes + n * k // 2 + n * groups * (4 + 1) + m * n * elem_bytes


def _dense_a16_memory_bytes(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    elem_bytes = dtype.itemsize
    return (m * k + n * k + m * n) * elem_bytes


def _marlin_w4a16_memory_bytes(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype = torch.float16,
    group_size: int = _W4A16_GROUP_SIZE,
) -> int:
    elem_bytes = dtype.itemsize
    qweight_bytes = (k // 16) * (n * 2) * torch.int32.itemsize
    scale_bytes = (k // group_size) * n * elem_bytes
    zero_bytes = (k // group_size) * (n // 8) * torch.int32.itemsize
    return m * k * elem_bytes + qweight_bytes + scale_bytes + zero_bytes + m * n * elem_bytes


def _flashinfer_fp8_blockscale_ref(test: GemmFp8Test, *inputs: torch.Tensor) -> torch.Tensor:
    from flashinfer.gemm import fp8_blockscale_gemm_sm90

    a, b, scale_a, scale_b = inputs[:4]
    if len(inputs) == 5:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline does not support bias.")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline requires float8_e4m3fn.")
    if test.out_dtype != torch.bfloat16:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline requires bfloat16 output.")
    if test.k % 128 != 0:
        raise ValueError("FlashInfer FP8 blockscale GEMM baseline requires k divisible by 128.")
    if scale_a.shape != (test.m, test.k // 128) or scale_b.shape != (test.n, test.k // 128):
        raise ValueError(
            "FlashInfer FP8 blockscale GEMM baseline requires exact "
            f"scale shapes {(test.m, test.k // 128)} and {(test.n, test.k // 128)}, "
            f"got {tuple(scale_a.shape)} and {tuple(scale_b.shape)}"
        )
    return fp8_blockscale_gemm_sm90(a, b, scale_a, scale_b, out_dtype=test.out_dtype)


def _prepare_flashinfer_fp8_per_tensor(
    test: GemmFp8Test, *inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    import flashinfer

    a, b, scale_a, scale_b = inputs[:4]
    if len(inputs) == 5:
        raise ValueError("FlashInfer FP8 per-tensor GEMM baseline does not support bias.")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("FlashInfer FP8 per-tensor GEMM baseline requires float8_e4m3fn.")
    if test.out_dtype != torch.bfloat16:
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


def _make_marlin_w4a16_callable(
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

    def marlin(
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

    return marlin, (activation, qweight, scales, zeros, workspace)


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
    test = GemmTest(m, n, k, dtype, trans_a, trans_b)
    a, b = test.gen_inputs()

    op = GemmOp(trans_a=trans_a, trans_b=trans_b)
    bm = ManifestBenchmark(_OP_NAME, op, test)

    # The benchmark framework warms up internally; eval_roofline() is read
    # lazily after profiling, by which point forward() has bound the dims.
    result = bm.profile(op, a, b)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, a, b)
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
    test = GemmFp8Test(m, n, k, dtype, scale_mode, out_dtype=out_dtype)
    inputs = test.gen_inputs()

    op = GemmFp8Op(out_dtype=out_dtype)
    bm = ManifestBenchmark(_FP8_OP_NAME, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-scaled-mm")

    if scale_mode == "per_tensor":
        unsupported_reason = _flashinfer_fp8_per_tensor_unsupported_reason(inputs[0].device)
        if unsupported_reason is not None:
            print(f"  [skip] flashinfer-mm-fp8: {unsupported_reason}")
            return
        flashinfer = pytest.importorskip("flashinfer")
        prepared_b, alpha = _prepare_flashinfer_fp8_per_tensor(test, *inputs)
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
            lambda *args: _flashinfer_fp8_blockscale_ref(test, *args), *inputs
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
    test = GemmW4A16Test(m, n, k, dtype, group_size=group_size)
    inputs = test.gen_inputs()

    op = GemmW4A16Op(group_size=group_size)
    bm = ManifestBenchmark(_W4A16_OP_NAME, op, test)

    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    dense_bm = _StaticGemmW4A16Benchmark(test, _dense_a16_memory_bytes(m, n, k, dtype=dtype))
    result_bl = dense_bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-dequantized-matmul")


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case.label) for case in _W4A16_DECODE_CASES],
)
def test_gemm_w4a16_decode_bench(case: _W4A16BenchmarkCase) -> None:
    """W4A16 decode comparison suite.

    This is intentionally not a complete public manifest benchmark. The suite
    fixes M=1, FP16 activation/output, affine UINT4 weights, group size 128,
    FP32 accumulation, pre-packed weights, and TileOps' cold-cache bench_kernel
    protocol. Quantization and repacking are outside the timed region.

    The long-K case (1,8192,81920) is the main mechanism stress test: it exceeds
    L2 by a wide margin, elongates the K dependency chain, and makes load,
    depack, compute overlap, activation reuse, and split-K policy visible in
    latency. The first three cases are retained so the stress test does not
    stand in for common model layers by itself.
    """
    dtype = torch.float16
    m, n, k = case.m, case.n, case.k
    scenario = case.scenario
    purpose = case.purpose
    weight_metadata_mib = case.weight_metadata_mib

    test = GemmW4A16Test(m, n, k, dtype)
    inputs = test.gen_inputs()
    op = GemmW4A16Op()

    w4_bm = _StaticGemmW4A16Benchmark(test, _w4a16_memory_bytes(m, n, k, dtype=dtype))
    result = w4_bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops-w4a16")

    dense_bm = _StaticGemmW4A16Benchmark(test, _dense_a16_memory_bytes(m, n, k, dtype=dtype))
    result_dense = dense_bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_dense, tag="torch-dequantized-a16")

    for reduce_mode, use_fp32_reduce in (("fp32", True), ("fp16", False)):
        try:
            marlin, marlin_inputs = _make_marlin_w4a16_callable(
                m, n, k, use_fp32_reduce=use_fp32_reduce
            )
        except (ImportError, ModuleNotFoundError) as exc:
            print(f"  [skip] marlin-{reduce_mode}: {exc}")
            continue
        marlin_bm = _StaticGemmW4A16Benchmark(
            test, _marlin_w4a16_memory_bytes(m, n, k, dtype=dtype)
        )
        actual = marlin(*marlin_inputs)
        if actual.shape != (m, n) or not torch.isfinite(actual).all():
            raise RuntimeError("Marlin W4A16 baseline smoke check failed")
        torch.cuda.synchronize()
        result_marlin = marlin_bm.profile(marlin, *marlin_inputs)
        BenchmarkReport.record(op, locals(), result_marlin, tag=f"marlin-{reduce_mode}")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
