from typing import Any, Callable, Optional

import pytest
import torch

    DEEPGEMM_TAG,
    FLAGGEMS_TAG,
    assert_matches_reference,
    deepgemm_op,
    flaggems_op,
    reference_tolerance,
)
from benchmarks.baselines import (
from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark, workload_params
from benchmarks.benchmark_base import ManifestBenchmark
from benchmarks.cublaslt_baseline import make_cublaslt_best
from tileops.manifest import load_workloads
from tileops.ops import GemmFp8FwdOp, GemmFwdOp, GemmW4A16FwdOp
from workloads.gemm import GemmFp8Workload, GemmW4A16Workload, GemmWorkload

_OP_NAME = "GemmFwdOp"
_FP8_OP_NAME = "GemmFp8FwdOp"
_W4A16_OP_NAME = "GemmW4A16FwdOp"
_W4A16_GROUP_SIZE = 128


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
            raise ValueError(f"unsupported FP8 scale shape {tuple(scale.shape)} for {(rows, cols)}")
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
        workload.n,
        workload.k // 128,
    ):
        raise ValueError(
            "FlashInfer FP8 blockscale GEMM baseline requires exact "
            f"scale shapes {(workload.m, workload.k // 128)} "
            f"and {(workload.n, workload.k // 128)}, "
            f"got {tuple(scale_a.shape)} and {tuple(scale_b.shape)}"
        )
    return fp8_blockscale_gemm_sm90(a, b, scale_a, scale_b, out_dtype=workload.out_dtype)


def _deepgemm_bf16_nt(
    workload: GemmBenchmarkWorkload, a: torch.Tensor, b: torch.Tensor
) -> Optional[Callable[..., torch.Tensor]]:
    """DeepGEMM's dense bf16 GEMM, or None for a row its kernel cannot take.

    It reads both operands K-major and bf16 only, which is the N-T layout here.
    """
    if workload.trans_a or not workload.trans_b or a.dtype != torch.bfloat16:
        return None

    gemm = deepgemm_op("bf16_gemm_nt")
    m, n = workload.m, workload.n

    def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        gemm(a, b, out)
        return out

    return run


def _deepgemm_fp8_per_tensor(
    workload: GemmFp8BenchmarkWorkload, *inputs: torch.Tensor
) -> Callable[..., torch.Tensor]:
    """DeepGEMM's dense FP8 GEMM over a per-tensor scale.

    It reads A's scale per token and B's scale per 128x128 block, so a per-tensor scale
    expands into both without changing the arithmetic.

    Raises:
        ValueError: When the row falls outside that path.
    """
    gemm = deepgemm_op("fp8_gemm_nt")
    align = deepgemm_op("get_mn_major_tma_aligned_tensor")

    scale_a, scale_b = inputs[2], inputs[3]
    if len(inputs) == 5:
        raise ValueError("DeepGEMM FP8 GEMM baseline does not support bias.")
    if scale_a.shape != (1, 1) or scale_b.shape != (1, 1):
        raise ValueError(
            "DeepGEMM FP8 GEMM baseline requires (1, 1) scales, "
            f"got {tuple(scale_a.shape)} and {tuple(scale_b.shape)}"
        )
    if workload.out_dtype != torch.bfloat16:
        raise ValueError("DeepGEMM FP8 GEMM baseline requires bfloat16 output.")
    if workload.n % 128 or workload.k % 128:
        raise ValueError(
            "DeepGEMM FP8 GEMM baseline requires n and k divisible by 128, "
            f"got n={workload.n} k={workload.k}"
        )

    m, n, k = workload.m, workload.n, workload.k
    aligned_scale_a = align(scale_a.expand(m, k // 128).contiguous())
    block_scale_b = scale_b.expand(n // 128, k // 128).contiguous()

    def run(a: torch.Tensor, b: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        out = torch.empty((m, n), dtype=workload.out_dtype, device=a.device)
        gemm((a, aligned_scale_a), (b, block_scale_b), out)
        return out

    return run


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


def _gemm_args(w: dict, dtype: torch.dtype) -> tuple:
    """``(m, n, k, trans_a, trans_b, dtype)``; the transposes default to N-T."""
    return (
        w["m"],
        w["n"],
        w["k"],
        bool(w.get("trans_a", False)),
        bool(w.get("trans_b", True)),
        dtype,
    )


def _gemm_fp8_args(w: dict, dtype: torch.dtype) -> tuple:
    """``(m, n, k, scale_mode, bias, dtype)``; ``bias_shape`` selects the bias epilogue."""
    return (w["m"], w["n"], w["k"], w["scale_mode"], bool(w.get("bias_shape")), dtype)


def _gemm_w4a16_args(w: dict, dtype: torch.dtype) -> tuple:
    return (w["m"], w["n"], w["k"], int(w.get("group_size", 128)), dtype)


@pytest.mark.parametrize(
    "m, n, k, trans_a, trans_b, dtype",
    workload_params(load_workloads(_OP_NAME), _gemm_args),
)
def test_gemm_bench(
    m: int,
    n: int,
    k: int,
    trans_a: bool,
    trans_b: bool,
    dtype: torch.dtype,
) -> None:
    workload = GemmBenchmarkWorkload(m, n, k, dtype, trans_a, trans_b)
    a, b = workload.gen_inputs()

    op = GemmFwdOp(trans_a=trans_a, trans_b=trans_b)
    bm = ManifestBenchmark(_OP_NAME, op, workload)

    # The benchmark framework warms up internally; eval_roofline() is read
    # lazily after profiling, by which point forward() has bound the dims.

    # Stronger baseline: cuBLASLt's fastest algorithm found by heuristic search
    # (falling back to the plain torch.matmul path when that is faster).
    # torch.matmul alone uses cuBLAS's default top-1 heuristic, measurably
    # slower than searchable algorithms on some shapes (small-M, tall-skinny,
    # awkward-N) — comparing against it can credit a kernel with a win a
    # cuBLASLt user would not concede. It goes in the same plan as the others
    # so every entry is timed in one forward-then-reversed pass: profiling it
    # separately would time it in whatever thermal state the algorithm search
    # left, against entries measured before that search ran. Absent from the
    # plan when cuBLASLt cannot be loaded, leaving the torch-cublas baseline.
    functors = {"tileops": op, "torch-cublas": workload.torch_matmul}
    best_fn = make_cublaslt_best(m, n, k, dtype, trans_a, trans_b)
    if best_fn is not None:
        functors["cublaslt-best"] = best_fn

    deepgemm_fn = _deepgemm_bf16_nt(workload, a, b)
    if deepgemm_fn is not None:
        assert_matches_reference(
            deepgemm_fn, workload.torch_matmul, a, b, **reference_tolerance(dtype)
        )
        functors[DEEPGEMM_TAG] = deepgemm_fn

    # flag_gems' mm takes row-major operands; a transposed row is its own layout,
    # which its kernel does not express, so those rows carry cuBLAS alone.
    if not trans_a and not trans_b:
        flaggems_mm = flaggems_op("mm")
        assert_matches_reference(
            flaggems_mm, workload.torch_matmul, a, b, **reference_tolerance(dtype)
        )
        functors[FLAGGEMS_TAG] = flaggems_mm

    bm.compare(functors, a, b, record_as=op, params=locals())


@pytest.mark.parametrize(
    "m, n, k, scale_mode, bias, dtype",
    workload_params(load_workloads(_FP8_OP_NAME), _gemm_fp8_args),
)
def test_gemm_fp8_bench(
    m: int,
    n: int,
    k: int,
    scale_mode: str,
    bias: bool,
    dtype: torch.dtype,
) -> None:
    out_dtype = torch.bfloat16
    workload = GemmFp8BenchmarkWorkload(m, n, k, dtype, scale_mode, out_dtype=out_dtype, bias=bias)
    inputs = workload.gen_inputs()

    op = GemmFp8FwdOp(out_dtype=out_dtype)
    bm = ManifestBenchmark(_FP8_OP_NAME, op, workload)

    if scale_mode not in ("per_tensor", "block128"):
        raise ValueError(f"unsupported FP8 GEMM scale_mode for benchmark: {scale_mode!r}")

    functors = {"tileops": op, "torch-scaled-mm": workload.torch_scaled_matmul}

    if scale_mode == "per_tensor":
        try:
            deepgemm_fn = _deepgemm_fp8_per_tensor(workload, *inputs)
        except ValueError as exc:
            print(f"  [skip] {DEEPGEMM_TAG}: {exc}")
        else:
            assert_matches_reference(
                deepgemm_fn,
                workload.torch_scaled_matmul,
                *inputs,
                **reference_tolerance(out_dtype),
            )
            functors[DEEPGEMM_TAG] = (deepgemm_fn, inputs[:2])

        unsupported_reason = _flashinfer_fp8_per_tensor_unsupported_reason(inputs[0].device)
        if unsupported_reason is not None:
            print(f"  [skip] flashinfer-mm-fp8: {unsupported_reason}")
        else:
            # Probe once and drop only the flashinfer row when it cannot run;
            # skipping would take the op's own numbers down with it.
            try:
                import flashinfer

                prepared_b, alpha = _prepare_flashinfer_fp8_per_tensor(workload, *inputs)

                def flashinfer_fn(a):
                    return flashinfer.mm_fp8(a, prepared_b, alpha, out_dtype=out_dtype)

                flashinfer_fn(inputs[0])
            except (ImportError, RuntimeError) as exc:
                print(f"  [skip] flashinfer-mm-fp8: {str(exc).splitlines()[0]}")
            else:
                functors["flashinfer-mm-fp8"] = (flashinfer_fn, (inputs[0],))
    else:
        try:
            import flashinfer  # noqa: F401
        except ImportError as exc:
            print(f"  [skip] flashinfer-fp8-blockscale-sm90: {exc}")
        else:
            functors["flashinfer-fp8-blockscale-sm90"] = (
                lambda *args: _flashinfer_fp8_blockscale_ref(workload, *args),
                inputs,
            )

    bm.compare(functors, *inputs, record_as=op, params=locals())


@pytest.mark.parametrize(
    "m, n, k, group_size, dtype", workload_params(load_workloads(_W4A16_OP_NAME), _gemm_w4a16_args)
)
def test_gemm_w4a16_bench(
    m: int,
    n: int,
    k: int,
    group_size: int,
    dtype: torch.dtype,
) -> None:
    workload = GemmW4A16BenchmarkWorkload(m, n, k, dtype, group_size=group_size)
    inputs = workload.gen_inputs()

    op = GemmW4A16FwdOp(group_size=group_size)
    bm = ManifestBenchmark(_W4A16_OP_NAME, op, workload)

    functors = {
        "tileops": op,
        "torch-dequantized-matmul": workload.torch_dequantized_matmul,
    }

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
            functors[f"marlin-{reduce_mode}"] = (marlin, marlin_inputs)

    bm.compare(functors, *inputs, record_as=op, params=locals())
