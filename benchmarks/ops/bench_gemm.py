"""Benchmark TileOPs GEMM, FP8 GEMM and W4A16 GEMM, one case per manifest call, against cuBLAS and the library kernels available for each."""

import contextlib
from typing import Any, Callable, Optional

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    DEEPGEMM_TAG,
    FLAGGEMS_TAG,
    QUACK_TAG,
    deepgemm_op,
    flaggems_op,
    flashinfer_op,
    quack_op,
)
from benchmarks.timing import bench_kernel, median_busy_ms
from tileops.kernels.gemm.w4a16 import GROUP_SIZE
from tileops.ops import GemmFP8FwdOp, GemmFwdOp, GemmW4A16FwdOp
from tileops.utils import get_sm_version
from workloads.gemm import GemmFP8Workload, GemmWorkload, dequantize_w4a16_weight

CUBLASLT_TAG = "cublaslt-best"


class _CublasLtBestGemm:
    """Callable bound to the fastest cuBLAS algorithm found for one shape.

    ``torch.matmul`` runs cuBLASLt's default heuristic, which under-uses split-K on
    small-m and awkward-n shapes; timing only against it credits wins a cuBLASLt user
    would not concede. Construction ranks the heuristic picks and ``torch.matmul``
    through ``bench_kernel``, so nothing wins selection on a timer the report does not
    use. The algorithm API comes from nvmath, which maps the ``libcublasLt`` torch
    already loaded and takes the caller's tensors, so the inputs stay byte-identical
    to the ``torch-cublas`` entry.

    Args:
        a: Left operand, ``[m, k]``.
        b: Right operand, ``[n, k]`` under NT or ``[k, n]`` under NN.
        trans_b: ``True`` for NT (``A @ Bᵀ``), ``False`` for NN (``A @ B``).

    Raises:
        RuntimeError: dtype unsupported, or the plan produced no algorithm.
    """

    WORKSPACE_BYTES = 256 * 1024 * 1024
    N_CANDIDATES = 8

    def __init__(self, a: torch.Tensor, b: torch.Tensor, trans_b: bool) -> None:
        from nvmath.linalg.advanced import (
            Matmul,
            MatmulComputeType,
            MatmulOptions,
            MatmulPlanPreferences,
        )

        if a.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(f"unsupported dtype {a.dtype}")
        self.trans_b = trans_b
        self._a = a
        self._b = b
        rhs = b.t() if trans_b else b

        self._mm = Matmul(
            a,
            rhs,
            options=MatmulOptions(
                compute_type=MatmulComputeType.COMPUTE_32F,
                memory_limit=self.WORKSPACE_BYTES,
            ),
        )
        algorithms = self._mm.plan(preferences=MatmulPlanPreferences(limit=self.N_CANDIDATES))
        if not algorithms:
            self._mm.free()
            raise RuntimeError("cuBLASLt returned no runnable algorithm")
        self.n_searched = len(algorithms)

        self._algorithm = min(
            algorithms,
            key=lambda al: median_busy_ms(bench_kernel(lambda: self._mm.execute(algorithm=al))),
        )
        torch_ms = median_busy_ms(bench_kernel(lambda: torch.matmul(a, rhs)))
        best_ms = median_busy_ms(bench_kernel(lambda: self._mm.execute(algorithm=self._algorithm)))
        self._use_torch = torch_ms < best_ms

    def free(self) -> None:
        """Release the plan and its 256 MB workspace, one per workload row."""
        mm = getattr(self, "_mm", None)
        if mm is not None:
            mm.free()
            self._mm = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.free()

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        rhs = b.t() if self.trans_b else b
        if a is not self._a or b is not self._b:
            self._mm.reset_operands(a=a, b=rhs)
            self._a, self._b = a, b
        if self._use_torch:
            return torch.matmul(a, rhs)
        return self._mm.execute(algorithm=self._algorithm)


def cublaslt_best(
    a: torch.Tensor, b: torch.Tensor, *, trans_a: bool, trans_b: bool
) -> Optional[Callable]:
    """Build a cuBLASLt searched-best GEMM callable, or None for a row it cannot take.

    Returns ``None`` — the caller keeps the plain ``torch-cublas`` baseline — when
    ``trans_a`` is True, the dtype is unsupported, or the plan produced no algorithm.
    A missing nvmath is not caught: it is a declared runner dependency, so its absence
    is a degraded image and has to fail the row rather than drop the tag.
    """
    if trans_a:
        return None
    try:
        return _CublasLtBestGemm(a, b, trans_b)
    except RuntimeError:
        return None


def _flashinfer_fp8_blockscale_1d2d(
    workload: GemmFP8Workload, *inputs: torch.Tensor
) -> Callable[..., torch.Tensor]:
    """FlashInfer's FP8 block-scale GEMM over 1D2D scales.

    It reads ``scale_a`` as M-contiguous rows padded to a multiple of 4, whatever the
    tensor's strides say, so the adapter lays the scales out that way under the
    ``[M, K/128]`` shape it checks.

    Raises:
        ValueError: When the row falls outside that path.
    """
    block_size = 128
    gemm = flashinfer_op("gemm.fp8_blockscale_gemm_sm90")
    if workload.k % block_size != 0:
        raise ValueError(f"FlashInfer FP8 blockscale GEMM requires k divisible by {block_size}.")

    def run(a, b, scale_a, scale_b, bias=None):
        m, scale_k = scale_a.shape
        padded_m = -(-m // 4) * 4
        m_major = torch.zeros((scale_k, padded_m), dtype=scale_a.dtype, device=scale_a.device)
        m_major[:, :m] = scale_a.T
        m_major_scale_a = torch.as_strided(m_major, (m, scale_k), (1, padded_m))
        out = gemm(a, b, m_major_scale_a, scale_b, out_dtype=torch.bfloat16)
        if bias is not None:
            out = out.float() + bias.float()
        return out.to(workload.out_dtype)

    return run


def _deepgemm_fp8(workload: GemmFP8Workload) -> Callable[..., torch.Tensor]:
    """FP8 GEMM with dynamic scale layout conversion and a full-precision epilogue."""
    gemm = deepgemm_op("fp8_gemm_nt")
    per_tensor = workload.scale_mode == "per_tensor"
    granularity = (
        1
        if workload.scale_mode == "block128" or workload.bias or workload.out_dtype == torch.float16
        else 128
    )
    intermediate_dtype = (
        torch.float32
        if workload.bias or workload.out_dtype == torch.float16 or granularity == 1
        else workload.out_dtype
    )

    def run(a, b, scale_a, scale_b, bias=None):
        if per_tensor:
            scale_a = scale_a.expand(a.shape[0], a.shape[1] // 128).contiguous()
            scale_b = scale_b.expand(
                (b.shape[0] + granularity - 1) // granularity, b.shape[1] // 128
            ).contiguous()
        elif granularity == 1 and workload.scale_mode == "block128x128":
            scale_b = scale_b.repeat_interleave(128, dim=0)[: b.shape[0]].contiguous()
        out = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=intermediate_dtype)
        accumulator = out.zero_() if granularity == 1 else None
        gemm((a, scale_a), (b, scale_b), out, c=accumulator, recipe=(1, granularity, 128))
        if bias is not None:
            out = out + bias.float()
        return out.to(workload.out_dtype)

    return run


def _deepgemm_bf16_nt(
    workload: GemmWorkload, a: torch.Tensor, b: torch.Tensor
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


def _flashinfer_fp8_per_tensor_unsupported_reason(device: torch.device) -> Optional[str]:
    arch = get_sm_version(device.index)
    if arch < 100:
        return (
            "TRTLLM low-latency GEMM requires Blackwell (sm100+), "
            f"but the current device is sm{arch}"
        )
    return None


def _prepare_marlin_w4a16_baseline(
    m: int,
    n: int,
    k: int,
    use_fp32_reduce: bool,
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero: torch.Tensor,
) -> tuple[Callable[..., torch.Tensor], tuple[Any, ...]]:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
        marlin_permute_scales,
        marlin_zero_points,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
        get_weight_perm,
        marlin_weights,
    )
    from vllm.scalar_type import scalar_types

    if not hasattr(torch.ops._C, "marlin_gemm"):
        raise RuntimeError("this vLLM build does not include marlin_gemm")
    if k % 16 or k % GROUP_SIZE or n % 64:
        raise ValueError("Marlin W4A16 benchmark requires K % 128 == 0 and N % 64 == 0")

    if tuple(activation.shape) != (m, k):
        raise ValueError(f"activation must have shape {(m, k)}, got {tuple(activation.shape)}")
    if tuple(packed_weight.shape) != (n, k // 2):
        raise ValueError(
            f"packed_weight must have shape {(n, k // 2)}, got {tuple(packed_weight.shape)}"
        )

    # TileOPs packs the two adjacent K values into each byte. Reconstruct the
    # exact logical q[N,K], transpose to Marlin's K-major convention, then use
    # vLLM's official Marlin test-layout and metadata permutation helpers.
    packed_i32 = packed_weight.to(torch.int32)
    logical_q = torch.stack(
        (packed_i32 & 0xF, packed_i32 >> 4),
        dim=-1,
    ).reshape(n, k)
    qweight = marlin_weights(
        logical_q.T.contiguous(),
        k,
        n,
        4,
        get_weight_perm(4),
    )
    scales = marlin_permute_scales(
        weight_scale.T.to(activation.dtype).contiguous(),
        k,
        n,
        GROUP_SIZE,
    )
    zeros = marlin_zero_points(
        weight_zero.T.to(torch.int32).contiguous(),
        k // GROUP_SIZE,
        n,
        4,
    )
    workspace = marlin_make_workspace_new(activation.device)

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


@pytest.mark.parametrize("case", bench.cases(GemmFwdOp), ids=lambda case: case.id)
def test_gemm_bench(case) -> None:
    workload = case.workload
    a, b = case.inputs
    trans_a, trans_b = workload.trans_a, workload.trans_b

    op = GemmFwdOp(**case.arguments)

    implementations = {"torch-cublas": case.reference}
    best_fn = cublaslt_best(a, b, trans_a=trans_a, trans_b=trans_b)
    if best_fn is not None:
        implementations[CUBLASLT_TAG] = best_fn

    deepgemm_fn = _deepgemm_bf16_nt(workload, a, b)
    if deepgemm_fn is not None:
        implementations[DEEPGEMM_TAG] = deepgemm_fn

    if not trans_a and not trans_b:
        flaggems_mm = flaggems_op("mm")
        implementations[FLAGGEMS_TAG] = flaggems_mm

    quack_gemm = quack_op("gemm", "quack.gemm_interface")

    def quack_fn(a, b):
        return quack_gemm(a.T if trans_a else a, b.T if trans_b else b)

    implementations[QUACK_TAG] = quack_fn
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(GemmFP8FwdOp), ids=lambda case: case.id)
def test_gemm_fp8_bench(case) -> None:
    workload = case.workload
    inputs = case.inputs
    scale_mode, out_dtype = (workload.scale_mode, workload.out_dtype)
    op = GemmFP8FwdOp(**case.arguments)
    implementations = {"torch-fp32-ref": case.reference}
    implementations[DEEPGEMM_TAG] = _deepgemm_fp8(workload)
    if scale_mode == "per_tensor":

        def scaled_mm(a, b, scale_a, scale_b, bias=None):
            return torch._scaled_mm(
                a, b.T, scale_a=scale_a, scale_b=scale_b, bias=bias, out_dtype=out_dtype
            )

        implementations["torch-scaled-mm"] = scaled_mm
        unsupported_reason = _flashinfer_fp8_per_tensor_unsupported_reason(inputs[0].device)
        if unsupported_reason is not None:
            print(f"  [skip] flashinfer-mm-fp8: {unsupported_reason}")
        else:
            import flashinfer

            prepared_b = flashinfer.prepare_low_latency_gemm_weights(inputs[1], {})

            def flashinfer_fn(a, b, scale_a, scale_b, bias=None):
                alpha = (scale_a * scale_b).reshape(())
                out = flashinfer.mm_fp8(a, prepared_b, alpha, out_dtype=torch.bfloat16)
                if bias is not None:
                    out = out.float() + bias.float()
                return out.to(out_dtype)

            implementations["flashinfer-mm-fp8"] = flashinfer_fn
    elif scale_mode == "block128x128":
        baselines = {"flashinfer-fp8-blockscale-sm90": _flashinfer_fp8_blockscale_1d2d}
        for tag, adapter in baselines.items():
            try:
                fn = adapter(workload, *inputs)
            except ValueError as exc:
                print(f"  [skip] {tag}: {str(exc).splitlines()[0]}")
            else:
                implementations[tag] = bench.Implementation(run=fn, args=inputs)
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(GemmW4A16FwdOp), ids=lambda case: case.id)
def test_gemm_w4a16_bench(case) -> None:
    workload = case.workload
    inputs = case.inputs
    m, n, k = workload.m, workload.n, workload.k

    op = GemmW4A16FwdOp(**case.arguments)

    # The torch row multiplies by a weight dequantized once, outside the timed call.
    weight = dequantize_w4a16_weight(*inputs[1:]).to(workload.dtype)

    def torch_dequantized_matmul(activation: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        return torch.matmul(activation, weight.T)

    implementations = {"torch-dequantized-matmul": torch_dequantized_matmul}

    logical = (inputs[0], workload.row_major_weight, inputs[2], inputs[3])
    for mode, use_fp32_reduce in (("fp32", True), ("fp16", False)):
        tag = f"marlin-{mode}"
        try:
            baseline, baseline_inputs = _prepare_marlin_w4a16_baseline(
                m, n, k, use_fp32_reduce, *logical
            )
        except ValueError as exc:
            print(f"  [skip] {tag}: {exc}")
            continue
        implementations[tag] = bench.Implementation(run=baseline, args=baseline_inputs)

    bench.Runner(op, case).compare(implementations)
