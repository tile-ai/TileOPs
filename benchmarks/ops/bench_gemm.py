"""Benchmark TileOPs GEMM, FP8 GEMM, INT8 W8A8 GEMM, W4A8 GEMM and W4A16 GEMM, one case per manifest call, against cuBLAS and the library kernels available for each."""

import contextlib
from typing import Any, Callable, Optional

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    DEEPGEMM_TAG,
    FLAGGEMS_TAG,
    QUACK_TAG,
    VLLM_TAG,
    baseline_module,
    deepgemm_op,
    flaggems_op,
    flashinfer_op,
    quack_op,
    vllm_op,
)
from benchmarks.timing import bench_kernel, median_busy_ms
from tileops.ops import (
    GemmFP8FwdOp,
    GemmFwdOp,
    GemmINT8W8A8FwdOp,
    GemmW4A8FwdOp,
    GemmW4A16FwdOp,
)
from tileops.utils import get_sm_version
from workloads.gemm import (
    GemmFP8Workload,
    GemmWorkload,
    dequantize_w4a16_weight,
    matmul_fp32_reduction,
)

CUBLASLT_TAG = "cublaslt-best"


class _CublasLtBestGemm:
    """Callable bound to the fastest cuBLAS algorithm found for one shape.

    Construction ranks cuBLASLt's candidate algorithms and a PyTorch fallback using
    the benchmark timer. Calls reuse the selected algorithm with the caller's operands.

    Args:
        a: Left operand, ``[m, k]``.
        b: Right operand, ``[n, k]`` under NT or ``[k, n]`` under NN.
        trans_b: ``True`` for NT (``A @ Bᵀ``), ``False`` for NN (``A @ B``).
        advanced: ``nvmath.linalg.advanced``, resolved by the caller.
        cublaslt: ``nvmath.bindings.cublasLt``, resolved by the caller.

    Raises:
        RuntimeError: dtype unsupported, or the plan produced no algorithm.
    """

    WORKSPACE_BYTES = 256 * 1024 * 1024
    N_CANDIDATES = 8

    def __init__(
        self, a: torch.Tensor, b: torch.Tensor, trans_b: bool, advanced: Any, cublaslt: Any
    ) -> None:
        if a.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(f"unsupported dtype {a.dtype}")
        self.trans_b = trans_b
        self._a = a
        self._b = b
        rhs = b.t() if trans_b else b

        self._mm = advanced.Matmul(
            a,
            rhs,
            options=advanced.MatmulOptions(
                compute_type=advanced.MatmulComputeType.COMPUTE_32F,
                memory_limit=self.WORKSPACE_BYTES,
            ),
        )
        # GemmFwdOp and the workload's reference accumulate in fp32, so a split-K algorithm
        # may reduce its partials only in the compute type: an in-place reduction sums them
        # in the 16-bit output.
        algorithms = self._mm.plan(
            preferences=advanced.MatmulPlanPreferences(
                limit=self.N_CANDIDATES,
                reduction_scheme_mask=cublaslt.ReductionScheme.COMPUTE_TYPE,
            )
        )
        if not algorithms:
            self._mm.free()
            raise RuntimeError("cuBLASLt returned no runnable algorithm")
        self.n_searched = len(algorithms)

        self._algorithm = min(
            algorithms,
            key=lambda al: median_busy_ms(bench_kernel(lambda: self._mm.execute(algorithm=al))),
        )
        torch_ms = median_busy_ms(bench_kernel(lambda: matmul_fp32_reduction(a, rhs)))
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
            return matmul_fp32_reduction(a, rhs)
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
    advanced = baseline_module("nvmath.linalg.advanced", "nvmath")
    cublaslt = baseline_module("nvmath.bindings.cublasLt", "nvmath")
    try:
        return _CublasLtBestGemm(a, b, trans_b, advanced, cublaslt)
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
    if get_sm_version() != 90:
        raise ValueError("FlashInfer FP8 blockscale GEMM requires SM90")
    if workload.n <= 0 or workload.n % 64:
        raise ValueError("FlashInfer FP8 blockscale GEMM requires positive N divisible by 64")
    gemm = flashinfer_op("gemm.fp8_blockscale_gemm_sm90")
    if workload.k <= 0 or workload.k % block_size != 0:
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

    arch = get_sm_version()
    output_alignment = 16 // torch.empty((), dtype=intermediate_dtype).element_size()
    if not (90 <= arch < 110) or workload.k <= 0 or workload.n <= 0:
        raise ValueError("DeepGEMM FP8 requires SM9x/SM10x and positive N/K")
    # TMA descriptors require 16-byte row strides for operands and the intermediate.
    if workload.k % 16 or workload.n % output_alignment:
        raise ValueError(
            f"DeepGEMM FP8 requires K divisible by 16 and N divisible by {output_alignment}"
        )
    gemm = deepgemm_op("fp8_gemm_nt")

    def run(a, b, scale_a, scale_b, bias=None):
        if per_tensor:
            scale_a = scale_a.expand(a.shape[0], (a.shape[1] + 127) // 128).contiguous()
            scale_b = scale_b.expand(
                (b.shape[0] + granularity - 1) // granularity, (b.shape[1] + 127) // 128
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

    It reads both operands K-major and bf16 only, which is the N-T layout here, and
    stores the output through TMA, which needs a whole number of 16-byte units a row:
    positive N and K divisible by 8 provide 16-byte output and operand row strides.
    """
    if (
        workload.trans_a
        or not workload.trans_b
        or a.dtype != torch.bfloat16
        or workload.n <= 0
        or workload.k <= 0
        or workload.n % 8
        or workload.k % 8
        or not 90 <= get_sm_version() < 110
    ):
        print("  [skip] deepgemm: dense BF16 NT needs SM9x/SM10x, positive N/K and N/K % 8 == 0")
        return None

    gemm = deepgemm_op("bf16_gemm_nt")
    m, n = workload.m, workload.n

    def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        gemm(a, b, out)
        return out

    return run


def _flashinfer_fp8_per_tensor_unsupported_reason(
    device: torch.device, n: int, k: int
) -> Optional[str]:
    if device.type != "cuda":
        return f"flashinfer runs on cuda, not {device}"
    arch = get_sm_version(device.index)
    if not 100 <= arch < 110:
        return f"TRTLLM low-latency GEMM requires SM10x, but the current device is sm{arch}"
    if n <= 0 or n % 32 or k <= 0 or k % 128:
        return "TRTLLM FP8 weight preparation requires positive N % 32 == 0 and K % 128 == 0"
    return None


def _quack_gemm_unsupported_reason(workload: GemmWorkload) -> Optional[str]:
    """Quack's non-unit operand strides and output row stride must be 16-byte aligned."""
    if not workload.m or not workload.n or not workload.k:
        return None
    alignment = 16 // torch.empty((), dtype=workload.dtype).element_size()
    a_stride = workload.m if workload.trans_a else workload.k
    b_stride = workload.k if workload.trans_b else workload.n
    if any(stride % alignment for stride in (a_stride, b_stride, workload.n)):
        return f"Quack requires operand/output leading strides divisible by {alignment}"
    return None


def _marlin_w4a16_unsupported_reason(n: int, k: int, group_size: int) -> Optional[str]:
    """Marlin's positive group sizes and aligned affine INT4 matrix domain."""
    if get_sm_version() < 80:
        return "Marlin W4A16 requires compute capability at least 80"
    # vLLM marlin_utils.MARLIN_SUPPORTED_GROUP_SIZES, excluding per-channel -1.
    if group_size not in (32, 64, 128):
        return f"Marlin W4A16 supports positive group sizes 32, 64 or 128, got {group_size}"
    if n <= 0 or n % 64 or k <= 0 or k % 128 or k % group_size:
        return f"Marlin W4A16 needs positive N % 64 == 0 and K % 128 == 0, got N={n}, K={k}"
    return None


def _prepare_marlin_w4a16_baseline(
    m: int,
    n: int,
    k: int,
    group_size: int,
    use_fp32_reduce: bool,
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero: torch.Tensor,
) -> tuple[Callable[..., torch.Tensor], tuple[Any, ...]]:
    marlin_gemm = vllm_op("marlin_gemm")
    marlin_utils = "model_executor.layers.quantization.utils.marlin_utils"
    marlin_make_workspace_new = vllm_op("marlin_make_workspace_new", marlin_utils)
    marlin_permute_scales = vllm_op("marlin_permute_scales", marlin_utils)
    marlin_zero_points = vllm_op("marlin_zero_points", marlin_utils)
    get_weight_perm = vllm_op("get_weight_perm", f"{marlin_utils}_test")
    marlin_weights = vllm_op("marlin_weights", f"{marlin_utils}_test")
    scalar_types = vllm_op("scalar_types", "scalar_type")

    if not hasattr(torch.ops._C, "marlin_gemm"):
        raise RuntimeError("this vLLM build does not include marlin_gemm")
    reason = _marlin_w4a16_unsupported_reason(n, k, group_size)
    if reason is not None:
        raise ValueError(reason)

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
        group_size,
    )
    zeros = marlin_zero_points(
        weight_zero.T.to(torch.int32).contiguous(),
        k // group_size,
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
        return marlin_gemm(
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


def _vllm_w4a8_unsupported_reason(n: int, k: int, group_size: int) -> Optional[str]:
    """Why vLLM's ``cutlass_w4a8_mm`` cannot take this call, or ``None`` when it can.

    CUTLASS's INT4 reorder copies a K row in groups of 128 threads times two nibbles
    without a tail predicate (``mixed_dtype_utils.hpp::reorder_tensor``). The GEMM also
    requires SM90, group size 128 and N divisible by 128.
    """
    if get_sm_version() != 90:
        return "CUTLASS W4A8 requires compute capability 90"
    if group_size != 128:
        return f"CUTLASS W4A8 supports group_size 128 only, got {group_size}"
    if n % 128:
        return f"CUTLASS W4A8 needs N divisible by 128, got N={n}"
    if k % 256:
        return f"CUTLASS INT4 reorder needs K divisible by 256, got K={k}"
    return None


def _prepare_vllm_w4a8_baseline(
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_group_scale: torch.Tensor,
    weight_channel_scale: torch.Tensor,
    activation_scale: torch.Tensor,
    group_size: int,
) -> tuple[Callable[..., torch.Tensor], tuple[Any, ...]]:
    """vLLM's ``cutlass_w4a8_mm`` on operands laid out as vLLM lays them out at load time.

    ``CutlassW4A8LinearKernel.process_weights_after_loading`` (vllm 0.27.1) reorders the
    column-major ``[K/8, N]`` int32 packing through ``cutlass_encode_and_reorder_int4b`` and
    the ``[K/group, N]`` fp8 group scales through ``cutlass_pack_scale_fp8``; its
    ``apply_weights`` passes channel scales ``[1, N]`` and token scales ``[M, 1]``. Both
    reorders run here, outside the timed call.
    """
    m, n = activation.shape[0], packed_weight.shape[0]
    reason = _vllm_w4a8_unsupported_reason(n, activation.shape[1], group_size)
    if reason is not None:
        raise ValueError(reason)
    # The row-major bytes read as int32 put K 8j + i in bits 4i..4i+3 of word j, the
    # signed INT4 packing the reorder takes; transposed, the [N, K/8] buffer is the
    # column-major [K/8, N] operand.
    packed_words = packed_weight.view(torch.int32)
    weight = vllm_op("cutlass_encode_and_reorder_int4b")(packed_words.t())
    group_scale = vllm_op("cutlass_pack_scale_fp8")(weight_group_scale.t().contiguous())
    cutlass_w4a8_mm = vllm_op("cutlass_w4a8_mm")

    def run(a, b_q, b_group_scales, b_channel_scales, a_token_scales):
        return cutlass_w4a8_mm(
            a=a,
            b_q=b_q,
            b_group_scales=b_group_scales,
            b_group_size=group_size,
            b_channel_scales=b_channel_scales,
            a_token_scales=a_token_scales,
        )

    return run, (
        activation,
        weight,
        group_scale,
        weight_channel_scale.view(1, n),
        activation_scale.view(m, 1),
    )


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

    quack_reason = _quack_gemm_unsupported_reason(workload)
    if quack_reason is None:
        quack_gemm = quack_op("gemm", "quack.gemm_interface")

        def quack_fn(a, b):
            return quack_gemm(a.T if trans_a else a, b.T if trans_b else b)

        implementations[QUACK_TAG] = quack_fn
    else:
        print(f"  [skip] {QUACK_TAG}: {quack_reason}")
    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(GemmFP8FwdOp), ids=lambda case: case.id)
def test_gemm_fp8_bench(case) -> None:
    workload = case.workload
    inputs = case.inputs
    scale_mode, out_dtype = (workload.scale_mode, workload.out_dtype)
    op = GemmFP8FwdOp(**case.arguments)
    implementations = {"torch-fp32-ref": case.reference}
    try:
        implementations[DEEPGEMM_TAG] = _deepgemm_fp8(workload)
    except ValueError as exc:
        print(f"  [skip] {DEEPGEMM_TAG}: {exc}")
    if scale_mode == "per_tensor":

        def scaled_mm(a, b, scale_a, scale_b, bias=None):
            return torch._scaled_mm(
                a, b.T, scale_a=scale_a, scale_b=scale_b, bias=bias, out_dtype=out_dtype
            )

        if workload.k % 16 or workload.n % 16:
            print("  [skip] torch-scaled-mm: CUDA scaled_mm requires K/N divisible by 16")
        else:
            implementations["torch-scaled-mm"] = scaled_mm
        unsupported_reason = _flashinfer_fp8_per_tensor_unsupported_reason(
            inputs[0].device, workload.n, workload.k
        )
        if unsupported_reason is not None:
            print(f"  [skip] flashinfer-mm-fp8: {unsupported_reason}")
        else:
            flashinfer = baseline_module("flashinfer", "flashinfer")

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


def _vllm_scaled_mm_unsupported_reason(n: int, k: int) -> Optional[str]:
    """Why ``cutlass_scaled_mm`` would not run its CUTLASS kernel here, or ``None``.

    Off N and K multiples of 16 it silently runs a Triton kernel instead, and the row would
    time that one.
    """
    if n % 16 or k % 16:
        return f"CUTLASS scaled_mm needs N and K divisible by 16, got N={n}, K={k}"
    return None


@pytest.mark.parametrize("case", bench.cases(GemmINT8W8A8FwdOp), ids=lambda case: case.id)
def test_gemm_int8_w8a8_bench(case) -> None:
    """Compare INT8 activations times INT8 weights, dequantized per token and per channel."""
    workload = case.workload
    op = GemmINT8W8A8FwdOp(**case.arguments)
    implementations = {}
    reason = _vllm_scaled_mm_unsupported_reason(workload.n, workload.k)
    if reason is not None:
        print(f"  [skip] {VLLM_TAG}: {reason}")
    else:
        cutlass_scaled_mm = vllm_op("cutlass_scaled_mm")
        out_dtype = workload.out_dtype

        # vLLM takes the weight as a column-major [K, N] operand: b's transpose, not a copy.
        def vllm_fn(a, b, scale_a, scale_b, bias=None):
            return cutlass_scaled_mm(a, b.t(), scale_a, scale_b, out_dtype, bias)

        implementations[VLLM_TAG] = vllm_fn
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
                m, n, k, workload.group_size, use_fp32_reduce, *logical
            )
        except ValueError as exc:
            print(f"  [skip] {tag}: {exc}")
            continue
        implementations[tag] = bench.Implementation(run=baseline, args=baseline_inputs)

    bench.Runner(op, case).compare(implementations)


@pytest.mark.parametrize("case", bench.cases(GemmW4A8FwdOp), ids=lambda case: case.id)
def test_gemm_w4a8_bench(case) -> None:
    """Compare FP8 e4m3 activations times signed INT4 weights, producing BF16 output.

    Every W4A8 row uses FP8 e4m3 group scales and FP32 channel/token scales.
    """
    workload = case.workload
    op = GemmW4A8FwdOp(**case.arguments)

    implementations = {}
    reason = _vllm_w4a8_unsupported_reason(workload.n, workload.k, workload.group_size)
    if reason is not None:
        print(f"  [skip] {VLLM_TAG}: {reason}")
    else:
        baseline, baseline_inputs = _prepare_vllm_w4a8_baseline(*case.inputs, workload.group_size)
        implementations[VLLM_TAG] = bench.Implementation(run=baseline, args=baseline_inputs)

    bench.Runner(op, case).compare(implementations)
