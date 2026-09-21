import contextlib
import functools
from typing import Any, Callable, Optional

import pytest
import torch

from benchmarks.baselines import (
    DEEPGEMM_TAG,
    FLAGGEMS_TAG,
    assert_matches_reference,
    deepgemm_op,
    flaggems_op,
    reference_tolerance,
)
from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from benchmarks.timing import bench_kernel, median_busy_ms
from tileops.kernels.gemm.w4a16 import GROUP_SIZE
from tileops.manifest import load_workloads
from tileops.ops import GemmFp8FwdOp, GemmFwdOp, GemmW4A16FwdOp
from workloads.gemm import (
    GemmFp8Workload,
    GemmW4A16Workload,
    GemmWorkload,
)

_FP8_BLOCK = 128


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
        scale_cols = -(-cols // _FP8_BLOCK)
        if tuple(scale.shape) != (rows, scale_cols):
            raise ValueError(f"unsupported FP8 scale shape {tuple(scale.shape)} for {(rows, cols)}")
        return scale.repeat_interleave(_FP8_BLOCK, dim=1)[:, :cols]

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
    if workload.k % _FP8_BLOCK != 0:
        raise ValueError(
            f"FlashInfer FP8 blockscale GEMM baseline requires k divisible by {_FP8_BLOCK}."
        )
    if scale_a.shape != (workload.m, workload.k // _FP8_BLOCK) or scale_b.shape != (
        workload.n,
        workload.k // _FP8_BLOCK,
    ):
        raise ValueError(
            "FlashInfer FP8 blockscale GEMM baseline requires exact "
            f"scale shapes {(workload.m, workload.k // _FP8_BLOCK)} "
            f"and {(workload.n, workload.k // _FP8_BLOCK)}, "
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
    if workload.n % _FP8_BLOCK or workload.k % _FP8_BLOCK:
        raise ValueError(
            f"DeepGEMM FP8 GEMM baseline requires n and k divisible by {_FP8_BLOCK}, "
            f"got n={workload.n} k={workload.k}"
        )

    m, n, k = workload.m, workload.n, workload.k
    aligned_scale_a = align(scale_a.expand(m, k // _FP8_BLOCK).contiguous())
    block_scale_b = scale_b.expand(n // _FP8_BLOCK, k // _FP8_BLOCK).contiguous()

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


# Relative to the output's own scale. Two implementations of the same
# dequantization differ by 1e-4 here; a baseline given the wrong weight layout
# differs by ~1.
_W4A16_BASELINE_MAX_DRIFT = 1e-2


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


def _prepare_machete_w4a16_baseline(
    m: int,
    n: int,
    k: int,
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero: torch.Tensor,
) -> tuple[Callable[..., torch.Tensor], tuple[Any, ...]]:
    """Machete, vLLM's CUTLASS mixed-input GEMM, on the same logical weights.

    Marlin is the faster of the two while the token count is small, and Machete
    takes over once the shape is a real GEMM, so the pair brackets the W4A16
    state of the art across this op's workloads.
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.machete_utils import (
        check_machete_supports_shape,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        pack_quantized_values_into_int32,
    )
    from vllm.scalar_type import scalar_types

    supported, reason = check_machete_supports_shape(k, n)
    if not supported:
        raise ValueError(f"Machete W4A16 baseline does not take this shape: {reason}")

    weight_type = scalar_types.uint4
    packed_i32 = packed_weight.to(torch.int32)
    logical_q = torch.stack((packed_i32 & 0xF, packed_i32 >> 4), dim=-1).reshape(n, k)
    # Machete reads B K-major with K packed into int32, through its own prepack.
    b_q = pack_quantized_values_into_int32(logical_q.T.contiguous(), weight_type, packed_dim=0)
    b_q = ops.machete_prepack_B(
        b_q.t().contiguous().t(),
        a_type=activation.dtype,
        b_type=weight_type,
        group_scales_type=activation.dtype,
    )
    scales = weight_scale.T.to(activation.dtype).contiguous()
    # Machete takes the zero point pre-scaled and negated: it adds this term
    # rather than subtracting the zero before the multiply.
    zeros = (-1.0 * scales * weight_zero.T.to(activation.dtype)).contiguous()

    def _run_machete(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor, z: torch.Tensor):
        return ops.machete_mm(
            a=a,
            b_q=b,
            b_type=weight_type,
            b_group_scales=s,
            b_group_zeros=z,
            b_group_size=GROUP_SIZE,
        )

    return _run_machete, (activation, b_q, scales, zeros)


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
    return (w["m"], w["n"], w["k"], int(w.get("group_size", GROUP_SIZE)), dtype)


@pytest.mark.parametrize(
    "m, n, k, trans_a, trans_b, dtype",
    workload_params(load_workloads(GemmFwdOp), _gemm_args),
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
    bm = ManifestBenchmark(op, workload)

    # The benchmark framework warms up internally; eval_roofline() is read
    # lazily after profiling, by which point forward() has bound the dims.

    functors = {"tileops": op, "torch-cublas": workload.torch_matmul}
    best_fn = cublaslt_best(a, b, trans_a=trans_a, trans_b=trans_b)
    if best_fn is not None:
        assert_matches_reference(best_fn, workload.torch_matmul, a, b, **reference_tolerance(dtype))
        functors[CUBLASLT_TAG] = best_fn

    deepgemm_fn = _deepgemm_bf16_nt(workload, a, b)
    if deepgemm_fn is not None:
        assert_matches_reference(
            deepgemm_fn, workload.torch_matmul, a, b, **reference_tolerance(dtype)
        )
        functors[DEEPGEMM_TAG] = deepgemm_fn

    if not trans_a and not trans_b:
        flaggems_mm = flaggems_op("mm")
        assert_matches_reference(
            flaggems_mm, workload.torch_matmul, a, b, **reference_tolerance(dtype)
        )
        functors[FLAGGEMS_TAG] = flaggems_mm

    bm.compare(functors, a, b)


@pytest.mark.parametrize(
    "m, n, k, scale_mode, bias, dtype",
    workload_params(load_workloads(GemmFp8FwdOp), _gemm_fp8_args),
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
    bm = ManifestBenchmark(op, workload)

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

            blockscale_fn = functools.partial(_flashinfer_fp8_blockscale_ref, workload)
            assert_matches_reference(
                blockscale_fn,
                workload.torch_scaled_matmul,
                *inputs,
                **reference_tolerance(out_dtype),
            )
        except (ImportError, ValueError) as exc:
            print(f"  [skip] flashinfer-fp8-blockscale-sm90: {str(exc).splitlines()[0]}")
        except AssertionError as exc:
            # Preferred, not selected: drop the tag rather than fail the row.
            print(
                "  [skip] flashinfer-fp8-blockscale-sm90: disagrees with the reference "
                f"({str(exc).splitlines()[0]})"
            )
        else:
            functors["flashinfer-fp8-blockscale-sm90"] = (blockscale_fn, inputs)

    bm.compare(functors, *inputs)


@pytest.mark.parametrize(
    "m, n, k, group_size, dtype", workload_params(load_workloads(GemmW4A16FwdOp), _gemm_w4a16_args)
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
    bm = ManifestBenchmark(op, workload)

    expected = workload.ref_program(*inputs)
    torch.testing.assert_close(op(*inputs), expected, atol=7e-2, rtol=5e-2)

    functors = {
        "tileops": op,
        "torch-dequantized-matmul": workload.torch_dequantized_matmul,
    }

    # Every arm reorders the same logical weight its own way, here, outside the
    # timed region. Both run on every row: Marlin leads at a few tokens and
    # Machete from a few dozen, so timing one of them reports a win the other
    # would have taken.
    logical = (inputs[0], workload.row_major_weight, inputs[2], inputs[3])
    candidates: list[tuple[str, Callable[..., Any]]] = [
        (f"marlin-{mode}", functools.partial(_prepare_marlin_w4a16_baseline, m, n, k, fp32))
        for mode, fp32 in (("fp32", True), ("fp16", False))
    ]
    candidates.append(("machete", functools.partial(_prepare_machete_w4a16_baseline, m, n, k)))

    for tag, prepare in candidates:
        try:
            baseline, baseline_inputs = prepare(*logical)
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            # A shape the baseline's packing cannot address drops its tag.
            print(f"  [skip] {tag}: {exc}")
            continue
        actual = baseline(*baseline_inputs)
        if actual.shape != (m, n) or not torch.isfinite(actual).all():
            raise RuntimeError(f"{tag} W4A16 baseline smoke check failed")
        # The question here is whether the baseline was handed the right weights,
        # not whether it is as accurate as this op, so the error is judged
        # against the output's own scale. A wrong layout reads as O(1); a
        # different accumulation order over a long K reads as O(1e-3). An
        # elementwise tolerance cannot tell them apart, because one output of a
        # long dot product lands near zero and no absolute bound survives it.
        drift = (actual.float() - expected.float()).abs().mean() / expected.float().abs().mean()
        if drift > _W4A16_BASELINE_MAX_DRIFT:
            print(f"  [skip] {tag} disagrees with the reference: mean error is {drift:.1%}")
            continue
        torch.cuda.synchronize()
        functors[tag] = (baseline, baseline_inputs)

    bm.compare(functors, *inputs)
