import contextlib
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
from tileops.kernels.gemm.fp8_1d2d import GemmFp81D2DKernel
from tileops.manifest import load_workloads
from tileops.ops import GemmFp8FwdOp, GemmFwdOp, GemmW4A16FwdOp
from workloads.gemm import GemmFp8Workload, GemmW4A16Workload, GemmWorkload

_W4A16_GROUP_SIZE = 128
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

    if k % 16 or k % _W4A16_GROUP_SIZE or n % 64:
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
        weight_scale.T.to(torch.float16).contiguous(),
        k,
        n,
        _W4A16_GROUP_SIZE,
    )
    zeros = marlin_zero_points(
        weight_zero.T.to(torch.int32).contiguous(),
        k // _W4A16_GROUP_SIZE,
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
    return (w["m"], w["n"], w["k"], int(w.get("group_size", _W4A16_GROUP_SIZE)), dtype)


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
        except ImportError as exc:
            print(f"  [skip] flashinfer-fp8-blockscale-sm90: {exc}")
        else:
            functors["flashinfer-fp8-blockscale-sm90"] = (
                lambda *args: _flashinfer_fp8_blockscale_ref(workload, *args),
                inputs,
            )

    bm.compare(functors, *inputs)


@pytest.mark.parametrize(
    "m,n,k",
    [
        pytest.param(128, 2112, 7168, id="ds-v3-decode-gate-up"),
        pytest.param(128, 7168, 2048, id="ds-v3-decode-down"),
        pytest.param(4096, 2112, 7168, id="ds-v3-prefill-gate-up"),
        pytest.param(4096, 7168, 2048, id="ds-v3-prefill-down"),
        pytest.param(4096, 4096, 7168, id="ds-v3-prefill-attn-proj"),
        pytest.param(4096, 7168, 16384, id="k-dominant-7168x16384"),
        pytest.param(4096, 24576, 1536, id="wide-n-24576"),
    ],
)
def test_gemm_fp8_1d2d_bench(m: int, n: int, k: int) -> None:
    """Fair 1D2D comparison: A 1x128 scales and B 128x128 scales."""
    from flashinfer.gemm import fp8_blockscale_gemm_sm90

    q = k // 128
    scale_n = (n + 127) // 128
    a = (torch.randn(m, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    scale_a = 0.5 + torch.rand(m, q, device="cuda")
    scale_a_k_major = scale_a.T.contiguous()
    scale_a_flashinfer = scale_a_k_major.view_as(scale_a)
    scale_b = 0.5 + torch.rand(scale_n, q, device="cuda")
    kernel = GemmFp81D2DKernel(m, n, k, torch.float8_e4m3fn, torch.bfloat16)

    def reference() -> torch.Tensor:
        return (
            (a.float() * scale_a.repeat_interleave(128, dim=1))
            @ (
                b.float() * scale_b.repeat_interleave(128, dim=0)[:n].repeat_interleave(128, dim=1)
            ).T
        ).to(torch.bfloat16)

    local = kernel(a, b, scale_a_k_major, scale_b)
    flashinfer = fp8_blockscale_gemm_sm90(
        a, b, scale_a_flashinfer, scale_b, out_dtype=torch.bfloat16
    )
    expected = reference()
    torch.testing.assert_close(local, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(flashinfer, expected, atol=2e-2, rtol=2e-2)
    local_ms = median_busy_ms(bench_kernel(lambda: kernel(a, b, scale_a_k_major, scale_b)))
    flashinfer_ms = median_busy_ms(
        bench_kernel(
            lambda: fp8_blockscale_gemm_sm90(
                a, b, scale_a_flashinfer, scale_b, out_dtype=torch.bfloat16
            )
        )
    )
    print(
        f"1D2D m={m} n={n} k={k}: tileops={local_ms:.5f} ms "
        f"flashinfer={flashinfer_ms:.5f} ms ratio={local_ms / flashinfer_ms:.3f}x"
    )


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

    if m == 1:
        for reduce_mode, use_fp32_reduce in (("fp32", True), ("fp16", False)):
            try:
                marlin, marlin_inputs = _prepare_marlin_w4a16_baseline(
                    m,
                    n,
                    k,
                    use_fp32_reduce,
                    *inputs,
                )
            except (ImportError, ModuleNotFoundError) as exc:
                print(f"  [skip] marlin-{reduce_mode}: {exc}")
                continue
            actual = marlin(*marlin_inputs)
            if actual.shape != (m, n) or not torch.isfinite(actual).all():
                raise RuntimeError("Marlin W4A16 baseline smoke check failed")
            # A baseline that does not reproduce the reference is dropped from
            # the comparison rather than compared against under a wrong layout.
            try:
                torch.testing.assert_close(actual, expected, atol=7e-2, rtol=5e-2)
            except AssertionError as exc:
                print(f"  [skip] marlin-{reduce_mode} disagrees with the reference: {exc}")
                continue
            torch.cuda.synchronize()
            functors[f"marlin-{reduce_mode}"] = (marlin, marlin_inputs)

    bm.compare(functors, *inputs)
