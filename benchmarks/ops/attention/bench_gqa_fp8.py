import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pytest
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[3]))

from benchmarks.benchmark_base import BenchmarkReport, bench_kernel
from tileops.kernels.attention import (
    GQAFwdFP8Fa3ContractKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224Kernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedPreRescaleKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaEmitterK224Kernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaSharedVKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapLocalPKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceBarrierKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel,
    GQAFwdFP8Fa3ContractPtxAccBN224WsKernel,
    GQAFwdFP8Fa3ContractPtxAccDirectStoreKernel,
    GQAFwdFP8Fa3ContractPtxAccFa3EpilogueStoreKernel,
    GQAFwdFP8Fa3ContractPtxAccFa3EpilogueReuseVSmemKernel,
    GQAFwdFP8Fa3ContractPtxAccKernel,
    GQAFwdFP8WgmmaKernel,
    GQAFwdFP8WsPersistentKernel,
)
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionPrefillFP8TensorCoreFwdOp

_FP8_TC_OP_NAME = "GroupedQueryAttentionPrefillFP8TensorCoreFwdOp"


@dataclass(frozen=True)
class GQAFp8BenchCase:
    batch: int
    seq_len: int
    heads: int
    heads_kv: int
    dim: int = 128
    out_dtype: torch.dtype = torch.bfloat16
    label: str = ""

    @property
    def name(self) -> str:
        if self.label:
            return self.label
        return f"b{self.batch}_s{self.seq_len}_h{self.heads}_hkv{self.heads_kv}_d{self.dim}"

    def flops(self) -> float:
        # Non-causal dense prefill: QK and PV.
        return 4.0 * self.batch * self.heads * self.seq_len * self.seq_len * self.dim


def _manifest_fp8_tensor_core_cases() -> list[GQAFp8BenchCase]:
    cases: list[GQAFp8BenchCase] = []
    for workload in load_workloads(_FP8_TC_OP_NAME):
        batch, seq_len, heads, dim = workload["q_shape"]
        _, _, heads_kv, _ = workload["kv_shape"]
        for dtype_name in workload["dtypes"]:
            out_dtype = getattr(torch, dtype_name)
            op = GroupedQueryAttentionPrefillFP8TensorCoreFwdOp(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                seq_len=seq_len,
                dim=dim,
                is_causal=workload.get("is_causal", False),
                dtype=out_dtype,
            )
            op.eval_roofline()
            cases.append(
                GQAFp8BenchCase(
                    batch=batch,
                    seq_len=seq_len,
                    heads=heads,
                    heads_kv=heads_kv,
                    dim=dim,
                    out_dtype=out_dtype,
                    label=f"{workload['label']}-{dtype_name}",
                ))
    return cases


def _block_quant_128(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, dim = x.shape
    x_blocked = x.reshape(batch, seq_len // 128, 128, heads,
                          dim).permute(0, 3, 1, 2, 4).contiguous()
    amax = x_blocked.abs().amax(dim=(3, 4)).clamp(min=1e-4)
    scale = amax / 448.0
    x_fp8 = torch.clamp(x_blocked / scale[..., None, None], -448.0,
                        448.0).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.permute(0, 2, 3, 1, 4).reshape(batch, seq_len, heads, dim).contiguous()
    return x_fp8, scale.contiguous().float()


def _quantize_kv_fa3_scale(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, dim = x.shape
    descale = x.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0
    x_fp8 = torch.clamp(x / descale[:, None, :, None], -448.0,
                        448.0).to(torch.float8_e4m3fn).contiguous()
    tileops_scale = descale[:, :, None].expand(batch, heads, seq_len // 128).contiguous()
    return x_fp8, tileops_scale.float(), descale.float().contiguous()


def _quantize_q_fa3_gqa_scale(
    x: torch.Tensor,
    heads_kv: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, heads, dim = x.shape
    group_size = heads // heads_kv
    x_grouped = x.reshape(batch, seq_len, heads_kv, group_size, dim)
    descale = x_grouped.abs().amax(dim=(1, 3, 4)).clamp(min=1e-4) / 448.0
    x_fp8 = torch.clamp(x_grouped / descale[:, None, :, None, None], -448.0,
                        448.0).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.reshape(batch, seq_len, heads, dim).contiguous()
    q_head_scale = descale.repeat_interleave(group_size, dim=1)
    tileops_scale = q_head_scale[:, :, None].expand(batch, heads, seq_len // 128).contiguous()
    return x_fp8, tileops_scale.float(), descale.float().contiguous()


def _make_inputs(
    case: GQAFp8BenchCase,
    *,
    scale_mode: str = "fa3",
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    shape_q = (case.batch, case.seq_len, case.heads, case.dim)
    shape_kv = (case.batch, case.seq_len, case.heads_kv, case.dim)
    q = torch.randn(shape_q, device="cuda", dtype=torch.float16) * 0.25
    k = torch.randn(shape_kv, device="cuda", dtype=torch.float16) * 0.25
    v = torch.randn(shape_kv, device="cuda", dtype=torch.float16) * 0.25
    if scale_mode == "block128":
        q_fp8, q_scale = _block_quant_128(q)
        k_fp8, k_scale = _block_quant_128(k)
        v_fp8, v_scale = _block_quant_128(v)
        return (q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale), ()
    if scale_mode != "fa3":
        raise ValueError(f"unknown scale_mode={scale_mode!r}")

    q_fp8, q_scale, q_descale = _quantize_q_fa3_gqa_scale(q, case.heads_kv)
    k_fp8, k_scale, k_descale = _quantize_kv_fa3_scale(k)
    v_fp8, v_scale, v_descale = _quantize_kv_fa3_scale(v)
    tileops_args = (q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale)
    fa3_args = (q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale)
    return tileops_args, fa3_args


def _fa3_fp8_gqa_fwd() -> Optional[Callable]:
    try:
        from flash_attn_interface import flash_attn_func  # noqa: PLC0415
    except ImportError:
        return None

    def baseline_fn(q, k, v, q_descale, k_descale, v_descale):
        out = flash_attn_func(
            q,
            k,
            v,
            causal=False,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return out[0] if isinstance(out, tuple) else out

    return baseline_fn


def _bench_one(
    case: GQAFp8BenchCase,
    *,
    ws_persistent: bool = False,
    use_register_p: bool,
    scale_mode: str = "fa3",
    n_warmup: int = 5,
    n_repeat: int = 20,
    n_trials: int = 3,
) -> dict[str, float | str | bool | int]:
    inputs, _ = _make_inputs(case, scale_mode=scale_mode)
    kernel_cls = GQAFwdFP8WsPersistentKernel if ws_persistent else GQAFwdFP8WgmmaKernel
    kernel = kernel_cls(
        case.batch,
        case.heads,
        case.heads_kv,
        case.seq_len,
        case.dim,
        case.out_dtype,
        config={"use_register_p": use_register_p},
    )

    # Compile before timing.
    kernel(*inputs)
    torch.cuda.synchronize()

    latency_ms = bench_kernel(
        kernel,
        args=inputs,
        n_warmup=n_warmup,
        n_repeat=n_repeat,
        n_trials=n_trials,
    )
    tflops = case.flops() / latency_ms * 1e-9
    family = "tileops_ws" if ws_persistent else "tileops_wgmma"
    p_path = "register_p" if use_register_p else "shared_p"
    return {
        "case": case.name,
        "impl": f"{family}_{p_path}",
        "seq_len": case.seq_len,
        "heads": case.heads,
        "heads_kv": case.heads_kv,
        "ws_persistent": ws_persistent,
        "use_register_p": use_register_p,
        "latency_ms": latency_ms,
        "tflops": tflops,
    }


def _bench_ws_config(
    case: GQAFp8BenchCase,
    *,
    impl: str,
    config: dict,
    scale_mode: str = "fa3",
    n_warmup: int = 5,
    n_repeat: int = 20,
    n_trials: int = 3,
) -> dict[str, float | str | bool | int]:
    inputs, _ = _make_inputs(case, scale_mode=scale_mode)
    kernel = GQAFwdFP8WsPersistentKernel(
        case.batch,
        case.heads,
        case.heads_kv,
        case.seq_len,
        case.dim,
        case.out_dtype,
        config=config,
    )

    kernel(*inputs)
    torch.cuda.synchronize()

    latency_ms = bench_kernel(
        kernel,
        args=inputs,
        n_warmup=n_warmup,
        n_repeat=n_repeat,
        n_trials=n_trials,
    )
    return {
        "case": case.name,
        "impl": impl,
        "seq_len": case.seq_len,
        "heads": case.heads,
        "heads_kv": case.heads_kv,
        "ws_persistent": True,
        "latency_ms": latency_ms,
        "tflops": case.flops() / latency_ms * 1e-9,
    }


_LLAMA_CASES = {
    "llama8b-1k": GQAFp8BenchCase(1, 1024, 32, 8, label="llama8b-1k"),
    "llama8b-4k": GQAFp8BenchCase(1, 4096, 32, 8, label="llama8b-4k"),
    "llama8b-8k": GQAFp8BenchCase(1, 8192, 32, 8, label="llama8b-8k"),
    "llama8b-32k": GQAFp8BenchCase(1, 32768, 32, 8, label="llama8b-32k"),
    "llama8b-128k": GQAFp8BenchCase(1, 131072, 32, 8, label="llama8b-128k"),
    "llama70b-4k": GQAFp8BenchCase(1, 4096, 64, 8, label="llama70b-4k"),
    "llama405b-4k": GQAFp8BenchCase(1, 4096, 128, 8, label="llama405b-4k"),
}

_PARAMS = [
    pytest.param(_LLAMA_CASES["llama8b-1k"], id="llama8b-1k"),
    pytest.param(_LLAMA_CASES["llama8b-4k"], id="llama8b-4k"),
    pytest.param(_LLAMA_CASES["llama70b-4k"], id="llama70b-4k"),
]


@pytest.mark.parametrize("case", _PARAMS)
def test_gqa_fp8_bench(case: GQAFp8BenchCase) -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch fp8 is unavailable")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires Hopper FP8 WGMMA")
    for ws_persistent, use_register_p in ((False, False), (False, True), (True, False)):
        result = _bench_one(case, ws_persistent=ws_persistent, use_register_p=use_register_p)
        BenchmarkReport.record(
            "gqa_fp8_wgmma",
            {"case": case.name, "impl": result["impl"]},
            result,
            tag=str(result["impl"]),
        )

    _tileops_args, fa3_args = _make_inputs(case, scale_mode="fa3")
    fa3_fn = _fa3_fp8_gqa_fwd()
    if fa3_fn is not None:
        fa3_fn(*fa3_args)
        torch.cuda.synchronize()
        latency_ms = bench_kernel(fa3_fn, args=fa3_args)
        result = {
            "case": case.name,
            "impl": "fa3",
            "seq_len": case.seq_len,
            "heads": case.heads,
            "heads_kv": case.heads_kv,
            "latency_ms": latency_ms,
            "tflops": case.flops() / latency_ms * 1e-9,
        }
        BenchmarkReport.record(
            "gqa_fp8_wgmma",
            {"case": case.name, "impl": "fa3"},
            result,
            tag="fa3",
        )


def _bench_fa3(
    case: GQAFp8BenchCase,
    *,
    n_warmup: int = 5,
    n_repeat: int = 20,
    n_trials: int = 3,
) -> Optional[dict[str, float | str | int]]:
    _tileops_args, fa3_args = _make_inputs(case, scale_mode="fa3")
    fa3_fn = _fa3_fp8_gqa_fwd()
    if fa3_fn is None:
        return None

    fa3_fn(*fa3_args)
    torch.cuda.synchronize()
    latency_ms = bench_kernel(
        fa3_fn,
        args=fa3_args,
        n_warmup=n_warmup,
        n_repeat=n_repeat,
        n_trials=n_trials,
    )
    return {
        "case": case.name,
        "impl": "fa3",
        "seq_len": case.seq_len,
        "heads": case.heads,
        "heads_kv": case.heads_kv,
        "latency_ms": latency_ms,
        "tflops": case.flops() / latency_ms * 1e-9,
    }


def _main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark FP8 GQA WGMMA variants.")
    parser.add_argument("--quick", action="store_true", help="Use fewer timing iterations.")
    parser.add_argument("--cases", default=None,
                        help="Comma-separated Llama-style cases. "
                             f"Available: {','.join(_LLAMA_CASES)}")
    parser.add_argument("--seq-lens", default=None,
                        help="Optional ad-hoc seq lengths; overrides --cases.")
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--heads-kv", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--impls", default="tileops_ws_shared_p,fa3",
                        help="Comma-separated implementations to run.")
    parser.add_argument("--scale-mode", choices=("fa3", "block128"), default="fa3",
                        help="Use FA3-compatible per-GQA-group descales, or TileOps block-128 scales.")
    args = parser.parse_args(argv)

    n_warmup = 3 if args.quick else 10
    n_repeat = 10 if args.quick else 50
    n_trials = 2 if args.quick else 3
    impls = {s.strip() for s in args.impls.split(",") if s.strip()}
    if args.scale_mode != "fa3" and "fa3" in impls:
        raise ValueError("FA3 comparison requires --scale-mode fa3")

    if args.seq_lens:
        cases = [
            GQAFp8BenchCase(args.batch, int(seq_len), args.heads, args.heads_kv)
            for seq_len in args.seq_lens.split(",")
            if seq_len
        ]
    else:
        case_names = args.cases
        if case_names is None:
            case_names = "llama8b-1k" if args.quick else "llama8b-1k,llama8b-4k,llama70b-4k"
        cases = []
        for name in (s.strip() for s in case_names.split(",") if s.strip()):
            if name not in _LLAMA_CASES:
                raise ValueError(f"unknown case {name!r}; available: {', '.join(_LLAMA_CASES)}")
            cases.append(_LLAMA_CASES[name])

    print(f"device={torch.cuda.get_device_name(0)}")
    print("case,impl,latency_ms,tflops")
    for case in cases:
        if "tileops_wgmma_shared_p" in impls:
            result = _bench_one(
                case,
                ws_persistent=False,
                use_register_p=False,
                scale_mode=args.scale_mode,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_wgmma_register_p" in impls:
            result = _bench_one(
                case,
                ws_persistent=False,
                use_register_p=True,
                scale_mode=args.scale_mode,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_shared_p" in impls:
            result = _bench_one(
                case,
                ws_persistent=True,
                use_register_p=False,
                scale_mode=args.scale_mode,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_register_p" in impls:
            result = _bench_one(
                case,
                ws_persistent=True,
                use_register_p=True,
                scale_mode=args.scale_mode,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_pv_extern" in impls:
            result = _bench_ws_config(
                case,
                impl="tileops_ws_fa3_pv_extern",
                config={"use_fa3_pv_extern": True},
                scale_mode=args.scale_mode,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_pv_extern_tma_direct" in impls:
            result = _bench_ws_config(
                case,
                impl="tileops_ws_fa3_pv_extern_tma_direct",
                config={"use_fa3_pv_extern": True, "use_fa3_v_tma_direct": True},
                scale_mode=args.scale_mode,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_pv" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
                config={"use_ptx_pv": True},
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_pv",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_direct_store" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccDirectStoreKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_direct_store",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_fa3_epilogue_store" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccFa3EpilogueStoreKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_fa3_epilogue_store",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_fa3_epilogue_reuse_v_smem" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccFa3EpilogueReuseVSmemKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_fa3_epilogue_reuse_v_smem",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224Kernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_tma_v" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_tma_v",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_tma_v_inplace_barrier" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsTmaVInplaceBarrierKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_tma_v_inplace_barrier",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_streaming_p" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_streaming_p",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_streaming_p_plain_wait" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapStreamingPPlainWaitKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_streaming_p_plain_wait",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_local_p" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapLocalPKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_local_p",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_fragment_delta" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragmentDeltaKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_fragment_delta",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_visible_pv_delta" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_visible_pv_delta",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_visible_pv_delta_shared_v" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaSharedVKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_visible_pv_delta_shared_v",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_visible_pv_delta_emitter_k224" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapVisiblePVDeltaEmitterK224Kernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_visible_pv_delta_emitter_k224",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_frag_p_local_delta" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsOverlapFragPLocalDeltaKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_overlap_frag_p_local_delta",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_pingpong" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_pingpong",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_pingpong_corrected" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_pingpong_corrected",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "tileops_ws_fa3_contract_ptx_acc_bn224_ws_pingpong_corrected_prerescale" in impls:
            inputs, _ = _make_inputs(case, scale_mode=args.scale_mode)
            kernel = GQAFwdFP8Fa3ContractPtxAccBN224WsPingpongCorrectedPreRescaleKernel(
                case.batch,
                case.heads,
                case.heads_kv,
                case.seq_len,
                case.dim,
                case.out_dtype,
            )
            kernel(*inputs)
            torch.cuda.synchronize()
            latency_ms = bench_kernel(
                kernel,
                args=inputs,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            result = {
                "case": case.name,
                "impl": "tileops_ws_fa3_contract_ptx_acc_bn224_ws_pingpong_corrected_prerescale",
                "latency_ms": latency_ms,
                "tflops": case.flops() / latency_ms * 1e-9,
            }
            print(
                f"{result['case']},{result['impl']},"
                f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                flush=True,
            )
        if "fa3" in impls:
            result = _bench_fa3(
                case,
                n_warmup=n_warmup,
                n_repeat=n_repeat,
                n_trials=n_trials,
            )
            if result is None:
                print(f"{case.name},fa3,NA,NA", flush=True)
            else:
                print(
                    f"{result['case']},{result['impl']},"
                    f"{result['latency_ms']:.6f},{result['tflops']:.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    _main()
