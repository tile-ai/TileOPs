import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkReport, ManifestBenchmark
from benchmarks.ops.attention.manifest_params import gqa_decode_args, manifest_params
from tileops.manifest import load_workloads
from tileops.ops import GroupedQueryAttentionDecodeWithKVCacheFwdOp
from workloads.attention.gqa_decode import GroupedQueryAttentionDecodeTest

_OP_NAME = "GroupedQueryAttentionDecodeWithKVCacheFwdOp"


class _GroupedQueryAttentionDecodeTestBaseline(GroupedQueryAttentionDecodeTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.unsqueeze(1).transpose(1, 2)  # [B, H, 1, D]
        groups = self.heads // self.heads_kv
        k_bhsd = k.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        v_bhsd = v.repeat_interleave(groups, dim=2).transpose(1, 2).float()
        scores = torch.matmul(q_bhsd.float(), k_bhsd.transpose(-2, -1)) * self.sm_scale
        if self.softcap > 0:
            scores = self.softcap * torch.tanh(scores / self.softcap)
        probs = torch.softmax(scores, dim=-1)
        output_bhsd = torch.matmul(probs, v_bhsd)
        return output_bhsd.transpose(1, 2).squeeze(1).to(q.dtype).contiguous()


def _fa3_gqa_decode_fwd(test):
    """Return FA3 KV-cache decode baseline callable, or None if not installed."""
    if test.sm_scale != test.dim**-0.5 or test.softcap != 0.0:
        return None
    try:
        from flash_attn_interface import flash_attn_with_kvcache  # noqa: PLC0415
    except ImportError:
        return None

    cache_seqlens = torch.full(
        (test.batch,), test.seq_len_kv, dtype=torch.int32, device="cuda")

    def baseline_fn(q, k, v):
        # Q is (B, H, D); FA3 KV-cache decode expects (B, S_q, H, D).
        out = flash_attn_with_kvcache(q.unsqueeze(1), k, v, cache_seqlens=cache_seqlens)
        out = out[0] if isinstance(out, tuple) else out
        return out.squeeze(1)

    return baseline_fn


def _flashinfer_gqa_decode_fwd(test, q, k, v):
    """Set up FlashInfer batched paged decode for non-paged KV cache.

    Reshapes the contiguous (B, S_kv, H_kv, D) KV into paged format with
    page_size=256, then uses the specialized decode kernel.
    FlashInfer decode kernel supports group_size (Q/KV head ratio) up to 8.
    """
    if test.sm_scale != test.dim**-0.5 or test.softcap != 0.0:
        return None
    try:
        from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper  # noqa: PLC0415
    except ImportError:
        return None

    # Q is (B, H, D) — single token per request
    # K/V is (B, S_kv, H_kv, D)
    B, H, D = q.shape
    Hkv = k.shape[2]
    if H // Hkv > 8:
        return None  # FlashInfer decode kernel does not support group_size > 8
    Skv = k.shape[1]
    page_size = 256
    pages_per_seq = (Skv + page_size - 1) // page_size

    # Reshape (B, S_kv, H_kv, D) → (B * pages_per_seq, page_size, H_kv, D)
    # Pad S_kv to a multiple of page_size if needed
    if Skv % page_size != 0:
        pad = page_size * pages_per_seq - Skv
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
    k_paged = k.reshape(B * pages_per_seq, page_size, Hkv, D)
    v_paged = v.reshape(B * pages_per_seq, page_size, Hkv, D)
    kv_data = (k_paged, v_paged)

    total_pages = B * pages_per_seq
    indptr = torch.arange(0, B + 1, dtype=torch.int32, device=q.device) * pages_per_seq
    indices = torch.arange(0, total_pages, dtype=torch.int32, device=q.device)
    last_page_len_val = Skv - (pages_per_seq - 1) * page_size
    last_page_len = torch.full((B,), last_page_len_val, dtype=torch.int32, device=q.device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        indptr=indptr, indices=indices, last_page_len=last_page_len,
        num_qo_heads=H, num_kv_heads=Hkv, head_dim=D,
        page_size=page_size,
        q_data_type=q.dtype,
    )

    def run_fn(q, k, v):
        return wrapper.run(q, kv_data)

    return run_fn


_GQA_DECODE_BENCH_PARAMS = manifest_params(load_workloads(_OP_NAME), gqa_decode_args)


@pytest.mark.parametrize(
    "batch, heads, heads_kv, seq_len_kv, dim, sm_scale, softcap, dtype, tune",
    _GQA_DECODE_BENCH_PARAMS,
)
def test_gqa_decode_bench(batch: int, heads: int, heads_kv: int, seq_len_kv: int, dim: int,
                          sm_scale: float | None, softcap: float | None, dtype: torch.dtype,
                          tune: bool) -> None:
    test = _GroupedQueryAttentionDecodeTestBaseline(
        batch, heads, heads_kv, seq_len_kv, dim, dtype, sm_scale=sm_scale, softcap=softcap)
    inputs = test.gen_inputs()

    op = GroupedQueryAttentionDecodeWithKVCacheFwdOp(
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        dtype,
        sm_scale=sm_scale,
        softcap=softcap,
        tune=tune,
    )
    bm = ManifestBenchmark(_OP_NAME, op, test)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    fa3_fn = _fa3_gqa_decode_fwd(test)
    if fa3_fn is not None:
        result_bl = bm.profile(fa3_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="fa3")

    fi_fn = _flashinfer_gqa_decode_fwd(test, *inputs)
    if fi_fn is not None:
        result_fi = bm.profile(fi_fn, *inputs)
        BenchmarkReport.record(op, locals(), result_fi, tag="flashinfer")

    if fa3_fn is None and fi_fn is None:
        result_bl = bm.profile(test.ref_program, *inputs)
        BenchmarkReport.record(op, locals(), result_bl, tag="torch-sdpa")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
