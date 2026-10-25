"""Warp-specialized SM90 dense GQA decode for batched calls.

``GQADecodeWSKernel`` runs one CTA per (KV head, batch entry, KV split): a TMA producer
warp streams K and V tiles through a two-slot ring while one WGMMA consumer warpgroup
keeps the head group's online softmax. One split normalizes in the CTA and writes the
output; more splits write FP32 partials that a combine kernel merges.
"""

import functools
import math
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.call_spec import AttentionCall, GQADenseFwdInterface
from tileops._kernels.attention.gqa.decode import GQADecodeKernel
from tileops._kernels.attention.gqa.decode_bs1_common import (
    COMPILE_FLAGS,
    make_gqa_decode_bs1_combine,
)
from tileops._kernels.attention.online_softmax import make_apply_softcap, make_dense_sink_scale
from tileops._kernels.constants import LOG2E
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._utils import get_shared_memory_optin, get_sm_count

__all__ = ["GQADecodeWSKernel"]


def _as_fp32(value: float) -> float:
    """*value* as the kernel's FP32 arithmetic holds it."""
    return float(torch.tensor(value, dtype=torch.float32))


@functools.lru_cache(maxsize=32)
def _gqa_decode_ws_kernel(batch, heads, groups, dim, sm_scale, softcap, dtype, has_sinks):
    score_scale = dim**-0.5 if sm_scale is None else sm_scale
    # The running maximum is taken over raw scores and needs a positive FP32 scale: a
    # negative one reverses their order, so the query is negated on load; one that is zero
    # in FP32 gives every key the same logit, which a zero query yields at any scale.
    vanishes = _as_fp32(abs(score_scale) * LOG2E) == 0.0
    query_sign = 0.0 if vanishes else (-1.0 if score_scale < 0 else 1.0)
    score_scale = 1.0 if vanishes else abs(score_scale)
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    accum_dtype = "float"
    group_size = heads // groups

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
        compile_flags=COMPILE_FLAGS,
    )
    def _func(block_m, block_n, num_split):
        # The consumer is one warpgroup: WGMMA tiles the head group 64 rows at a time.
        consumer_threads = 128
        producer_threads = 32
        # Two slots let the producer fill one while the consumer reads the other, and
        # keep two CTAs resident on an SM wherever the tiles fit twice.
        ring = 2
        row_tiles = -(-group_size // block_m)
        fused = num_split == 1
        seqlen_kv = T.dynamic("seqlen_kv")
        shape_q = [batch, heads, dim]
        shape_kv = [batch, seqlen_kv, groups, dim]
        apply_softcap = (
            make_apply_softcap(score_scale, softcap, accum_dtype, block_m, block_n)
            if use_softcap
            else None
        )
        sink_scale = make_dense_sink_scale(scale, block_m)
        combine = make_gqa_decode_bs1_combine(
            batch, heads, num_split, dim, dtype, accum_dtype, has_sinks=has_sinks
        )

        @T.prim_func
        def gqa_decode_ws(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_kv, dtype),
            V: T.Tensor(shape_kv, dtype),
            glse: T.Tensor([batch, heads, num_split], accum_dtype),
            Output_partial: T.Tensor([batch, heads, num_split, dim], accum_dtype),
            sinks: T.Tensor([heads] if has_sinks else shape_q, "float32" if has_sinks else dtype),
            Output: T.Tensor(shape_q, dtype),
        ):
            # The KV head varies fastest across CTAs: the heads of one batch entry
            # interleave row by row in K and V, so CTAs launched together read the
            # same DRAM pages.
            with T.Kernel(
                groups * row_tiles, batch, num_split, threads=consumer_threads + producer_threads
            ) as (head_tile, bid, sid):
                hid = head_tile // row_tiles
                row0 = hid * group_size + head_tile % row_tiles * block_m
                rows = T.min(group_size - head_tile % row_tiles * block_m, block_m)

                q_s = T.alloc_shared([block_m, dim], dtype)
                k_s = T.alloc_shared([ring, block_n, dim], dtype)
                v_s = T.alloc_shared([ring, block_n, dim], dtype)
                p_s = T.alloc_shared([block_m, block_n], dtype)
                T.annotate_layout(
                    {
                        q_s: tilelang.layout.make_swizzled_layout(q_s),
                        k_s: tilelang.layout.make_swizzled_layout(k_s),
                        v_s: tilelang.layout.make_swizzled_layout(v_s),
                        p_s: tilelang.layout.make_swizzled_layout(p_s),
                    }
                )
                ready = T.alloc_barrier([producer_threads] * ring)
                free = T.alloc_barrier([consumer_threads] * ring)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)

                # Whole KV tiles split as evenly as possible: the first ``extra_tiles``
                # splits take one tile more than the rest.
                num_tiles = T.ceildiv(seqlen_kv, block_n)
                base_tiles = num_tiles // num_split
                extra_tiles = num_tiles % num_split
                tiles_this_split = base_tiles + T.if_then_else(sid < extra_tiles, 1, 0)
                base = (sid * base_tiles + T.min(sid, extra_tiles)) * block_n
                this_len = T.max(T.min(seqlen_kv - base, tiles_this_split * block_n), 0)
                loop_range = T.ceildiv(this_len, block_n)

                tx = T.get_thread_binding()
                if tx >= consumer_threads:
                    for k in T.serial(loop_range):
                        T.mbarrier_wait_parity(free[k % ring], (k // ring) % 2 ^ 1)
                        T.tma_copy(
                            K[bid, base + k * block_n : base + (k + 1) * block_n, hid, :],
                            k_s[k % ring, :, :],
                            barrier=ready[k % ring],
                        )
                        T.tma_copy(
                            V[bid, base + k * block_n : base + (k + 1) * block_n, hid, :],
                            v_s[k % ring, :, :],
                            barrier=ready[k % ring],
                        )
                        T.mbarrier_arrive(ready[k % ring])
                else:
                    T.fill(acc_o, 0)
                    T.fill(logsum, 0)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    for i, j in T.Parallel(block_m, dim):
                        q_s[i, j] = T.if_then_else(
                            i < rows,
                            Q[bid, row0 + i, j] * T.cast(query_sign, dtype),
                            T.cast(0, dtype),
                        )
                    for k in T.serial(loop_range):
                        T.mbarrier_wait_parity(ready[k % ring], (k // ring) % 2)
                        T.wgmma_gemm(
                            q_s,
                            k_s[k % ring, :, :],
                            acc_s,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=True,
                        )
                        T.wait_wgmma(0)
                        if use_softcap:
                            apply_softcap(acc_s)
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.if_then_else(
                                k * block_n + j < this_len, acc_s[i, j], -T.infinity(accum_dtype)
                            )
                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(block_m):
                            scores_scale[i] = T.exp2(
                                scores_max_prev[i] * scale - scores_max[i] * scale
                            )
                        for i, j in T.Parallel(block_m, block_n):
                            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(block_m):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                        for i, j in T.Parallel(block_m, dim):
                            acc_o[i, j] *= scores_scale[i]
                        T.copy(acc_s, p_s)
                        T.wgmma_gemm(
                            p_s,
                            v_s[k % ring, :, :],
                            acc_o,
                            policy=T.GemmWarpPolicy.FullRow,
                            clear_accum=False,
                        )
                        T.wait_wgmma(0)
                        T.mbarrier_arrive(free[k % ring])
                    if fused:
                        # One split holds every key, so it normalizes here; the sink
                        # factor needs the completed row sum.
                        if has_sinks:
                            sink_scale(logsum, scores_max, sinks, scores_scale, row0, 1, rows)
                            for i, j in T.Parallel(block_m, dim):
                                acc_o[i, j] *= scores_scale[i]
                        for i, j in T.Parallel(block_m, dim):
                            acc_o[i, j] /= logsum[i]
                        for i, j in T.Parallel(block_m, dim):
                            if i < rows:
                                Output[bid, row0 + i, j] = acc_o[i, j]
                    else:
                        # An empty split weighs nothing in the combine: its log-sum-exp
                        # is -inf and its partial output zero.
                        for i, j in T.Parallel(block_m, dim):
                            acc_o[i, j] /= T.if_then_else(this_len > 0, logsum[i], 1.0)
                        for i in T.Parallel(block_m):
                            if i < rows:
                                glse[bid, row0 + i, sid] = T.if_then_else(
                                    this_len > 0,
                                    T.log2(logsum[i]) + scores_max[i] * scale,
                                    -T.infinity(accum_dtype),
                                )
                        for i, j in T.Parallel(block_m, dim):
                            if i < rows:
                                Output_partial[bid, row0 + i, sid, j] = T.if_then_else(
                                    this_len > 0, acc_o[i, j], 0.0
                                )
            if not fused:
                combine(glse, Output_partial, Output, sinks)

        return gqa_decode_ws

    return _func


class GQADecodeWSKernel(Kernel, GQADenseFwdInterface):
    """SM90 TMA-fed, warp-specialized dense decode; the split count follows the grid.

    The batch-1 kernels keep the calls they serve: they split one sequence over more
    CTAs than the batch and KV heads give this plan.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_dense_decode"})
    # Head dimensions whose tiles this plan is built and tested at.
    _DIMS = (64, 128, 256)

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        """Unrotated 16-bit dense decode at the head dimensions this plan builds for."""
        if call.return_lse:
            return "does not return the log-sum-exp"
        if not call.dense_decode_region:
            return "does not serve this call"
        if call.fuse_rope:
            return "does not fuse RoPE"
        if call.dim not in cls._DIMS:
            return f"requires head dimension in {cls._DIMS}"
        if call.sm_scale is not None and math.isinf(_as_fp32(abs(call.sm_scale) * LOG2E)):
            return "requires a score scale whose base-2 form FP32 holds"
        return GQADecodeKernel.refusal(call)

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """The cache length is taken at runtime; the split count follows it per call."""
        args = GQADecodeKernel.construction_args(call)
        for rope_arg in call.rope_args:
            args.pop(rope_arg)
        return tuple(v for k, v in args.items() if k != "seq_len_kv"), lambda: cls(**args)

    def __init__(
        self,
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        dtype=torch.float16,
        sm_scale: Optional[float] = None,
        softcap: float = 0.0,
        config: Optional[dict] = None,
        *,
        has_sinks: bool = False,
        device_index: Optional[int] = None,
    ):
        super().__init__(device_index=device_index)
        # The program takes the cache length per call.
        del seq_len_kv
        if heads_kv <= 0 or heads % heads_kv != 0:
            raise ValueError("heads must be a positive multiple of heads_kv")
        if dim not in self._DIMS:
            raise ValueError(f"{type(self).__name__} requires head dimension in {self._DIMS}")
        self.batch = batch
        self.heads = heads
        self.groups = heads_kv
        self.dim = dim
        self.dtype = dtype
        self.sm_scale = sm_scale
        self.softcap = softcap
        self.has_sinks = has_sinks
        self._sm_count = get_sm_count(device_index)
        self._smem_budget = get_shared_memory_optin(device_index)
        self.jit = _gqa_decode_ws_kernel(
            batch, heads, heads_kv, dim, sm_scale, softcap, self.dtype_str, has_sinks
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # A tile of 128 keys at D64 holds as many bytes as 64 keys at D128, and halves the
        # per-tile barrier and softmax steps of the narrow head.
        return {"block_m": 64, "block_n": 128 if self.dim <= 64 else 64}

    def _shared_bytes(self) -> int:
        """The query rows, two K and V slots and the probability tile, in bytes."""
        c = self.config
        elems = c["block_m"] * self.dim + 4 * c["block_n"] * self.dim + c["block_m"] * c["block_n"]
        return elems * self.dtype.itemsize

    def num_split(self, seqlen_kv: int) -> int:
        """The power-of-two split count that fills the resident CTA slots once.

        The slots are the SMs times the CTAs whose shared memory fits on one. Each split
        keeps at least four KV tiles, so its producer has work to run ahead of the
        consumer.
        """
        c = self.config
        row_tiles = -(-(self.heads // self.groups) // c["block_m"])
        units = self.batch * self.groups * row_tiles
        slots = self._sm_count * max(1, self._smem_budget // self._shared_bytes())
        tiles = -(-seqlen_kv // c["block_n"])
        limit = max(1, min(slots // units, tiles // 4))
        return 1 << (limit.bit_length() - 1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._require_devices(q=q, k=k, v=v)
        del q_scale, k_scale, v_scale, rope_cos, rope_sin
        query = q.squeeze(1)
        num_split = self.num_split(k.shape[1])
        glse = torch.empty(
            (self.batch, self.heads, num_split), dtype=torch.float32, device=q.device
        )
        partial = torch.empty(
            (self.batch, self.heads, num_split, self.dim), dtype=torch.float32, device=q.device
        )
        program = self.program(self._compile, num_split)
        output = program(query, k, v, glse, partial, sinks if self.has_sinks else query)
        return output.unsqueeze(1)

    def _compile(self, cfg: dict, num_split: int):
        """The program for *cfg* over *num_split* splits."""
        return self.jit(cfg["block_m"], cfg["block_n"], num_split)
