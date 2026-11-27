"""Warp-specialized batch=1 GQA decode kernel (SM90), context-split.

``GQADecodeBs1Kernel`` dispatches on the runtime K/V sequence extent: RoPE calls always
use the context split; plain calls keep the generic single-kernel path below a fixed cache
length.
"""

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.attention.call_spec import AttentionCall, GQADenseFwdInterface
from tileops._kernels.attention.gqa.decode import (
    GQADecodeKernel,
    gqa_decode_kernel,
)
from tileops._kernels.attention.gqa.decode_bs1_common import (
    COMPILE_FLAGS,
    RING_DEPTH,
    allocate_partials,
    make_gqa_decode_bs1_combine,
    make_gqa_decode_bs1_split,
)
from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import LOG2E
from tileops._kernels.kernel_base import Entry, Kernel

__all__ = ["GQADecodeBs1Kernel"]


@target_cache(maxsize=32)
def gqa_decode_bs1_ctx_kernel(
    batch,
    heads,
    groups,
    dim,
    sm_scale,
    softcap,
    dtype,
    fuse_rope=False,
    max_position=1,
    rotary_dim=0,
    rope_layout="neox",
    has_sinks=False,
):
    scale = LOG2E if softcap > 0.0 else sm_scale * LOG2E
    accum_dtype = "float"
    kv_group_num = heads // groups

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=COMPILE_FLAGS,
    )
    def _func(block_M, block_N, ctx_splits, threads):
        seqlen_kv = T.dynamic("seqlen_kv")
        shape_q = [batch, heads, dim]
        shape_k = [batch, seqlen_kv, groups, dim]
        shape_o = [batch, heads, dim]
        rope_shape = [max_position, rotary_dim // 2]
        part_shape = [batch, heads, ctx_splits, dim]
        lse_shape = [batch, heads, ctx_splits]

        @T.macro
        def load_kv(K, V, unused_layout, bid, hid, base, k, Ks, Vs, ready):
            T.tma_copy(
                K[bid, base + k * block_N : base + (k + 1) * block_N, hid, :],
                Ks[k % RING_DEPTH, :, :],
                barrier=ready[k % RING_DEPTH],
            )
            T.tma_copy(
                V[bid, base + k * block_N : base + (k + 1) * block_N, hid, :],
                Vs[k % RING_DEPTH, :, :],
                barrier=ready[k % RING_DEPTH],
            )

        split = make_gqa_decode_bs1_split(
            batch,
            groups,
            block_M,
            block_N,
            dim,
            dtype,
            scale,
            kv_group_num,
            ctx_splits,
            threads,
            accum_dtype,
            load_kv,
            paged=False,
            score_scale=sm_scale,
            softcap=softcap,
            fuse_rope=fuse_rope,
            rotary_dim=rotary_dim,
            rope_layout=rope_layout,
        )
        combine = make_gqa_decode_bs1_combine(
            batch,
            heads,
            ctx_splits,
            dim,
            dtype,
            accum_dtype,
            has_sinks=has_sinks,
        )

        if fuse_rope:

            @T.prim_func
            def gqa_decode_bs1_ctx_rope(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_k, dtype),
                rope_cos: T.Tensor(rope_shape, dtype),
                rope_sin: T.Tensor(rope_shape, dtype),
                glse: T.Tensor(lse_shape, accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                sinks: T.Tensor(
                    [heads] if has_sinks else shape_q, "float32" if has_sinks else dtype
                ),
                Output: T.Tensor(shape_o, dtype),
            ):
                split(
                    Q,
                    K,
                    V,
                    K,
                    seqlen_kv,
                    rope_cos,
                    rope_sin,
                    glse,
                    Output_partial,
                )
                combine(glse, Output_partial, Output, sinks)

            return gqa_decode_bs1_ctx_rope

        @T.prim_func
        def gqa_decode_bs1_ctx(
            Q: T.Tensor(shape_q, dtype),
            K: T.Tensor(shape_k, dtype),
            V: T.Tensor(shape_k, dtype),
            glse: T.Tensor(lse_shape, accum_dtype),
            Output_partial: T.Tensor(part_shape, accum_dtype),
            sinks: T.Tensor([heads] if has_sinks else shape_q, "float32" if has_sinks else dtype),
            Output: T.Tensor(shape_o, dtype),
        ):
            split(Q, K, V, K, seqlen_kv, Q, Q, glse, Output_partial)
            combine(glse, Output_partial, Output, sinks)

        return gqa_decode_bs1_ctx

    return _func


class GQADecodeBs1Kernel(Kernel, GQADenseFwdInterface):
    """SM90 warp-specialized batch=1 GQA decode kernel with a context-length switch.

    ``forward`` always uses the context pipeline for RoPE calls. Plain calls below 640
    retain the generic single-kernel path.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"gqa_dense_decode_ws"})
    # Threads of the RoPE pipeline, whose producer is a warpgroup.
    _ROPE_PIPELINE_THREADS = 256
    _PLAIN_CTX_MIN = 640
    _TARGET_PARTIAL_CTAS = 128
    _CTX_SPLIT_CANDIDATES = (1, 2, 4, 8, 16, 32, 64)

    def _dense_ctx_splits_for(self, real_seqlen_kv: int) -> int:
        block_n = self.config["block_N"]
        num_tiles = (real_seqlen_kv + block_n - 1) // block_n
        split_capacity = max(
            1,
            min(self._CTX_SPLIT_CANDIDATES[-1], self._TARGET_PARTIAL_CTAS // self.groups),
        )
        selected = 1
        for ctx_splits in self._CTX_SPLIT_CANDIDATES:
            if ctx_splits > split_capacity:
                break
            selected = ctx_splits
            if ctx_splits >= num_tiles:
                break
        return selected

    @classmethod
    def refusal(cls, call: AttentionCall) -> str | None:
        """The calls :attr:`AttentionCall.dense_decode_program` names ``"batch1"``, within
        the group and head dimension one 64-row consumer tile holds."""
        if call.return_lse:
            return "does not return the log-sum-exp"
        if call.dense_decode_program != "batch1":
            return "does not serve this call"
        if call.heads // call.heads_kv > 64 or call.dim > 128:
            return "requires at most 64 heads a KV head and head dimension at most 128"
        return GQADecodeKernel.refusal(call)

    @classmethod
    def entry_for(cls, call: AttentionCall) -> Entry:
        """This program takes the cache length at runtime and tiers on nothing."""
        args = GQADecodeKernel.construction_args(call)
        return tuple(v for k, v in args.items() if k != "seq_len_kv"), lambda: cls(**args)

    def __init__(
        self,
        batch,
        heads,
        heads_kv,
        seq_len_kv,
        dim,
        dtype="float16",
        sm_scale: float | None = None,
        softcap: float = 0.0,
        config: dict | None = None,
        *,
        fuse_rope: bool = False,
        max_position: int = 1,
        rotary_dim: int = 0,
        rope_layout: str = "neox",
        has_sinks: bool = False,
        device_index: int | None = None,
    ):
        super().__init__(device_index=device_index)
        self.has_sinks = has_sinks
        self.batch = batch
        self.heads = heads
        self.groups = heads_kv
        self.seqlen_kv = seq_len_kv
        self.dim = dim
        self.dtype = dtype
        self.sm_scale = dim**-0.5 if sm_scale is None else sm_scale
        self.softcap = softcap
        self.fuse_rope = fuse_rope
        self.max_position = max_position
        self.rotary_dim = rotary_dim
        self.rope_layout = rope_layout
        if self.groups <= 0:
            raise ValueError("heads_kv must be positive")
        if self.heads % self.groups != 0:
            raise ValueError("heads must be divisible by heads_kv")
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"block_M": 64, "block_N": 128, "threads": 160}

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_scale: torch.Tensor | None = None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._require_devices(q=q, k=k, v=v)
        del q_scale, k_scale, v_scale
        if self.fuse_rope and (rope_cos is None or rope_sin is None):
            raise ValueError("fused RoPE requires rope_cos and rope_sin")
        Q = q.squeeze(1)
        real_seqlen_kv = k.shape[1]
        if not self.fuse_rope and real_seqlen_kv < self._PLAIN_CTX_MIN:
            program = self.program(self._compile_short, sinks is not None)
            return program(Q, k, v, sinks if sinks is not None else Q).unsqueeze(1)

        ctx_splits = self._dense_ctx_splits_for(real_seqlen_kv)
        program = self.program(self._compile, ctx_splits)
        glse, Output_partial = allocate_partials(Q, self.batch, self.heads, self.dim, ctx_splits)
        if self.fuse_rope:
            output = program(
                Q, k, v, rope_cos, rope_sin, glse, Output_partial, sinks if self.has_sinks else Q
            )
        else:
            output = program(Q, k, v, glse, Output_partial, sinks if self.has_sinks else Q)
        return output.unsqueeze(1)

    def _compile_short(self, cfg: dict, with_sinks: bool):
        """The generic single-kernel program for a plain call below ``_PLAIN_CTX_MIN`` keys;
        *with_sinks* says whether the call passes sinks."""
        return gqa_decode_kernel(
            self.batch,
            self.heads,
            self.groups,
            self.dim,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            has_sinks=with_sinks,
        )(64, 128, 2, 128)

    def _compile(self, cfg: dict, ctx_splits: int):
        """The context pipeline for *cfg* over *ctx_splits* splits, with RoPE when built for it."""
        if self.fuse_rope:
            rope = (True, self.max_position, self.rotary_dim, self.rope_layout)
            threads = self._ROPE_PIPELINE_THREADS
        else:
            rope = (False, 1, 0, "neox")
            threads = cfg["threads"]
        return gqa_decode_bs1_ctx_kernel(
            self.batch,
            self.heads,
            self.groups,
            self.dim,
            self.sm_scale,
            self.softcap,
            self.dtype_str,
            *rope,
            has_sinks=self.has_sinks,
        )(cfg["block_M"], cfg["block_N"], ctx_splits, threads)
