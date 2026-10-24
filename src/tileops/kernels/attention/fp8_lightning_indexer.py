import dataclasses
import functools
import itertools
from abc import abstractmethod
from typing import Optional

import tilelang
import torch
from tilelang import language as T
from tilelang.layout import make_swizzled_layout

from tileops._csrc import csrc_include
from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Entry, Kernel, KernelInterface
from tileops.utils import get_shared_memory_optin

__all__ = [
    "FP8LightningIndexerCall",
    "FP8LightningIndexerFwdInterface",
    "FP8LightningIndexerKernel",
    "FP8LightningIndexerWSKernel",
]


@dataclasses.dataclass(frozen=True)
class FP8LightningIndexerCall(CallSpec):
    """One lightning-indexer call over FP8 index keys."""

    batch: int = 0
    seq_len: int = 0
    heads: int = 0
    index_dim: int = 0
    seq_len_kv: int = 0
    kv_group: int = 0
    clean_logits: bool = True


class FP8LightningIndexerFwdInterface(KernelInterface):
    """Lightning-indexer logits over FP8 index keys."""

    request = FP8LightningIndexerCall

    @abstractmethod
    def forward(
        self,
        IndexQ: torch.Tensor,
        IndexK: torch.Tensor,
        IndexKScale: torch.Tensor,
        Weights: torch.Tensor,
        CuSeqLenKS: torch.Tensor,
        CuSeqLenKE: torch.Tensor,
    ) -> torch.Tensor:
        """Score each key inside its query's window; nothing is written in place.

        The logit of query ``s`` and key ``t`` of group ``g`` sums ``Weights[s, h]`` times
        ``relu(IndexQ[s, h] . IndexK[t, g]) * IndexKScale[t, g]`` over the heads of group
        ``g``. Where ``call.clean_logits``, a key outside ``[CuSeqLenKS[s], CuSeqLenKE[s])``
        is set to negative infinity; otherwise its slot holds whatever the scan left.
        Nothing is written in place, and the output aliases no input.

        Every tensor is on ``call.device``, contiguous and row-major in the axis order
        given; the op makes them contiguous before the call.

        Args:
            IndexQ: ``float8_e4m3fn``
                ``(call.batch, call.seq_len, call.heads, call.index_dim)``.
            IndexK: ``float8_e4m3fn``
                ``(call.batch, call.seq_len_kv, call.kv_group, call.index_dim)``.
            IndexKScale: ``float32`` ``(call.batch, call.seq_len_kv, call.kv_group)``.
            Weights: ``float32`` ``(call.seq_len, call.heads)``.
            CuSeqLenKS: ``int32`` ``(call.seq_len,)``, the first key of each window.
            CuSeqLenKE: ``int32`` ``(call.seq_len,)``, one past the last key.

        Returns:
            A new contiguous ``float32``
            ``(call.batch, call.seq_len, call.seq_len_kv, call.kv_group)`` logit tensor.
        """


@functools.lru_cache(maxsize=32)
def _fp8_lightning_indexer_kernel(
    batch, seq_len, heads, index_dim, seq_len_kv, kv_group, min_pipelined_block_q, clean_logits=True
):
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
    )
    def _fp8_lightning_indexer_func(
        block_N=256,
        num_stages=3,
        threads=512,
        block_Q=None,
    ):
        if block_Q is None:
            block_Q = max(1, 128 // heads)
        # Guards every entry point, incl. the default block_Q == 1 for heads >= 65.
        if block_Q < min_pipelined_block_q:
            num_stages = 0
        dtype = T.float8_e4m3fn
        accum_dtype = T.float32
        index_dtype = T.int32

        seq_len = T.dynamic("seq_len")
        seq_len_kv = T.dynamic("seq_len_kv")

        index_q_shape = [batch, seq_len * heads, index_dim]
        index_k_shape = [batch, seq_len_kv, kv_group, index_dim]
        index_k_scale_shape = [batch, seq_len_kv, kv_group]
        logits_shape = [batch, seq_len, seq_len_kv, kv_group]

        @T.prim_func
        def _fp8_lightning_indexer_main(
            IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
            IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
            IndexKScale: T.Tensor(index_k_scale_shape, accum_dtype),  # type: ignore
            Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
            Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
            CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
            CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
        ):
            heads_per_group = heads // kv_group
            with T.Kernel(T.ceildiv(seq_len, block_Q), batch, threads=threads) as (bx, by):
                index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
                index_k_scale_fragment = T.alloc_fragment([block_N, kv_group], accum_dtype)
                s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)  #
                s_reshaped = T.reshape(s, (block_N, block_Q, heads_per_group, kv_group))
                logits = T.alloc_fragment([block_N, block_Q, kv_group], accum_dtype)
                weights = T.alloc_fragment([block_Q, heads], accum_dtype)
                index_k_group_shared = T.alloc_shared([block_N, index_dim], dtype)
                if kv_group > 1:
                    index_k_shared = T.alloc_shared([block_N, kv_group, index_dim], dtype)
                    index_q_group_shared = T.alloc_shared(
                        [block_Q * heads_per_group, index_dim], dtype
                    )
                    s_tmp = T.alloc_fragment([block_N, block_Q * heads_per_group], accum_dtype)

                seq_len_i = bx * block_Q
                b_i = by

                cu_k_s_min = T.alloc_var(index_dtype)
                cu_k_e_max = T.alloc_var(index_dtype)

                cu_k_s_min = 2147483647
                cu_k_e_max = -2147483648

                for bq_i in T.serial(block_Q):
                    cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
                for bq_i in T.serial(block_Q):
                    cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

                T.copy(IndexQ[b_i, seq_len_i * heads, 0], index_q_shared)
                T.copy(Weights[seq_len_i, 0], weights)

                for nbn_i in T.Pipelined(
                    T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages
                ):
                    if kv_group > 1:
                        T.copy(IndexK[b_i, cu_k_s_min + nbn_i * block_N, 0, 0], index_k_shared)
                    T.copy(
                        IndexKScale[b_i, cu_k_s_min + nbn_i * block_N, 0], index_k_scale_fragment
                    )

                    if kv_group == 1:
                        T.copy(
                            IndexK[
                                b_i,
                                cu_k_s_min + nbn_i * block_N : cu_k_s_min + (nbn_i + 1) * block_N,
                                0,
                                0:index_dim,
                            ],
                            index_k_group_shared,
                        )
                        T.gemm(
                            index_k_group_shared,
                            index_q_shared,
                            s,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                    else:
                        for g in T.Serial(kv_group):
                            for bn_i, d_i in T.Parallel(block_N, index_dim):
                                index_k_group_shared[bn_i, d_i] = index_k_shared[bn_i, g, d_i]  #
                            for bq_i, h_i, d_i in T.Parallel(block_Q, heads_per_group, index_dim):
                                index_q_group_shared[bq_i * heads_per_group + h_i, d_i] = (
                                    index_q_shared[bq_i * heads + g * heads_per_group + h_i, d_i]
                                )
                            T.gemm(
                                index_k_group_shared,
                                index_q_group_shared,
                                s_tmp,
                                transpose_B=True,
                                clear_accum=True,
                                policy=T.GemmWarpPolicy.FullCol,
                            )
                            for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads_per_group):
                                s_reshaped[bn_i, bq_i, h_i, g] = s_tmp[
                                    bn_i, bq_i * heads_per_group + h_i
                                ]

                    for bn_i, bq_i, h_i, g in T.Parallel(
                        block_N, block_Q, heads_per_group, kv_group
                    ):
                        s_reshaped[bn_i, bq_i, h_i, g] = (
                            T.max(s_reshaped[bn_i, bq_i, h_i, g], 0)
                            * weights[bq_i, g * heads_per_group + h_i]
                        ) * index_k_scale_fragment[bn_i, g]

                    T.reduce_sum(s_reshaped, logits, dim=-2, clear=True)

                    for bq_i, bn_i, g in T.Parallel(block_Q, block_N, kv_group):
                        Logits[b_i, seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i, g] = (
                            logits[bn_i, bq_i, g]
                        )

        return _fp8_lightning_indexer_main

    return _fp8_lightning_indexer_func


@functools.lru_cache(maxsize=32)
@tilelang.jit
def _clean_logits_(
    threads: int = 512,
):
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")
    kv_group = T.dynamic("kv_group")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([batch, seq_len, seq_len_kv, kv_group], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, batch, threads=threads) as (bx, by):
            tx = T.get_thread_binding()
            cu_k_s = T.max(T.min(CuSeqLenKS[bx], seq_len_kv), 0)
            cu_k_e = T.max(T.min(CuSeqLenKE[bx], seq_len_kv), cu_k_s)

            for n_i in T.serial(T.ceildiv(cu_k_s, threads)):
                idx = n_i * threads + tx
                if idx < cu_k_s:
                    for g in T.serial(kv_group):
                        Logits[by, bx, idx, g] = -T.infinity(dtype)
            for n_i in T.serial(T.ceildiv(seq_len_kv - cu_k_e, threads)):
                idx = cu_k_e + n_i * threads + tx
                if idx < seq_len_kv:
                    for g in T.serial(kv_group):
                        Logits[by, bx, idx, g] = -T.infinity(dtype)

    return clean_logits_kernel


class FP8LightningIndexerKernel(Kernel, FP8LightningIndexerFwdInterface):
    """FP8 lightning indexer: per-query logits over an fp8-quantized index cache.

    Args:
        batch: Batch size.
        seq_len: Query sequence length.
        heads: Number of index heads.
        index_dim: Index head dimension.
        seq_len_kv: Key/value sequence length.
        kv_group: Number of key/value groups.
        clean_logits: Whether to zero the logits outside each query's
            ``[cu_seqlen_ks, cu_seqlen_ke)`` window.
        dtype: Torch dtype of the quantized index tensors. The tensor-core
            path this kernel emits accepts ``torch.float8_e4m3fn`` only; the
            weights, scales and logits are fixed at ``torch.float32``.
        config: Optional dict with "block_N", "num_stages", "threads" and
            "block_Q".

    Raises:
        ValueError: If *dtype* is not ``torch.float8_e4m3fn``.
    """

    @classmethod
    def _fp8_lightning_indexer_run(
        cls,
        batch: int,
        seq_len: int,
        heads: int,
        index_dim: int,
        seq_len_kv: int,
        kv_group: int,
        clean_logits: bool,
        block_N: int,
        num_stages: int,
        threads: int,
        block_Q: int,
        IndexQ: torch.Tensor,
        IndexK: torch.Tensor,
        IndexKScale: torch.Tensor,
        Logits: torch.Tensor,
        Weights: torch.Tensor,
        CuSeqLenKS: torch.Tensor,
        CuSeqLenKE: torch.Tensor,
    ) -> None:
        _fp8_lightning_indexer_kernel(
            batch,
            seq_len,
            heads,
            index_dim,
            seq_len_kv,
            kv_group,
            cls._MIN_PIPELINED_BLOCK_Q,
        )(block_N, num_stages, threads, block_Q)(
            IndexQ.view(batch, seq_len * heads, index_dim),
            IndexK,
            IndexKScale,
            Logits,
            Weights,
            CuSeqLenKS,
            CuSeqLenKE,
        )
        if clean_logits:
            _clean_logits_(threads=threads)(Logits, CuSeqLenKS, CuSeqLenKE)

    supported_archs: list[int] = [89, 90]
    # Queries a block takes by default, widest first.
    _BLOCK_QS = (4, 2, 1)
    _BLOCK_N = 64
    _NUM_STAGES = 2
    _THREADS = 128
    # block_Q == 1 deadlocks the software pipeline (some warps never reach the
    # pipeline barrier); such tiles must run unpipelined. block_Q >= 2 is safe.
    _MIN_PIPELINED_BLOCK_Q = 2

    @classmethod
    def refusal(cls, call: FP8LightningIndexerCall) -> Optional[str]:
        """Why the call cannot fit the device's shared memory at one query a block, or ``None``."""
        if not call.smem_budget:
            return None
        need = cls._shared_bytes(1, call.heads, call.index_dim, call.kv_group)
        if need <= call.smem_budget:
            return None
        return (
            f"needs {need} bytes of shared memory per block at {call.heads} heads of dim "
            f"{call.index_dim} in {call.kv_group} groups; the device gives {call.smem_budget}"
        )

    @classmethod
    def _shared_bytes(cls, block_q: int, heads: int, index_dim: int, kv_group: int) -> int:
        """Shared memory of the default program, a byte an element: the block's queries, a key
        tile a pipeline stage (one tile at one query, which runs unpipelined), with several
        groups one group's keys and queries, and a reduction workspace of 4 bytes a thread."""
        stages = cls._NUM_STAGES if block_q >= cls._MIN_PIPELINED_BLOCK_Q else 1
        n = cls._BLOCK_N
        tiles = block_q * heads * index_dim + stages * n * kv_group * index_dim
        if kv_group > 1:
            tiles += n * index_dim + block_q * (heads // kv_group) * index_dim
        return tiles + 4 * cls._THREADS

    @classmethod
    def entry_for(cls, call: FP8LightningIndexerCall) -> Entry:
        return call, lambda: cls(
            call.batch,
            call.seq_len,
            call.heads,
            call.index_dim,
            call.seq_len_kv,
            call.kv_group,
            call.clean_logits,
        )

    def __init__(
        self,
        batch,
        seq_len,
        heads,
        index_dim,
        seq_len_kv,
        kv_group,
        clean_logits=True,
        dtype: torch.dtype = torch.float8_e4m3fn,
        config: Optional[dict] = None,
    ):
        super().__init__()
        if dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"FP8LightningIndexerKernel indexes float8_e4m3fn tensors, got {dtype}"
            )
        self.dtype = dtype
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.kv_group = kv_group
        self.clean_logits = clean_logits
        self.config = config

        self.kernel = _fp8_lightning_indexer_kernel(
            self.batch,
            self.seq_len,
            self.heads,
            self.index_dim,
            self.seq_len_kv,
            self.kv_group,
            self._MIN_PIPELINED_BLOCK_Q,
            self.clean_logits,
        )

        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return self._default_config_for(
            get_shared_memory_optin(self.device_index),
            self.heads,
            self.index_dim,
            self.kv_group,
        )

    @classmethod
    def _default_config_for(cls, budget: int, heads: int, index_dim: int, kv_group: int) -> dict:
        """The config this kernel builds at *budget* bytes of shared memory per block."""
        block_q = next(
            (
                q
                for q in cls._BLOCK_QS
                if cls._shared_bytes(q, heads, index_dim, kv_group) <= budget
            ),
            cls._BLOCK_QS[-1],
        )
        return {
            "block_N": cls._BLOCK_N,
            "num_stages": cls._NUM_STAGES,
            "threads": cls._THREADS,
            "block_Q": block_q,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        block_N = [32, 64, 128]
        num_stages = [0, 1, 2]
        threads = [128, 256]
        block_Q = [1, 2, 4]
        # block_Q == 1 runs unpipelined (above), so it is tried at num_stages 0 alone.
        _configs = [
            c
            for c in itertools.product(block_N, num_stages, threads, block_Q)
            if c[3] >= self._MIN_PIPELINED_BLOCK_Q or c[1] == 0
        ]

        configs = [
            {
                "block_N": c[0],
                "num_stages": c[1],
                "threads": c[2],
                "block_Q": c[3],
            }
            for c in _configs
        ]
        return configs

    def forward(
        self,
        IndexQ: torch.Tensor,  # type: ignore
        IndexK: torch.Tensor,  # type: ignore
        IndexKScale: torch.Tensor,  # type: ignore
        Weights: torch.Tensor,  # type: ignore
        CuSeqLenKS: torch.Tensor,  # type: ignore
        CuSeqLenKE: torch.Tensor,  # type: ignore
    ) -> torch.Tensor:
        Logits = torch.empty(
            [self.batch, self.seq_len, self.seq_len_kv, self.kv_group],
            device=IndexQ.device,
            dtype=torch.float32,
        )
        self._fp8_lightning_indexer_run(
            self.batch,
            self.seq_len,
            self.heads,
            self.index_dim,
            self.seq_len_kv,
            self.kv_group,
            self.clean_logits,
            self.config["block_N"],
            self.config["num_stages"],
            self.config["threads"],
            self.config["block_Q"],
            IndexQ,
            IndexK,
            IndexKScale,
            Logits,
            Weights,
            CuSeqLenKS,
            CuSeqLenKE,
        )
        return Logits

    def supply_prog(
        self, *args, **kwargs
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # TileLang invokes supply_prog with the candidate JIT params during autotuning;
        # ignore them and build benchmark inputs from the kernel's shape attributes.
        batch, seq_len, heads, index_dim = self.batch, self.seq_len, self.heads, self.index_dim
        seq_len_kv = self.seq_len_kv
        accum_dtype = torch.float32
        index_dtype = torch.int32
        fp8_dtype = torch.float8_e4m3fn  # IndexQ/IndexK are fp8 in the kernel signature
        # torch.randn does not support fp8 dtypes, so generate in fp16 then cast.
        IndexQ = torch.randn(
            batch, seq_len * heads, index_dim, device="cuda", dtype=torch.float16
        ).to(fp8_dtype)
        IndexK = torch.randn(
            batch, seq_len_kv, self.kv_group, index_dim, device="cuda", dtype=torch.float16
        ).to(fp8_dtype)
        IndexKScale = torch.randn(
            batch, seq_len_kv, self.kv_group, device="cuda", dtype=accum_dtype
        )
        Logits = torch.empty(
            batch, seq_len, seq_len_kv, self.kv_group, device="cuda", dtype=accum_dtype
        )
        Weights = torch.randn(seq_len, heads, device="cuda", dtype=accum_dtype)
        CuSeqLenKS = torch.zeros(seq_len, device="cuda", dtype=index_dtype)
        CuSeqLenKE = torch.full((seq_len,), fill_value=seq_len_kv, device="cuda", dtype=index_dtype)

        return IndexQ, IndexK, IndexKScale, Logits, Weights, CuSeqLenKS, CuSeqLenKE

    @property
    def autotune_supply_prog(self):
        # The kernel has scalar (int32) params; the default tensor-only input
        # generation cannot build them, so supply benchmark inputs explicitly.
        return self.supply_prog

    def autotune(self, warmup=10, rep=10):
        super().autotune(warmup=warmup, rep=rep)


@functools.lru_cache(maxsize=32)
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        # The roles synchronize by mbarriers; an inserted CTA-wide barrier between the serving
        # warpgroup's loop and the consumers' would deadlock them.
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
    compile_flags=["-O3", "-DNDEBUG", *csrc_include("lightning_indexer_helper.h")],
)
def _fp8_lightning_indexer_ws_kernel(
    batch, heads, index_dim, block_q, splits, clean_logits, prefetch_q, mixed=False
):
    """Warp-specialized logits over 64-key tiles.

    Warpgroups 0 and 1 take alternate tiles: one m64n``rows`` WGMMA each, then the
    relu-weighted head sums, stored. Warpgroup 2 serves them: warp 0 loads q and the key tiles
    by TMA, and warps 1 to 3 write -inf over the keys outside every window.

    ``mixed`` builds the one-batch-row launch that runs some query blocks whole and splits the
    rest: the launch arguments ``whole``, ``shift`` and ``per`` give its CTA mapping, its
    epilogue sums (x + |x|) * weight on the FMA pipe and halves the sum, and a consumer
    releases its key stage before the epilogue. Without it the launch arguments go unread.
    """
    dtype = T.float8_e4m3fn
    accum = T.float32
    rows = block_q * heads  # the WGMMA's N
    block_n = 64  # one m64 WGMMA a tile
    stages = 4  # two tiles in flight a consumer
    fill_n = 256  # the widest TMA box
    serving_regs = 24  # the serving warpgroup only issues TMA
    # setmaxnreg moves registers within the CTA's launch allocation: what the serving
    # warpgroup gives back goes to the two consumer warpgroups, in steps of 8.
    launch_regs = 65536 // 384 // 8 * 8
    consumer_regs = (launch_regs + (launch_regs - serving_regs) // 2) // 8 * 8
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    def acc_layout(thread_offset):
        """The m64n``rows`` WGMMA accumulator as the PTX arranges it, which the helper reads.

        Row i, column c sits at register 4 (c / 8) + 2 ((i % 16) / 8) + c % 2 of thread
        32 (i / 16) + 4 (i % 8) + (c % 8) / 2 of the warpgroup starting at *thread_offset*.
        """

        def forward_fn(i, c):
            thread = thread_offset + (i // 16) * 32 + (i % 8) * 4 + (c % 8) // 2
            return thread, (c // 8) * 4 + ((i % 16) // 8) * 2 + c % 2

        return tilelang.layout.Fragment([block_n, rows], forward_fn=forward_fn)

    @T.macro
    def consumer(wg, Logits, Weights, Qs, Ks, Ss, Win, q_full, k_full, k_free, b, s0, c0, count, in_lo, in_hi, tx):  # fmt: skip
        """Score tiles wg, wg + 2, ... of this CTA's computed range and store them."""
        T.set_max_nreg(consumer_regs, 1)
        acc = T.alloc_fragment([block_n, rows], accum)
        T.annotate_layout({acc: acc_layout(128 * wg)})
        # The head weight of each accumulator column this thread holds: column
        # 8 j + 2 (t % 4) + i at 2 j + i.
        wt = T.alloc_local([rows // 4], accum)
        t = tx % 128
        for j in T.unroll(rows // 8):
            for i in T.unroll(2):
                col = 8 * j + 2 * (t % 4) + i
                query = T.min(s0 + col // heads, seq_len - 1)
                if mixed:
                    # The clamp at 0 changes no index; it lets the compiler drop the bounds
                    # checks a run-time CTA mapping would otherwise put on the loads.
                    query = T.max(query, 0)
                wt[2 * j + i] = Weights[query, col % heads]
        # The offset of this thread's first stored logit at the walk's first key: query
        # (t % 4) / 2 at tile row 16 (t / 32) + (t % 32) / 4 + 8 (t % 2), the row the helper
        # derives from t; a tile adds its first key.
        row = 16 * (t // 32) + (t % 32) // 4 + 8 * (t % 2)
        off = T.alloc_var("int64", init=T.Cast("int64", s0 + (t % 4) // 2) * seq_len_kv + row)
        if count > 0:
            T.mbarrier_wait_parity(q_full[0], 0)
        mine = T.ceildiv(count - wg, 2)
        # Unrolled by the two stages this warpgroup owns, so each one's addresses are constants.
        for jj2 in T.serial(T.ceildiv(mine, 2)):
            for u in T.unroll(2):
                jj = 2 * jj2 + u
                if jj < mine:
                    stage = wg + 2 * u
                    T.mbarrier_wait_parity(k_full[stage], jj2 % 2)
                    T.wgmma_gemm(Ks[stage, :, :], Qs, acc, transpose_B=True, clear_accum=True)
                    T.wait_wgmma(0)
                    if mixed:
                        # The tile's last read of the stage is the scale of this thread's row:
                        # the stage goes back to the producer before the epilogue.
                        key_scale = T.alloc_var(accum, init=Ss[stage, row])
                        T.mbarrier_arrive(k_free[stage])
                    k0 = T.alloc_var("int32", init=(c0 + wg + 2 * jj) * block_n)
                    if mixed:
                        T.call_extern(
                            "handle",
                            f"tileops::indexer_tile_epilogue_doubled<{heads}, {block_q}, {str(clean_logits).lower()}>",
                            T.address_of(acc[0, 0]), T.address_of(wt[0]), key_scale,
                            T.address_of(Logits[b, 0, 0]), off, s0, k0, seq_len, seq_len_kv,
                            T.address_of(Win[0, 0]), k0 >= in_lo and k0 + block_n <= in_hi, t,
                        )  # fmt: skip
                    else:
                        T.call_extern(
                            "handle",
                            f"tileops::indexer_tile_epilogue<{heads}, {block_q}, {str(clean_logits).lower()}>",
                            T.address_of(acc[0, 0]), T.address_of(wt[0]), T.address_of(Ss[stage, 0]),
                            T.address_of(Logits[b, 0, 0]), off, s0, k0, seq_len, seq_len_kv,
                            T.address_of(Win[0, 0]), k0 >= in_lo and k0 + block_n <= in_hi, t,
                        )  # fmt: skip
                        T.mbarrier_arrive(k_free[stage])

    @T.prim_func
    def main(
        IndexQ: T.Tensor([batch, seq_len * heads, index_dim], dtype),  # type: ignore
        IndexK: T.Tensor([batch, seq_len_kv, index_dim], dtype),  # type: ignore
        IndexKScale: T.Tensor([batch, seq_len_kv], accum),  # type: ignore
        Logits: T.Tensor([batch, seq_len, seq_len_kv], accum),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], "int32"),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], "int32"),  # type: ignore
        whole: T.int32,
        shift: T.int32,
        per: T.int32,
    ):
        q_blocks = T.ceildiv(seq_len, block_q)
        grid = whole + T.shift_left(q_blocks - whole, shift) if mixed else q_blocks * splits
        with T.Kernel(grid, batch, threads=384) as (bx, b):
            tiles = T.ceildiv(seq_len_kv, block_n)
            if mixed:
                # Query blocks below `whole` run on one CTA each and own every key tile; each
                # later one splits its tiles over 2^shift CTAs, `per` a CTA. A power-of-two
                # split keeps division off the prologue.
                qi = T.alloc_var("int32", init=bx)
                t0 = T.alloc_var("int32", init=0)
                t1 = T.alloc_var("int32", init=tiles)
                if bx >= whole:
                    qi = whole + T.shift_right(bx - whole, shift)
                    t0 = T.min(T.bitwise_and(bx - whole, T.shift_left(1, shift) - 1) * per, tiles)
                    t1 = T.min(t0 + per, tiles)
                # The longest windows start first: a causal window grows with its query.
                qb = q_blocks - 1 - qi
            else:
                # The longest windows start first: a causal window grows with its query.
                qb = T.ceildiv(seq_len, block_q) - 1 - bx // splits
                sp = bx % splits
            s0 = qb * block_q
            Qs = T.alloc_shared([rows, index_dim], dtype)
            Ks = T.alloc_shared([stages, block_n, index_dim], dtype)
            Ss = T.alloc_shared([stages, block_n], accum)
            Neg = T.alloc_shared([block_q, fill_n], accum)
            NegN = T.alloc_shared([block_q, block_n], accum)
            Win = T.alloc_shared([2, block_q], "int32")
            T.annotate_layout({Qs: make_swizzled_layout(Qs), Ks: make_swizzled_layout(Ks)})
            q_full = T.alloc_barrier([32])
            k_full = T.alloc_barrier([32] * stages)
            k_free = T.alloc_barrier([128] * stages)
            tx = T.get_thread_binding()
            if prefetch_q or mixed:  # noqa: SIM102
                if tx >= 256 and tx < 288:
                    T.tma_copy(IndexQ[b, s0 * heads : s0 * heads + rows, :], Qs, barrier=q_full[0])
                    T.mbarrier_arrive(q_full[0])
            if tx < block_q:
                # A row past seq_len has an empty window.
                s = T.min(s0 + tx, seq_len - 1)
                if mixed:
                    s = T.max(s, 0)  # changes no index; drops the loads' bounds checks
                lo = T.max(T.min(CuSeqLenKS[s], seq_len_kv), 0)
                hi = T.max(T.min(CuSeqLenKE[s], seq_len_kv), lo)
                Win[0, tx] = T.if_then_else(s0 + tx < seq_len, lo, seq_len_kv)
                Win[1, tx] = T.if_then_else(s0 + tx < seq_len, hi, 0)
            T.sync_threads()
            lo_b = T.alloc_var("int32", init=seq_len_kv)
            hi_b = T.alloc_var("int32", init=0)
            # [in_lo, in_hi): keys inside every row's window, empty when a row is past seq_len.
            in_lo = T.alloc_var("int32", init=0)
            in_hi = T.alloc_var("int32", init=seq_len_kv)
            for q in T.unroll(block_q):
                lo_b = T.min(lo_b, Win[0, q])
                hi_b = T.max(hi_b, Win[1, q])
                in_lo = T.max(in_lo, Win[0, q])
                in_hi = T.min(in_hi, Win[1, q])
            if not clean_logits:
                # Without -inf, only the tensor's own end bounds a store.
                in_lo = 0
                in_hi = T.if_then_else(s0 + block_q <= seq_len, seq_len_kv, 0)
            if not mixed:
                # Split sp owns key tiles [t0, t1); it scores [c0, c0 + count) and fills the
                # rest. Variables, not expressions: inlined into every index and guard, the
                # expressions cost layout inference minutes of proofs.
                share = T.ceildiv(tiles, splits)
                t0 = T.alloc_var("int32", init=T.min(sp * share, tiles))
                t1 = T.alloc_var("int32", init=T.min(t0 + share, tiles))
            c0 = T.alloc_var("int32", init=T.max(t0, lo_b // block_n))
            count = T.alloc_var("int32", init=0)
            if lo_b < hi_b:
                count = T.max(T.min(t1, T.ceildiv(hi_b, block_n)) - c0, 0)
            if count == 0:
                c0 = t1
            if tx >= 256:
                T.set_max_nreg(serving_regs, 0)
                p = tx - 256
                if p < 32:
                    if prefetch_q or mixed:
                        if count == 0:
                            # A pending TMA load must complete before CTA exit.
                            T.mbarrier_wait_parity(q_full[0], 0)
                    elif count > 0:
                        T.tma_copy(
                            IndexQ[b, s0 * heads : s0 * heads + rows, :], Qs, barrier=q_full[0]
                        )
                        T.mbarrier_arrive(q_full[0])
                    for j in T.serial(count):
                        stage = j % stages
                        k0 = (c0 + j) * block_n
                        T.mbarrier_wait_parity(k_free[stage], ((j // stages) % 2) ^ 1)
                        T.tma_copy(
                            IndexK[b, k0 : k0 + block_n, :], Ks[stage, :, :],
                            barrier=k_full[stage],
                        )  # fmt: skip
                        T.tma_copy(
                            IndexKScale[b, k0 : k0 + block_n], Ss[stage, :],
                            barrier=k_full[stage],
                        )  # fmt: skip
                        T.mbarrier_arrive(k_full[stage])
                elif clean_logits:
                    # Keys outside every window: TMA stores from an -inf tile, clipped at
                    # seq_len and seq_len_kv.
                    for i, k in T.Parallel(block_q, fill_n):
                        Neg[i, k] = -T.infinity(accum)
                    for i, k in T.Parallel(block_q, block_n):
                        NegN[i, k] = -T.infinity(accum)
                    T.fence_proxy_async()
                    T.sync_threads(1, 96)
                    if p == 32:
                        for part in T.unroll(2):
                            k_lo = T.if_then_else(part == 0, t0, c0 + count) * block_n
                            k_hi = T.if_then_else(part == 0, c0, t1) * block_n
                            wide = (k_hi - k_lo) // fill_n
                            for f in T.serial(wide):
                                k = k_lo + f * fill_n
                                T.tma_copy(Neg, Logits[b, s0 : s0 + block_q, k : k + fill_n])
                            for f in T.serial((k_hi - k_lo - wide * fill_n) // block_n):
                                k = k_lo + wide * fill_n + f * block_n
                                T.tma_copy(
                                    NegN,
                                    Logits[b, s0 : s0 + block_q, k : k + block_n],
                                )  # fmt: skip
                        T.tma_store_wait()
            with T.ws(0):
                consumer(0, Logits, Weights, Qs, Ks, Ss, Win, q_full, k_full, k_free, b, s0, c0, count, in_lo, in_hi, tx)  # fmt: skip
            with T.ws(1):
                consumer(1, Logits, Weights, Qs, Ks, Ss, Win, q_full, k_full, k_free, b, s0, c0, count, in_lo, in_hi, tx)  # fmt: skip

    return main


class FP8LightningIndexerWSKernel(Kernel, FP8LightningIndexerFwdInterface):
    """SM90 warp-specialized lightning indexer over one key group.

    A thread block (CTA) holds 256 query-head rows, ``256 // heads`` queries, in shared memory.
    A producer warp streams 64-key tiles by TMA; two consumer warpgroups score alternate tiles,
    one m64n256 WGMMA each, and store the relu-weighted head sums. The producer warpgroup's
    other warps write -inf over the keys outside every window, so no clean pass follows. When
    the query blocks fall short of the multiprocessors, each block's keys split over CTAs.
    """

    supported_archs: list[int] = [90]
    preferred_over = frozenset({"fp8_lightning_indexer_kernel"})
    _ROWS = 256  # query-head rows a CTA holds: the WGMMA's N

    @classmethod
    def refusal(cls, call: FP8LightningIndexerCall) -> Optional[str]:
        """Why *call* is outside this implementation's region, or ``None`` when it is inside."""
        if call.kv_group != 1:
            return "scores one key group"
        if call.heads % 8 or cls._ROWS % call.heads or (cls._ROWS // call.heads) % 2:
            return "requires heads a multiple of 8 dividing 128"
        if call.index_dim not in (64, 128):
            return "requires index dimension 64 or 128"
        if call.seq_len_kv % 4:
            return "requires seq_len_kv a multiple of 4, the TMA stride of a logit row"
        return None

    @classmethod
    def splits_for(cls, call: FP8LightningIndexerCall) -> int:
        """CTAs a query block's keys split over: one while the blocks fill the multiprocessors,
        else enough to fill them, with at least two tiles a CTA. A split costs a CTA its own
        q load and prologue, which measured slower than idle multiprocessors past one block
        each."""
        q_blocks = -(-call.seq_len // (cls._ROWS // call.heads))
        if q_blocks >= call.sm_count:
            return 1
        tiles = -(-call.seq_len_kv // 64)  # the kernel's 64-key tiles
        return max(1, min(-(-call.sm_count // q_blocks), tiles // 2))

    @classmethod
    def partial_wave_for(cls, call: FP8LightningIndexerCall) -> Optional[tuple[int, int]]:
        """``(whole, pieces)`` when the call's query blocks leave a partial last wave on one
        batch row: the first ``whole`` blocks run on one CTA each and every later one splits its
        key tiles over ``pieces`` CTAs, a power of two. Only at 512 key tiles a block or more,
        in pieces of at least 64 tiles and at most 16 a block, the counts that measured
        fastest. ``None`` keeps every block whole."""
        q_blocks = -(-call.seq_len // (cls._ROWS // call.heads))
        tiles = -(-call.seq_len_kv // 64)  # the kernel's 64-key tiles
        if call.batch != 1 or q_blocks < call.sm_count or tiles < 512:
            return None
        partial = q_blocks % call.sm_count
        if not partial:
            return None
        return q_blocks - partial, min(16, 1 << ((tiles // 64).bit_length() - 1))

    @classmethod
    def entry_for(cls, call: FP8LightningIndexerCall) -> Entry:
        splits = cls.splits_for(call)
        partial_wave = cls.partial_wave_for(call)
        index = call.device.index if call.device is not None else None
        return (call, splits, partial_wave, index), lambda: cls(
            call.batch,
            call.seq_len,
            call.heads,
            call.index_dim,
            call.seq_len_kv,
            call.clean_logits,
            splits,
            partial_wave,
            device_index=index,
        )

    def __init__(
        self,
        batch: int,
        seq_len: int,
        heads: int,
        index_dim: int,
        seq_len_kv: int,
        clean_logits: bool = True,
        splits: int = 1,
        partial_wave: Optional[tuple[int, int]] = None,
        device_index: Optional[int] = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dtype = torch.float8_e4m3fn
        self.batch = batch
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.clean_logits = clean_logits
        if partial_wave is None:
            self.launch_args = (0, 0, 0)  # unread
        else:
            whole, pieces = partial_wave
            tiles = -(-seq_len_kv // 64)
            self.launch_args = (whole, pieces.bit_length() - 1, -(-tiles // pieces))
        self.kernel = _fp8_lightning_indexer_ws_kernel(
            batch,
            heads,
            index_dim,
            self._ROWS // heads,
            splits,
            clean_logits,
            prefetch_q=seq_len_kv <= 4096 * splits,
            mixed=partial_wave is not None,
        )

    def forward(
        self,
        IndexQ: torch.Tensor,
        IndexK: torch.Tensor,
        IndexKScale: torch.Tensor,
        Weights: torch.Tensor,
        CuSeqLenKS: torch.Tensor,
        CuSeqLenKE: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, heads, index_dim = IndexQ.shape
        seq_len_kv = IndexK.shape[1]
        logits = torch.empty(
            [batch, seq_len, seq_len_kv, 1], device=IndexQ.device, dtype=torch.float32
        )
        self.kernel(
            IndexQ.view(batch, seq_len * heads, index_dim),
            IndexK.view(batch, seq_len_kv, index_dim),
            IndexKScale.view(batch, seq_len_kv),
            logits.view(batch, seq_len, seq_len_kv),
            Weights,
            CuSeqLenKS,
            CuSeqLenKE,
            *self.launch_args,
        )
        return logits
