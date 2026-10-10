import dataclasses
import functools
import itertools
from abc import abstractmethod
from typing import Optional

import tilelang
import torch
from tilelang import language as T

from tileops.kernels.call_spec import CallSpec
from tileops.kernels.kernel_base import Entry, Kernel, KernelInterface
from tileops.utils import get_shared_memory_optin

__all__ = [
    "FP8LightningIndexerCall",
    "FP8LightningIndexerFwdInterface",
    "FP8LightningIndexerKernel",
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


# block_Q == 1 deadlocks the software pipeline (some warps never reach the
# pipeline barrier); such tiles must run unpipelined. block_Q >= 2 is safe.
_MIN_PIPELINED_BLOCK_Q = 2


@functools.lru_cache(maxsize=32)
def _fp8_lightning_indexer_kernel(
    batch, seq_len, heads, index_dim, seq_len_kv, kv_group, clean_logits=True
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
        if block_Q < _MIN_PIPELINED_BLOCK_Q:
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

    @staticmethod
    def _fp8_lightning_indexer_run(
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
        _fp8_lightning_indexer_kernel(batch, seq_len, heads, index_dim, seq_len_kv, kv_group)(
            block_N, num_stages, threads, block_Q
        )(
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
        stages = cls._NUM_STAGES if block_q >= _MIN_PIPELINED_BLOCK_Q else 1
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
            if c[3] >= _MIN_PIPELINED_BLOCK_Q or c[1] == 0
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
