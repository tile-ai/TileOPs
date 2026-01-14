from typing import Tuple
from typing import Dict, Optional

import torch
import tilelang
from tilelang import language as T


from benchmarks.benchmark import Benchmark
from top.ops import Fp8LightingIndexerOp

class Fp8LightingIndexerBenchmark(Benchmark):

    op_type = Fp8LightingIndexerOp

    def __init__(self,
                seq_len, 
                heads, 
                index_dim, 
                seq_len_kv,  
                clean_logits=True,
                config: Optional[dict] = None,
                # kernel_map: Optional[Dict[str, Kernel]] = None,
                is_causal: bool = True,
                ):
        # self.q = q
        # self.kv = kv
        # self.kv_scales = kv_scales
        # self.weights = weights
        # self.cu_seqlen_ks = cu_seqlen_ks
        # self.cu_seqlen_ke = cu_seqlen_ke
        self.seq_len = seq_len
        self.heads = heads
        self.index_dim = index_dim
        self.seq_len_kv = seq_len_kv
        self.clean_logits = clean_logits
        self.config = config
        # self.dtype = torch.float8_e4m3fn
        self.accum_dtype = torch.float32
        self.index_dtype = torch.int32


    @property
    def total_flops(self) -> float: #todo 
        flops = self.seq_len * self.heads * self.seq_len_kv * self.index_dim *2 
        return flops

    @property
    def total_memory(self) -> float: #unidentified dtype
        # IndexQ: seq_len * heads, index_dim
        # IndexK: seq_len_kv, index_dim
        # IndexKScale: seq_len_kv
        # Logits: seq_len, seq_len_kv
        # Weights: seq_len, heads
        # CuSeqLenKS: seq_len
        # CuSeqLenKE: seq_len
        seq_len, heads, index_dim = self.q.shape
        seq_len_kv = kv.shape[0]

        index_q_memory = self.seq_len * self.heads * self.index_dim * self.dtype.itemsize
        index_k_memory = self.seq_len_kv * self.index_dim * self.dtype.itemsize
        index_k_scale_memory = self.seq_len_kv * self.accum_dtype.itemsize
        logits_memory = self.seq_len * self.seq_len_kv * self.accum_dtype.itemsize
        weights_memory = self.seq_len * self.head * self.accum_dtype.itemsize
        cu_seqlens_ks_memory = self.seq_len * self.index_dtype.itemsize
        cu_seqlens_ke_memory = self.seq_len * self.index_dtype.itemsize

        return index_q_memory + index_k_memory + index_k_scale_memory + logits_memory + weights_memory + cu_seqlens_ks_memory + cu_seqlens_ke_memory


    def gen_inputs(self, 
                    params=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor]:
        IndexQ = torch.randn(self.seq_len * self.heads, self.index_dim, device='cuda', dtype=torch.bfloat16)
        IndexK = torch.randn(self.seq_len_kv, self.index_dim, device='cuda', dtype=torch.bfloat16)
        # IndexKScale = torch.randn(self.seq_len_kv, device='cuda', dtype=self.accum_dtype)
        Weights = torch.randn(self.seq_len, self.heads, device='cuda', dtype=self.accum_dtype)
        CuSeqLenKS = torch.zeros(self.seq_len, device='cuda', dtype=self.index_dtype)
        CuSeqLenKE = torch.full((self.seq_len,), fill_value=self.seq_len_kv - 1, device='cuda', dtype=self.index_dtype)

        return IndexQ, IndexK, Weights, CuSeqLenKS, CuSeqLenKE


    def ref_program(q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor, cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor):
        k = kv
        q = q.float()
        k = k.float()

        seq_len_kv = kv.shape[0]
        mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
        mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
        mask = mask_lo & mask_hi

        score = torch.einsum("mhd,nd->hmn", q, k)
        logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
        logits = logits.masked_fill(~mask, float("-inf"))

        cost = mask.sum()
        return logits, cost

    def per_custom_dims_cast_to_fp8(x: torch.Tensor, dims: Tuple[int], use_ue8m0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
        x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
        sf = x_amax / 448.0
        sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
        x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
        return x_scaled, sf.squeeze()