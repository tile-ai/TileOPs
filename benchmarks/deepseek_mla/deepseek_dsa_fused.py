import torch
from typing import Dict, Any, Optional


class DeepSeekDSAFusedBenchmark:
    """Benchmark for FusedDeepSeekSparseAttentionFunc"""

    def __init__(self,
                 seq_len_kv: int,
                 index_dim: int,
                 seq_len: int,
                 heads: int,
                 batch: int,
                 topk: int,
                 dim: int,
                 dim_tail: int,
                 stride_kv: int,
                 group_kv: int,
                 q_start_index_s: int,
                 quant_in_dtype: torch.dtype = torch.float16,
                 clean_logits: bool = True,
                 indexer_config: Optional[dict] = None,
                 in_dtype: str = "float16",
                 out_dtype: str = "int32",
                 sm_scale: Any = None,
                 is_causal: bool = True,
                 dsa_dtype: torch.dtype = torch.float16,
                 tune: bool = False):

        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.index_dim = index_dim
        self.heads = heads
        self.batch = batch
        self.topk = topk
        self.dim = dim
        self.dim_tail = dim_tail
        self.stride_kv = stride_kv
        self.group_kv = group_kv
        self.q_start_index_s = q_start_index_s

        self.quant_in_dtype = quant_in_dtype
        self.clean_logits = clean_logits
        self.indexer_config = indexer_config
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.dsa_dtype = dsa_dtype
        self.tune = tune

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def gen_inputs(self) -> Dict[str, torch.Tensor]:
        """Generate input tensors for DSA fused function"""
        return {
            "index_q":
                torch.randn(
                    self.batch,
                    self.seq_len,
                    self.heads,
                    self.index_dim,
                    dtype=self.in_dtype,
                    device=self.device).to(torch.float8_e4m3fn),
            "index_k":
                torch.randn(
                    self.batch,
                    self.seq_len_kv,
                    self.index_dim,
                    dtype=self.in_dtype,
                    device=self.device),
            "indexer_weights":
                torch.rand(self.seq_len, self.heads, dtype=torch.float32, device=self.device),
            "cu_seqlen_ks":
                torch.zeros(self.seq_len, dtype=torch.int32, device=self.device),
            "cu_seqlen_ke":
                torch.full((self.seq_len,), self.seq_len_kv, dtype=torch.int32, device=self.device),
            "starts":
                torch.zeros(self.batch, self.seq_len, dtype=torch.int32, device=self.device),
            "ends":
                torch.full((
                    self.batch,
                    self.seq_len,
                ),
                           self.seq_len_kv,
                           dtype=torch.int32,
                           device=self.device),
            "query":
                torch.randn(
                    self.batch,
                    self.seq_len,
                    self.heads,
                    self.dim,
                    dtype=self.in_dtype,
                    device=self.device),
            "kv_cache":
                torch.randn(
                    self.batch,
                    self.seq_len_kv,
                    self.heads,
                    self.dim + 8,
                    dtype=self.in_dtype,
                    device=self.device)
        }

    def check_fn(self,
                 fn: Any,
                 inputs: Dict[str, torch.Tensor],
                 atol: float = 1e-2,
                 rtol: float = 1e-2,
                 grad: bool = False) -> None:
        """Check function correctness"""
        output = fn(inputs["index_q"], inputs["index_k"], inputs["indexer_weights"],
                    inputs["cu_seqlen_ks"], inputs["cu_seqlen_ke"], inputs["starts"],
                    inputs["ends"], inputs["query"], inputs["kv_cache"])

        try:
            outputs_ref, cost = self.ref_program(*inputs)
            self.cost = cost
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  Skipped checking {self.__class__.__name__} due to OOM in ref: {e}")
                return
            raise e

        if isinstance(outputs_ref, torch.Tensor):
            outputs_ref = (outputs_ref,)
        elif not isinstance(outputs_ref, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs_ref)}")

        if not grad:
            with torch.no_grad():
                outputs = fn(*inputs)
        else:
            output = fn(*inputs)
            loss = output.sum()
            loss.backward()
            outputs = []
            outputs.append(output)
            for inp in inputs:
                outputs.append(inp.grad)

        if isinstance(outputs, list):
            outputs = tuple(outputs)
        elif isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        elif not isinstance(outputs, tuple):
            raise ValueError(f"Unsupported output type: {type(outputs)}")

        assert len(outputs) == len(outputs_ref), \
            f"outputs: {len(outputs)} and outputs_ref: {len(outputs_ref)} have different size"
        self.validate_tensor_match(outputs_ref, outputs, tolerance=1e-3, tensor_name="logits")

        print(f"All checks passed for {fn.__class__.__name__}.✅")

    def ref_program(self, index_q, index_k, indexer_weights, cu_seqlen_ks, cu_seqlen_ke, starts,
                    ends, query, kv_cache) -> torch.Tensor:
        k = index_k
        index_q = index_q.float()
        k = k.float()

        seq_len_kv = index_k.shape[0]
        mask_lo = torch.arange(0, seq_len_kv, device="cuda")[None, :] >= cu_seqlen_ks[:, None]
        mask_hi = torch.arange(0, seq_len_kv, device="cuda")[None, :] < cu_seqlen_ke[:, None]
        mask = mask_lo & mask_hi

        score = torch.einsum("mhd,nd->hmn", index_q, k)
        logits = (score.relu() * indexer_weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
        logits = logits.masked_fill(~mask, float("-inf"))
        indices = torch.topk(logits, self.topk, dim=-1)[1]

        query = query.float()
        kv_cache = kv_cache.float()
        indices = indices.transpose(1, 2)
        b, sq, h, dim_q = query.shape
        b, sk, g, _ = kv_cache.shape
        q_start_index_s = self.q_start_index_s
        if self.q_start_index_s is None:
            q_start_index_s = sk * self.stride_kv - sq

        assert kv_cache.shape[-1] == self.dim + self.dim_tail, 'you should assign dim otherwise'
        dim = self.dim
        k = kv_cache
        v = kv_cache[..., :dim]

        b, _, _, dim_v = v.shape
        g_index = g
        h_index = h // g
        compressed_causal_mask = torch.arange(
            q_start_index_s, sq + q_start_index_s, dtype=torch.int32,
            device="cuda").view(-1, 1) >= torch.arange(
                self.stride_kv - 1,
                sk * self.stride_kv,
                self.stride_kv,
                dtype=torch.int32,
                device="cuda").view(1, -1)

        mask = query.new_zeros(
            b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_causal_mask.view(1, 1, sq, sk)
        mask[:, :, :self.stride_kv - 1, 0] = True
        mask = mask.view(b, g_index, 1, sq, sk)

        query = query.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", query, k)
        sm_scale = dim_q**-0.5 if self.sm_scale is None else self.sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk)
        p = p.view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        return o.to(torch.float16)
