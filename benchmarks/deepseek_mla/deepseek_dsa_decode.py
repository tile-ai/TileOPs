from benchmarks.benchmark import Benchmark
from top.ops import DeepSeekSparseAttentionDecodeWithKVCacheOp
import torch


class DeepSeekSparseAttentionDecodeBenchmark(Benchmark):

    op_type = DeepSeekSparseAttentionDecodeWithKVCacheOp

    def __init__(self,
                 batch,
                 heads,
                 seq_len,
                 seq_len_kv,
                 dim,
                 tail_dim,
                 topk,
                 kv_stride,
                 kv_group,
                 q_start_index_s,
                 sm_scale=None,
                 is_causal=True,
                 dtype=torch.float16):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.seq_len_kv = seq_len_kv
        self.dim = dim
        self.tail_dim = tail_dim
        self.topk = topk
        self.kv_stride = kv_stride
        self.kv_group = kv_group
        self.sm_scale = sm_scale
        self.is_causal = is_causal
        self.dtype = dtype
        self.q_start_index_s = q_start_index_s

    @property
    def total_flops(self):
        flops = self.batch * self.seq_len * (2 * self.dim +
                                             self.tail_dim) * self.topk * 2 * self.heads
        return flops

    @property
    def total_memory(self):
        # Q: batch, seq_len, heads, dim + tail_dim
        # KV: batch, seq_len_kv, kv_group, dim + tail_dim
        # Indices: batch, seq_len, kv_group, topk
        # Output: batch, seq_len, heads, dim
        q_memory = self.batch * self.seq_len * self.heads * (self.dim +
                                                             self.tail_dim) * self.dtype.itemsize
        kv_memory = self.batch * self.seq_len_kv * self.kv_group * (
            self.dim + self.tail_dim) * self.dtype.itemsize
        indices_memory = self.batch * self.seq_len * self.kv_group * self.topk * 4  # int32
        output_memory = self.batch * self.seq_len * self.heads * self.dim * self.dtype.itemsize
        return q_memory + kv_memory + indices_memory + output_memory

    def gen_inputs(self):
        Q = torch.randn(
            self.batch,
            self.seq_len,
            self.heads,
            self.dim + self.tail_dim,
            device='cuda',
            dtype=self.dtype)
        KV = torch.randn(
            self.batch,
            self.seq_len_kv,
            self.kv_group,
            self.dim + self.tail_dim,
            device='cuda',
            dtype=self.dtype)
        Indices = torch.full((self.batch, self.seq_len, self.kv_group, self.topk),
                             self.seq_len_kv,
                             dtype=torch.int32,
                             device='cuda')
        for b in range(self.batch):
            for t in range(self.seq_len):
                for h in range(self.kv_group):
                    i_i = torch.randperm(
                        min(
                            max(1, ((t + int(self.q_start_index_s)) // self.kv_stride)),
                            self.seq_len_kv))[:self.topk]
                    Indices[b, t, h, :len(i_i)] = i_i
        return Q, KV, Indices

    def ref_program(self, Q: torch.Tensor, KV: torch.Tensor, Indices: torch.Tensor):
        q = Q.float()
        kv = KV.float()
        indices = Indices.transpose(1, 2)
        b, sq, h, dim_q = q.shape
        b, sk, g, _ = kv.shape
        q_start_index_s = self.q_start_index_s
        if self.q_start_index_s is None:
            q_start_index_s = sk * self.kv_stride - sq

        assert kv.shape[-1] == self.dim + self.tail_dim, 'you should assign dim otherwise'
        dim = self.dim
        k = kv
        v = kv[..., :dim]

        b, _, _, dim_v = v.shape
        g_index = g
        h_index = h // g
        compressed_causal_mask = torch.arange(
            q_start_index_s, sq + q_start_index_s, dtype=torch.int32,
            device="cuda").view(-1, 1) >= torch.arange(
                self.kv_stride - 1,
                sk * self.kv_stride,
                self.kv_stride,
                dtype=torch.int32,
                device="cuda").view(1, -1)

        mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_causal_mask.view(1, 1, sq, sk)
        mask[:, :, :self.kv_stride - 1, 0] = True
        mask = mask.view(b, g_index, 1, sq, sk)

        q = q.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", q, k)
        sm_scale = dim_q**-0.5 if self.sm_scale is None else self.sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk)
        p = p.view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        return o.to(torch.float16)
