from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention.call_spec import DeltaNetFwdInterface, GDNCall
from tileops.kernels.linear_attention.chunk_prefill import chunk_refusal, run_chunk_prefill
from tileops.kernels.linear_attention.deltanet.dense_prefill import DeltaNetDensePrefillFwdKernel


def gated_call(call):
    return GDNCall(
        batch=call.batch,
        seq_len=call.seq_len,
        heads=call.heads,
        value_heads=call.heads,
        dim_k=call.dim_k,
        dim_v=call.dim_v,
        dtype=call.dtype,
        scale=call.scale,
        l2norm=call.l2norm,
        varlen=call.varlen,
        has_initial_state=call.has_initial_state,
        num_sequences=call.num_sequences,
        device=call.device,
    )


class DeltaNetChunkPrefillFwdKernel(Kernel, DeltaNetFwdInterface):
    supported_archs = [89]

    @classmethod
    def refusal(cls, call):
        return DeltaNetDensePrefillFwdKernel.refusal(call) or chunk_refusal(call)

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(gated_call(call))

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(self, q, k, v, beta, initial_state=None, cu_seqlens=None, cu_seqlens_cpu=None):
        self._require_devices(q=q, k=k, v=v, beta=beta)
        inputs = (q, k, v, None, beta, initial_state, cu_seqlens, None, None)
        return run_chunk_prefill(self.call, inputs, gate_mode="none")
