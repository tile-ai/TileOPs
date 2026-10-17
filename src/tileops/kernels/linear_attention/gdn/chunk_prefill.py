from tileops.kernels.kernel_base import Kernel
from tileops.kernels.linear_attention.call_spec import GDNFwdInterface
from tileops.kernels.linear_attention.chunk_prefill import chunk_refusal, run_chunk_prefill
from tileops.kernels.linear_attention.gdn.prefill import GDNDensePrefillFwdKernel


class GDNChunkPrefillFwdKernel(Kernel, GDNFwdInterface):
    supported_archs = [89]

    @classmethod
    def refusal(cls, call):
        return GDNDensePrefillFwdKernel.refusal(call) or chunk_refusal(call)

    @classmethod
    def entry_for(cls, call):
        return call, lambda: cls(call)

    def __init__(self, call):
        super().__init__(device_index=call.device.index)
        self.call = call

    def forward(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        A_log=None,
        dt_bias=None,
    ):
        self._require_devices(q=q, k=k, v=v, g=g, beta=beta)
        inputs = (q, k, v, g, beta, initial_state, cu_seqlens, A_log, dt_bias)
        return run_chunk_prefill(self.call, inputs, gate_mode="head")
