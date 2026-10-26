from tileops._kernels.kernel_base import Kernel
from tileops._kernels.linear_attention.call_spec import GDNFwdInterface
from tileops._kernels.linear_attention.chunk_prefill import chunk_refusal, run_chunk_prefill
from tileops._kernels.linear_attention.gdn.prefill import GDNDensePrefillFwdKernel


class GDNChunkPrefillFwdKernel(Kernel, GDNFwdInterface):
    supported_archs = [89]
    split_output = False

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
        return run_chunk_prefill(
            self, self.call, inputs, gate_mode="head", split_output=self.split_output
        )


class GDNSplitPrefillFwdKernel(GDNChunkPrefillFwdKernel):
    """Walk long sparse grids while forming their outputs in parallel over chunks."""

    preferred_over = frozenset({"gdn_chunk_prefill"})
    split_output = True

    @classmethod
    def refusal(cls, call):
        reason = super().refusal(call)
        if reason is not None:
            return reason
        if call.varlen or call.state_v_first or call.dim_k != 128:
            return "uses dense sequences with a key-major 128-wide state"
        # Below 256 chunks the extra output launch can outweigh the shorter walk.
        if call.seq_len < 16384:
            return "uses sequences of at least 256 chunks"
        # Sixteen value columns per CTA leave room for parallel output only on sparse grids.
        ctas = call.num_sequences * call.value_heads * (call.dim_v // 16)
        if ctas * 4 > call.sm_count:
            return "uses state walks occupying at most a quarter of the SMs"
        return None
