"""RMSNorm of each group of a row, scaled, shifted and gated by ``act(z)``.

For a row ``x``, its gate row ``z`` and a group of ``G`` columns of the row::

    h = x * act(z)                              (gate before the norm)
    y = h * rsqrt(mean(h^2) + eps) * w + b      (over the group)
    y = y * act(z)                              (gate after the norm)

``act`` is silu or sigmoid. Without ``z`` there is no gate, without ``bias`` no shift. The
arithmetic is float32 and ``y`` rounds once to the dtype of ``x``.
"""

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import LOG2E, VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.norm.call_spec import RMSNormGatedCall, RMSNormGatedFwdInterface
from tileops._utils import WARP_LANES

__all__ = ["RMSNormGatedKernel"]


@target_cache(maxsize=64)
def _rms_norm_gated_kernel(
    n: int,
    group: int,
    eps: float,
    norm_before_gate: bool,
    activation: str,
    dtype: str,
    weight_dtype: str,
    bias_dtype: "str | None",
    z_dtype: "str | None",
):
    groups = n // group
    # An access moves the widest of 16 bytes of x, or of a halving of it, that divides the
    # group.
    vec = VECTOR_ACCESS_BYTES // torch.empty((), dtype=getattr(torch, dtype)).element_size()
    while group % vec:
        vec //= 2
    accesses = group // vec
    # A lane makes two accesses where the group has two, so it has two loads of each tensor
    # in flight, and one where the group has one; a group of up to 64 accesses then fits a
    # warp or part of one. A wider group doubles its lanes until a lane makes at most four
    # accesses or the lanes reach 1024; past that a lane makes ceil(accesses / 1024).
    lanes = min(1 << (-(-accesses // 2) - 1).bit_length(), WARP_LANES)
    while -(-accesses // lanes) > 4 and lanes < 1024:
        lanes *= 2
    steps = -(-accesses // lanes)
    whole = lanes * steps * vec == group
    # Lanes of one warp that hold a group, and the warps a group spans.
    warp_lanes = min(lanes, WARP_LANES)
    group_warps = lanes // warp_lanes
    has_bias = bias_dtype is not None
    gated = z_dtype is not None
    # An absent bias or z is a one-element placeholder in the dtype of x, never read.
    b_dtype = bias_dtype if has_bias else dtype
    g_dtype = z_dtype if gated else dtype

    @tilelang.jit(out_idx=[-1])
    def _func(threads: int):
        # int64: a signature-valid call may hold 2^31 rows or more.
        rows = T.dynamic("rows", "int64")
        threads = max(threads, lanes)
        per_cta = threads // lanes
        held = steps * vec
        # The gate's exponential is formed scaled by 2^shift and scaled back once; see the
        # gate below. 2^64 keeps exp(-|z|) normal down to z = -132, past any float32 gate.
        shift = 64
        unshift = T.float32(2.0**-shift)

        @T.macro
        def add_lanes(ss, mask):
            ss[0] = ss[0] + T.shfl_xor(ss[0], mask)

        def warp_sum(ss):
            """Sum ``ss[0]`` over the warp's lanes of one group. A plain function: the builder
            would turn a loop inside the prim_func into a TIR loop."""
            mask = warp_lanes // 2
            while mask:
                add_lanes(ss, mask)
                mask //= 2

        @T.prim_func
        def main(
            x: T.Tensor((rows, n), dtype),
            weight: T.Tensor((n,), weight_dtype),
            bias: T.Tensor((n if has_bias else 1,), b_dtype),
            z: T.Tensor((rows, n) if gated else (1,), g_dtype),
            y: T.Tensor((rows, n), dtype),
        ):
            # A CTA normalizes `per_cta` consecutive groups, `lanes` threads a group.
            with T.Kernel(T.ceildiv(rows * groups, per_cta), threads=threads) as b:
                tx = T.get_thread_binding()
                lane = tx % lanes
                slot = tx // lanes
                s = T.cast(b, "int64") * per_cta + slot
                live = s < rows * groups
                row = s // groups
                start = s % groups * group
                xs = T.alloc_local((held,), dtype)
                ws = T.alloc_local((held,), weight_dtype)
                h = T.alloc_local((held,), "float32")
                ss = T.alloc_local((1,), "float32")
                # Allocated whether or not the call reads them; one it does not read is
                # dropped by the compiler.
                bs = T.alloc_local((held,), b_dtype)
                zs = T.alloc_local((held,), g_dtype)
                gate = T.alloc_local((held,), "float32")
                part = T.alloc_shared((max(threads // WARP_LANES, 1),), "float32")
                for k in T.unroll(steps):
                    base = (k * lanes + lane) * vec
                    # A slot no access fills adds nothing to the sum: a zero x, and a finite
                    # gate, since a gate before the norm multiplies x by it.
                    for i in T.unroll(vec):
                        xs[k * vec + i] = T.cast(0, dtype)
                        zs[k * vec + i] = T.cast(0, g_dtype)
                    if live and (whole or base < group):
                        for i in T.vectorized(vec):
                            xs[k * vec + i] = x[row, start + base + i]
                        if gated:
                            for i in T.vectorized(vec):
                                zs[k * vec + i] = z[row, start + base + i]
                        for i in T.vectorized(vec):
                            ws[k * vec + i] = weight[start + base + i]
                        if has_bias:
                            for i in T.vectorized(vec):
                                bs[k * vec + i] = bias[start + base + i]
                ss[0] = T.float32(0)
                for j in T.unroll(held):
                    h[j] = T.cast(xs[j], "float32")
                    if gated:
                        # With e = 2^64 exp(-|v|) and r = 1 / (1 + exp(-|v|)): sigmoid(v) is r
                        # for v >= 0 and e r 2^-64 below, silu(v) is v times that. The 2^64
                        # keeps e and v e normal down to v = -132, so a gate exp(-|v|) alone
                        # would flush or overflow comes from one final IEEE multiply, and a
                        # gate before the norm keeps its relative size. The approximate
                        # reciprocal, which flushes subnormals, only sees [1, 2].
                        v = T.cast(zs[j], "float32")
                        e = T.exp2(T.abs(v) * T.float32(-LOG2E) + T.float32(shift))
                        r = T.fast_rcp(T.float32(1) + e * unshift)
                        if activation == "silu":
                            gate[j] = T.if_then_else(v >= T.float32(0), v * r, v * e * r * unshift)
                        else:
                            gate[j] = T.if_then_else(v >= T.float32(0), r, e * r * unshift)
                        if not norm_before_gate:
                            h[j] = h[j] * gate[j]
                    ss[0] += h[j] * h[j]
                warp_sum(ss)
                if group_warps > 1:
                    if tx % WARP_LANES == 0:
                        part[tx // WARP_LANES] = ss[0]
                    T.sync_threads()
                    ss[0] = T.float32(0)
                    for w in T.unroll(group_warps):
                        ss[0] += part[slot * group_warps + w]
                rstd = T.rsqrt(ss[0] / T.float32(group) + T.float32(eps))
                for k in T.unroll(steps):
                    base = (k * lanes + lane) * vec
                    if live and (whole or base < group):
                        for i in T.unroll(vec):
                            j = k * vec + i
                            h[j] = h[j] * rstd * T.cast(ws[j], "float32")
                            if has_bias:
                                h[j] = h[j] + T.cast(bs[j], "float32")
                            if gated and norm_before_gate:
                                h[j] = h[j] * gate[j]
                            xs[j] = T.cast(h[j], dtype)
                        for i in T.vectorized(vec):
                            y[row, start + base + i] = xs[k * vec + i]

        return main

    return _func


class RMSNormGatedKernel(Kernel, RMSNormGatedFwdInterface):
    """Each group of a row held by a power-of-two group of lanes, up to 1024 of them.

    A lane makes one access in a group of one access, two to four in a group of up to 4096
    accesses, and ``ceil(accesses / 1024)`` in a wider group. Supports SM80+ architectures.
    A group of up to 32 lanes shares a warp with others and sums its squares by xor
    shuffles; a wider one spans warps, which add their sums through shared memory. The row count is a runtime value, so one build serves every batch.
    ``config["threads"]`` is the fewest threads a CTA takes; a CTA always holds whole groups.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: RMSNormGatedCall) -> Entry:
        identity = (
            call.n,
            call.group,
            call.eps,
            call.norm_before_gate,
            call.activation,
            call.dtype,
            call.weight_dtype,
            call.bias_dtype,
            call.z_dtype,
        )
        return identity, lambda: cls(*identity)

    def __init__(
        self,
        n: int,
        group: int,
        eps: float,
        norm_before_gate: bool,
        activation: str,
        dtype: torch.dtype,
        weight_dtype: torch.dtype,
        bias_dtype: "torch.dtype | None",
        z_dtype: "torch.dtype | None",
        config: dict | None = None,
    ) -> None:
        """Build for a row width, its group width, the gate and the tensors' dtypes.

        Args:
            n: Elements of a row.
            group: Elements each norm runs over; divides *n*.
            eps: Added to the mean square.
            norm_before_gate: Whether the gate multiplies the normalized row, else its input.
            activation: ``"silu"`` or ``"sigmoid"``.
            dtype: Of ``x`` and the output.
            weight_dtype: Of ``weight``.
            bias_dtype: Of ``bias``, or ``None`` when no bias is passed.
            z_dtype: Of ``z``, or ``None`` when no gate is passed.
            config: Optional ``{"threads"}`` override.
        """
        super().__init__()
        self.n = n
        self.dtype = dtype
        self.kernel = _rms_norm_gated_kernel(
            n,
            group,
            eps,
            norm_before_gate,
            activation,
            self.dtype_str,
            self.dtype_to_str(weight_dtype),
            None if bias_dtype is None else self.dtype_to_str(bias_dtype),
            None if z_dtype is None else self.dtype_to_str(z_dtype),
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # Four warps: at D = 128 a CTA holds sixteen groups, and on SM80 and SM90 sixteen
        # CTAs fill an SM's 64 warps.
        return {"threads": 128}

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        z: torch.Tensor | None,
    ) -> torch.Tensor:
        self._require_devices(x=x, weight=weight, bias=bias, z=z)
        rows = x.reshape(-1, self.n)
        placeholder = rows.new_empty(1)
        y = self.program(self._compile)(
            rows,
            weight,
            placeholder if bias is None else bias,
            placeholder if z is None else z.reshape(-1, self.n),
        )
        return y.reshape(x.shape)

    def _compile(self, cfg: dict):
        """The program, launched with *cfg*'s threads."""
        return self.kernel(**cfg)
