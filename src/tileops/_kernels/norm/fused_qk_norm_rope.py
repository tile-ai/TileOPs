"""Per-head RMSNorm of the q and k heads of a packed projection, then RoPE, in place.

``qkv`` is ``[N, (H + 2 Hk) D]``: H q heads, Hk k heads, Hk v heads, D values each. For token
``n`` and q or k head ``h``, over the head's values ``x``::

    y = x * rsqrt(mean(x^2) + eps) * w            (w = q_weight or k_weight; 1 + w zero-centered)
    c, s = cos_sin_cache[positions[n], :R/2], cos_sin_cache[positions[n], R/2:]
    neox:        (y[i], y[i + R/2]) <- (y[i] c[i] - y[i + R/2] s[i], y[i + R/2] c[i] + y[i] s[i])
    interleaved: (y[2i], y[2i + 1]) <- (y[2i] c[i] - y[2i + 1] s[i], y[2i + 1] c[i] + y[2i] s[i])

With interleaved multimodal sections, ``positions`` is ``[3, N]`` and frequency ``i`` takes
``c[i]``, ``s[i]`` from the row of ``positions[1, n]`` where ``i % 3 == 1 and i < 3 * sections[1]``,
of ``positions[2, n]`` where ``i % 3 == 2 and i < 3 * sections[2]``, else of ``positions[0, n]``.

Values past R pass through normalized; the v heads are not touched. The arithmetic is float32
and each value rounds once to the qkv dtype.
"""

import operator

import tilelang
import tilelang.language as T
import torch

from tileops._kernels.build_target import target_cache
from tileops._kernels.constants import VECTOR_ACCESS_BYTES
from tileops._kernels.kernel_base import Entry, Kernel
from tileops._kernels.norm._config import rms_scale
from tileops._kernels.norm.call_spec import FusedQKNormRoPECall, FusedQKNormRoPEFwdInterface
from tileops._kernels.warp_reduce import warp_reduce
from tileops._utils import WARP_LANES

__all__ = ["FusedQKNormRoPEKernel"]


def _vector(dim: int, rot: int, dtype: torch.dtype) -> int:
    """Values a lane moves per access: the widest of one 16-byte vector, 4, 2 or 1 whose pairs
    tile both the rotated and the passed-through values. Needs an even ``dim``."""
    widest = VECTOR_ACCESS_BYTES // dtype.itemsize
    return next(v for v in (widest, 4, 2, 1) if dim % (2 * v) == 0 and rot % (2 * v) == 0)


@target_cache(maxsize=64)
def _fused_qk_norm_rope_kernel(
    heads: int,
    kv_heads: int,
    dim: int,
    rot: int,
    interleaved: bool,
    eps: float,
    dtype: str,
    cache_dtype: str,
    vec: int,
    zero_centered: bool,
    mrope: bool,
    height_end: int,
    width_end: int,
):
    """``mrope`` reads ``[3, N]`` positions of interleaved multimodal sections, whose height and
    width frequencies end at ``height_end = 3 * sections[1]`` and ``width_end = 3 * sections[2]``."""
    qk = heads + kv_heads
    width = (heads + 2 * kv_heads) * dim
    half = rot // 2
    # A lane holds two accesses of vec values, so a head is dim / (2 vec) lanes, padded to a
    # power of two so its lanes reduce by xor shuffles inside one warp.
    lanes = dim // (2 * vec)
    group = 1 << (lanes - 1).bit_length()
    # NeoX: lane l < rot_lanes holds values [l vec, +vec) and the vec values R/2 after them, a
    # rotated pair; lane rot_lanes + j holds [R + j vec, +vec) and the vec values (D - R)/2
    # after them, passed through. Both offsets stay multiples of vec, so the accesses vectorize.
    # Interleaved: lane l holds 2 vec consecutive values, whose pairs are adjacent.
    rot_lanes = half // vec
    rest = (dim - rot) // 2

    # A multimodal row picks each value's cache row with selects the bounds check would branch
    # around; the op's contract keeps positions in [0, P), so that program carries none.
    unchecked = {tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True}

    @tilelang.jit(out_idx=[], pass_configs=unchecked if mrope else None)
    def _func(threads: int):
        tokens = T.dynamic("tokens")
        positions_max = T.dynamic("positions_max")
        position_shape = (3, tokens) if mrope else (tokens,)
        rows = threads // group
        blocks = -(-qk // rows)

        @T.macro
        def rotate(x, i, j, c, s):
            """Rotate the pair (x[i], x[j]) by the angle whose cosine and sine are c and s."""
            a = x[i]
            x[i] = a * c - x[j] * s
            x[j] = x[j] * c + a * s

        @T.prim_func
        def main(
            qkv: T.Tensor((tokens, width), dtype),
            q_weight: T.Tensor((dim,), dtype),
            k_weight: T.Tensor((dim,), dtype),
            cos_sin_cache: T.Tensor((positions_max, rot), cache_dtype),
            positions: T.Tensor(position_shape, "int64"),
        ):
            # A CTA is `rows` consecutive q/k heads of one token, `group` lanes a head. CTAs
            # are numbered token-major, so the CTAs resident at once read one stretch of qkv.
            with T.Kernel(tokens * blocks, threads=threads) as b:
                n = b // blocks
                bh = b % blocks
                tx = T.get_thread_binding()
                lane = tx % group
                h = bh * rows + tx // group
                live = T.And(h < qk, lane < lanes)
                # A lane's pairs all rotate or all pass through: R is a multiple of 2 vec.
                rotates = T.And(live, lane < rot_lanes)
                if interleaved:
                    o1 = lane * (2 * vec)
                    o2 = o1 + vec
                    cbase = lane * vec
                else:
                    passed = T.Cast("int32", lane >= rot_lanes)
                    o1 = lane * vec + passed * half
                    o2 = o1 + half + passed * (rest - half)
                    cbase = o1
                va = T.alloc_local((vec,), dtype)
                vb = T.alloc_local((vec,), dtype)
                wa = T.alloc_local((vec,), dtype)
                wb = T.alloc_local((vec,), dtype)
                cr = T.alloc_local((vec,), cache_dtype)
                sr = T.alloc_local((vec,), cache_dtype)
                x = T.alloc_local((2 * vec,), "float32")
                ss = T.alloc_local((1,), "float32")
                pos = T.alloc_local((3 if mrope else 1,), "int64")
                if mrope:
                    phase_row = T.alloc_local((3,), "int64")
                    phase_end = T.alloc_local((3,), "int32")
                for v in T.unroll(vec):
                    va[v] = T.cast(0, dtype)
                    vb[v] = T.cast(0, dtype)
                # Every load is issued before the first one is waited on: the position first,
                # whose cos/sin row is the one dependent load, then the weights and the row.
                if rotates:
                    if mrope:
                        for a in T.unroll(3):
                            pos[a] = positions[a, n]
                    else:
                        pos[0] = positions[n]
                if live:
                    if h < heads:
                        for v in T.vectorized(vec):
                            wa[v] = q_weight[o1 + v]
                        for v in T.vectorized(vec):
                            wb[v] = q_weight[o2 + v]
                    else:
                        for v in T.vectorized(vec):
                            wa[v] = k_weight[o1 + v]
                        for v in T.vectorized(vec):
                            wb[v] = k_weight[o2 + v]
                    for v in T.vectorized(vec):
                        va[v] = qkv[n, h * dim + o1 + v]
                    for v in T.vectorized(vec):
                        vb[v] = qkv[n, h * dim + o2 + v]
                if rotates:
                    if mrope:
                        # The lane's first frequency fixes v's residue, so the row and section
                        # end are picked once a residue (k = v % 3); each value reads its row.
                        first = cbase % 3
                        temporal = pos[0]
                        for k in T.unroll(3):
                            residue = (first + k) % 3
                            phase_row[k] = T.if_then_else(
                                residue == 1,
                                pos[1],
                                T.if_then_else(residue == 2, pos[2], temporal),
                            )
                            phase_end[k] = T.if_then_else(
                                residue == 1,
                                height_end,
                                T.if_then_else(residue == 2, width_end, half),
                            )
                        for v in T.unroll(vec):
                            row = T.if_then_else(
                                cbase + v < phase_end[v % 3], phase_row[v % 3], temporal
                            )
                            cr[v] = cos_sin_cache[row, cbase + v]
                            sr[v] = cos_sin_cache[row, half + cbase + v]
                    else:
                        for v in T.vectorized(vec):
                            cr[v] = cos_sin_cache[pos[0], cbase + v]
                        for v in T.vectorized(vec):
                            sr[v] = cos_sin_cache[pos[0], half + cbase + v]
                ss[0] = T.float32(0)
                for v in T.unroll(vec):
                    x[v] = T.cast(va[v], "float32")
                    x[vec + v] = T.cast(vb[v], "float32")
                    ss[0] += x[v] * x[v] + x[vec + v] * x[vec + v]
                warp_reduce(ss, operator.add, lanes=group)
                scale = T.rsqrt(ss[0] / T.float32(dim) + T.float32(eps))
                for v in T.unroll(vec):
                    x[v] = x[v] * scale * rms_scale(T.cast(wa[v], "float32"), zero_centered)
                    x[vec + v] = (
                        x[vec + v] * scale * rms_scale(T.cast(wb[v], "float32"), zero_centered)
                    )
                if rotates:
                    for u in T.unroll(vec):
                        c = T.cast(cr[u], "float32")
                        s = T.cast(sr[u], "float32")
                        if interleaved:
                            # Pair u is values 2u, 2u + 1 of the lane's 2 vec.
                            rotate(x, 2 * u, 2 * u + 1, c, s)
                        else:
                            rotate(x, u, vec + u, c, s)
                for v in T.unroll(vec):
                    va[v] = T.cast(x[v], dtype)
                    vb[v] = T.cast(x[vec + v], dtype)
                if live:
                    for v in T.vectorized(vec):
                        qkv[n, h * dim + o1 + v] = va[v]
                    for v in T.vectorized(vec):
                        qkv[n, h * dim + o2 + v] = vb[v]

        return main

    return _func


class FusedQKNormRoPEKernel(Kernel, FusedQKNormRoPEFwdInterface):
    """Each (token, q or k head) row held by a group of lanes in one warp, two accesses a lane.

    Args:
        heads: q heads, ``H``.
        kv_heads: k heads, ``Hk`` (as many v heads follow, untouched).
        dim: Values a head, ``D``.
        rot: Rotated values a head, ``R``.
        interleaved: Whether the rotation pairs adjacent values instead of NeoX halves.
        eps: Added to the mean square.
        dtype: float16 or bfloat16, of qkv and the weights.
        cache_dtype: float16, bfloat16 or float32, of the cos/sin cache.
        config: Optional ``{"threads"}`` override.
        device_index: The device the kernel runs on.
        zero_centered: Whether to scale by ``1 + weight``, formed in float32.
        mrope_ends: ``(3 * sections[1], 3 * sections[2])`` of interleaved multimodal sections,
            whose ``positions`` are ``[3, N]``; ``None`` for one position a token.
    """

    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def refusal(cls, call: FusedQKNormRoPECall) -> str | None:
        if call.mrope_section is not None and not call.mrope_interleaved:
            return (
                "the kernel assigns the three positions' frequencies interleaved; "
                "consecutive sections (mrope_interleaved=False) are not served"
            )
        if call.dim % 2:
            return f"a lane holds pairs of values, so the head dim must be even, not {call.dim}"
        lanes = call.dim // (2 * _vector(call.dim, call.rot, call.dtype))
        if lanes > WARP_LANES:
            return (
                f"a head of {call.dim} values with {call.rot} rotated needs {lanes} lanes; "
                "its sum of squares is reduced inside one warp of 32"
            )
        return super().refusal(call)

    @classmethod
    def entry_for(cls, call: FusedQKNormRoPECall) -> Entry:
        args = (
            call.heads,
            call.kv_heads,
            call.dim,
            call.rot,
            call.interleaved,
            call.eps,
            call.dtype,
            call.cache_dtype,
        )
        sections = call.mrope_section
        options = {
            "zero_centered": call.zero_centered,
            "mrope_ends": None if sections is None else (3 * sections[1], 3 * sections[2]),
        }
        index = call.device_index
        key = (*args, *options.values(), index)
        return key, lambda: cls(*args, **options, device_index=index)

    def __init__(
        self,
        heads: int,
        kv_heads: int,
        dim: int,
        rot: int,
        interleaved: bool,
        eps: float,
        dtype: torch.dtype,
        cache_dtype: torch.dtype,
        config: dict | None = None,
        device_index: int | None = None,
        *,
        zero_centered: bool = False,
        mrope_ends: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(device_index=device_index)
        self.dtype = dtype
        self.vec = _vector(dim, rot, dtype)
        self.kernel = _fused_qk_norm_rope_kernel(
            heads,
            kv_heads,
            dim,
            rot,
            interleaved,
            eps,
            self.dtype_str,
            str(cache_dtype).removeprefix("torch."),
            self.vec,
            zero_centered,
            mrope_ends is not None,
            *(mrope_ends or (0, 0)),
        )
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        # Eight heads of one token at D = 128.
        return {"threads": 64}

    def forward(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        self.program(self._compile)(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
        )

    def _compile(self, cfg: dict):
        """The program, launched with *cfg*'s threads."""
        return self.kernel(**cfg)
