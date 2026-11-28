"""Rotary position embedding ops.

The ops of vLLM's ``RotaryEmbedding`` family hold a cos/sin cache computed from their
construction parameters and rotate ``query`` and ``key`` at the rows ``positions`` names,
as ``RotaryEmbedding.forward_native(positions, query, key)`` does:

- **RoPEFwdOp**: standard frequencies ``base ** (-2k / rotary_dim)``
- **RoPELlama31FwdOp**: Llama 3.1 piecewise-scaled frequencies
- **YaRNFwdOp**: YaRN interpolated frequencies and magnitude scaling
- **LongRoPEFwdOp**: per-pair rescaled frequencies, the Phi-3 LongRoPE computation once one
  set of factors is chosen

``query`` and ``key`` hold each token's heads flattened, ``head_size`` columns a head; the
first ``rotary_dim`` columns of each head rotate and the rest are copied.

**RoPECosSinFwdOp** rotates by cos and sin rows the caller passes, as flash-attn's
``apply_rotary_emb`` does.
"""

import math
from collections.abc import Mapping, Sequence
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.rope import (
    RoPECall,
    RoPECosSinCall,
    RoPECosSinFwdInterface,
    RoPECosSinKernel,
    RoPENeoxPositionIdsFwdInterface,
    RoPENeoxPositionIdsKernel,
    RoPENeoxStagedRowsKernel,
    RoPENonNeoxFwdInterface,
    RoPENonNeoxKernel,
    RoPENonNeoxStagedRowsKernel,
)
from tileops._ops._signature_codegen import param_state
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = [
    "LongRoPEFwdOp",
    "RoPECosSinFwdOp",
    "RoPEFwdOp",
    "RoPELlama31FwdOp",
    "YaRNFwdOp",
]

# The interface of each rotation convention.
_NEOX, _GPTJ = "neox", "gptj"


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    """The rotary dimension that turns *num_rotations* times over the context."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
    truncate: bool,
) -> tuple[float, float]:
    """The rotary dimensions between which YaRN blends interpolation into extrapolation."""
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    return max(low, 0), min(high, dim - 1)


class _RotaryEmbeddingOp(Op):
    """The rotation every RoPE op shares: cos/sin cache rows read at each token's position.

    A subclass gives its inverse frequencies, its cache rows and its magnitude scale. The
    cache is computed in float32, rounded to the ``dtype`` parameter and then to the query's
    dtype, as vLLM's cache is.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "rope_neox_positions": RoPENeoxPositionIdsKernel,
        "rope_gptj": RoPENonNeoxKernel,
        "rope_neox_staged_rows": RoPENeoxStagedRowsKernel,
        "rope_gptj_staged_rows": RoPENonNeoxStagedRowsKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        _NEOX: RoPENeoxPositionIdsFwdInterface,
        _GPTJ: RoPENonNeoxFwdInterface,
    }

    head_size: int
    rotary_dim: int
    is_neox_style: bool
    dtype: torch.dtype
    # Rewrite the rotary columns of query and key; only RoPEFwdOp takes it as a parameter.
    inplace: bool = False

    def _start(self, target: Target) -> None:
        self._freq_cache: dict[tuple, tuple] = {}
        super().__init__(target=target)

    def _inv_freq(self) -> torch.Tensor:
        """The ``rotary_dim // 2`` inverse frequencies, float32 on the CPU."""
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )

    def _cache_rows(self) -> int:
        """The rows of the cos/sin cache, which positions index."""
        return self.max_position_embeddings

    def _mscale(self) -> float:
        """The magnitude every cos and sin value is scaled by."""
        return 1.0

    def _tables(self, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
        """The cos and sin tables, ``(cache rows, rotary_dim // 2)`` in *dtype* on *device*.

        They are computed again once a parameter changes.
        """
        key = (dtype, device)
        sources = param_state(self, type(self)._signature.sig)
        held = self._freq_cache.get(key)
        if held is None or held[0] != sources:
            t = torch.arange(self._cache_rows(), dtype=torch.float32)
            freqs = torch.outer(t, self._inv_freq())
            mscale = self._mscale()
            tables = tuple(
                (table * mscale).to(self.dtype).to(device=device, dtype=dtype)
                for table in (freqs.cos(), freqs.sin())
            )
            held = self._freq_cache[key] = (sources, tables)
        return held[1]

    def _forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``query`` and ``key`` with each token's heads rotated at its cache row; in place,
        ``query`` and ``key`` themselves."""
        positions = positions.contiguous()
        offsets = None if offsets is None else offsets.contiguous()
        row_strides = None
        if self.inplace:
            row_strides = (_row_stride(query), 0 if key is None else _row_stride(key))
        else:
            query = query.contiguous()
            key = None if key is None else key.contiguous()
        cos, sin = self._tables(query.dtype, query.device)
        kernel = self.kernel_for(
            _NEOX if self.is_neox_style else _GPTJ,
            lambda: RoPECall(
                query_heads=query.shape[-1] // self.head_size,
                key_heads=0 if key is None else key.shape[-1] // self.head_size,
                head_size=self.head_size,
                rotary_dim=self.rotary_dim,
                max_position=cos.shape[0],
                dtype=query.dtype,
                position_dtype=positions.dtype,
                offset_dtype=None if offsets is None else offsets.dtype,
                row_strides=row_strides,
                device=query.device,
            ),
        )
        return kernel(positions, query, key, offsets, cos, sin)

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Rotate ``query`` and ``key`` at ``positions``.

        Args:
            positions: Each token's position, ``[*S]``, ``int32`` or ``int64``, in
                ``[0, cache rows)``.
            query: ``[*S, head_size * Hq]``, each token's heads flattened.
            key: ``[*S, head_size * Hk]``, or None.

        Returns:
            ``(query_out, key_out)``, shaped as ``query`` and ``key``; ``key_out`` is None
            where ``key`` is. In place they are ``query`` and ``key``.
        """
        return self._forward(positions, query, key)


def _row_stride(x: torch.Tensor) -> int:
    """The elements from one token's row of *x* to the next, for a call that writes *x*.

    Raises:
        ValueError: *x*'s token dimensions do not collapse to one stride, or its columns are
            not contiguous.
    """
    columns = x.shape[-1]
    if columns > 1 and x.stride(-1) != 1:
        raise ValueError(f"in place, the columns must be contiguous, got strides {x.stride()}")
    tokens = x.numel() // columns if columns else 0
    # A lone row's stride is never stepped, and a row of no columns is never read.
    if tokens <= 1:
        return columns
    try:
        return x.view(tokens, columns).stride(0)
    except RuntimeError as exc:
        raise ValueError(
            f"in place, the token dimensions of a {tuple(x.shape)} tensor of strides "
            f"{x.stride()} must collapse to one row stride"
        ) from exc


class RoPEFwdOp(_RotaryEmbeddingOp):
    """RoPE with standard frequencies, either rotation convention, in place when ``inplace``
    (vLLM ``RotaryEmbedding``).

    ``is_neox_style`` rotates the two halves of each head's rotary columns against each
    other; otherwise each adjacent pair rotates (GPT-J). ``inplace`` rewrites the rotary
    columns of ``query`` and ``key`` and leaves the rest untouched, as
    ``RotaryEmbedding.forward_cuda`` does; ``query`` and ``key`` may then be views whose
    token rows lie any stride apart, each row's columns contiguous.

    Positions in the cache are the caller's obligation, checked only with
    ``validate_inputs=True``; unchecked, a position outside the cache reads its nearest row,
    row 0 or the last.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        init_cache: bool = True,
        *,
        inplace: bool = False,
        validate_inputs: bool = False,
        target: Target = None,
    ):
        """Build the op and its cos/sin cache settings.

        Args:
            head_size: Columns per head.
            rotary_dim: Leading columns of each head that rotate; even, at most head_size.
            max_position_embeddings: Rows of the cos/sin cache.
            base: Frequency base.
            is_neox_style: Rotate halves (NeoX) rather than adjacent pairs (GPT-J).
            dtype: The cos/sin cache's dtype, before it takes the query's.
            init_cache: Must be True: the op computes its own cache.
            inplace: Rewrite the rotary columns of ``query`` and ``key`` and return them.
            validate_inputs: Check on the CPU that every position lies in the cache before
                the kernels run; this synchronizes the device and cannot run inside CUDA
                Graph capture.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.init_cache = init_cache
        self.inplace = inplace
        self.validate_inputs = validate_inputs
        self._start(target)


class RoPELlama31FwdOp(_RotaryEmbeddingOp):
    """Llama 3.1 RoPE with piecewise-scaled frequencies (vLLM ``Llama3RotaryEmbedding``).

    A frequency whose wavelength is below ``orig_max_position / high_freq_factor`` is kept,
    one above ``orig_max_position / low_freq_factor`` is divided by ``scaling_factor``, and
    one between blends the two.

    Positions in the cache are the caller's obligation, checked only with
    ``validate_inputs=True``; unchecked, a position outside the cache reads its nearest row,
    row 0 or the last.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ):
        """Build the op and its cos/sin cache settings.

        Args:
            head_size: Columns per head.
            rotary_dim: Leading columns of each head that rotate; even, at most head_size.
            max_position_embeddings: Rows of the cos/sin cache.
            base: Frequency base.
            is_neox_style: Rotate halves (NeoX) rather than adjacent pairs (GPT-J).
            dtype: The cos/sin cache's dtype, before it takes the query's.
            scaling_factor: Divisor of the low frequencies.
            low_freq_factor: Bounds the low-frequency band.
            high_freq_factor: Bounds the high-frequency band.
            orig_max_position: The context length the bands are measured against.
            validate_inputs: Check on the CPU that every position lies in the cache before
                the kernels run; this synchronizes the device and cannot run inside CUDA
                Graph capture.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        self.validate_inputs = validate_inputs
        self._start(target)

    def _inv_freq(self) -> torch.Tensor:
        inv_freqs = super()._inv_freq()
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor
        wave_len = 2 * math.pi / inv_freqs
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
        else:
            smooth = 0
        return torch.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor + smooth * inv_freqs,
            ),
        )


class YaRNFwdOp(_RotaryEmbeddingOp):
    """YaRN RoPE (vLLM ``YaRNScalingRotaryEmbedding``).

    Frequencies blend interpolation by ``scaling_factor`` into extrapolation between the
    correction dimensions ``beta_fast`` and ``beta_slow`` name; cos and sin are scaled by
    the YaRN magnitude. The cache holds ``max_position_embeddings * scaling_factor`` rows,
    rounded up.

    Positions in the cache are the caller's obligation, checked only with
    ``validate_inputs=True``; unchecked, a position outside the cache reads its nearest row,
    row 0 or the last.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        apply_yarn_scaling: bool = True,
        truncate: bool = True,
        validate_inputs: bool = False,
        target: Target = None,
    ):
        """Build the op and its cos/sin cache settings.

        Args:
            head_size: Columns per head.
            rotary_dim: Leading columns of each head that rotate; even, at most head_size.
            max_position_embeddings: The original context length.
            base: Frequency base.
            is_neox_style: Rotate halves (NeoX) rather than adjacent pairs (GPT-J).
            scaling_factor: Context extension factor.
            dtype: The cos/sin cache's dtype, before it takes the query's.
            extrapolation_factor: Weight of the extrapolated frequencies.
            attn_factor: Magnitude scale of cos and sin.
            beta_fast: Rotations bounding the extrapolated band.
            beta_slow: Rotations bounding the interpolated band.
            apply_yarn_scaling: Scale the magnitude by YaRN's ``0.1 * ln(scale) + 1``.
            truncate: Round the correction dimensions outward to integers.
            validate_inputs: Check on the CPU that every position lies in the cache before
                the kernels run; this synchronizes the device and cannot run inside CUDA
                Graph capture.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.scaling_factor = scaling_factor
        self.dtype = dtype
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.apply_yarn_scaling = apply_yarn_scaling
        self.truncate = truncate
        self.validate_inputs = validate_inputs
        self._start(target)

    def _cache_rows(self) -> int:
        return math.ceil(self.max_position_embeddings * self.scaling_factor)

    def _mscale(self) -> float:
        if not self.apply_yarn_scaling:
            return float(self.attn_factor)
        scale = self.scaling_factor
        return float((1.0 if scale <= 1 else 0.1 * math.log(scale) + 1.0) * self.attn_factor)

    def _inv_freq(self) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim
        )
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
            self.truncate,
        )
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(self.rotary_dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1
        )
        mask = (1 - ramp) * self.extrapolation_factor
        return (1.0 / (self.scaling_factor * pos_freqs)) * (1 - mask) + (1.0 / pos_freqs) * mask


class LongRoPEFwdOp(_RotaryEmbeddingOp):
    """LongRoPE: vLLM ``Phi3LongRoPEScaledRotaryEmbedding`` once one set of factors is chosen.

    Phi-3 chooses at construction its long factors and long magnitude when the serving
    context exceeds ``original_max_position_embeddings``, else its short ones; the caller
    passes the chosen factors as ``rescale_factors`` (None for every factor 1) and the chosen
    magnitude as ``mscale`` (None for ``sqrt(1 + ln(scale) / ln(original))`` with
    ``scale = max_position_embeddings / original_max_position_embeddings``, 1 where
    ``scale <= 1``). Token n reads cache row ``positions[n] + offsets[n]``, which lies in
    ``[0, max_position_embeddings)``, and in ``[0, original_max_position_embeddings)`` with
    the short factors. Both bounds are the caller's obligation; ``validate_inputs=True``
    checks ``positions`` alone against the cache, not the sum. Unchecked, a row outside the
    cache reads its nearest row, row 0 or the last. The rotation is NeoX, and the cache dtype
    is the query's.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        original_max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        rescale_factors: Sequence[float] | None,
        mscale: float | None = None,
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ):
        """Build the op and its cos/sin cache settings.

        Args:
            head_size: Columns per head.
            rotary_dim: Leading columns of each head that rotate; even, at most head_size.
            max_position_embeddings: Rows of the cos/sin cache.
            original_max_position_embeddings: The original context length.
            base: Frequency base.
            is_neox_style: Must be True.
            dtype: The cos/sin cache's dtype, the query's.
            rescale_factors: ``rotary_dim // 2`` divisors of the frequencies, or None for 1.
            mscale: Magnitude scale of cos and sin, or None for the default.
            validate_inputs: Check on the CPU that every position lies in the cache before
                the kernels run; this synchronizes the device and cannot run inside CUDA
                Graph capture.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.

        Raises:
            ValueError: A rescale factor is zero or NaN; the frequency divides by it.
        """
        if rescale_factors is not None and any(f == 0 or math.isnan(f) for f in rescale_factors):
            raise ValueError("rescale_factors must be nonzero and not NaN")
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.rescale_factors = rescale_factors
        self.mscale = mscale
        self.validate_inputs = validate_inputs
        self._start(target)

    def _inv_freq(self) -> torch.Tensor:
        inv_freq = super()._inv_freq()
        if self.rescale_factors is None:
            return inv_freq
        return inv_freq / torch.tensor(self.rescale_factors, dtype=torch.float32)

    def _mscale(self) -> float:
        if self.mscale is not None:
            return self.mscale
        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Rotate ``query`` and ``key`` at ``positions + offsets``.

        Args:
            positions: Each token's position, ``[N]``, ``int32`` or ``int64``.
            query: ``[N, head_size * Hq]``, each token's heads flattened.
            key: ``[N, head_size * Hk]``; required, as Phi-3 requires it.
            offsets: ``[N]`` added to ``positions``, or None.

        Returns:
            ``(query_out, key_out)``, shaped as ``query`` and ``key``.
        """
        return self._forward(positions, query, key, offsets)


class RoPECosSinFwdOp(Op):
    """Rotary embedding by caller-given cos and sin rows, in place when ``inplace``.

    Follows ``flash_attn.layers.rotary.apply_rotary_emb``. Token ``s`` of sequence ``b``
    reads row ``s + seqlen_offsets`` of ``cos`` and ``sin`` ``[P, Rh]`` (one offset for
    every sequence, or one per sequence) and rotates the first ``2 * Rh`` columns of each
    head; the rest are copied, or left as they are in place. ``interleaved`` pairs
    ``(x[2i], x[2i + 1])``, otherwise ``(x[i], x[i + Rh])`` (NeoX). The rotation is computed
    in float32 and rounded once to ``x``'s dtype; ``cos`` and ``sin`` may each have their own
    dtype.

    ``x`` is dense ``[B, S, H, D]``, or packed ``[Tt, H, D]`` with ``cu_seqlens``, ``s``
    counting from each sequence's start. ``x`` may be a strided view, such as the trailing
    columns of a wider head, which ``inplace`` rotates without a copy. Passing ``-sin``
    applies the inverse rotation.

    A row inside the tables is the caller's obligation, checked for tensor offsets only with
    ``validate_inputs=True``. Unchecked, a row outside the tables reads its nearest row, row 0
    or the last. ``max_seqlen`` bounds the packed sequences for the caller; the rotation does
    not read it.
    """

    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"rope_cos_sin": RoPECosSinKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "rope_cos_sin": RoPECosSinFwdInterface
    }

    def __init__(
        self,
        interleaved: bool = False,
        inplace: bool = False,
        max_seqlen: int | None = None,
        *,
        validate_inputs: bool = False,
        target: Target = None,
    ):
        """Build the op.

        Args:
            interleaved: Rotate adjacent pairs rather than the two halves (NeoX).
            inplace: Write the rotation into ``x`` and return ``x``.
            max_seqlen: The longest packed sequence; required with ``cu_seqlens``.
            validate_inputs: Check on the CPU that ``cu_seqlens`` starts at 0, never
                decreases, ends at the token count and holds no sequence longer than
                ``max_seqlen``, and that each tensor of
                ``seqlen_offsets`` lies in ``[0, P - L]``, before the kernel runs; this
                synchronizes the device and cannot run inside CUDA Graph capture.
            target: Which set of kernels serves this op — a target name, ``BUILTIN``
                for the in-tree kernels, or ``None`` to decide from the input device.
        """
        self.interleaved = interleaved
        self.inplace = inplace
        self.max_seqlen = max_seqlen
        self.validate_inputs = validate_inputs
        super().__init__(target=target)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        seqlen_offsets: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        """Rotate the leading ``2 * Rh`` columns of each head of ``x``.

        Args:
            x: ``[B, S, H, D]``, or ``[Tt, H, D]`` with ``cu_seqlens``; any strides.
            cos: ``[P, Rh]``.
            sin: ``[P, Rh]``.
            cu_seqlens: ``[B + 1]`` sequence starts of packed ``x``, ``int32`` or
                ``int64``, or None for dense ``x``.
            seqlen_offsets: The table row of each sequence's first token: an int, or
                ``[B]`` ``int32`` or ``int64``.

        Returns:
            ``x`` when ``inplace``, else a new tensor of its shape and dtype.
        """
        cos, sin = cos.contiguous(), sin.contiguous()
        cu_seqlens = None if cu_seqlens is None else cu_seqlens.contiguous()
        offsets_tensor = isinstance(seqlen_offsets, torch.Tensor)
        if offsets_tensor:
            seqlen_offsets = seqlen_offsets.contiguous()
        kernel = self.kernel_for(
            "rope_cos_sin",
            lambda: RoPECosSinCall(
                heads=x.shape[-2],
                head_dim=x.shape[-1],
                rotary_half=cos.shape[-1],
                interleaved=self.interleaved,
                inplace=self.inplace,
                dtype=x.dtype,
                cos_dtype=cos.dtype,
                sin_dtype=sin.dtype,
                seqlens_dtype=None if cu_seqlens is None else cu_seqlens.dtype,
                offset_dtype=seqlen_offsets.dtype if offsets_tensor else None,
                device=x.device,
            ),
        )
        return kernel(x, cos, sin, cu_seqlens, seqlen_offsets)
