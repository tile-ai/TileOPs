"""Rotary position embedding ops, following vLLM's ``RotaryEmbedding`` family.

Each op holds a cos/sin cache computed from its construction parameters and rotates
``query`` and ``key`` at the rows ``positions`` names, as
``RotaryEmbedding.forward_native(positions, query, key)`` does:

- **RoPEFwdOp**: standard frequencies ``base ** (-2k / rotary_dim)``
- **RoPELlama31FwdOp**: Llama 3.1 piecewise-scaled frequencies
- **YaRNFwdOp**: YaRN interpolated frequencies and magnitude scaling
- **LongRoPEFwdOp**: per-pair rescaled frequencies, the Phi-3 LongRoPE computation once one
  set of factors is chosen

``query`` and ``key`` hold each token's heads flattened, ``head_size`` columns a head; the
first ``rotary_dim`` columns of each head rotate and the rest are copied.
"""

import math
from collections.abc import Mapping, Sequence
from typing import ClassVar

import torch

from tileops._kernels.kernel_base import Kernel, KernelInterface
from tileops._kernels.rope import (
    RoPECall,
    RoPENeoxPositionIdsCall,
    RoPENeoxPositionIdsFwdInterface,
    RoPENeoxPositionIdsKernel,
    RoPENonNeoxFwdInterface,
    RoPENonNeoxKernel,
)
from tileops._ops._signature_codegen import param_state
from tileops._ops.op_base import Op
from tileops.backend import Target

__all__ = [
    "LongRoPEFwdOp",
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
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        _NEOX: RoPENeoxPositionIdsFwdInterface,
        _GPTJ: RoPENonNeoxFwdInterface,
    }

    head_size: int
    rotary_dim: int
    is_neox_style: bool
    dtype: torch.dtype

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

    def _rotate(
        self, x: torch.Tensor, index: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """*x*, ``[*S, head_size * H]``, with each token's heads rotated at its row *index*.

        A row outside the cache is clamped to it.
        """
        tokens, rows = index.numel(), cos.shape[0]
        heads = x.numel() // max(tokens * self.head_size, 1)
        flat = x.reshape(tokens, heads, self.head_size).contiguous()
        if self.is_neox_style:
            kernel = self.kernel_for(
                _NEOX,
                lambda: RoPENeoxPositionIdsCall(
                    num_tokens=tokens,
                    num_heads=heads,
                    head_dim=self.head_size,
                    rotary_dim=self.rotary_dim,
                    max_position=rows,
                    dtype=flat.dtype,
                    position_dtype=index.dtype,
                    device=flat.device,
                ),
            )
            return kernel(flat, cos, sin, index.contiguous()).reshape(x.shape)
        part = flat[..., : self.rotary_dim].contiguous().unsqueeze(0)
        kernel = self.kernel_for(
            _GPTJ,
            lambda: RoPECall(
                seq_len=tokens,
                head_dim=self.rotary_dim,
                batch=1,
                num_heads=heads,
                dtype=flat.dtype,
                device=flat.device,
            ),
        )
        # Clamped as the NeoX kernel clamps, so the gather never reads past the cache.
        clamped = index.clamp(0, rows - 1)
        rotated = kernel(part, cos[clamped].contiguous(), sin[clamped].contiguous())[0]
        if self.rotary_dim < self.head_size:
            rotated = torch.cat((rotated, flat[..., self.rotary_dim :]), dim=-1)
        return rotated.reshape(x.shape)

    def _forward(
        self, index: torch.Tensor, query: torch.Tensor, key: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        index = index.reshape(-1)
        cos, sin = self._tables(query.dtype, query.device)
        query_out = self._rotate(query, index, cos, sin)
        key_out = None if key is None else self._rotate(key, index, cos, sin)
        return query_out, key_out

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
            where ``key`` is.
        """
        return self._forward(positions, query, key)


class RoPEFwdOp(_RotaryEmbeddingOp):
    """RoPE with standard frequencies, either rotation convention (vLLM ``RotaryEmbedding``).

    ``is_neox_style`` rotates the two halves of each head's rotary columns against each
    other; otherwise each adjacent pair rotates (GPT-J).

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
        index = positions if offsets is None else positions + offsets
        return self._forward(index, query, key)
