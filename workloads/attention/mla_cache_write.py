"""Inputs and the independent reference for the MLA latent cache write."""

import math

import torch

from workloads.device import run_device
from workloads.numerics import Custom, Exact
from workloads.workload_base import CallWorkload, WorkloadBase

__all__ = [
    "MLAKVCacheWriteCall",
    "MLAKVCacheWriteWorkload",
    "mla_cache_write_result",
]


def mla_cache_write_result(subject, *inputs):
    """Expose the declared cache write as the numeric result of an in-place op."""
    subject(*inputs)
    return inputs[2]


def _split_pairs(x: torch.Tensor, layout: str):
    """The two members of each rotation pair: halves for neox, even and odd for interleaved."""
    half = x.shape[1] // 2
    if layout == "neox":
        return x[:, :half], x[:, half:]
    return x[:, 0::2], x[:, 1::2]


def _join_pairs(first: torch.Tensor, second: torch.Tensor, layout: str) -> torch.Tensor:
    """The inverse of ``_split_pairs``."""
    if layout == "neox":
        return torch.cat([first, second], -1)
    return torch.stack([first, second], -1).flatten(1)


def _float32_outward(value: torch.Tensor, direction: float) -> torch.Tensor:
    """``value`` rounded to float32 away from zero in ``direction``, so no bound shrinks."""
    rounded = value.float()
    short = rounded.double() > value if direction < 0 else rounded.double() < value
    return torch.where(
        short, torch.nextafter(rounded, torch.full_like(rounded, direction)), rounded
    )


class MLAKVCacheWriteWorkload(WorkloadBase):
    """New latent rows scattered into a paged cache by slot, the cache's other rows left alone.

    ``slots[n]`` is the cache row token ``n`` writes, or ``-1`` for a token that writes
    nothing. Omitted, the slots are distinct random rows, which is what a serving step hands
    over; a caller passing its own list exercises the skip path, a slot at either end of a
    page, and slots out of order. ``positions`` likewise default to random rows of the
    cos/sin table.
    """

    def __init__(
        self,
        tokens: int,
        latent_dim: int,
        rope_dim: int,
        page_size: int,
        num_pages: int,
        dtype: torch.dtype,
        cache_dtype: torch.dtype | None = None,
        fuse_rope: bool = False,
        rope_layout: str = "neox",
        cos_sin_dtype: torch.dtype | None = None,
        max_position: int = 0,
        slots: list[int] | None = None,
        positions: list[int] | None = None,
    ) -> None:
        self.tokens, self.latent_dim, self.rope_dim = tokens, latent_dim, rope_dim
        self.page_size, self.num_pages = page_size, num_pages
        self.dtype = dtype
        self.cache_dtype = dtype if cache_dtype is None else cache_dtype
        self.fuse_rope, self.rope_layout = fuse_rope, rope_layout
        self.cos_sin_dtype = cos_sin_dtype
        self.max_position = max_position
        self.slots, self.positions = slots, positions

    @property
    def physical_tokens(self) -> int:
        return self.num_pages * self.page_size

    @property
    def quantized(self) -> bool:
        return self.cache_dtype == torch.float8_e4m3fn

    def _metadata(self, values: list[int] | None, name: str, high: int, distinct: bool):
        device = run_device()
        if values is not None:
            if len(values) != self.tokens:
                raise ValueError(f"{name} holds {len(values)} rows for {self.tokens} tokens")
            return torch.tensor(values, dtype=torch.int64, device=device)
        if distinct:
            if self.tokens > high:
                raise ValueError(
                    f"{self.tokens} tokens need {self.tokens} distinct slots but the cache "
                    f"holds {high} rows"
                )
            drawn = torch.randperm(high, generator=self.rng(name))[: self.tokens]
        else:
            drawn = torch.randint(0, high, (self.tokens,), generator=self.rng(name))
        return drawn.to(device=device, dtype=torch.int64)

    def gen_inputs(self):
        device = run_device()
        kv_c = torch.randn(
            (self.tokens, self.latent_dim),
            dtype=torch.float32,
            device=device,
            generator=self.rng("kv_c", device=device),
        ).to(self.dtype)
        k_pe = torch.randn(
            (self.tokens, self.rope_dim),
            dtype=torch.float32,
            device=device,
            generator=self.rng("k_pe", device=device),
        ).to(self.dtype)
        kv_cache = torch.randn(
            (self.num_pages, self.page_size, self.latent_dim + self.rope_dim),
            dtype=torch.float32,
            device=device,
            generator=self.rng("kv_cache", device=device),
        ).to(self.cache_dtype)
        slots = self._metadata(self.slots, "slots", self.physical_tokens, distinct=True)
        # A non-unit divisor: a kernel that drops it stores something the reference does not.
        scale = torch.tensor([0.75], dtype=torch.float32, device=device) if self.quantized else None
        positions = cos_sin = None
        if self.fuse_rope:
            positions = self._metadata(self.positions, "positions", self.max_position, False)
            # Real angles, base 10000: cos^2 + sin^2 = 1, the magnitude a rotation evaluated
            # in 16-bit arithmetic is bounded by in verification().
            steps = torch.arange(0, self.rope_dim, 2, dtype=torch.float64) / self.rope_dim
            angles = torch.arange(self.max_position, dtype=torch.float64)[:, None] / (
                10000.0 ** steps[None, :]
            )
            cos_sin = torch.cat([angles.cos(), angles.sin()], -1)
            cos_sin = cos_sin.to(device=device, dtype=self.cos_sin_dtype)
        return kv_c, k_pe, kv_cache, slots, scale, positions, cos_sin

    def ref_program(
        self, kv_c, k_pe, kv_cache, slot_mapping, scale=None, positions=None, cos_sin_cache=None
    ):
        """Scatter by index, with saturating FP8 conversion that preserves NaN.

        The cache is cloned before it is written: the reference runs on the shared inputs,
        and a call that wrote them would leave nothing to compare against. A token whose
        slot is ``-1`` writes nothing, and a row no token names keeps the value it was
        handed, which the caller's comparison reads back.
        """
        kv_cache = kv_cache.clone()
        written = (slot_mapping >= 0).nonzero().squeeze(1)
        pe = k_pe[written]
        if positions is not None:
            # Each pair rotated in float32 and rounded once to k_pe's dtype.
            cos, sin = cos_sin_cache[positions[written]].float().chunk(2, -1)
            a, b = _split_pairs(pe.float(), self.rope_layout)
            pe = _join_pairs(a * cos - b * sin, b * cos + a * sin, self.rope_layout)
            pe = pe.to(k_pe.dtype)
        stored = torch.cat([kv_c[written], pe], -1)
        if scale is not None:
            # Express saturation explicitly: some torch FP8 casts turn overflow into NaN.
            # torch.clamp preserves NaN while bounding other values.
            limit = torch.finfo(kv_cache.dtype).max
            stored = (stored.float() / scale[0]).clamp(-limit, limit)
        kv_cache.view(-1, kv_cache.shape[-1])[slot_mapping[written]] = stored.to(kv_cache.dtype)
        return kv_cache

    def verification(
        self, kv_c, k_pe, kv_cache, slot_mapping, scale=None, positions=None, cos_sin_cache=None
    ):
        if positions is None:
            # The stored bytes are the whole result: a named row is written exactly and a
            # row no slot names must still hold what it held, so no tolerance is owed.
            return Exact(rtol=0, atol=0)
        written = (slot_mapping >= 0).nonzero().squeeze(1)
        rows = slot_mapping[written]
        layout, dtype, cache_dtype = self.rope_layout, self.dtype, self.cache_dtype
        latent = self.latent_dim
        table = cos_sin_cache[positions[written]]
        cos, sin = table.double().chunk(2, -1)
        a, b = _split_pairs(k_pe[written].double(), layout)
        exact = _join_pairs(a * cos - b * sin, b * cos + a * sin, layout)

        def stored(value):
            landed = value.to(dtype).float()
            if scale is None:
                return landed.double()
            limit = torch.finfo(cache_dtype).max
            return (landed / scale[0]).clamp(-limit, limit).to(cache_dtype).double()

        # Two evaluations are the rotation. In float32, each operation rounds once with unit
        # roundoff 2^-24: the sum always, a product only where it is not exact in float32
        # (a float32 table). Rounding to the dtype, dividing by the scale and the FP8 cast
        # are monotone, so the stored value lies between the images of the ends of that
        # interval around the exact rotation.
        bits = sum(round(-math.log2(torch.finfo(t).eps)) + 1 for t in (dtype, table.dtype))
        slack = exact.abs()
        if bits > 24:
            slack = slack + _join_pairs(
                (a * cos).abs() + (b * sin).abs(), (b * cos).abs() + (a * sin).abs(), layout
            )
        slack = 2.0**-24 * slack
        # An infinite product leaves no rounding to bound.
        slack = torch.where(torch.isfinite(slack), slack, torch.zeros_like(slack))
        low = stored(_float32_outward(exact - slack, -float("inf")))
        high = stored(_float32_outward(exact + slack, float("inf")))
        # In the input dtype, with a table of that dtype: each product rounds to the dtype,
        # then their sum does, one value per element.
        in_dtype = None
        if table.dtype == dtype:
            c, s = table.chunk(2, -1)
            x, y = _split_pairs(k_pe[written], layout)
            in_dtype = stored(_join_pairs(x * c - y * s, y * c + x * s, layout).float())

        def within_rotation_rounding(got, expected):
            def raw(t):
                return t.view(torch.uint8) if scale is not None else t

            # Every value outside the rotated columns of the written rows is exact.
            flat_got, flat_expected = got.view(-1, got.shape[-1]), expected.view(-1, got.shape[-1])
            unrotated = torch.ones(flat_got.shape, dtype=torch.bool, device=got.device)
            unrotated[rows, latent:] = False
            assert torch.equal(raw(flat_got)[unrotated], raw(flat_expected)[unrotated]), (
                "a value outside the rotated k_pe columns differs from the reference"
            )
            value = flat_got[rows, latent:].double()
            ok = ((value >= low) & (value <= high)) | (value.isnan() & exact.isnan())
            if in_dtype is not None:
                ok |= (value == in_dtype) | (value.isnan() & in_dtype.isnan())
            assert bool(ok.all()), (
                f"{int((~ok).sum())} rotated values match neither a float32 rotation nor one "
                "in the input dtype"
            )

        return Custom(
            within_rotation_rounding,
            reason="a rotation evaluated in float32 or, with a table of the input dtype, in "
            "that dtype",
        )


class MLAKVCacheWriteCall(CallWorkload, MLAKVCacheWriteWorkload):
    """The write manifest call, including the device-resident slots and positions."""

    def __init__(self, call):
        CallWorkload.__init__(self, call)
        fuse_rope = bool(call.params.get("fuse_rope", False))
        MLAKVCacheWriteWorkload.__init__(
            self,
            tokens=call.indices["N"],
            latent_dim=call.indices["DC"],
            rope_dim=call.indices["PE"],
            page_size=call.indices["PS"],
            num_pages=call.indices["NP"],
            dtype=getattr(torch, call.tensors["kv_c"][1]),
            cache_dtype=getattr(torch, call.tensors["kv_cache"][1]),
            fuse_rope=fuse_rope,
            rope_layout=call.params.get("rope_layout", "neox"),
            cos_sin_dtype=getattr(torch, call.tensors["cos_sin_cache"][1]) if fuse_rope else None,
            max_position=call.indices["P"] if fuse_rope else 0,
            slots=call.metadata_values("slot_mapping"),
            positions=call.metadata_values("positions") if fuse_rope else None,
        )

    gen_inputs = MLAKVCacheWriteWorkload.gen_inputs

    def arguments(self):
        """The op is built from its two parameters alone, so a case needs no tensor for it.

        The base materializes the call to hand a construction-time tensor over; this entry
        declares none, and materializing would allocate the cache and the cos/sin table for
        nothing.
        """
        return self.call.arguments({})
