# GQA Prefill Interface

**Status:** Accepted interface decision
**Date:** 2026-08-14

## Scope

This document defines the public Op and manifest boundary for GQA prefill. It
does not decide how the underlying kernels are factored, shared, fused, or
scheduled. Those kernel-layer decisions follow from, but do not change, the
interfaces below.

## Decision

GQA prefill has three public Ops, separated by input data topology:

| Op | Public layout | Sequence structure |
| --- | --- | --- |
| `GroupedQueryAttentionPrefillDenseFwdOp` | BSHD | Fixed query and KV lengths within a batch |
| `GroupedQueryAttentionPrefillVarlenFwdOp` | Packed THD plus cumulative sequence lengths | Ragged query and KV lengths |
| `GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` | Packed THD query/new KV plus paged KV storage | Ragged query lengths and an existing paged cache |

The Op boundary does not dynamically convert or dispatch between these three
topologies. In particular:

- the dense Op directly accepts BSHD tensors;
- the varlen Op never attempts to recover a BSHD view from packed inputs;
- the paged Op always addresses KV through the page table;
- no hot-path CPU inspection of cumulative-sequence tensor values is required
  to select an Op topology.

Full attention, causal masking, sliding-window masking, softcap, and numeric
format are semantic or implementation dimensions within each applicable Op.
They do not create additional public Ops.

## Public interfaces

Dense, Varlen, and Paged all expose the same optional fused-RoPE semantics.
Callers provide prepared cosine and sine tables so model-specific frequency
scaling (Llama 3, YaRN, LongRoPE, and similar policies) stays outside the
attention Op. Paged additionally owns KV-cache persistence and therefore
defines whether the stored K is rotated. RoPE does not create a fourth Op or a
separate manifest dispatch role.

The common semantic parameters are:

```text
is_causal: bool = True
sm_scale: float | None = None
softcap: float | None = None
window_size_left: int = -1
window_size_right: int = -1
dtype: torch.dtype
fuse_rope: bool = False
rotary_dim: int | None = None
rope_layout: str = "neox"
```

The three forward interfaces are:

```text
Dense.forward(
    q, k, v,                         # BSHD
    q_scale, k_scale, v_scale,
    rope_cos=None, rope_sin=None,
) -> o

Varlen.forward(
    q, k, v,                         # packed THD
    cu_seqlens_q, cu_seqlens_kv,
    q_scale, k_scale, v_scale,
    rope_cos=None, rope_sin=None,
) -> o

Paged.forward(
    q, k_new, v_new,                 # packed THD current chunk
    k_pages, v_pages,
    q_scale, k_scale, v_scale,
    cu_seqlens_q, cache_seqlens, block_table,
    rope_cos=None, rope_sin=None,
) -> o
```

Dense derives K positions from `0..Skv-1` and bottom-right-aligns Q positions.
Varlen applies the same rule independently within every packed request. Paged
assumes existing cached K is already rotated when fused RoPE is enabled and
uses `cache_seqlens[b] + local_position` for Q and `k_new`.

`rope_layout="neox"` pairs the first and second halves of the rotary channels;
`rope_layout="interleaved"` pairs adjacent even/odd channels. The layout is an
explicit semantic choice and part of kernel dispatch and caching.

## Common attention semantics

All three Ops use the common semantic parameters in the public interfaces
above. The manifest spells `None` defaults as `null` and uses its existing
`dtype | None` syntax where applicable; no new manifest field kind is needed.

`dtype` names the output element type. For FP16 or BF16 inputs, it must equal
the input element type. For FP8 inputs, it selects an FP16 or BF16 output.

`sm_scale=None` means the standard `1 / sqrt(head_dim)` scale. `softcap=None`
or zero disables softcap; a positive value applies:

```text
scores = softcap * tanh((QK^T * sm_scale) / softcap)
```

Masking is applied after softcap and before softmax. A negative softcap is
invalid.

## Sliding-window semantics

A window bound of `-1` is unlimited. Any other valid bound is non-negative.
Sliding-window attention is enabled when either bound is not `-1`; there is no
separate `sliding_window` or implementation-selection parameter.

For dense prefill, query position `i` and key position `j` use:

```text
offset = sequence_length_kv - sequence_length_q
center = i + offset
```

For varlen prefill, the same equations are evaluated independently for each
request using the lengths described by its cumulative-sequence tensors.

For paged prefill:

```text
center = cache_sequence_length + local_query_position
```

The key at logical position `j` is visible when every applicable condition is
true:

```text
0 <= j < sequence_length_kv

is_causal:
    j <= center

window_size_left >= 0:
    j >= center - window_size_left

window_size_right >= 0:
    j <= center + window_size_right
```

Paged masking is defined entirely in logical token coordinates. Page-table
translation happens only after the logical key range is established.

## Fixed superset tensor ABI

The current manifest format requires a fixed ordered tensor input list and does
not support `Optional[Tensor]`. Each Op therefore uses one stable superset ABI
for scaled and unscaled formats.

### Dense prefill

```text
q        [B, Sq, H, D]
k        [B, Skv, Hkv, D]
v        [B, Skv, Hkv, D]
q_scale  [B, Hkv]
k_scale  [B, Hkv]
v_scale  [B, Hkv]
rope_cos [max_position, rotary_dim / 2]
rope_sin [max_position, rotary_dim / 2]
```

### Varlen prefill

```text
q                [total_q, H, D]
k                [total_kv, Hkv, D]
v                [total_kv, Hkv, D]
cu_seqlens_q      [B + 1]
cu_seqlens_kv     [B + 1]
q_scale           [B, Hkv]
k_scale           [B, Hkv]
v_scale           [B, Hkv]
rope_cos          [max_position, rotary_dim / 2]
rope_sin          [max_position, rotary_dim / 2]
```

### Paged prefill

```text
q                [total_q, H, D]
k_new            [total_q, Hkv, D]
v_new            [total_q, Hkv, D]
k_pages          [physical_tokens, Hkv, D]
v_pages          [physical_tokens, Hkv, D]
q_scale           [B, Hkv]
k_scale           [B, Hkv]
v_scale           [B, Hkv]
cu_seqlens_q      [B + 1]
cache_seqlens     [B]
block_table       [B, max_pages_per_request]
rope_cos          [max_position, rotary_dim / 2]
rope_sin          [max_position, rotary_dim / 2]
```

## Paged state mutation and fused transformations

`GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp` declares one additional
semantic parameter:

```text
append_kv: bool = True
fuse_rope: bool = False
rotary_dim: int | None = None
rope_layout: str = "neox"
```

`k_new` and `v_new` always participate as the current logical suffix of the
attention request. `append_kv` controls persistence only:

- `append_kv=True` stores the current suffix into `k_pages` and `v_pages`;
- `append_kv=False` leaves the cache unchanged after computing the same
  attention result.

This avoids the ambiguous alternative where disabling append also silently
removes the current chunk from attention. With append enabled, the public
contract includes all of the following behavior in one invocation:

1. read the existing logical KV prefix described by `cache_seqlens` and
   `block_table`;
2. append `k_new` and `v_new` to the physical locations selected by the page
   table;
3. compute attention over the existing prefix and the newly appended tokens;
4. return the attention output while retaining the appended KV in
   `k_pages` and `v_pages`.

Consequently, `k_pages` and `v_pages` are conditionally read-write inputs.
`cache_seqlens` is always the pre-append length and remains read-only; the
caller owns updating sequence-length metadata after the call. The current
manifest schema has no conditional-mutation field, so this contract is stated
in the Op documentation and tested without introducing a new manifest schema.

RoPE is also part of the Op-level semantics when enabled; it is not merely a
kernel-selection detail. The paged Op must define whether `q` and `k_new` are
already rotated or are rotated by the invocation, which absolute positions are
used, and which representation is stored in the cache. For fused RoPE, the
contract is:

- `q` and `k_new` are rotated at absolute position
  `cache_seqlens[b] + local_query_position`;
- the rotated `k_new` representation is appended to `k_pages`;
- `v_new` is appended unchanged;
- attention observes the rotated query and cached keys.

The public Op therefore exposes the semantic RoPE controls and required table
inputs, if any. Whether append, RoPE, and attention are executed by one GPU
kernel or several internal sub-kernels is an implementation decision. An
internal append or RoPE-append kernel is not a separate manifest dispatch item;
the manifest maps only the top-level callable role owned by the Op.

FP8 kernels consume the scale tensors. Unscaled FP16/BF16 paths receive
identity scales and may ignore them. Callers must reuse prepared identity
scale tensors rather than allocating them on each invocation.

For GQA, one query scale is shared by the query-head group associated with a
KV head. The scale contract can be generalized later without changing the Op
topology, but changing scale tensor shapes is an ABI change and requires an
explicit manifest revision.

## Numeric-format coverage

`signature.dtype_combos` exhaustively declares the supported cross-tensor
formats without introducing dtype-specific Ops.

The three interfaces are intended to cover:

| Op topology | Required numeric modes |
| --- | --- |
| Dense | FP16, BF16, native FP8 Tensor Core |
| Varlen | FP16, BF16, native FP8 Tensor Core |
| Paged | FP16/BF16 pages, FP8 cache with 16-bit query/new KV, and native FP8 query/new KV with FP8 pages |

Native FP8 kernels must cover the manifest-supported causal, sliding-window,
softcap, shape, and ragged semantics for their topology. Dequantizing the full
FP8 inputs and routing into a 16-bit attention kernel is not the fallback
strategy for these Ops; unsupported combinations remain outside the manifest
until the native Tensor Core family implements them.

## Dispatch boundary

Public parameters describe attention semantics, never implementation names.
The interfaces therefore do not expose parameters such as `backend="dense"`,
`backend="fp8"`, or `backend="sliding_window"`.

Each Op constructs a call description from its fixed topology, parameters, and
input tensor metadata. Kernel implementations declare the region they serve.
Selection is capability-based and order-independent:

- zero applicable implementations is an unsupported-call error;
- two overlapping specialized implementations are an ambiguity error;
- a general implementation provides correctness where no specialization
  applies.

Kernel-map keys remain internal dispatch roles and are not part of the user
interface.

## Public API consolidation

The three topology-specific Ops replace public interfaces that differ only by
attention semantics or duplicate one of the three data layouts. Sliding-window
and generic compatibility wrappers may exist temporarily as deprecated
adapters, but they must not own independent kernel maps or implementations.

The target public prefill surface is exactly:

```text
GroupedQueryAttentionPrefillDenseFwdOp
GroupedQueryAttentionPrefillVarlenFwdOp
GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp
```

## Kernel family plan

The three public Ops do not imply exactly three GPU kernels. Kernel boundaries
follow physical data movement and execution pipelines, while causal masking,
sliding windows, softcap, and Paged fused RoPE remain policies within those
families.

```text
Dense Op
├─ dense general 16-bit
├─ dense Hopper warp-specialized causal/window specializations
└─ dense native-FP8 Tensor Core family
   ├─ BN224 producer/consumer fast schedule
   └─ general causal/rectangular/tail/window/softcap schedule

Varlen Op
├─ ragged general 16-bit, packed-KV loader
├─ Hopper causal left-window specialization
└─ native-FP8 general schedule, packed-KV loader

Paged Op
├─ ragged general 16-bit, paged-KV loader
├─ paged FP8-cache loader with online dequantization
├─ internal append / RoPE-append pass when required
└─ native-FP8 general schedule, paged-KV loader and raw-FP8 append
```

The native-FP8 general schedules now cover the declared causal, rectangular
tail, window, softcap, ragged, and append semantics without materializing a
16-bit Q/K/V fallback. Generated SM90 code uses FP8 WGMMA for QK and FP8
`mma.sync` for PV. This is the correctness-complete schedule, not the final
performance schedule.

The dense BN224 fast region follows the producer/consumer and raw-PTX fragment
contract developed in TileOps PR #1873: PV stays in its native WGMMA register
layout, accumulator rescaling is in-place, and the row reduction is deferred
for shorter schedules. That contract is currently specialized to a
`64 x 128 x 224` consumer tile. Generalizing it to dynamic tail/window tiles is
the required next step before replacing the shared-memory probability bridge
in Dense-general, Varlen, and Paged native-FP8 schedules. The general paths must
not copy the fixed BN224 helper under a different name or claim equivalent
performance before that work is complete.

Dense-general, Varlen, and Paged native-FP8 implementations call the same
`make_native_fp8_prefill_tile_update()` macro for the computation below:

```text
Q tile
→ logical Q/K coordinates
→ causal/window mask policy
→ QK
→ scale/softcap policy
→ online softmax
→ PV
→ output
```

They instantiate different loaders around that shared math tile. Dense reads
fixed BSHD ranges, Varlen reads contiguous packed
THD ranges. Paged translates logical positions through `block_table`, chooses
between existing pages and the current suffix, and may persist that suffix.
These address calculations should not be forced into one runtime kernel or one
runtime branch; they are separate compiled instantiations of shared source.

The following features do not independently justify another general kernel:

- full, causal, left/right sliding-window masking;
- softcap enabled or disabled;
- Paged fused RoPE enabled or disabled;
- FP16 versus BF16 when the same MMA and data-movement pipeline serves both.

They may still appear as compile-time specializations. A separate class remains
only when benchmark evidence establishes a materially different schedule. The
measured Hopper causal left-window Varlen implementation is such a
specialization: it is about 2.5% faster for the long square case and 8.5%
faster when Q is shorter than KV, while the general kernel is about 4.6%
faster for a bidirectional left-and-right window.

The Hopper Dense sliding-window implementation is also a justified physical
specialization. On an H200 locked at 1500 MHz, two native-CUPTI A/B runs found
it 1.09x faster than the window-capable general Dense kernel for short windows,
1.22--1.23x faster for long windows, 1.20x faster for a bidirectional window,
and 1.23x faster for the tested BF16 case. Every measured implementation was a
single kernel with zero inter-kernel gap. The standalone public sliding-window
Op can therefore be removed, but the specialized kernel remains an internal
Dense dispatch target. It applies to supported Hopper 16-bit, square-Q/KV
window calls and implements the same custom scale and softcap semantics as the
general kernel; the general Dense kernel is the correctness fallback outside
that physical region.

Two FP8 cases must remain distinct:

1. FP8 cache storage with FP16/BF16 Q and new KV dequantizes pages into a
   16-bit MMA pipeline. It is a Paged KV-loader policy, not a native-FP8
   attention family.
2. FP8 Q/K/V consumed by FP8 Tensor Cores changes the MMA, accumulation,
   scaling, and pipeline contract. It remains a separate specialization for
   every topology it supports.

Likewise, Hopper warp-specialized producer/consumer kernels retain independent
physical pipelines. They can reuse semantic helpers, but should not be folded
into the general kernel merely to reduce the number of classes.

Consolidation status and remaining targets are:

1. the standalone public Dense and Varlen sliding-window Ops are absorbed by
   the topology Ops; their justified Hopper kernels remain internal roles;
2. Paged RoPE reuses the general Paged attention body after its internal
   RoPE-append pass;
3. 16-bit pages, storage-only FP8 pages, and native-FP8 pages instantiate one
   Paged source skeleton with compile-time loader/dequantization policies;
4. keep the retained Hopper Dense sliding-window specialization behind the
   Dense Op's internal capability dispatch and cover scale/softcap/window
   combinations in the shared Dense benchmark surface;
5. the correctness schedule now shares one native-FP8 computation macro across
   Dense-general, Varlen, and Paged while retaining distinct loaders;
6. as a performance follow-up, generalize PR #1873's raw-PTX WGMMA-PV fragment
   contract and replace the shared-memory probability bridge behind that same
   semantic macro boundary.

## Multi-backend dispatch boundary

Each public Op exposes exactly one target-facing callable role. An external
builder receives the tensors above in manifest order plus the normalized
manifest parameters; it never receives an in-tree kernel key, `AttentionCall`,
architecture flag, packed view, or launch-only scalar.

The in-tree path adapts the same public ABI behind its returned callable:

- Dense converts BSHD tensors to packed views and creates uniform cumulative
  sequence lengths;
- Paged supplies `max_seqlen_q` to the TileLang launch wrapper;
- contiguous Decode performs its fixed-capacity padding and supplies the real
  sequence length.

These are NVIDIA implementation details, so an external target sees none of
them. Its callable is memoized by device plus the complete manifest input
signature. The memo is bounded; a backend that wants coarser or longer-lived
reuse owns that cache behind its builder.

`rope_layout` remains an explicit string enum (`"neox"` or
`"interleaved"`) across every target. It is not inferred from model family or
encoded as a boolean.

## Native-FP8 validation snapshot

The three topology-specific general schedules were compiled and executed on an
H200 with the SM clock locked to 1500 MHz. Fresh-cache generated CUDA contains
native FP8 WGMMA for QK and native FP8 `mma.sync` for PV in Dense, Varlen, and
Paged; none of the three paths materializes full FP16/BF16 Q/K/V tensors.

Representative native-CUPTI measurements use one timed operator sequence and
report the complete GPU activity envelope:

| topology | representative semantics | TileOps latency | FA3 envelope | relative performance | kernels | GPU gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Dense | rectangular tail, causal, left window, custom scale, softcap | 0.0492 ms | 0.0497 ms | 101.0% | 1 | 0 ms |
| Varlen | ragged tails, causal, left window, softcap | 0.0628 ms | 0.0517 ms | 82.3% | 1 | 0 ms |
| Paged | ragged tails, causal, left window, softcap, raw-FP8 append | 0.0770 ms | 0.0718 ms | 93.2% | 1 | 0 ms |

The Dense case uses `S_q=513` and `S_kv=769`, so it cannot enter the fixed
square BN224 fast schedule. The separate BN224 `S=896` fast-path check measures
0.0333 ms versus FA3 at 0.0285 ms (85.6%). FA3's Varlen and Paged figures above
are complete two-kernel envelopes, including their inter-kernel gap; comparing
only device-busy sums would understate the baseline Op latency. These
comparisons are validation evidence rather than a claim that every schedule
has reached the performance ceiling.

## Deferred performance work

The public topology boundary and correctness-complete kernel families are now
settled. Remaining work is deliberately performance-scoped:

1. generalize PR #1873's raw-PTX WGMMA-PV fragment contract beyond the fixed
   BN224 Dense schedule and evaluate it independently for Dense-general,
   Varlen, and Paged loaders;
2. add a native-FP8 Paged fused-RoPE specialization before declaring that
   dtype/transform combination in the manifest;
3. retain or remove physical specializations only after topology-local A/B
   benchmarks show whether their different pipeline remains worthwhile.
