"""Hardware and math constants shared across kernel families."""

# Widest vectorized global access the kernels plan for: 128 bits.
VECTOR_ACCESS_BYTES: int = 16

# Address range the shared-memory banks cover before repeating: 32 banks of 4 bytes.
SHARED_BANK_SPAN_BYTES: int = 128

# Alignment of a shared buffer a 128-byte-swizzled TMA copy targets. TileLang aligns every
# shared buffer to it, so a buffer's footprint is its size rounded up to this.
SHARED_BUFFER_ALIGN_BYTES: int = 1024

# Threads in an SM90 warpgroup, the four warps a WGMMA instruction issues across.
WARPGROUP_THREADS: int = 128

# Rows one WGMMA instruction computes: its M extent.
WGMMA_ROWS: int = 64

# Rows one warp-level MMA instruction computes: the M extent TileLang lowers to below
# WGMMA_ROWS, and so the narrowest a matrix operand may be.
WARP_MMA_ROWS: int = 16

# Shared memory one block may take without opting in to the dynamic allocation.
STATIC_SHARED_BYTES: int = 48 * 1024

# Shared memory one block may be given after opting in, by architecture.
BLOCK_SHARED_BYTES_OPT_IN: dict[int, int] = {
    80: 163 * 1024,
    86: 99 * 1024,
    89: 99 * 1024,
    90: 227 * 1024,
}

# Threads one block may hold.
MAX_BLOCK_THREADS: int = 1024

# Blocks one thread-block cluster may hold without opting in to a non-portable size.
MAX_PORTABLE_CLUSTER_BLOCKS: int = 8

# Blocks one SM may hold resident at once, by architecture.
SM_RESIDENT_BLOCKS: dict[int, int] = {
    80: 32,
    86: 16,
    89: 24,
    90: 32,
}

# log2(e), to fold exp(x) into the single-instruction exp2(x * LOG2E).
LOG2E: float = 1.4426950408889634

# Widest exponent gap a pair of bfloat16 factors can carry between them. A product written
# as exp(a) * exp(-a) holds only while both factors are representable, and bfloat16 runs to
# 2**127; the margin leaves room for the operands the factors scale.
BF16_SPLIT_EXP2_SPAN: float = 120.0

# 1/sqrt(2), for the erf form of GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
INV_SQRT2: float = 0.7071067811865476

# sqrt(2/pi) and the cubic coefficient of torch's tanh-approximate GELU.
SQRT_2_OVER_PI: float = 0.7978845608028654
GELU_TANH_COEFF: float = 0.044715

# Largest finite float8_e4m3fn value; quantizers clamp to +-FP8_E4M3_MAX.
FP8_E4M3_MAX: float = 448.0

# Elements one scale covers in the block-scaled quantization formats the manifest fixes:
# a run along K for the INT8 and FP8 activation forms, both axes of a tile for the FP8
# weight form.
QUANT_SCALE_BLOCK: int = 128
