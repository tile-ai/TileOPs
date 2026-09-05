"""Hardware and math constants shared across kernel families."""

# Widest vectorized global access the kernels plan for: 128 bits.
VECTOR_ACCESS_BYTES: int = 16

# Address range the shared-memory banks cover before repeating: 32 banks of 4 bytes.
SHARED_BANK_SPAN_BYTES: int = 128

# Shared memory one block may take without opting in to the dynamic allocation.
STATIC_SHARED_BYTES: int = 48 * 1024

# log2(e), to fold exp(x) into the single-instruction exp2(x * LOG2E).
LOG2E: float = 1.4426950408889634

# 1/sqrt(2), for the erf form of GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
INV_SQRT2: float = 0.7071067811865476

# sqrt(2/pi) and the cubic coefficient of torch's tanh-approximate GELU.
SQRT_2_OVER_PI: float = 0.7978845608028654
GELU_TANH_COEFF: float = 0.044715

# Largest finite float8_e4m3fn value; quantizers clamp to +-FP8_E4M3_MAX.
FP8_E4M3_MAX: float = 448.0
