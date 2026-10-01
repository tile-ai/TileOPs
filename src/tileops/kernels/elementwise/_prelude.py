"""Device source an elementwise body calls into, and the calls that reach it.

Each helper is ``static __device__ __forceinline__``, so a kernel that calls none of
them compiles to the same SASS as one built without the prelude.
"""

import tilelang.language as T

__all__ = ["PRELUDE", "approx_reciprocal"]

PRELUDE = """
// One MUFU.RCP. ``__fdividef`` reaches the same instruction for a normal divisor,
// but its PTX is ``div.approx.f32``, which ptxas guards with a test, a scale and two
// selects so that a subnormal divisor still returns a subnormal result.
static __device__ __forceinline__ float tl_approx_reciprocal(float x) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
"""


def approx_reciprocal(x):
    """``1 / x`` to two ulp for a normal *x* whose reciprocal is normal, else a zero.

    A subnormal *x* reads as a zero of its sign and returns an infinity; a reciprocal
    that would be subnormal returns a zero. A caller takes those two answers as a
    refusal and computes the value another way.
    """
    return T.call_extern("float32", "tl_approx_reciprocal", x)
