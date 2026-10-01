"""The Philox4x32-10 stream the sampling references draw from.

Salmon et al., SC'11. A kernel that counts by the draw and the row alone, naming no launch
fact, draws the same uniform however the launch splits the row, and so reproduces
``workloads/sampling.py`` exactly.
"""

import tilelang.language as T

__all__ = ["UNIFORM_BITS", "mix"]

_EVEN_MUL, _ODD_MUL = 0xD2511F53, 0xCD9E8D57
_EVEN_WEYL, _ODD_WEYL = 0x9E3779B9, 0xBB67AE85
_ROUNDS = 10

# Top bits of one 32-bit output word that make the uniform, as the references take them.
UNIFORM_BITS = 24


@T.macro
def mix(counter, key, bump):
    """Run the ten rounds in place over *counter*, leaving the uniform's word in ``counter[0]``.

    *counter* and *key* are the caller's, already set; *bump* is two words of scratch.
    """
    for step in T.serial(_ROUNDS):
        if step > 0:
            key[0] = key[0] + T.uint32(_EVEN_WEYL)
            key[1] = key[1] + T.uint32(_ODD_WEYL)
        bump[0] = (
            T.call_extern("uint32", "__umulhi", counter[2], T.uint32(_ODD_MUL))
            ^ counter[1]
            ^ key[0]
        )
        bump[1] = (
            T.call_extern("uint32", "__umulhi", counter[0], T.uint32(_EVEN_MUL))
            ^ counter[3]
            ^ key[1]
        )
        counter[1] = counter[2] * T.uint32(_ODD_MUL)
        counter[3] = counter[0] * T.uint32(_EVEN_MUL)
        counter[0] = bump[0]
        counter[2] = bump[1]
