"""The Philox4x32-10 stream the sampling references draw from.

Salmon et al., SC'11. A kernel that counts by the draw and the row alone, naming no launch
fact, draws the same uniform however the launch splits the row, and so reproduces
``workloads/sampling.py`` exactly.
"""

import tilelang.language as T

__all__ = ["draw", "mix", "uniform_of"]

_EVEN_MUL, _ODD_MUL = 0xD2511F53, 0xCD9E8D57
_EVEN_WEYL, _ODD_WEYL = 0x9E3779B9, 0xBB67AE85
_ROUNDS = 10


@T.macro
def mix(counter, key, bump):
    """Run the ten rounds in place over *counter*, leaving the four output words in it; the
    uniform's word is ``counter[0]``.

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


@T.macro
def draw(counter, key, bump, seed, offset, index, row):
    """Leave in ``counter[0]`` the first output word of draw ``index`` of row ``row``.

    The stream is keyed by the ``int64`` *seed* and counted by the draw, the row and the
    ``int64`` *offset*; *counter*, *key* and *bump* are the caller's scratch.
    """
    key[0] = T.cast(seed[0] & T.int64(0xFFFFFFFF), "uint32")
    key[1] = T.cast((seed[0] >> T.int64(32)) & T.int64(0xFFFFFFFF), "uint32")
    counter[0] = T.cast(index, "uint32")
    counter[1] = T.cast(row, "uint32")
    counter[2] = T.cast(offset[0] & T.int64(0xFFFFFFFF), "uint32")
    counter[3] = T.cast((offset[0] >> T.int64(32)) & T.int64(0xFFFFFFFF), "uint32")
    mix(counter, key, bump)


def uniform_of(word):
    """The float32 uniform in ``[0, 1)`` that 32-bit output *word* makes."""
    # The references take the word's top 24 bits.
    bits = 24
    return T.cast(word >> T.uint32(32 - bits), "float32") * T.float32(2.0**-bits)
