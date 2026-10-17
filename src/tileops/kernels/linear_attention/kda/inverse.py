"""Stable blockwise inversion of the KDA unit-lower triangular system."""

import tilelang.language as T

__all__ = ["inverse_program"]


def _diagonal_program(*, chunk_size, block_size, dtype, output_offset):
    @T.macro
    def diagonal(coeff, inverse):
        x = T.alloc_local([block_size], "float32")
        lane = T.get_thread_binding()
        if lane < chunk_size:
            block = lane // block_size
            col = lane % block_size
            for row in T.unroll(block_size):
                value = T.alloc_var("float32")
                value = T.if_then_else(row == col, T.float32(1.0), T.float32(0.0))
                for prior in T.unroll(row):
                    value += (
                        T.cast(
                            coeff[block * block_size + row, block * block_size + prior], "float32"
                        )
                        * x[prior]
                    )
                x[row] = value
                inverse[block * block_size + row, lane + output_offset] = T.cast(value, dtype)
        T.sync_threads()

    return diagonal


def inverse_program(*, chunk_size, block_size, threads, dtype, output_offset=0):
    """Reuse the caller's coefficient, inverse and FP32 GEMM scratch buffers."""
    assert threads >= chunk_size, "inverse requires at least one thread per column"
    assert chunk_size % block_size == 0, "inverse requires complete diagonal blocks"
    diagonal = _diagonal_program(
        chunk_size=chunk_size, block_size=block_size, dtype=dtype, output_offset=output_offset
    )
    levels = (chunk_size // block_size - 1).bit_length()

    def matrix(buffer, offset):
        return buffer[:, offset : offset + chunk_size]

    @T.macro
    def invert(coeff, inverse, scratch):
        for i, j in T.Parallel(chunk_size, chunk_size):
            inverse[i, j + output_offset] = T.cast(0, dtype)
        T.sync_threads()
        diagonal(coeff, inverse)
        for i, j in T.Parallel(chunk_size, chunk_size):
            coeff[i, j] = T.if_then_else(
                i // block_size > j // block_size, coeff[i, j], T.cast(0, dtype)
            )
        T.sync_threads()
        T.clear(scratch)
        T.gemm(matrix(coeff, 0), matrix(inverse, output_offset), scratch)
        T.sync_threads()
        for i, j in T.Parallel(chunk_size, chunk_size):
            coeff[i, j] = T.cast(scratch[i, j], dtype)
        T.sync_threads()
        for level in T.serial(levels):
            for i, j in T.Parallel(chunk_size, chunk_size):
                coeff[i, j] = coeff[i, j] + T.cast(T.if_then_else(i == j, 1.0, 0.0), dtype)
            T.sync_threads()
            T.clear(scratch)
            T.gemm(matrix(inverse, output_offset), matrix(coeff, 0), scratch)
            T.sync_threads()
            for i, j in T.Parallel(chunk_size, chunk_size):
                inverse[i, j + output_offset] = T.cast(scratch[i, j], dtype)
                coeff[i, j] = T.if_then_else(
                    i // block_size > j // block_size, coeff[i, j], T.cast(0, dtype)
                )
            T.sync_threads()
            if level < levels - 1:
                T.clear(scratch)
                T.gemm(matrix(coeff, 0), matrix(coeff, 0), scratch)
                T.sync_threads()
                for i, j in T.Parallel(chunk_size, chunk_size):
                    coeff[i, j] = T.cast(scratch[i, j], dtype)
                T.sync_threads()

    return invert
