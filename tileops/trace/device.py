"""Device-side timing helper injection for the in-kernel trace.

The timestamp source is ``clock64()`` (a CUDA builtin, per-SM cycle counter),
wrapped in a ``__device__`` helper that the emit path calls via
``T.call_extern("uint64", "__tl_now")``. The helper is injected with
``T.import_source``, which must run inside a ``with T.Kernel(...)`` scope.

Also provides ``__tl_thread_idx_x()`` helper for implicit thread blocks.
"""

import tilelang.language as T

__all__ = ["inject_helper"]

# clock64() is a CUDA builtin (no inline asm); the cast keeps the return type a
# plain u64.
_HELPER = r"""
__device__ __forceinline__ unsigned long long __tl_now() {
    return (unsigned long long)clock64();  // per-SM cycle counter (CUDA builtin)
}

__device__ __forceinline__ int __tl_thread_idx_x() {
    return threadIdx.x;  // CUDA builtin thread index
}
"""


def inject_helper() -> None:
    """Inject the ``__tl_now()`` and ``__tl_thread_idx_x()`` device helpers into the current kernel.

    Emits the ``clock64()`` wrapper and threadIdx.x accessor via ``T.import_source`` so that
    ``T.call_extern("uint64", "__tl_now")`` and ``T.call_extern("int32", "__tl_thread_idx_x")``
    resolve at codegen.

    Note:
        MUST be called inside a ``with T.Kernel(...)`` scope. Calling it before
        the kernel block raises "No builder in current scope".
    """
    T.import_source(_HELPER)
