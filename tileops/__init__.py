"""TileOPs package initialization.

Hotfix: preload the CUDA driver into the global symbol namespace before any
submodule imports tilelang's native runtime. See ``_preload_cuda_driver``.
"""
from __future__ import annotations


def _preload_cuda_driver() -> None:
    """Load libcuda into the global symbol scope before tilelang's runtime.

    tilelang built from source (we pin a main commit) ships ``libtvm_runtime.so``
    with unresolved CUDA *driver* symbols such as ``cuModuleLoadData``: it links
    the CUDA *runtime* stub (``libcudart_stub.so``) but not the *driver* stub
    (``libcuda_stub.so``), and assumes the driver is already present in the
    global symbol scope. When nothing else loads it globally — e.g. torch's CUDA
    libraries are opened ``RTLD_LOCAL`` — importing tilelang fails with
    ``OSError: undefined symbol: cuModuleLoadData``.

    Loading the real driver with ``RTLD_GLOBAL`` here makes those symbols
    resolvable, exactly as an external ``LD_PRELOAD=libcuda.so.1`` would. This is
    best effort: on non-Linux or CPU-only hosts the driver is absent, so we leave
    the import to proceed (and surface tilelang's own error if one is raised).

    Stopgap for a tilelang source-build packaging gap — remove once tilelang
    links the driver stub (or preloads it) itself.
    """
    import ctypes
    import sys

    if not sys.platform.startswith("linux"):
        return
    for soname in ("libcuda.so.1", "libcuda.so"):
        try:
            ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


_preload_cuda_driver()
del _preload_cuda_driver

__all__: list[str] = []
