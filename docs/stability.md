# API Stability

This page defines what TileOPs supports as public API and what a release may change.

## Public modules

| Module              | What it holds                                                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `tileops`           | The family modules, `Op`, and the backend names it re-exports.                                                                    |
| `tileops.<family>`  | The ops of one family, and the value types their constructors take (for example `tileops.moe.MaskedLayoutSpec`).                  |
| `tileops.backend`   | Target selection and registration of out-of-tree kernels.                                                                         |
| `tileops.authoring` | `Op`, `Kernel`, `KernelInterface`, `CallSpec`, `Entry`, `SignatureCall`: the classes code outside TileOPs uses to write a new op. |
| `tileops.manifest`  | The names in its `__all__`, the `tileops.manifest.primitives` module, and the spec format.                                        |
| `tileops.trace`     | The in-kernel timeline tracer.                                                                                                    |

A name is public when it is listed in the `__all__` of a public module.
Every module, function, class, attribute or method whose name starts with an underscore is internal, including `tileops._ops`, `tileops._kernels`, `tileops._utils` and `tileops._perf`.
Internal names change in any release without notice.

A concrete op such as `tileops.elementwise.HardtanhFwdOp` is public to construct and call, not to subclass.

## Compatibility

Within 0.x, minor and patch releases change the public API only additively:

- Public names, signatures, parameter defaults, return structure and documented behavior stay as they are.
- A release may add a module, a name, an optional parameter, or a new accepted input.
- Any other change waits for a major release.

`scripts/ci/check_api_compat.py` checks every pull request against the latest release and fails on a change that breaks a call written against it.

## Support scope

A call is supported under these conditions:

- An op instance is used by one thread and one CUDA stream at a time.
- A CUDA graph replay of an instance does not overlap that instance's eager calls.
- A compiled callable (`torch.compile` output) does not outlive the op instance it calls.
- An input the call does not write is contiguous, unless the op documents otherwise.
- A `Bwd` op is called directly; it does not plug into `torch.autograd`.
- CUDA Graph capture is supported for the ops [`tests/cuda_graph_contract.py`](../tests/cuda_graph_contract.py) lists, on the calls their `cuda_graph` tests make ([CUDA Graph validation](development.md#cuda-graph-validation)).

## Caching and logging

- Compiled kernels and tuning results are cached on disk under `TILELANG_CACHE_DIR`, by default `~/.tilelang/cache`, and reused across processes.
- Tuning progress goes through the `logging` module, under the `tilelang` logger, for the application to route; nothing is printed to stdout or written to the working directory.
- A tuning candidate that fails to compile or run is skipped; running out of device memory or an error in the calling code stops the tuning.

## Writing an op outside TileOPs

Code outside TileOPs writes a new op by subclassing `tileops.authoring.Op`:

- The class provides its spec as the class attribute `spec`, in the same schema as a manifest entry ([manifest.md](design/manifest.md)).
- `__init__` assigns the spec's parameters, then calls `super().__init__(target=target)`.
- The class declares its `Kernel` subclasses in `kernel_types` and its `KernelInterface` call sites in `interfaces`, and writes `forward`, which gets a kernel with `kernel_for(interface, call)`.

The op is then called as `op(...)`, like a built-in op.
