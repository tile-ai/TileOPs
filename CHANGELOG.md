# Changelog

All notable changes to TileOPs are listed here, newest first. Versions follow [Semantic Versioning](https://semver.org/).

## 0.1.0

The first release. [docs/stability.md](https://github.com/tile-ai/TileOPs/blob/main/docs/stability.md) states what is public and what later 0.x releases may change.

- Ops are imported as `tileops.<family>.<Op>`; every op has a manifest entry in `tileops.manifest` that states its signature, workloads and roofline.
- `tileops.backend` lets a backend distribution serve whole ops on its own device through `register_kernel_builder`.
- `tileops.authoring` lets code outside TileOPs write a new op by subclassing `Op` with its own `spec`.
- `tileops.trace` records an in-kernel timeline of a TileLang kernel.
- Verified on SM90 (H200) with Python 3.12 and 3.13, PyTorch 2.13, TileLang 0.1.12 and CUDA Toolkit 13.2.
