# CI runner image

Multi-stage image for the self-hosted GPU runner. It bakes a tilelang wheel (compiled once
from a pinned commit, or installed from a PyPI release) plus the test/benchmark stack onto a
public CUDA base, so CI never recompiles tilelang per PR.

Built **manually on a GPU host** (needs `nvcc`), then pushed to `ghcr.io`. It is **not**
built in CI.

## Prerequisites

- An NVIDIA GPU host with a CUDA 12.9-capable driver and `nvcc`.
- Docker with BuildKit enabled (`DOCKER_BUILDKIT=1`).
- Run from the **repository root** — the build context must contain `constraints.txt`,
  `scripts/ci/verify_runtime_stack.py`, and `.github/runner/entrypoint.sh` (the Dockerfile
  copies all three).

## Build

Provide tilelang one of two ways — pass **exactly one** of these build-args:

- **main commit**: `--build-arg TILELANG_GIT_SHA=<commit>` — shallow-fetches and compiles that
  commit. The Dockerfile carries no commit literal; the commit you pass is the single source
  of truth, and the image tag records it.
- **release**: `--build-arg TILELANG_VERSION=<version>` — `pip install tilelang==<version>`.

```bash
# from the repository root (main-commit mode)
DOCKER_BUILDKIT=1 docker build \
  -f .github/runner/Dockerfile \
  --target final \
  --build-arg TILELANG_GIT_SHA=65dbc9837beedf6882a40a08e18ea571d92fd6a5 \
  -t ghcr.io/tile-ai/tileops-runner:65dbc98 \
  .
```

Tag with the tilelang commit's **short SHA** (`:65dbc98`). If you rebuild the same commit,
add a numeric suffix (`:65dbc98-2`).

## Roll out an updated runner image

Changes to this Dockerfile are not picked up by CI automatically. After a Dockerfile change
lands, rebuild and tag the image from a GPU host using the build command above, then push it:

```bash
docker push ghcr.io/tile-ai/tileops-runner:<new-tag>
```

Then update the runner launcher configuration in TileOpsGov (`ci/runner-launcher/common.env`,
the `IMAGE=` value) to point at the new tag, and restart/redeploy the local runner launcher.
Merging the TileOPs PR only changes the image recipe; the live self-hosted runners keep using
their existing image until that local Docker rollout is done.

### Build args

| Arg                | Default                                | Purpose                                                   |
| ------------------ | -------------------------------------- | --------------------------------------------------------- |
| `TILELANG_GIT_SHA` | *(none)*                               | tilelang commit to shallow-clone and compile (main mode). |
| `TILELANG_VERSION` | *(none)*                               | tilelang PyPI version to `pip install` (release mode).    |
| `BASE_IMAGE`       | `nvidia/cuda:12.9.1-devel-ubuntu22.04` | Public CUDA `devel` base (Python 3.12 via deadsnakes).    |
| `MAX_JOBS`         | `64`                                   | Parallelism for the tilelang / FA2 / FA3 source builds.   |
| `NVCC_THREADS`     | `4`                                    | Per-`nvcc` threads.                                       |
| `RUNNER_VERSION`   | `2.334.0`                              | GitHub Actions runner version baked into `final`.         |

Set exactly one of `TILELANG_GIT_SHA` / `TILELANG_VERSION`; the build fails fast if neither is set.

### Stages (`--target`)

| Stage       | Contents                                                                                                                                                                                 |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `runtime`   | Python 3.12 + torch / torchvision / torchaudio `2.10.0 / 0.25.0 / 2.10.0 +cu129` + triton `3.6.0` + tilelang build/runtime deps (incl. `apache-tvm-ffi 0.1.11`). **No tilelang itself.** |
| `post-fa3`  | `runtime` + pytest / pytest-xdist / ruff + FlashAttention-3 (built from the `hopper/` source).                                                                                           |
| `fa2`       | `post-fa3` + FlashAttention-2 (`flash-attn 2.8.3`, source-built in its own layer so changes to the bench loop never recompile it).                                                       |
| `fullstack` | `fa2` + flash-linear-attention `0.4.2` + vLLM `0.19.1` + mamba-ssm `2.3.1`, then flashinfer-python/-cubin upgraded to `0.6.11.post2` (`--no-deps`, so torch stays +cu129). sgl-kernel is not installed. |
| `tilelang`  | `fullstack` + the tilelang wheel (`--no-deps`), then the build-time guard. Built **last** so a SHA bump rebuilds only this layer.                                                        |
| `final`     | `tilelang` + the GitHub Actions runner (no TileOPs source baked).                                                                                                                        |

Build an earlier stage for debugging with `--target runtime` (etc.).

The `tilelang` stage ends by running `scripts/ci/verify_runtime_stack.py` (GPU-free): it fails
the build unless tilelang imports, the installed `apache-tvm-ffi` sits inside the tilelang
wheel's declared range, and torch is still the cu129 build.

## Verify the built image

```bash
docker run --rm --gpus all ghcr.io/tile-ai/tileops-runner:65dbc98 python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)   # expect 2.10.0+cu129, cuda 12.9
import tilelang; print("tilelang", tilelang.__version__)

# cuBLAS probe: matmul / bmm / einsum on the GPU
a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
assert torch.matmul(a, b).isfinite().all()
ab = torch.randn(8, 128, 128, device="cuda", dtype=torch.float16)
assert torch.bmm(ab, ab).isfinite().all()
assert torch.einsum("bik,bkj->bij", ab, ab).isfinite().all()
print("cuBLAS probe OK")
PY
```

Then run the smoke tests against a checkout of this repo:

```bash
docker run --rm --gpus all -v "$PWD:/src" -w /src \
  ghcr.io/tile-ai/tileops-runner:65dbc98 \
  bash -c 'scripts/ci/install_tileops.sh && pytest -m smoke'
```

`install_tileops.sh` installs TileOPs `--no-deps` against the baked stack; it fails fast if
tilelang is missing (the image provides it).

## Run as a self-hosted runner

`entrypoint.sh` registers an ephemeral runner (one job per container), then deregisters on
exit. Provide a registration token and the target URL; bind-mount the host cache. The
entrypoint removes `RUNNER_TOKEN` from the environment before the runner starts, so jobs
cannot read the registration token.

```bash
docker run -d --gpus all \
  -e RUNNER_URL=https://github.com/tile-ai/TileOPs \
  -e RUNNER_TOKEN=<registration-token> \
  -e RUNNER_LABELS=self-hosted,tile-ops,venv \
  -v <host-cache-dir>:/ci-cache \
  ghcr.io/tile-ai/tileops-runner:65dbc98
```

The image sets cache env vars (`TILELANG_CACHE_DIR`, `TRITON_CACHE_DIR`, `PIP_CACHE_DIR`, …)
under `/ci-cache`; the directories are pre-created so the container also works unmounted.

## Bumping the tilelang commit

A commit (or release) bump always rebuilds, but **never edits the Dockerfile**: rebuild with a
new `--build-arg TILELANG_GIT_SHA=<commit>` (or `TILELANG_VERSION=<version>`) and a new
`:<short-sha>` tag, push to `ghcr.io`, then point the runner at the new tag. Because tilelang
is the last stage, only its layer recompiles — the bench layers (FA2 / FA3 / vLLM / …) stay
cached. Switching between a release and a main commit is the same — only the build-arg and tag
change.
