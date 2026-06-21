# CI runner image

Multi-stage image for the self-hosted GPU runner. It bakes a tilelang wheel (compiled
once from a pinned commit) plus the test/benchmark stack onto a public CUDA base, so CI
never recompiles tilelang per PR.

Built **manually on a GPU host** (needs `nvcc`), then pushed to `ghcr.io`. It is **not**
built in CI.

## Prerequisites

- An NVIDIA GPU host with a CUDA 12.9-capable driver and `nvcc`.
- Docker with BuildKit enabled (`DOCKER_BUILDKIT=1`).
- Run from the **repository root** — the build context must contain `constraints.txt`
  and `.github/runner/entrypoint.sh` (the Dockerfile copies both).

## Build

`TILELANG_GIT_SHA` is **required** (no default). The Dockerfile carries no commit literal;
the commit you pass is the single source of truth, and the image tag records it.

```bash
# from the repository root
DOCKER_BUILDKIT=1 docker build \
  -f .github/runner/Dockerfile \
  --target final \
  --build-arg TILELANG_GIT_SHA=65dbc9837beedf6882a40a08e18ea571d92fd6a5 \
  -t ghcr.io/tile-ai/tileops-runner:65dbc98 \
  .
```

Tag with the tilelang commit's **short SHA** (`:65dbc98`). If you rebuild the same commit,
add a numeric suffix (`:65dbc98-2`).

### Build args

| Arg                | Default                                | Purpose                                                                                                                                                                                           |
| ------------------ | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TILELANG_GIT_SHA` | *(required)*                           | tilelang commit to compile and bake.                                                                                                                                                              |
| `BASE_IMAGE`       | `nvidia/cuda:12.9.1-devel-ubuntu22.04` | Public CUDA `devel` base (Python 3.12 via deadsnakes).                                                                                                                                            |
| `MAX_JOBS`         | `64`                                   | Parallelism for the tilelang / extension builds.                                                                                                                                                  |
| `NVCC_THREADS`     | `4`                                    | Per-`nvcc` threads.                                                                                                                                                                               |
| `RUNNER_VERSION`   | `2.334.0`                              | GitHub Actions runner version baked into `final`.                                                                                                                                                 |
| `RUNNER_SHA256`    | *(empty → skipped)*                    | Optional: published SHA-256 of the runner tarball; when set, the build verifies the download before extracting. Find it on the [runner release page](https://github.com/actions/runner/releases). |

### Stages (`--target`)

| Stage       | Contents                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------ |
| `runtime`   | Python 3.12 + torch `2.10.0+cu129` + tilelang runtime deps + tilelang wheel (`--no-deps`). |
| `post-fa3`  | `runtime` + pytest/ruff + FlashAttention-3.                                                |
| `fullstack` | `post-fa3` + vLLM / flashinfer / sgl-kernel / FLA.                                         |
| `final`     | `fullstack` + the GitHub Actions runner (no TileOPs source baked).                         |

Build an earlier stage for debugging with `--target runtime` (etc.).

## Verify the built image

```bash
docker run --rm --gpus all ghcr.io/tile-ai/tileops-runner:65dbc98 python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)   # expect cuda 12.9
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

`install_tileops.sh` installs TileOPs `--no-deps` against the baked stack; it fails fast
if tilelang is missing (the image provides it).

## Run as a self-hosted runner

`entrypoint.sh` registers an ephemeral runner (one job per container), then deregisters on
exit. Provide a registration token and the target URL; bind-mount the host cache.

```bash
docker run -d --gpus all \
  -e RUNNER_URL=https://github.com/tile-ai/TileOPs \
  -e RUNNER_TOKEN=<registration-token> \
  -e RUNNER_LABELS=self-hosted,tile-ops \
  -v <host-cache-dir>:/ci-cache \
  ghcr.io/tile-ai/tileops-runner:65dbc98
```

The image sets cache env vars (`TILELANG_CACHE_DIR`, `TRITON_CACHE_DIR`, `PIP_CACHE_DIR`, …)
under `/ci-cache`; the directories are pre-created so the container also works unmounted.

## Bumping the tilelang commit

A commit bump always rebuilds (the wheel is baked), but **never edits the Dockerfile**:
rebuild with a new `--build-arg TILELANG_GIT_SHA=<commit>` and a new `:<short-sha>` tag,
push to `ghcr.io`, then point the runner at the new tag. Switching between a release and a
main commit is the same — only the image tag changes.
