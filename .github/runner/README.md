# CI runner image

Multi-stage image for the self-hosted GPU runner. It bakes a tilelang wheel plus the
test/benchmark stack onto a public CUDA base, so CI never recompiles tilelang per PR.

Built **manually on a GPU host** — never in CI.

One package holds both images the project uses, told apart by tag:

| Tag                            | `--target` | Who runs it                                             |
| ------------------------------ | ---------- | ------------------------------------------------------- |
| `<tilelang-sha>`               | `final`    | The self-hosted CI runners — includes the Actions agent |
| `<tilelang-sha>-torch2.13-dev` | `tilelang` | Local development — same stack, no agent                |

Both come from the same tilelang commit, so the two stay in step.

## Build and roll out

You need a GPU host with a CUDA 13.2-capable driver and `nvcc`, and Docker with BuildKit.

**1. Build, from the repository root.** The context must contain `constraints.txt`,
`constraints-runner-lock.txt`, `scripts/ci/`, and `.github/runner/entrypoint.sh`.

```bash
DOCKER_BUILDKIT=1 docker build \
  -f .github/runner/Dockerfile \
  --target final \
  --build-arg TILELANG_GIT_SHA=<commit> \
  -t ghcr.io/tile-ai/tileops-runner:<short-sha> \
  .
```

Pass **exactly one** tilelang source: `TILELANG_GIT_SHA=<commit>` compiles that main commit,
`TILELANG_VERSION=<version>` installs that release. The build fails fast if neither is set.
The Dockerfile carries no commit literal — what you pass is the single source of truth, and
the tag records it. Rebuilding the same commit takes a numeric suffix: `:<short-sha>-2`.

**2. Verify it.** The build already ran the GPU-free stack check; this one needs the GPU.

```bash
docker run --rm --gpus all -v "$PWD:/src" \
  ghcr.io/tile-ai/tileops-runner:<short-sha> \
  python /src/scripts/ci/verify_runner_image.py
```

**3. Run the smoke tests** against a checkout.

```bash
docker run --rm --gpus all -v "$PWD:/src" -w /src \
  ghcr.io/tile-ai/tileops-runner:<short-sha> \
  bash -c 'scripts/ci/install_tileops.sh && pytest -m smoke'
```

**4. Push.**

```bash
docker push ghcr.io/tile-ai/tileops-runner:<short-sha>
```

**5. Build and push the dev tag** the same way, with `--target tilelang` and
`-t ghcr.io/tile-ai/tileops-runner:<short-sha>-torch2.13-dev`.

**6. Point the runners at the new tag.** A maintainer task, done outside this repository.
Merging a TileOPs PR only changes the recipe — the live runners keep their existing image
until this happens.

## Bump the tilelang commit

Rebuild with the new `--build-arg` and a new tag; **never edit the Dockerfile**. tilelang is
the last stage, so only its layer recompiles and the bench layers stay cached. Switching
between a release and a main commit changes nothing but the build-arg and the tag.

Then update the tag in the `docker run` line of
[`docs/development.md`](../../docs/development.md#dev-docker-image) — the one place in the repo
that echoes it. The two mentions in `src/tileops/kernels/` record where vendored code and a
verified behaviour came from; they are frozen, so leave them alone.

## Pinning

Two files, both reaching every `pip install` in the build through `PIP_CONSTRAINT`:

| File                          | Written by | Holds                                                                          |
| ----------------------------- | ---------- | ------------------------------------------------------------------------------ |
| `constraints.txt`             | by hand    | The versions the project chooses, and why. Also used by the CPU preflight.     |
| `constraints-runner-lock.txt` | generated  | The full transitive closure of that choice — every package the image installs. |

The lock is what makes a version stick: an install step that would move a version an earlier
step settled on fails the build instead of winning silently. Regenerate it with
`scripts/ci/lock_runner_stack.sh` after changing a version or an install list, and read the diff.

tilelang is absent from the lock by design: it is compiled from source in its own stage, and a
pin would reject that wheel. vLLM depends on an exact tilelang release, so a PyPI tilelang is
present earlier in the build; the source build replaces it with `--force-reinstall`.

## Register a self-hosted runner

`entrypoint.sh` registers an ephemeral runner (one job per container) and deregisters on exit.
It strips `RUNNER_TOKEN` from the environment before the runner starts, so jobs cannot read it.

```bash
docker run -d --gpus all \
  -e RUNNER_URL=https://github.com/tile-ai/TileOPs \
  -e RUNNER_TOKEN=<registration-token> \
  -e RUNNER_LABELS=self-hosted,tile-ops,nightly \
  -v <host-cache-dir>:/ci-cache \
  ghcr.io/tile-ai/tileops-runner:<short-sha>
```

The third label decides which jobs the runner takes: `nightly` for the shared pool, `fork`
for the pool that serves pull requests from forks. A runner registered with any other label
sits idle — no workflow requests one.

Cache env vars (`TILELANG_CACHE_DIR`, `TRITON_CACHE_DIR`, `PIP_CACHE_DIR`, …) point under
`/ci-cache`, pre-created so the container also works unmounted.

## Reference

### Build args

| Arg                | Default                                    | Purpose                                                                   |
| ------------------ | ------------------------------------------ | ------------------------------------------------------------------------- |
| `TILELANG_GIT_SHA` | *(none)*                                   | tilelang commit to shallow-clone and compile (main mode).                 |
| `TILELANG_VERSION` | *(none)*                                   | tilelang PyPI version to `pip install` (release mode).                    |
| `BASE_IMAGE`       | `nvidia/cuda:13.2.1-devel-ubuntu22.04`     | Public CUDA `devel` base (Python 3.12 via deadsnakes).                    |
| `MAX_JOBS`         | `64`                                       | Parallelism for the tilelang / FA2 / FA3 source builds.                   |
| `NVCC_THREADS`     | `4`                                        | Per-`nvcc` threads.                                                       |
| `DEEPGEMM_GIT_SHA` | `c9f8b34dcdacc20aa746b786f983492c51072870` | DeepGEMM commit for the grouped-GEMM benchmark baseline (`v2.1.1.post3`). |
| `RUNNER_VERSION`   | `2.336.0`                                  | GitHub Actions runner version baked into `final`.                         |

### Stages (`--target`)

Build an earlier stage to debug with `--target runtime` (etc.).

| Stage       | Contents                                                                                                                                                                                                          |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `runtime`   | Python 3.12 + torch / torchvision `2.13.0 / 0.28.0 +cu132` + triton `3.7.1` + tilelang build/runtime deps (incl. `apache-tvm-ffi 0.1.11`). No torchaudio: no cu132 build exists. **No tilelang itself.**          |
| `post-fa3`  | `runtime` + pytest / pytest-xdist / ruff / pytest-timeout / py-spy + FlashAttention-3 (built from the `hopper/` source).                                                                                          |
| `fa2`       | `post-fa3` + FlashAttention-2 (`flash-attn 2.8.3.post1`, source-built in its own layer so changes to the bench loop never recompile it).                                                                          |
| `fullstack` | `fa2` + flash-linear-attention `0.5.2` + vLLM `0.27.1` + mamba-ssm `2.3.2.post1` + DeepGEMM `2.1.1.post3`. flashinfer comes in at vLLM's pin (`0.6.16.post3`) — no separate upgrade. sgl-kernel is not installed. |
| `tilelang`  | `fullstack` + the tilelang wheel (`--no-deps`), then the build-time guard. Built **last** so a SHA bump rebuilds only this layer.                                                                                 |
| `final`     | `tilelang` + the GitHub Actions runner (no TileOPs source baked).                                                                                                                                                 |

The `tilelang` stage ends by running `scripts/ci/verify_runtime_stack.py` (GPU-free): the build
fails unless tilelang imports, the installed `apache-tvm-ffi` sits inside the tilelang wheel's
declared range, and torch is still the cu132 build.
