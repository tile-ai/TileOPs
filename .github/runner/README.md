# CI runner image

Multi-stage image for the self-hosted GPU runner. It bakes a tilelang wheel plus the
test/benchmark stack onto a public CUDA base, so CI never recompiles tilelang per PR.

Built **manually on a GPU host** — never in CI. One package, two tags from the same tilelang
commit, both naming the three versions that decide what the image can run:

```
cu<cuda-minor>-torch<major.minor>-tl-<tilelang-short-sha>[-dev]
```

`--target final` takes the bare tag: the CI runners, with the Actions agent. `--target tilelang`
takes the `-dev` suffix: local development, no agent. Rebuilding the same three versions appends
a numeric suffix, `…-tl-<sha>-2`.

The Dockerfile carries no commit literal. Pass **exactly one** tilelang source —
`TILELANG_GIT_SHA=<commit>` compiles that main commit, `TILELANG_VERSION=<version>` installs
that release — and the tag records what you passed. The build fails fast if neither is set.
The `ARG` block in the Dockerfile lists the rest, with defaults.

## Build and roll out

Needs a GPU host with a CUDA 13.2-capable driver and `nvcc`, and Docker with BuildKit. Run from
the repository root; the context must contain `constraints.txt`, `constraints-runner-lock.txt`,
`scripts/ci/`, and `.github/runner/entrypoint.sh`.

```bash
IMG=ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-<short-sha>

# 1. Build. --target tilelang for the dev tag, otherwise the same command.
DOCKER_BUILDKIT=1 docker build -f .github/runner/Dockerfile --target final \
  --provenance=false --sbom=false \
  --build-arg TILELANG_GIT_SHA=<commit> \
  --build-arg TILEOPS_RUNNER_IMAGE="$IMG" -t "$IMG" .

# 2. Verify on GPU (the build already ran the GPU-free stack check).
docker run --rm --gpus all -v "$PWD:/src" "$IMG" python /src/scripts/ci/verify_runner_image.py

# 3. Smoke-test against a checkout.
docker run --rm --gpus all -v "$PWD:/src" -w /src --user root "$IMG" \
  bash -c 'scripts/ci/install_tileops.sh && pytest -m smoke'

# 4. Push, then repeat 1 and 4 with --target tilelang and -t "$IMG-dev".
docker push "$IMG"
```

**Point the runners at the new tag** — a maintainer task outside this repository. Merging a
TileOPs PR only changes the recipe; the live runners keep their image until this happens.

What the flags are for:

- `--user root` — `final` runs as `ci-runner`, which cannot write the editable install's
  `src/tileops.egg-info` into a bind mount owned by the host user. `--target tilelang` sets no
  user and needs no override.
- `--provenance=false --sbom=false` — keeps the tag one manifest. BuildKit's default
  attestations add two untagged versions per tag and nothing reads them.
- `--build-arg TILEOPS_RUNNER_IMAGE` — bakes the tag in, so a run reports which image produced
  it; the registry cannot answer that later. `-e TILEOPS_RUNNER_IMAGE=<tag>` overrides it for an
  image built before this. With neither, the nightly reports the image as unknown.
- `--build-arg FLASH_ATTENTION_FORCE_BUILD=TRUE` — compiles FlashAttention-2 instead of
  taking the prebuilt wheel from its GitHub releases. Slower; reach for it only where that
  download keeps failing.
- `--target runtime`, `--target fa2`, … — build an earlier stage to debug.

## Bump the tilelang commit

Rebuild with the new `--build-arg` and a new tag; **never edit the Dockerfile**. tilelang is the
last stage, so only its layer recompiles. Then update the tag in the `docker run` line of
[`docs/development.md`](../../docs/development.md#dev-docker-image) — the one place in the repo
that echoes it. The two mentions in `src/tileops/kernels/` are frozen records; leave them alone.

## Pinning

Three files. `PIP_CONSTRAINT` hands the latter two to every `pip install` in the build.

| File                          | Written by | Holds                                                    |
| ----------------------------- | ---------- | -------------------------------------------------------- |
| `requirements.in`             | by hand    | Direct requirements, for the lock compile alone.         |
| `constraints.txt`             | by hand    | Chosen versions and why. Also used by the CPU preflight. |
| `constraints-runner-lock.txt` | generated  | The transitive closure of both.                          |

The lock is what makes a version stick: an install that would move a settled version fails the
build instead of winning silently. After changing a version or a requirement, regenerate it with
the `uv pip compile` command in its own header and read the diff. tilelang is excluded there: it
is source-built in its own stage, and vLLM's exact-release dependency on it is overwritten by
`--force-reinstall`.

## Runner registration

`entrypoint.sh` registers an ephemeral runner — one job per container — and deregisters on
exit. It strips `RUNNER_TOKEN` from the environment before the runner starts, so jobs cannot
read it.

The image expects a cache directory bind-mounted at `/ci-cache`; `TILELANG_CACHE_DIR`,
`TRITON_CACHE_DIR` and friends point under it and are pre-created, so the container also works
unmounted. Which labels a runner registers with, and how the pools are provisioned, is a
maintainer task outside this repository.
