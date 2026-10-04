# CI runner image

Multi-stage image for the self-hosted GPU runner. It bakes a tilelang wheel plus the
test/benchmark stack onto a public CUDA base, so CI never recompiles tilelang per PR.

Built **manually on a GPU host** — never in CI. One package, two tags off the same layers,
both naming the three versions that decide what the image can run:

```
cu<cuda-minor>-torch<major.minor>-tl-<tilelang>[-dev]
```

`<tilelang>` is the release `constraints.txt` pins — the one `pyproject.toml` declares and a
PyPI install of tileops resolves — or, for an image built with `TILELANG_GIT_SHA=<commit>`, that
main commit's short SHA. The release build is the standard one: it is what users get. A SHA
build compiles a main commit in place of the release, for work that needs tilelang ahead of it.

`--target final` takes the bare tag: the CI runners, with the Actions agent. `--target tilelang`
takes the `-dev` suffix: local development, no agent. The `ARG` block in the Dockerfile lists the
other options, with defaults.

## Build and roll out

Needs a GPU host with a CUDA 13.2-capable driver and `nvcc`, and Docker with BuildKit. Run from
the repository root; the context must contain `constraints.txt`, `constraints-runner-lock.txt`,
`scripts/ci/`, and `.github/runner/entrypoint.sh`.

```bash
IMG=ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-<tilelang>

# 1. Build both tags. The second reuses the first's layers.
DOCKER_BUILDKIT=1 docker build -f .github/runner/Dockerfile --target final \
  --provenance=false --sbom=false \
  --build-arg TILEOPS_RUNNER_IMAGE="$IMG" -t "$IMG" .
DOCKER_BUILDKIT=1 docker build -f .github/runner/Dockerfile --target tilelang \
  --provenance=false --sbom=false \
  --build-arg TILEOPS_RUNNER_IMAGE="$IMG-dev" -t "$IMG-dev" .

# 2. Verify both on GPU (the build already ran the GPU-free stack check).
docker run --rm --gpus all -v "$PWD:/src" "$IMG" python /src/scripts/ci/verify_runner_image.py
docker run --rm --gpus all -v "$PWD:/src" "$IMG-dev" python /src/scripts/ci/verify_runner_image.py

# 3. Smoke-test against a checkout.
docker run --rm --gpus all -v "$PWD:/src" -w /src --user root "$IMG" \
  bash -c 'scripts/ci/install_tileops.sh && pytest -m smoke'

# 4. Push.
docker push "$IMG"
docker push "$IMG-dev"
```

**Point the runners at the new tag** — a maintainer task outside this repository. Merging a
TileOPs PR only changes the recipe; the live runners keep their image until this happens.

**Pass the digest to the runner.** The tag is a moving name; the digest is what identifies the
image a night's numbers were produced on, and a container cannot read its own. The host that
starts the runner reads it once and passes it in:

```bash
DIGEST="$(docker image inspect "$IMG" --format '{{index .RepoDigests 0}}')"
docker run ... -e TILEOPS_RUNNER_IMAGE="$IMG" -e TILEOPS_RUNNER_IMAGE_DIGEST="${DIGEST#*@}" ...
```

`scripts/ci/collect_env.py` records it, and the published `meta.json` carries it beside the tag.
Without it the nightly still publishes; the snapshot simply cannot say which image it ran on.

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

Add `--build-arg TILELANG_GIT_SHA=<commit>` to both builds in step 1 for a SHA build.

## Bump tilelang

A release: change the `tilelang` pin in `pyproject.toml` and `constraints.txt` together,
regenerate the lock, and rebuild under a new tag. The constraints reach the first layer, so the
whole image rebuilds. Check the bench baselines the new tilelang affects before rebuilding — vllm
pins an exact tilelang release, and its install is non-fatal, so an unsatisfiable pin drops the
baseline from a green image. A main commit: rebuild with a new `TILELANG_GIT_SHA` and a new tag;
every layer before tilelang is reused. **Never edit the Dockerfile** for either.

Then update the tag in [`docs/development.md`](../../docs/development.md#dev-docker-image) — the
one place in the repo that echoes it. The two mentions in `src/tileops/kernels/` are frozen
records; leave them alone.

## Pinning

Three files. `PIP_CONSTRAINT` hands the latter two to every `pip install` in the build.

| File                          | Written by | Holds                                                    |
| ----------------------------- | ---------- | -------------------------------------------------------- |
| `requirements.in`             | by hand    | Direct requirements, for the lock compile alone.         |
| `constraints.txt`             | by hand    | Chosen versions and why. Also used by the CPU preflight. |
| `constraints-runner-lock.txt` | generated  | The transitive closure of both.                          |

The lock is what makes a version stick: an install that would move a settled version fails the
build instead of winning silently. After changing a version or a requirement, regenerate it with
the `uv pip compile` command in its own header and read the diff. A SHA build installs its
tilelang wheel outside the constraints, so it leaves the locked release and vLLM's pin on it
unsatisfied.

## Runner registration

`entrypoint.sh` registers an ephemeral runner — one job per container — and deregisters on
exit. It strips `RUNNER_TOKEN` from the environment before the runner starts, so jobs cannot
read it.

The image expects a cache directory bind-mounted at `/ci-cache`; `TILELANG_CACHE_DIR`,
`TRITON_CACHE_DIR` and friends point under it and are pre-created, so the container also works
unmounted. Which labels a runner registers with, and how the pools are provisioned, is a
maintainer task outside this repository.
