# Nightly CI Host Setup

This document records the current host-side setup for TileOPs nightly CI after migrating the runner service to `tileops_ci`.

## Current topology

- GitHub Actions is the scheduler.
- The self-hosted runner service receives jobs on the host.
- The runner service runs as `tileops_ci`.
- The runner asks the host Docker daemon to create the job container.
- The nightly job container runs the repo workload inside the local image `tileops-runner:latest`.

In other words:

`GitHub workflow -> self-hosted runner (tileops_ci) -> host Docker daemon -> job container`

## What starts automatically on boot

The following services are currently configured for auto-start:

- `docker.service`: `enabled`
- `actions.runner.tile-ai-TileOPs.tile-ops-nightly.service`: `enabled`

That means the nightly runner should come back automatically after a host reboot as long as both services remain enabled.

## Docker on this host

The nightly CI uses the host's system Docker daemon, not a rootless per-user Docker instance.

- Service unit: `docker.service`
- Service file: `/lib/systemd/system/docker.service`
- Start command: `/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock`
- Runtime owner: system service managed by `systemd` as root
- Docker socket: `/var/run/docker.sock`
- Socket owner/mode: `root:docker 660`
- Docker data root: `/var/lib/docker`
- Storage driver: `overlay2`

This means:

- Docker itself is started by `systemd` on the host
- Nightly workflows do not start Docker themselves
- The runner talks to the already-running host daemon through `/var/run/docker.sock`

## Which account launches CI containers

Nightly GitHub Actions jobs are now launched by the self-hosted runner service:

- Service: `actions.runner.tile-ai-TileOPs.tile-ops-nightly.service`
- Runner directory: `/home/tileops_ci/actions-runner`
- Runner user: `tileops_ci`
- ExecStart: `/home/tileops_ci/actions-runner/runsvc.sh`

The important boundary is:

- `root` / `systemd` starts `dockerd`
- `tileops_ci` runs the GitHub runner service
- `tileops_ci` is in the `docker` group and can talk to `/var/run/docker.sock`
- the workflow uses `container:` instead of calling `docker run` directly

## Local image used by nightly

For the current migration stage, nightly uses the local host image:

- `tileops-runner:latest`

This image must already exist on the runner host before the workflow starts. GitHub Actions will not build it automatically for `container:` jobs.

## Host cache mounts

The nightly host cache directories are:

- `/data/ci-cache/pip`
- `/data/ci-cache/wheels`
- `/data/ci-cache/tilelang`
- `/data/ci-cache/triton`

Current ownership for the cache directories used by nightly:

- `tileops_ci:tileops_ci`

The parent directory remains:

- `/data/ci-cache` -> `ga:ga`

This is acceptable as long as the mounted subdirectories themselves stay writable by `tileops_ci`.

## Manual control commands

Check service status:

```bash
systemctl status docker --no-pager
systemctl status actions.runner.tile-ai-TileOPs.tile-ops-nightly.service --no-pager
```

Start services:

```bash
sudo systemctl start docker
sudo systemctl start actions.runner.tile-ai-TileOPs.tile-ops-nightly.service
```

Restart services:

```bash
sudo systemctl restart docker
sudo systemctl restart actions.runner.tile-ai-TileOPs.tile-ops-nightly.service
```

Check auto-start:

```bash
systemctl is-enabled docker
systemctl is-enabled actions.runner.tile-ai-TileOPs.tile-ops-nightly.service
```

Verify the runner user and path:

```bash
systemctl show actions.runner.tile-ai-TileOPs.tile-ops-nightly.service -p User -p WorkingDirectory -p ExecStart
```

Verify the local image is visible to the runner account:

```bash
sudo -u tileops_ci docker images --format '{{.Repository}}:{{.Tag}}' | rg '^tileops-runner:'
```

Verify cache write access:

```bash
sudo -u tileops_ci bash -lc 'touch /data/ci-cache/pip/.write-test && rm /data/ci-cache/pip/.write-test'
```

## Host-specific cautions

- `tileops_ci` being in the `docker` group is a high-privilege setup. Treat it as a dedicated CI account, not a general-purpose login.
- The nightly workflow should not call `docker run` directly anymore once migrated to `container:`.
- Do not mount `/var/run/docker.sock` into the job container.
- Do not introduce extra host bind mounts beyond the cache directories and the workspace managed by GitHub Actions.
- Keep `tileops-runner:latest` present on the host, or the container job will fail to start.
- If the cache directories are recreated, make sure they stay writable by `tileops_ci`.
- The host currently emits `Failed to allocate directory watch: Too many open files` during some `systemctl` operations. It did not block the runner from starting, but it should be cleaned up separately if it becomes frequent.
- The runner copy under `/home/tileops_ci/actions-runner` uses local symlinks for `bin` and `externals`. If the runner is updated or copied again, verify those symlinks still point inside `/home/tileops_ci/actions-runner`.

## Recommended workflow boundary

For nightly CI, keep responsibilities split like this:

- Host: Docker daemon, runner service, local image, GPU runtime, cache directories
- Workflow YAML: `runs-on`, `container`, `checkout`, test commands, artifact upload
- Job container: build, benchmark, test, report generation

This keeps Docker control on the host side and keeps repo logic inside the job container.
