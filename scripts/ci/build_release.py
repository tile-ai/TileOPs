#!/usr/bin/env python3
"""Build the release sdist and the wheel from that sdist, byte-identical across builds.

    SOURCE_DATE_EPOCH=$(git log -1 --format=%ct) \\
        python scripts/ci/build_release.py --source-dir . --out-dir dist --version 0.1.0

The sdist is repacked because setuptools stamps its entries with the checkout's mtimes and
the generated entries with the time the build ran. ``--version`` is the version both
artifacts must declare; ``--pretend-version`` also supplies it to setuptools-scm, for a
commit that carries no tag.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email import message_from_string
from pathlib import Path


def _run(argv: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"+ {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=str(cwd), env=env, check=True)


def _normalize_sdist(sdist: Path, destination: Path, epoch: int) -> Path:
    """Rewrite *sdist* into *destination* with every ordering and timestamp fixed."""
    out = destination / sdist.name
    with tarfile.open(sdist) as source:
        members = sorted(source.getmembers(), key=lambda m: m.name)
        # mtime=0 in the gzip header: gzip records the compression time otherwise.
        with (
            gzip.GzipFile(filename="", mode="wb", fileobj=out.open("wb"), mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as target,
        ):
            for member in members:
                # The pax header carries an mtime of its own, which setting the field below
                # does not reach.
                member.pax_headers = {}
                member.mtime = epoch
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                # One mode per kind, so the builder's umask cannot reach the archive.
                member.mode = 0o755 if member.isdir() or member.mode & 0o111 else 0o644
                target.addfile(member, source.extractfile(member) if member.isreg() else None)
    return out


def _metadata_version(archive: Path) -> str:
    """The version the built artifact declares, read from its own metadata."""
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
            text = zf.read(name).decode("utf-8")
    else:
        with tarfile.open(archive) as tf:
            name = next(n for n in tf.getnames() if n.count("/") == 1 and n.endswith("/PKG-INFO"))
            text = tf.extractfile(name).read().decode("utf-8")
    return message_from_string(text)["Version"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--version", required=True, help="the version both artifacts must declare")
    parser.add_argument(
        "--pretend-version",
        action="store_true",
        help="pass --version to setuptools-scm instead of reading the source tree's tag",
    )
    args = parser.parse_args(argv)

    epoch = os.environ.get("SOURCE_DATE_EPOCH", "")
    if not epoch.isdigit():
        print("SOURCE_DATE_EPOCH must be set to the commit timestamp", file=sys.stderr)
        return 1

    env = dict(os.environ)
    if args.pretend_version:
        env["SETUPTOOLS_SCM_PRETEND_VERSION"] = args.version
    else:
        env.pop("SETUPTOOLS_SCM_PRETEND_VERSION", None)

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    staging = out_dir.parent / f"{out_dir.name}-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    _run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(staging)],
        args.source_dir.resolve(),
        env,
    )
    sdist = _normalize_sdist(next(staging.glob("*.tar.gz")), out_dir, int(epoch))

    unpacked = staging / "unpacked"
    with tarfile.open(sdist) as tf:
        tf.extractall(unpacked, filter="data")
    root = next(p for p in unpacked.iterdir() if p.is_dir())
    _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir)], root, env)

    wheels = sorted(out_dir.glob("*.whl"))
    sdists = sorted(out_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        print(
            f"expected one wheel and one sdist, got {len(wheels)} and {len(sdists)}",
            file=sys.stderr,
        )
        return 1
    for artifact in (*sdists, *wheels):
        declared = _metadata_version(artifact)
        if declared != args.version:
            print(
                f"{artifact.name} declares version {declared}, not {args.version}", file=sys.stderr
            )
            return 1
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        print(f"{digest}  {artifact.name}")
    shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
