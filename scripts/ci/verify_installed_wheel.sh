#!/usr/bin/env bash
# Install one tileops wheel into a new venv and test it there, outside the checkout.
#
#   scripts/ci/verify_installed_wheel.sh --wheel <wheel> --deps image|pypi \
#       [--python <interpreter>] [--work-dir <dir>]
#
# --deps image  the venv sees the system site-packages and the wheel goes in with --no-deps,
#               so torch and tilelang are the ones the runner image bakes. No `pip check`:
#               the image's tilelang snapshot leaves its own pins unsatisfied by design.
# --deps pypi   the venv sees nothing else and `pip check` must pass, so the wheel's declared
#               dependencies are what a user would resolve.
#
# A skipped packaging case fails the run: its family would go untested while the run reads
# green. The work dir holds freeze.txt, env.json and packaging.xml.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONSTRAINTS="${REPO_ROOT}/constraints.txt"
PYPI_INDEX="https://pypi.org/simple"
# The device and toolkit README.md names as the verified combination.
REQUIRED_CAPABILITY="9.0"
REQUIRED_CUDA_TOOLKIT="13.2"

WHEEL=""
DEPS=""
PYTHON="python3"
WORK_DIR=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --wheel) WHEEL="$2"; shift 2 ;;
    --deps) DEPS="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [ -z "${WHEEL}" ] || [ ! -f "${WHEEL}" ]; then
  echo "--wheel must name an existing wheel file" >&2
  exit 2
fi
case "${DEPS}" in
  image|pypi) ;;
  *) echo "--deps must be image or pypi" >&2; exit 2 ;;
esac
WHEEL="$(cd "$(dirname "${WHEEL}")" && pwd)/$(basename "${WHEEL}")"
if [ -z "${WORK_DIR}" ]; then
  WORK_DIR="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/verify-wheel.XXXXXX")"
fi
mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
VENV="${WORK_DIR}/venv"

# The runner image points PIP_CONSTRAINT at its own lock, which would override the wheel's
# declared dependencies.
unset PYTHONPATH PIP_CONSTRAINT PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_REQUIRE_VIRTUALENV

echo "=== Create venv (${DEPS}) with ${PYTHON} ==="
venv_args=()
if [ "${DEPS}" = "image" ]; then
  venv_args+=(--system-site-packages)
fi
rm -rf "${VENV}"
"${PYTHON}" -m venv "${venv_args[@]}" "${VENV}"
VPY="${VENV}/bin/python"
pip_install=("${VPY}" -m pip install --disable-pip-version-check --index-url "${PYPI_INDEX}")

echo "=== Install test tools ==="
# Explicit, so the venv never runs a pytest it inherited; workloads/ imports einops.
"${pip_install[@]}" -c "${CONSTRAINTS}" pytest pytest-timeout einops packaging

echo "=== Install $(basename "${WHEEL}") ==="
if [ "${DEPS}" = "image" ]; then
  "${pip_install[@]}" --no-deps "${WHEEL}"
else
  "${pip_install[@]}" "${WHEEL}"
fi

if [ "${DEPS}" = "pypi" ]; then
  echo "=== pip check ==="
  "${VPY}" -m pip check
fi

"${VPY}" -m pip freeze --all > "${WORK_DIR}/freeze.txt"
echo "=== Resolved environment ==="
cat "${WORK_DIR}/freeze.txt"

# Run from an empty directory, so no checkout is on sys.path.
PROBE_DIR="${WORK_DIR}/probe"
mkdir -p "${PROBE_DIR}"
cd "${PROBE_DIR}"

echo "=== Version and import cost ==="
"${VPY}" - "${WHEEL}" <<'PY'
import importlib.metadata
import sys
import zipfile
from email import message_from_bytes

import tileops

with zipfile.ZipFile(sys.argv[1]) as zf:
    meta = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
    wheel_version = message_from_bytes(zf.read(meta))["Version"]
assert tileops.__version__ == importlib.metadata.version("tileops") == wheel_version, (
    tileops.__version__,
    importlib.metadata.version("tileops"),
    wheel_version,
)
assert "torch" not in sys.modules, "reading tileops.__version__ imported torch"
print(f"tileops {tileops.__version__}")
PY

echo "=== Machine, installed paths, resources and family imports ==="
"${VPY}" - "${VENV}" "${WORK_DIR}/env.json" "${WHEEL}" \
  "${REQUIRED_CAPABILITY}" "${REQUIRED_CUDA_TOOLKIT}" <<'PY'
import importlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import sysconfig
import zipfile
from email import message_from_bytes
from importlib import resources
from pathlib import Path

from packaging.specifiers import SpecifierSet

venv, report = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
wheel, want_capability, want_toolkit = Path(sys.argv[3]), sys.argv[4], sys.argv[5]

with zipfile.ZipFile(wheel) as zf:
    name = next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))
    requires_python = message_from_bytes(zf.read(name))["Requires-Python"]
assert platform.python_version() in SpecifierSet(requires_python), (
    f"interpreter {platform.python_version()} is outside Requires-Python {requires_python}"
)
nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout
toolkit = re.search(r"release (\d+\.\d+)", nvcc)
assert toolkit and toolkit.group(1) == want_toolkit, f"CUDA toolkit {nvcc}, not {want_toolkit}"

site = Path(sysconfig.get_paths()["purelib"]).resolve()
assert site.is_relative_to(venv), f"purelib {site} is not inside {venv}"

import tileops
from tileops._csrc import csrc_path
from tileops.manifest import load_manifest

package = Path(tileops.__file__).resolve().parent
spec = Path(str(resources.files("tileops.manifest") / "spec")).resolve()
headers = sorted((package / "csrc").glob("*.h"))
assert headers, f"no csrc headers under {package / 'csrc'}"
header = Path(csrc_path(headers[0].name)).resolve()
for what, path in (("package", package), ("manifest spec", spec), ("csrc header", header)):
    assert path.exists(), f"{what} {path} does not exist"
    assert path.is_relative_to(site), f"{what} {path} is outside the venv's {site}"
assert load_manifest(), "the manifest spec directory holds no ops"

for family in tileops._FAMILIES:
    importlib.import_module(f"tileops.{family}")

import tilelang
import torch

assert torch.cuda.is_available(), "no CUDA device"
capability = "{}.{}".format(*torch.cuda.get_device_capability(0))
assert capability == want_capability, f"device capability {capability}, not {want_capability}"

env = {
    "python": platform.python_version(),
    "tileops": tileops.__version__,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "tilelang": tilelang.__version__,
    "pyyaml": importlib.metadata.version("pyyaml"),
    "device": torch.cuda.get_device_name(0),
    "device_capability": capability,
    "cuda_toolkit": toolkit.group(1),
}
report.write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
print(json.dumps(env, indent=2, sort_keys=True))
PY

echo "=== pytest -m packaging ==="
TEST_DIR="${WORK_DIR}/tests-copy"
rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}"
cp -r "${REPO_ROOT}/tests" "${REPO_ROOT}/workloads" "${TEST_DIR}/"
cp "${REPO_ROOT}/pyproject.toml" "${REPO_ROOT}/conftest.py" "${TEST_DIR}/"
cd "${TEST_DIR}"
REPORT="${WORK_DIR}/packaging.xml"
"${VPY}" -m pytest -q tests/ops -m packaging -p no:cacheprovider \
  --timeout=900 --timeout-method=thread --junit-xml="${REPORT}"

"${VPY}" - "${REPORT}" <<'PY'
import sys
from xml.etree import ElementTree

report = ElementTree.parse(sys.argv[1])
skipped = [
    f"{case.get('classname')}::{case.get('name')}"
    for case in report.iter("testcase")
    if case.find("skipped") is not None
]
if skipped:
    sys.exit("packaging cases skipped, so their family went untested:\n  " + "\n  ".join(skipped))
cases = len(list(report.iter("testcase")))
print(f"{cases} packaging cases ran, none skipped")
PY
