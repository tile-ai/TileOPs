"""Package layout that ``pyproject.toml`` cannot express: native sources under
``src/csrc`` ship inside the wheel as ``tileops/csrc``."""

from setuptools import find_namespace_packages, setup

setup(
    packages=[
        *find_namespace_packages(where="src", include=["tileops", "tileops.*"]),
        "tileops.csrc",
    ],
    package_dir={"": "src", "tileops.csrc": "src/csrc"},
)
