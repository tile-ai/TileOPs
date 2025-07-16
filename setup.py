# Copyright (c) Tile-AI.
# Licensed under the MIT License.

import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install


class CustomInstall(install):

    def run(self):
        # Step 1: Install system dependencies (optional, warning: requires sudo)
        # print("Installing system dependencies for TileLang...")
        # subprocess.run(["sudo", "apt-get", "update"])
        # subprocess.run([
        #     "sudo", "apt-get", "install", "-y", "python3-setuptools", "gcc",
        #     "libtinfo-dev", "zlib1g-dev", "build-essential", "cmake",
        #     "libedit-dev", "libxml2-dev"
        # ])

        # Step 2: Install TileLang via pip from local path
        # print("Installing TileLang from submodule...")
        # subprocess.run(["pip", "install", "-e", "3rdparty/tilelang"])

        # Continue normal installation
        install.run(self)


setup(name="top",
      version="0.1.0",
      packages=find_packages(),
      install_requires=["torch", "tilelang"],
      cmdclass={'install': CustomInstall})
