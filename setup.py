import os
import sys
from setuptools import setup


# Control whether to use custom installation logic
# Custom installation is enabled only when:
# 1. TILEOPS_DEV_INSTALL environment variable is set to "1"
# 2. The current command is not for building distribution packages (bdist_wheel, sdist)
# This ensures that custom installation logic is skipped during package building
USE_CUSTOM_INSTALL = (
os.environ.get("TILEOPS_DEV_INSTALL") == "1"
and not any(cmd in sys.argv for cmd in ["bdist_wheel", "sdist"])
)

cmdclass = {}

if USE_CUSTOM_INSTALL:
    from setuptools.command.install import install
    import subprocess
    class CustomInstall(install):

        def run(self):
            # Step 1: Install system dependencies (optional, warning: requires sudo)
            print("Installing system dependencies for TileLang...")
            subprocess.run(["sudo", "apt-get", "update"])
            subprocess.run([
                "sudo", "apt-get", "install", "-y", "python3-setuptools", "gcc",
                "libtinfo-dev", "zlib1g-dev", "build-essential", "cmake",
                "libedit-dev", "libxml2-dev"
            ])

            # Step 2: Install TileLang via pip from local path
            print("Installing TileLang from submodule...")
            subprocess.run(["pip", "install", "-e", "3rdparty/tilelang"])

            # Continue normal installation
            install.run(self)

    cmdclass = {"install": CustomInstall}

setup(name = "tileops",
      version = "0.0.1.dev1",
      cmdclass=cmdclass)
