#!/bin/bash
# Development environment setup script for TileOPs.
# This script installs system dependencies and the TileLang submodule
# for local development.
#
# Usage:
#   bash scripts/dev_setup.sh

set -e

echo "==> Installing system dependencies for TileLang..."
sudo apt-get update
sudo apt-get install -y \
    python3-setuptools \
    gcc \
    libtinfo-dev \
    zlib1g-dev \
    build-essential \
    cmake \
    libedit-dev \
    libxml2-dev

echo "==> Installing TileLang from submodule..."
pip install -e 3rdparty/tilelang

echo "==> Installing TileOPs in editable mode with dev dependencies..."
pip install -e '.[dev]'

echo "==> Done! Development environment is ready."
