from setuptools import setup, find_packages

setup(
    name="tileops",
    version="0.1.0",
    description="TileOPs kernels for efficient inference",
    author="Tile-AI Corporation",
    license="MIT",
    packages=find_packages("."),
    package_dir={"": "."},
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)