"""Setup module for SWV_AnyPeakFinder.
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Get version info from SWV_AnyPeakFinder.py
version = {}
with open(here / "SWV_AnyPeakFinder" / "SWV_AnyPeakFinder.py") as f:
    exec(f.read(), version)

# Setup via pip
setup(
    name="SWV_AnyPeakFinder",
    version=version["__version__"],
    author=version["__author__"],
    author_email=version["__email__"],
    description=version["__description__"],
    license=version["__license__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=version["__url__"],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    keywords="voltammetry, biosensor, science",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    entry_points={
        "gui_scripts": ["swv_anypeakfinder = SWV_AnyPeakFinder.SWV_AnyPeakFinder:main"],
    },
)
