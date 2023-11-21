import os

from setuptools import find_packages, setup

ROOT = os.path.realpath(os.path.dirname(__file__))


if __name__ == "__main__":
    with open(os.path.join(ROOT, "VERSION")) as fd:
        version = fd.read().strip()

    setup(version=version, packages=find_packages())
