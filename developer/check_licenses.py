#!/usr/bin/env python3
"""
Check that all dependencies, including extras, have licenses compatible with
BSD-3-Clause.
"""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]

# These packages do not declare a license yet
IGNORED = ["torch-spex", "physical-basis"]


def main() -> int:
    """
    Run licensecheck on ``pyproject.toml`` with all extras.

    :return: The exit code of licensecheck.
    """
    if sys.version_info < (3, 12):
        print("Skipping license check: licensecheck requires Python >= 3.12")
        return 0

    import tomllib

    with open(ROOT / "pyproject.toml", "rb") as fd:
        extras = tomllib.load(fd)["project"]["optional-dependencies"]

    return subprocess.call(
        [sys.executable, "-m", "licensecheck", "--zero", "--show-only-failing"]
        + ["--extras", *extras, "--ignore-packages", *IGNORED],
        cwd=ROOT,
    )


if __name__ == "__main__":
    sys.exit(main())
